import numpy as np
import torch
from tqdm import tqdm

from Node2vecWalk import random_walk
from Preprocess import preprocess_transition_probs
from Skip_gram_optim import (
    CommunityHead,
    NegativeSamplingLoss,
    SkipGram,
    collapse_regularization,
    soft_modularity,
)


class Node2Vec:
    def __init__(self, G, p, q, dimension, walks_per_node, walk_length,
                 context_size, num_communities=10):
        self.G = G
        self.p = p
        self.q = q
        self.dimension = dimension
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size
        self.num_communities = num_communities

        self.model = None
        self.head = None
        self.history = []
        self.node_ids = list(G.nodes())
        self.node2id = {node: idx for idx, node in enumerate(self.node_ids)}

    # ------------------------------------------------------------------
    # 1. Random walk (giữ nguyên)
    # ------------------------------------------------------------------
    def create_walks(self):
        alias_edges = preprocess_transition_probs(self.G, self.p, self.q)
        walks = []
        for _ in range(self.walks_per_node):
            for node in self.G.nodes():
                walks.append(random_walk(self.G, node, self.walk_length, alias_edges))
        return walks

    def walks_to_ids(self, walks):
        """list[list[node]] -> ndarray (num_walks, walk_length + 1) toàn index."""
        return np.array(
            [[self.node2id[n] for n in walk] for walk in walks],
            dtype=np.int64,
        )

    # ------------------------------------------------------------------
    # 2. Positive samples: vector hóa theo offset (thay vòng for 3 tầng)
    # ------------------------------------------------------------------
    def positive_samples(self, walk_ids):
        sources, contexts = [], []
        for offset in range(1, self.context_size + 1):
            if offset >= walk_ids.shape[1]:
                break
            sources.append(walk_ids[:, :-offset].ravel())
            contexts.append(walk_ids[:, offset:].ravel())
        return np.concatenate(sources), np.concatenate(contexts)

    # ------------------------------------------------------------------
    # 3. Negative samples: O(P*K) thay vì O(P*N)
    #    Lấy mẫu một lần từ phân phối noise freq^0.75, sau đó chỉ lấy lại
    #    những vị trí bị trùng source / positive context.
    # ------------------------------------------------------------------
    def negative_samples(self, walk_ids, sources, pos_contexts, num_negative=5,
                         max_resample=20):
        n = len(self.node_ids)

        counts = np.bincount(walk_ids.ravel(), minlength=n).astype(np.float64)
        probs = counts ** 0.75
        probs /= probs.sum()

        num_pairs = len(sources)
        negatives = np.random.choice(n, size=(num_pairs, num_negative), p=probs)

        for _ in range(max_resample):
            bad = (negatives == sources[:, None]) | (negatives == pos_contexts[:, None])
            num_bad = int(bad.sum())
            if num_bad == 0:
                break
            negatives[bad] = np.random.choice(n, size=num_bad, p=probs)

        return negatives  # (P, K)

    # ------------------------------------------------------------------
    # 4. Tensor đồ thị cho modularity: dựng MỘT LẦN, đặt sẵn lên device
    # ------------------------------------------------------------------
    def build_graph_tensors(self, device):
        src, dst, w = [], [], []
        for u, v, data in self.G.edges(data=True):
            weight = data.get("weight", 1.0)
            iu, iv = self.node2id[u], self.node2id[v]
            src += [iu, iv]
            dst += [iv, iu]
            w += [weight, weight]

        src = torch.tensor(src, dtype=torch.long, device=device)
        dst = torch.tensor(dst, dtype=torch.long, device=device)
        w = torch.tensor(w, dtype=torch.float32, device=device)
        degree = torch.zeros(len(self.node_ids), device=device).scatter_add_(0, src, w)
        return src, dst, w, degree

    # ------------------------------------------------------------------
    # 5. Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        num_negative=5,
        batch_size=256,
        epochs=5,
        lr=0.001,
        lam=1.0,             # trọng số của Q_soft
        beta=0.1,            # trọng số collapse regularization
        mod_every=20,        # chỉ tính modularity mỗi mod_every batch
        edge_batch=None,     # số cạnh lấy mẫu mỗi lần tính Q (None = toàn bộ)
        warmup_epochs=1,     # các epoch đầu chỉ train Skip-gram (lam = 0)
        device=None,
    ):
        """L = L_NS - lam * Q_soft + beta * L_collapse

        Trả về embedding (N, d) theo thứ tự self.node_ids.
        Cộng đồng: self.get_communities().
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- dữ liệu huấn luyện (toàn bộ dạng ndarray / tensor) ----
        walks = self.create_walks()
        walk_ids = self.walks_to_ids(walks)
        sources, pos_contexts = self.positive_samples(walk_ids)
        negatives = self.negative_samples(
            walk_ids, sources, pos_contexts, num_negative=num_negative
        )

        # Đặt thẳng lên device, tránh DataLoader/TensorDataset (collate từng mẫu)
        sources = torch.from_numpy(sources).to(device)
        pos_contexts = torch.from_numpy(pos_contexts).to(device)
        negatives = torch.from_numpy(negatives).to(device)
        graph = self.build_graph_tensors(device)

        # ---- model ----
        self.model = SkipGram(len(self.node_ids), self.dimension).to(device)
        self.head = CommunityHead(self.dimension, self.num_communities).to(device)

        self.history = self.train_joint(
            sources, pos_contexts, negatives, graph,
            epochs=epochs, batch_size=batch_size, lr=lr,
            lam=lam, beta=beta, mod_every=mod_every,
            edge_batch=edge_batch, warmup_epochs=warmup_epochs,
            device=device,
        )
        return self.model.get_embeddings().detach().cpu().numpy()

    # ------------------------------------------------------------------
    # 6. Joint training
    # ------------------------------------------------------------------
    def train_joint(self, sources, pos_contexts, negatives, graph, epochs,
                    batch_size, lr, lam, beta, mod_every, edge_batch,
                    warmup_epochs, device):
        src, dst, w, degree = graph
        model, head = self.model, self.head
        criterion = NegativeSamplingLoss()
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(head.parameters()), lr=lr
        )

        num_pairs = sources.size(0)
        num_edges = src.numel()
        history = []

        for epoch in range(epochs):
            model.train()
            head.train()
            use_modularity = epoch >= warmup_epochs and lam > 0

            perm = torch.randperm(num_pairs, device=device)
            total_ns, total_samples = 0.0, 0
            last_q = float("nan")

            progress = tqdm(
                range(0, num_pairs, batch_size),
                desc=f"Epoch [{epoch + 1}/{epochs}]",
            )
            for step, start in enumerate(progress):
                idx = perm[start:start + batch_size]
                s, p, n = sources[idx], pos_contexts[idx], negatives[idx]

                optimizer.zero_grad()

                # --- Skip-gram + Negative Sampling (mỗi batch) ---
                l_ns = criterion(model(s, p), model(s.unsqueeze(1), n))
                loss = l_ns

                # --- Modularity (chỉ mỗi mod_every batch) ---
                if use_modularity and step % mod_every == 0:
                    S = head(model.get_embeddings())  # (N, K)

                    edge_idx = None
                    if edge_batch is not None and edge_batch < num_edges:
                        edge_idx = torch.randint(
                            0, num_edges, (edge_batch,), device=device
                        )

                    q = soft_modularity(S, src, dst, w, degree, edge_idx)
                    l_col = collapse_regularization(S)
                    loss = loss - lam * q + beta * l_col
                    last_q = q.item()

                loss.backward()
                optimizer.step()

                bs = s.size(0)
                total_ns += l_ns.item() * bs
                total_samples += bs
                progress.set_postfix(ns=f"{l_ns.item():.4f}", Q=f"{last_q:.4f}")

            epoch_ns = total_ns / total_samples
            history.append({"ns_loss": epoch_ns, "modularity": last_q})
            print(
                f"Epoch [{epoch + 1}/{epochs}] NS loss: {epoch_ns:.4f} "
                f"| Q_soft: {last_q:.4f}"
            )

        return history

    # ------------------------------------------------------------------
    # 7. Kết quả cộng đồng
    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_communities(self):
        self.model.eval()
        self.head.eval()
        S = self.head(self.model.get_embeddings())
        labels = S.argmax(dim=-1).cpu().numpy()
        return {node: int(labels[i]) for i, node in enumerate(self.node_ids)}