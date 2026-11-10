from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from Node2vecWalk import random_walk
from Preprocess import preprocess_transition_probs
from Skip_gram import NegativeSamplingLoss, SkipGram


class Node2Vec:
    def __init__(self, G,p,q,dimension,walks_per_node,walk_length, context_size):
        self.G = G
        self.p = p
        self.q = q
        self.dimension = dimension
        self.walks_per_node = walks_per_node
        self.walk_length = walk_length
        self.context_size = context_size
        self.model = None
        self.history = []
        self.node_ids = list(G.nodes())

    def create_walks(self):
        alias_edges = preprocess_transition_probs(self.G,self.p,self.q)
        walks=[]
        for i in range(self.walks_per_node):
            for node in self.G.nodes():
                walk=random_walk(self.G,node,self.walk_length,alias_edges)
                walks.append(walk)
        return walks

    def positive_samples(self, walks):
        positive_samples = []

        for walk in walks:
            for i, source in enumerate(walk):
                # Lấy tối đa context_size node phía sau source
                end = min(i + self.context_size + 1, len(walk))

                contexts = walk[i + 1:end]

                for context in contexts:
                    positive_samples.append((source, context))

        return positive_samples



    def negative_samples(self, walks, positive_pairs, num_negative=5):

        # Tính tần suất xuất hiện của từng node trong corpus
        node_counts = Counter(
            node
            for walk in walks
            for node in walk
        )

        nodes = list(node_counts.keys())

        # Phân phối noise: frequency ^ 0.75
        frequencies = np.array([
            node_counts[node] ** 0.75
            for node in nodes
        ], dtype=np.float64)

        probs = frequencies / frequencies.sum()

        negative_samples = []

        for source, positive_context in positive_pairs:

            # Không lấy source và positive context hiện tại
            valid_mask = np.array([
                node != source and node != positive_context
                for node in nodes
            ])

            candidates = np.array(nodes)[valid_mask]
            candidate_probs = probs[valid_mask]

            if len(candidates) == 0:
                continue

            candidate_probs /= candidate_probs.sum()

            negatives = np.random.choice(
                candidates,
                size=num_negative,
                replace=True,
                p=candidate_probs
            )

            for negative in negatives:
                negative_samples.append((source, negative))

        return negative_samples

    def fit(
            self,
            num_negative=5,
            batch_size=256,
            epochs=1,
            lr=0.001,
            device=None,
    ):
        """Generate and train Node2Vec embeddings for the configured graph.

        The returned rows follow ``self.node_ids``. The trained Skip-gram
        model and per-epoch losses are also available as ``self.model`` and
        ``self.history``.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        walks = self.create_walks()
        positive_pairs = self.positive_samples(walks)
        negative_pairs = self.negative_samples(
            walks=walks,
            positive_pairs=positive_pairs,
            num_negative=num_negative,
        )

        expected_negative_count = len(positive_pairs) * num_negative
        if len(negative_pairs) != expected_negative_count:
            raise ValueError(
                "Unexpected number of negative pairs: "
                f"expected {expected_negative_count}, got {len(negative_pairs)}. "
                "Check whether some source nodes have no valid negative candidates."
            )

        node2id = {node: idx for idx, node in enumerate(self.node_ids)}
        source_ids = torch.tensor(
            [node2id[source] for source, _ in positive_pairs],
            dtype=torch.long,
        )
        positive_context_ids = torch.tensor(
            [node2id[context] for _, context in positive_pairs],
            dtype=torch.long,
        )
        negative_context_ids = torch.tensor(
            [
                [node2id[negative] for _, negative in negative_pairs[
                    i * num_negative : (i + 1) * num_negative
                ]]
                for i in range(len(positive_pairs))
            ],
            dtype=torch.long,
        )

        dataset = TensorDataset(
            source_ids,
            positive_context_ids,
            negative_context_ids,
        )
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.model = SkipGram(
            num_nodes=self.G.number_of_nodes(),
            embedding_dim=self.dimension,
        )
        self.model, self.history = self.train_skipgram(
            model=self.model,
            train_loader=train_loader,
            epochs=epochs,
            lr=lr,
            device=device,
        )

        return self.model.get_embeddings().detach().cpu().numpy()


    @staticmethod
    def train_skipgram(
            model,
            train_loader,
            epochs=5,
            lr=0.001,
            device="cpu"
    ):
        model = model.to(device)

        criterion = NegativeSamplingLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        history = []

        for epoch in range(epochs):
            model.train()

            total_loss = 0.0
            total_samples = 0

            # Bọc train_loader bằng tqdm để hiển thị thanh tiến trình
            progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]")

            for sources, positive_contexts, negative_contexts in progress_bar:
                sources = sources.to(device)
                positive_contexts = positive_contexts.to(device)
                negative_contexts = negative_contexts.to(device)

                optimizer.zero_grad()

                # Positive pairs: (B,)
                positive_logits = model(
                    sources,
                    positive_contexts
                )

                # Negative pairs: (B, K)
                negative_logits = model(
                    sources.unsqueeze(1),
                    negative_contexts
                )

                loss = criterion(
                    positive_logits,
                    negative_logits
                )

                loss.backward()
                optimizer.step()

                batch_size = sources.size(0)

                total_loss += loss.item() * batch_size
                total_samples += batch_size

                # Cập nhật loss hiện tại lên thanh tiến trình
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

            epoch_loss = total_loss / total_samples
            history.append(epoch_loss)

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Average Loss: {epoch_loss:.4f}"
            )

        return model, history
