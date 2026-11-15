import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()

        self.source_embedding = nn.Embedding(num_nodes, embedding_dim)
        self.context_embedding = nn.Embedding(num_nodes, embedding_dim)
        
        nn.init.uniform_(
            self.source_embedding.weight,
            -0.5 / embedding_dim,
            0.5 / embedding_dim,
        )
        nn.init.zeros_(self.context_embedding.weight)

    def forward(self, sources, contexts):
        """
        sources:  (B,) hoặc (B, 1)
        contexts: (B,) hoặc (B, K)
        """
        source_vectors = self.source_embedding(sources)
        context_vectors = self.context_embedding(contexts)
        return (source_vectors * context_vectors).sum(dim=-1)

    def get_embeddings(self):
        return self.source_embedding.weight


class NegativeSamplingLoss(nn.Module):
    def forward(self, positive_logits, negative_logits):
        positive_loss = -F.logsigmoid(positive_logits)
        negative_loss = -F.logsigmoid(-negative_logits).sum(dim=1)
        return (positive_loss + negative_loss).mean()


# ---------------------------------------------------------------------------
# Community detection head
# ---------------------------------------------------------------------------
class CommunityHead(nn.Module):
    """Z (N, d) -> soft assignment S (N, K)."""

    def __init__(self, embedding_dim, num_communities, temperature=1.0):
        super().__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(embedding_dim, num_communities)
        self.temperature = temperature
    

    def forward(self, z):
        z = self.dropout(z)
        return F.softmax(self.linear(z) / self.temperature, dim=-1)


def soft_modularity(S, src, dst, edge_w, degree, edge_idx=None):
    """
    Q_soft = 1/(2m) * [ Tr(S^T A S) - ||S^T k||^2 / (2m) ]
    Không tạo ma trận B = A - kk^T/2m (N x N).

    S:       (N, K)
    src,dst: (E2,) cạnh theo CẢ HAI chiều
    edge_w:  (E2,)
    degree:  (N,) weighted degree, sum = 2m
    edge_idx: (M,) chỉ số các cạnh được lấy mẫu. Nếu có, trace term được
              ước lượng không chệch bằng cách nhân E2 / M.
    """
    two_m = degree.sum()

    if edge_idx is None:
        s, d, w = src, dst, edge_w
        scale = 1.0
    else:
        s, d, w = src[edge_idx], dst[edge_idx], edge_w[edge_idx]
        scale = src.numel() / edge_idx.numel()

    trace_term = (S[s] * S[d]).sum(dim=-1).mul(w).sum() * scale  # Tr(S^T A S)
    ks = S.t() @ degree                                         # (K,)
    null_term = (ks ** 2).sum() / two_m                         # Tr(S^T kk^T S)/2m
    return (trace_term - null_term) / two_m


def collapse_regularization(S):
    """Phạt khi các node dồn vào ít cụm (giống DMoN). Bằng 0 khi cụm cân bằng."""
    n, k = S.shape
    return (k ** 0.5 / n) * torch.linalg.norm(S.sum(dim=0)) - 1.0