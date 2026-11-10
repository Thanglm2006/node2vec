
import torch.nn as nn
import torch

import torch.nn.functional as F

class SkipGram(nn.Module):
    def __init__(self, num_nodes, embedding_dim):
        super().__init__()

        self.source_embedding = nn.Embedding(
            num_nodes,
            embedding_dim
        )

        self.context_embedding = nn.Embedding(
            num_nodes,
            embedding_dim
        )

        # Khởi tạo embedding
        nn.init.uniform_(
            self.source_embedding.weight,
            -0.5 / embedding_dim,
            0.5 / embedding_dim
        )

        nn.init.zeros_(
            self.context_embedding.weight
        )

    def forward(self, sources, contexts):
        """
        sources:  (batch_size,)
        contexts: (batch_size,) hoặc (batch_size, num_negative)
        """

        source_vectors = self.source_embedding(sources)
        context_vectors = self.context_embedding(contexts)

        # Dot product
        logits = (source_vectors * context_vectors).sum(dim=-1)

        return logits

    def get_embeddings(self):
        return self.source_embedding.weight




class NegativeSamplingLoss(nn.Module):
    def forward(self, positive_logits, negative_logits):

        positive_loss = -F.logsigmoid(positive_logits)

        negative_loss = -F.logsigmoid(-negative_logits).sum(dim=1)

        return (positive_loss + negative_loss).mean()

