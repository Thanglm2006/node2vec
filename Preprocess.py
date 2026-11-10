import numpy as np
import networkx as nx
def preprocess_transition_probs(G, p, q):
    alias_edges = {}
    neighbors = {
        v: list(G.neighbors(v))
        for v in G.nodes()
    }

    # Set giúp kiểm tra adjacency nhanh
    neighbor_sets = {
        v: set(neighbors[v])
        for v in G.nodes()
    }

    # Mỗi cạnh vô hướng tạo ra 2 trạng thái có hướng
    for u, v in G.edges():

        for t, curr in [(u, v), (v, u)]:

            candidates = neighbors[curr]

            # Edge weights từ curr -> candidate
            weights = np.array([
                G[curr][x].get("weight", 1.0)
                for x in candidates
            ])

            # Khoảng cách t -> candidate
            is_return = np.array([
                x == t for x in candidates
            ])

            is_close = np.array([
                x in neighbor_sets[t]
                for x in candidates
            ])

            # Tính alpha
            alpha = np.where(
                is_return,
                1.0 / p,
                np.where(is_close, 1.0, 1.0 / q)
            )

            # Modified transition weights
            probs = alpha * weights

            # Normalize
            probs = probs / probs.sum()

            alias_edges[(t, curr)] = {
                "candidates": candidates,
                "probs": probs
            }

    return alias_edges