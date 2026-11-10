import random
import networkx as nx
import numpy as np
import torch
from node2vec import Node2Vec as OriginalNode2Vec

from Node2Vec import Node2Vec


def main():
    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Load Karate Club graph
    G = nx.karate_club_graph()
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Directed: {G.is_directed()}")

    # 2. Node2Vec configuration
    p = 1.0
    q = 1.0
    embedding_dim = 128
    walks_per_node = 10
    walk_length = 80
    context_size = 5
    num_negative = 5

    node2vec = Node2Vec(
        G=G,
        p=p,
        q=q,
        dimension=embedding_dim,
        walks_per_node=walks_per_node,
        walk_length=walk_length,
        context_size=context_size,
    )

    # 3. Generate and train custom embeddings.
    print("\nGenerating and training custom embeddings...")
    custom_embeddings = node2vec.fit(
        num_negative=num_negative,
        batch_size=256,
        epochs=1,
        lr=0.001,
        device=device,
    )

    print("\nCustom training finished.")
    print(f"Embedding shape: {custom_embeddings.shape}")
    print(f"Loss history: {node2vec.history}")

    # 4. Generate embeddings with the reference node2vec package.
    print("\nGenerating and training original Node2Vec embeddings...")
    original_node2vec = OriginalNode2Vec(
        graph=G,
        dimensions=embedding_dim,
        walk_length=walk_length,
        num_walks=walks_per_node,
        p=p,
        q=q,
        workers=1,
        quiet=True,
        seed=seed,
    )
    original_model = original_node2vec.fit(
        window=context_size,
        min_count=1,
        batch_words=256,
        epochs=1,
        seed=seed,
    )
    original_embeddings = np.asarray(
        [original_model.wv.get_vector(str(node)) for node in G.nodes()]
    )

    print(f"Original embedding shape: {original_embeddings.shape}")

    # Save both embeddings and the shared node-ID mapping for evaluation.
    np.save("karate_embeddings_custom.npy", custom_embeddings)
    np.save("karate_embeddings_original.npy", original_embeddings)

    # Keep the original node labels in the same order as embedding rows.
    # Karate Club labels are integers, so this can be saved directly.
    np.save("karate_node_ids.npy", np.asarray(list(G.nodes()), dtype=object))

    print("\nSaved:")
    print("- karate_embeddings_custom.npy")
    print("- karate_embeddings_original.npy")
    print("- karate_node_ids.npy")


if __name__ == "__main__":
    main()
