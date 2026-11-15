import random
import networkx as nx
import numpy as np
import torch
from Node2vec_optim import Node2Vec
from collections import defaultdict
from pathlib import Path



def leiden_partition(G, seed=42):
    """Run Leiden through python-igraph and return NetworkX-style groups."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as exc:
        raise RuntimeError(
            "Leiden requires python-igraph and leidenalg. "
            "Install them with: pip install igraph leidenalg"
        ) from exc

    node_ids = list(G.nodes())
    node2id = {node: idx for idx, node in enumerate(node_ids)}
    edges = [(node2id[u], node2id[v]) for u, v in G.edges()]
    weights = [data.get("weight", 1.0) for _, _, data in G.edges(data=True)]

    ig_graph = ig.Graph(
        n=len(node_ids),
        edges=edges,
        directed=G.is_directed(),
    )
    partition = leidenalg.find_partition(
        ig_graph,
        leidenalg.ModularityVertexPartition,
        weights=weights,
        seed=seed,
    )

    return [{node_ids[idx] for idx in community} for community in partition]


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

    # data_dir = Path(__file__).resolve().parent / "Data" / "football"
    # txt = (data_dir / "football.txt").read_text(encoding="utf-8")
    # gml = (data_dir / "football.gml").read_text(encoding="utf-8")

    # # The dataset has a metadata line before the actual GML graph.
    # gml_lines = gml.splitlines()
    # if gml_lines and gml_lines[0].startswith("Creator "):
    #     gml_lines = gml_lines[1:]
    # G = nx.parse_gml(gml_lines)
    G= nx.karate_club_graph()  # small graph for testing
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Node2Vec configuration
    p = 1.0
    q = 1.0
    embedding_dim = 128
    walks_per_node = 10
    walk_length = 80
    context_size = 5
    num_negative = 5
    lam = 1.5
    lr = 0.0005

    n2v = Node2Vec(G, p=1.0, q=1.0, dimension=128, walks_per_node=10,
                walk_length=80, context_size=5, num_communities=G.number_of_nodes())

    print("\nStarting custom training...")
    emb = n2v.fit(epochs=20, lam=1.5, beta=0.1,batch_size=256, lr=0.0005,
                mod_every=5, edge_batch=50_000, warmup_epochs=4, device=device)
    communities = n2v.get_communities()   # {node: community_id}

    print("\nCustom training finished.")
    print(f"Embedding shape: {emb.shape}")
    print(f"Communities: {communities.values()}")
    print(f"Community counts: {np.bincount(list(communities.values()))}")


    groups = defaultdict(set)
    for node, c in communities.items():
        groups[c].add(node)

    custom_partition = list(groups.values())
    custom_q = nx.community.modularity(
        G, custom_partition, weight="weight", resolution=1
    )

    # Chạy Leiden trên chính graph và chấm điểm bằng cùng hàm modularity.
    leiden_groups = leiden_partition(G, seed=seed)
    leiden_q = nx.community.modularity(
        G, leiden_groups, weight="weight", resolution=1
    )

    custom_sizes = sorted((len(group) for group in custom_partition), reverse=True)
    leiden_sizes = sorted((len(group) for group in leiden_groups), reverse=True)

    print("\nCommunity comparison")
    print(f"Custom: {len(custom_partition)} communities")
    print(f"Custom sizes: {custom_sizes}")
    print(f"Custom modularity Q: {custom_q:.6f}")
    print(f"Leiden: {len(leiden_groups)} communities")
    print(f"Leiden sizes: {leiden_sizes}")
    print(f"Leiden modularity Q: {leiden_q:.6f}")
    print(f"Delta Q (custom - Leiden): {custom_q - leiden_q:+.6f}")
if __name__ == "__main__":
    main()
