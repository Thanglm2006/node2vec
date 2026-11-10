import numpy as np
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score

EMBEDDING_FILES = {
    "custom": "karate_embeddings_custom.npy",
    "original": "karate_embeddings_original.npy",
}

# Ground-truth labels
G = nx.karate_club_graph()
node_list = list(G.nodes())
node_ids = np.load("karate_node_ids.npy", allow_pickle=True).tolist()
if node_ids != node_list:
    raise ValueError("Embedding node IDs do not match the graph node order.")

y = np.array([
    0 if G.nodes[node]["club"] == "Mr. Hi" else 1
    for node in node_list
])

# Repeated evaluation
splitter = StratifiedShuffleSplit(
    n_splits=30,
    test_size=0.3,
    random_state=42
)

for name, path in EMBEDDING_FILES.items():
    X = np.load(path)
    if X.shape[0] != len(y):
        raise ValueError(
            f"{name} embeddings have {X.shape[0]} rows, expected {len(y)}."
        )

    accuracies = []
    f1_scores = []

    for train_idx, test_idx in splitter.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        accuracies.append(
            accuracy_score(y_test, predictions)
        )

        f1_scores.append(
            f1_score(y_test, predictions)
        )

    print(
        f"{name.capitalize():<9} | "
        f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f} | "
        f"F1: {np.mean(f1_scores):.4f} +/- {np.std(f1_scores):.4f}"
    )

