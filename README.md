# Node2Vec Embedding Comparison

This project compares a custom Node2Vec implementation with the reference
`node2vec` Python package on the Zachary Karate Club graph.

## Implementations

- `Node2Vec.py` contains the custom implementation, including biased random
  walks, positive and negative sampling, Skip-gram training, and a `fit()`
  method for generating embeddings.
- `main.py` trains both implementations with the same graph and comparable
  hyperparameters.
- `test.py` evaluates both embedding sets with repeated logistic-regression
  classification using the known Karate Club labels.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install numpy networkx torch tqdm node2vec scikit-learn
```

The `node2vec` package uses gensim internally. Install it explicitly if your
package manager does not install it automatically:

```bash
pip install gensim
```

## Usage

Generate embeddings with both implementations:

```bash
python main.py
```

This creates:

- `karate_embeddings_custom.npy`
- `karate_embeddings_original.npy`
- `karate_node_ids.npy`

Evaluate both embedding sets:

```bash
python test.py
```

The evaluator uses 30 stratified train/test splits and reports mean accuracy
and F1 score with standard deviations.

## Example Results

Using the current configuration:

```text
Custom    | Accuracy: 0.9788 +/- 0.0385 | F1: 0.9807 +/- 0.0352
Original  | Accuracy: 0.9788 +/- 0.0385 | F1: 0.9797 +/- 0.0369
```

## Main Hyperparameters

| Parameter | Value |
| --- | ---: |
| Embedding dimension | 128 |
| Walks per node | 10 |
| Walk length | 80 |
| Context size | 5 |
| Negative samples | 5 |
| Return parameter `p` | 1.0 |
| In-out parameter `q` | 1.0 |

Generated NumPy arrays, model outputs, Python caches, virtual environments,
and IDE files are excluded by `.gitignore`.
