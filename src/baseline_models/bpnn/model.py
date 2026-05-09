from __future__ import annotations

from sklearn.neural_network import MLPRegressor

MODEL_NAME = "bpnn"


def build_model(n_train_samples: int) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=max(2000, 200 * max(1, n_train_samples)),
        random_state=42,
    )
