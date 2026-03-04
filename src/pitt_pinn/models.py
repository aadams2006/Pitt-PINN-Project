from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: Optional[float]


def build_baseline_models(n_train_samples: int) -> dict:
    n_neighbors = max(1, min(5, n_train_samples))
    return {
        "linear_regression": LinearRegression(),
        "knn_regression": KNeighborsRegressor(n_neighbors=n_neighbors),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
        "mlp_regression": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, random_state=42),
    }


class PINNRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, depth: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        d_in = input_dim
        for _ in range(depth):
            layers.extend([nn.Linear(d_in, hidden_dim), nn.Tanh()])
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    r2 = float(r2_score(y_true, y_pred))
    if not np.isfinite(r2):
        r2 = None
    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=r2,
    )
