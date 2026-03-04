from dataclasses import dataclass


@dataclass
class DataConfig:
    test_size: float = 0.2
    random_state: int = 42
    ood_quantile: float = 0.85


@dataclass
class PINNConfig:
    hidden_dim: int = 64
    depth: int = 3
    lr: float = 1e-3
    epochs: int = 1500
    lambda_phys: float = 1.0
    lambda_smooth: float = 1e-2
    lambda_l2: float = 1e-5
    monotonic_margin: float = 0.0


@dataclass
class BaselineConfig:
    random_state: int = 42
