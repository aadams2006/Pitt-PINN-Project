from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

MODEL_NAME = "rfr"


def build_model(_: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        random_state=42,
    )
