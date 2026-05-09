from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CONCENTRATION_COLUMN = "pdms_concentration"
UNCOATED_LAYER_COLUMN = "uncoated_layer_thickness"
TOTAL_THICKNESS_COLUMN = "total_film_thickness"
TARGET_COLUMN = "bonded_film_thickness"

COLUMN_ALIASES = {
    "Concentration (g/mL)": CONCENTRATION_COLUMN,
    "Uncoated Layer (nm)": UNCOATED_LAYER_COLUMN,
    "Total Thickness (nm)": TOTAL_THICKNESS_COLUMN,
    "Bonded Thickness (nm)": TARGET_COLUMN,
}

REQUIRED_COLUMNS = [
    CONCENTRATION_COLUMN,
    UNCOATED_LAYER_COLUMN,
    TOTAL_THICKNESS_COLUMN,
    TARGET_COLUMN,
]
FEATURE_COLUMNS = [
    CONCENTRATION_COLUMN,
    UNCOATED_LAYER_COLUMN,
    TOTAL_THICKNESS_COLUMN,
]


@dataclass
class PreparedData:
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    x_scaler: StandardScaler
    y_scaler: StandardScaler


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={col: COLUMN_ALIASES.get(col, col) for col in df.columns})


def load_dataset(path: str) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError("Required columns contain missing values.")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in FEATURE_COLUMNS if col in df.columns]


def prepare_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> PreparedData:
    x = df[feature_columns(df)]
    y = df[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train_scaled = pd.DataFrame(
        x_scaler.fit_transform(x_train),
        columns=x_train.columns,
        index=x_train.index,
    )
    x_test_scaled = pd.DataFrame(
        x_scaler.transform(x_test),
        columns=x_test.columns,
        index=x_test.index,
    )

    y_train_scaled = pd.Series(
        y_scaler.fit_transform(y_train.to_frame()).ravel(),
        index=y_train.index,
        name=TARGET_COLUMN,
    )
    y_test_scaled = pd.Series(
        y_scaler.transform(y_test.to_frame()).ravel(),
        index=y_test.index,
        name=TARGET_COLUMN,
    )

    return PreparedData(
        x_train=x_train_scaled,
        x_test=x_test_scaled,
        y_train=y_train_scaled,
        y_test=y_test_scaled,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
    )


def split_ood_by_quantile(
    df: pd.DataFrame,
    column: str,
    quantile: float = 0.85,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    threshold = df[column].quantile(quantile)
    id_df = df[df[column] <= threshold].copy()
    ood_df = df[df[column] > threshold].copy()
    return id_df, ood_df
