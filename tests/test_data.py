from pathlib import Path

from pitt_pinn.data import TARGET_COLUMN, feature_columns, load_dataset


def test_load_dataset_normalizes_improved_csv_columns():
    dataset_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_data_improved.csv"
    df = load_dataset(str(dataset_path))

    assert feature_columns(df) == [
        "pdms_concentration",
        "uncoated_layer_thickness",
        "total_film_thickness",
    ]
    assert TARGET_COLUMN == "bonded_film_thickness"
    assert TARGET_COLUMN in df.columns
