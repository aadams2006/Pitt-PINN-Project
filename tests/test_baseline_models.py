from pathlib import Path

from baseline_models import MODEL_FILE_BY_NAME, build_baseline_models


def test_build_baseline_models_returns_expected_registry():
    models = build_baseline_models(8)

    assert list(models) == ["linear_regression", "rfr", "bpnn"]


def test_each_baseline_folder_has_model_definition():
    for model_file in MODEL_FILE_BY_NAME.values():
        assert Path(model_file).is_file()
