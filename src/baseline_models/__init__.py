from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType

MODEL_FILE_BY_NAME = {
    "linear_regression": Path(__file__).resolve().parent / "linear regression" / "model.py",
    "rfr": Path(__file__).resolve().parent / "rfr" / "model.py",
    "bpnn": Path(__file__).resolve().parent / "bpnn" / "model.py",
}


@lru_cache(maxsize=None)
def _load_model_module(model_name: str) -> ModuleType:
    model_file = MODEL_FILE_BY_NAME[model_name]
    spec = spec_from_file_location(f"baseline_models.{model_name}", model_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load baseline model module from {model_file}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_baseline_models(n_train_samples: int) -> dict[str, object]:
    """Return the baseline-model registry used by the training pipeline."""

    models: dict[str, object] = {}
    for model_name in ("linear_regression", "rfr", "bpnn"):
        module = _load_model_module(model_name)
        builder = getattr(module, "build_model", None)
        declared_name = getattr(module, "MODEL_NAME", None)
        if declared_name != model_name:
            raise ValueError(f"Baseline model module {module.__file__} declared unexpected name {declared_name!r}")
        if builder is None:
            raise AttributeError(f"Baseline model module {module.__file__} is missing build_model().")
        models[model_name] = builder(n_train_samples)
    return models
