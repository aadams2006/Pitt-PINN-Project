from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import dump

from .data import load_dataset, prepare_train_test
from .models import build_baseline_models, evaluate_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline regressors.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--outdir", default="outputs/baselines", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    prepared = prepare_train_test(df)

    metrics = {}

    for name, model in build_baseline_models(len(prepared.x_train)).items():
        try:
            model.fit(prepared.x_train, prepared.y_train)
            preds = model.predict(prepared.x_test)
            m = evaluate_regression(prepared.y_test.to_numpy(), preds)
            metrics[name] = {"mae": m.mae, "rmse": m.rmse, "r2": m.r2}
            dump(model, outdir / f"{name}.joblib")
        except Exception as exc:
            metrics[name] = {"error": str(exc)}

    dump(prepared.x_scaler, outdir / "x_scaler.joblib")
    dump(prepared.y_scaler, outdir / "y_scaler.joblib")

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
