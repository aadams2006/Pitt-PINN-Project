from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import load

from .data import load_dataset, prepare_train_test, split_ood_by_quantile
from .models import PINNRegressor, evaluate_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baselines + PINN")
    parser.add_argument("--data", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--pinn-model", required=True)
    parser.add_argument("--ood-column", default="withdrawal_velocity")
    parser.add_argument("--outdir", default="outputs/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    prepared = prepare_train_test(df)

    baseline_dir = Path(args.baseline_dir)
    baseline_files = sorted(baseline_dir.glob("*.joblib"))
    results = {"baselines": {}, "pinn": {}, "ood": {}}

    for model_file in baseline_files:
        if "scaler" in model_file.name:
            continue
        model = load(model_file)
        pred = model.predict(prepared.x_test)
        m = evaluate_regression(prepared.y_test.to_numpy(), pred)
        results["baselines"][model_file.stem] = {"mae": m.mae, "rmse": m.rmse, "r2": m.r2}

    payload = torch.load(args.pinn_model, map_location="cpu")
    pinn = PINNRegressor(input_dim=payload["input_dim"])
    pinn.load_state_dict(payload["state_dict"])
    pinn.eval()

    x_test = torch.tensor(prepared.x_test.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        pred = pinn(x_test).numpy().reshape(-1)
    m = evaluate_regression(prepared.y_test.to_numpy(), pred)
    results["pinn"] = {"mae": m.mae, "rmse": m.rmse, "r2": m.r2}

    if args.ood_column in df.columns:
        id_df, ood_df = split_ood_by_quantile(df, args.ood_column)
        results["ood"] = {
            "column": args.ood_column,
            "id_count": int(len(id_df)),
            "ood_count": int(len(ood_df)),
            "quantile": 0.85,
        }

    with open(outdir / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
