from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import TARGET_COLUMN, load_dataset, prepare_train_test
from .models import PINNRegressor
from .physics import (
    bounded_by_total_penalty,
    constraint_report,
    monotonic_increasing_penalty,
    smoothness_penalty,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PINN regressor.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--outdir", default="outputs/pinn", help="Output directory")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-phys", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    prepared = prepare_train_test(df)

    x_cols = prepared.x_train.columns.tolist()
    x_train = torch.tensor(prepared.x_train.to_numpy(), dtype=torch.float32, requires_grad=True)
    y_train = torch.tensor(prepared.y_train.to_numpy(), dtype=torch.float32).view(-1, 1)

    model = PINNRegressor(input_dim=x_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []

    total_idx = x_cols.index("total_film_thickness") if "total_film_thickness" in x_cols else None
    conc_idx = x_cols.index("pdms_concentration") if "pdms_concentration" in x_cols else None
    visc_idx = x_cols.index("viscosity") if "viscosity" in x_cols else None
    vel_idx = x_cols.index("withdrawal_velocity") if "withdrawal_velocity" in x_cols else None

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pred = model(x_train)
        data_loss = F.mse_loss(pred, y_train)

        physics_loss = torch.tensor(0.0)

        if total_idx is not None:
            total_col = x_train[:, total_idx].view(-1, 1)
            physics_loss = physics_loss + bounded_by_total_penalty(pred, total_col)

        grads = torch.autograd.grad(
            outputs=pred,
            inputs=x_train,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        if conc_idx is not None:
            physics_loss = physics_loss + monotonic_increasing_penalty(grads[:, conc_idx])
        if visc_idx is not None:
            physics_loss = physics_loss + monotonic_increasing_penalty(grads[:, visc_idx])

        smooth_loss = torch.tensor(0.0)
        if vel_idx is not None:
            smooth_loss = smoothness_penalty(grads[:, vel_idx])

        loss = data_loss + args.lambda_phys * physics_loss + args.lambda_smooth * smooth_loss
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == args.epochs - 1:
            history.append(
                {
                    "epoch": epoch,
                    "total_loss": float(loss.item()),
                    "data_loss": float(data_loss.item()),
                    "physics_loss": float(physics_loss.item()),
                    "smooth_loss": float(smooth_loss.item()),
                }
            )

    x_test = torch.tensor(prepared.x_test.to_numpy(), dtype=torch.float32)
    with torch.no_grad():
        pred_test = model(x_test).numpy()

    pred_test_unscaled = prepared.y_scaler.inverse_transform(pred_test)
    y_test_unscaled = prepared.y_scaler.inverse_transform(prepared.y_test.to_numpy().reshape(-1, 1))

    report = {
        "test_rmse": float(np.sqrt(np.mean((pred_test_unscaled - y_test_unscaled) ** 2))),
    }

    if total_idx is not None:
        total_test = x_test[:, total_idx].view(-1, 1)
        report.update(
            constraint_report(
                torch.tensor(pred_test, dtype=torch.float32),
                total_test,
            )
        )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": x_train.shape[1],
            "columns": x_cols,
        },
        outdir / "pinn_model.pt",
    )

    with open(outdir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(outdir / "constraint_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
