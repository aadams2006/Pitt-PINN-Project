# Inverse Modeling of Interfacial Thin-Film Formation with PINNs

This repository is set up for a full **8-week research workflow** to model bonded thin-film thickness from dip-coating experiments.

It supports:
- Data preprocessing and validation
- Baseline ML models from `src/baseline_models` (`linear_regression`, `rfr`, `bpnn`)
- A physics-informed neural network (PINN) with soft constraints
- Evaluation on in-distribution and out-of-distribution (OOD) splits
- Constraint-violation reporting and simple interpretability hooks

## 1) Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Prepare your dataset as CSV with these core columns:
- `Concentration (g/mL)` or `pdms_concentration`
- `Uncoated Layer (nm)` or `uncoated_layer_thickness`
- `Total Thickness (nm)` or `total_film_thickness`
- `Bonded Thickness (nm)` or `bonded_film_thickness`

The training pipeline normalizes these names internally and uses:
- Features: concentration, uncoated layer thickness, total thickness
- Target: bonded thickness

Then run:

```bash
train-baselines --data data/your_dataset.csv --outdir outputs/baselines
train-pinn --data data/your_dataset.csv --outdir outputs/pinn
evaluate-pinn --data data/your_dataset.csv --pinn-model outputs/pinn/pinn_model.pt --baseline-dir outputs/baselines
```

## 2) Project Layout

```text
src/pitt_pinn/
  config.py            # Dataclasses + run settings
  data.py              # Dataset loading, checks, scaling, split
  physics.py           # Physics-informed penalties/metrics
  models.py            # Baseline models + PINN model class
  train_baselines.py   # Baseline training CLI
  train_pinn.py        # PINN training CLI
  evaluate.py          # Unified evaluation CLI

data/
  template_thinfilm_dataset.csv

tests/
  test_physics.py
```

## 3) Physics-Informed Objective

The PINN objective implemented here is:

\[
\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}}\mathcal{L}_{\text{physics}} + \lambda_{\text{smooth}}\mathcal{L}_{\text{smooth}}
\]

Included soft constraints:
- `bonded <= total` (hard-prior penalty)
- Monotonic tendency with concentration (positive derivative target)

> These are encoded as penalties to remain robust under sparse/noisy experiments.

## 4) Typical Workflow (aligned with your timeline)

- Week 1: Use `data.py` checks and template schema to clean/refine data.
- Week 2: Run `train_baselines.py` and compare metrics from `metrics.json`.
- Weeks 3–4: Train PINN with `train_pinn.py`; tune λ weights and architecture.
- Weeks 5–6: Run OOD slices using custom split bounds in `evaluate.py`.
- Week 7: Inspect `constraint_report.json` and feature sensitivities.
- Week 8: Export figures/tables from JSON outputs for final report.

## 5) Outputs

Each run writes JSON and artifacts (model weights / scalers):
- `outputs/baselines/metrics.json`
- `outputs/baselines/metadata.json`
- `outputs/pinn/train_history.json`
- `outputs/pinn/pinn_model.pt`
- `outputs/pinn/constraint_report.json`

## 6) Notes

- Keep all thickness units consistent.
- If your dataset is very small, lower model width/depth and increase regularization.
- Random Forest is included as a strong baseline from your preliminary results.
