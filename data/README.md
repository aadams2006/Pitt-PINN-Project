# Data

This folder stores datasets and documentation for thin-film ML experiments.

## Datasets

- `agg.data.xlsx`: Main aggregated dataset used for model training and evaluation.
- `placeholder_film_data.csv`: Legacy placeholder data (for reference only).
- `Hexane+PDMS-new.xlsx`: Additional experimental data.
- `Hexane_PDMS_ML_Training_Dataset.xlsx`: Training dataset archive.

## Data Processing

The training code now normalizes either the legacy snake_case schema or the
`synthetic_data_improved.csv` column names into a shared internal format:
- **Feature columns**: concentration, uncoated layer thickness, total thickness
- **Target column**: bonded thickness

## Model Evaluation

All models now generate:
- **Evaluation Metrics**: MSE, RMSE, MAE, R² Score
- **Visualizations**: 
  - Predictions vs Actual scatter plot
  - Residual analysis plot
  - Residual distribution histogram
  - Performance metrics summary

Outputs are saved to each model's `results/` directory.
