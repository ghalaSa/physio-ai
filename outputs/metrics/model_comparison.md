# Model comparison (validation split, participant-independent)

## Exercise classification

| run | model | params | macro F1 | accuracy | train s |
|---|---|---|---|---|---|
| bilstm_cls | bilstm | 426441.0 | 0.986 | 0.982 | 135.8 |
| tcn_multi | tcn | 387850.0 | 0.983 | 0.982 | 153.1 |
| bilstm_multi | bilstm | 426634.0 | 0.983 | 0.979 | 129.1 |
| transformer_cls | transformer | 190953.0 | 0.979 | 0.975 | 201.9 |
| tcn_cls | tcn | 387753.0 | 0.978 | 0.975 | 205.8 |
| rf_stats | RandomForestClassifier | nan | 0.964 | 0.966 | 2.8 |
| transformer_multi | transformer | 191050.0 | 0.959 | 0.960 | 116.9 |
| logreg_stats | LogisticRegression | nan | 0.909 | 0.899 | 23.8 |

## Movement quality (0-100 physiotherapist average)

| run | model | params | MAE | RMSE | R² | train s |
|---|---|---|---|---|---|---|
| tcn_reg | tcn | 386977.0 | 10.28 | 13.32 | 0.045 | 103.6 |
| tcn_multi | tcn | 387850.0 | 10.62 | 14.07 | -0.066 | 153.1 |
| transformer_reg | transformer | 190177.0 | 10.70 | 13.68 | -0.008 | 101.4 |
| bilstm_reg | bilstm | 424897.0 | 10.82 | 14.19 | -0.085 | 49.0 |
| mean_predictor | train-mean | nan | 10.94 | 14.01 | -0.057 | nan |
| rf_reg_stats | RandomForestRegressor | nan | 11.37 | 14.71 | -0.166 | 47.9 |
| bilstm_multi | bilstm | 426634.0 | 11.56 | 15.52 | -0.298 | 129.1 |
| transformer_multi | transformer | 191050.0 | 11.63 | 15.49 | -0.292 | 116.9 |
| ridge_stats | Ridge | nan | 12.50 | 15.78 | -0.342 | 0.1 |

## Selected models

```json
{
  "classification": {
    "run_name": "bilstm_cls",
    "family": "torch",
    "artifact": "bilstm_cls.pt",
    "val_macro_f1": 0.9855557949675595,
    "task": "cls"
  },
  "regression": {
    "run_name": "tcn_reg",
    "family": "torch",
    "artifact": "tcn_reg.pt",
    "val_mae": 10.279015210106113,
    "task": "reg"
  }
}
```
