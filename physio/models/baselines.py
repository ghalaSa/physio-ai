"""Phase 5 baselines: non-temporal models on per-clip summary statistics.

These set the bar the temporal models must beat. They use exactly the same
participant-level splits as everything else.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


def classifier_baselines(seed: int = 0) -> dict[str, Pipeline]:
    return {
        "logreg_stats": make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=seed)),
        "rf_stats": make_pipeline(StandardScaler(),
                                  RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=seed, n_jobs=-1)),
    }


def regressor_baselines(seed: int = 0) -> dict[str, Pipeline]:
    return {
        "ridge_stats": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "rf_reg_stats": make_pipeline(StandardScaler(),
                                      RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=seed, n_jobs=-1)),
    }
