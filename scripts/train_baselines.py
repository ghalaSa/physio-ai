"""Phase 5: non-temporal baselines on summary statistics (participant-level splits)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio.data.manifest import load_manifest  # noqa: E402
from physio.data.splits import assert_no_leakage  # noqa: E402
from physio.models.baselines import classifier_baselines, regressor_baselines  # noqa: E402
from physio.models.train import fit_sklearn  # noqa: E402
from physio.pose.dataset import load_arrays  # noqa: E402


def main() -> int:
    man = load_manifest()
    assert_no_leakage(man)
    arrays = load_arrays(man)
    tr, va = arrays.for_split("train"), arrays.for_split("val")
    print(f"train={len(tr)} val={len(va)} clips; classes={arrays.classes}")

    for name, model in classifier_baselines().items():
        rec = fit_sklearn(name, model, "cls", tr.X_stat, tr.y_cls, va.X_stat, va.y_cls, arrays.classes)
        print(f"{name:14s} val macro-F1={rec['val']['macro_f1']:.3f} acc={rec['val']['accuracy']:.3f}")

    trs, vas = tr.scored(), va.scored()
    print(f"scored: train={len(trs)} val={len(vas)}")
    for name, model in regressor_baselines().items():
        rec = fit_sklearn(name, model, "reg", trs.X_stat, trs.y_reg, vas.X_stat, vas.y_reg, arrays.classes,
                          ex_va=vas.index["exercise"].to_numpy())
        print(f"{name:14s} val MAE={rec['val']['mae']:.2f} RMSE={rec['val']['rmse']:.2f} R2={rec['val']['r2']:.3f}")
    # trivial reference: predicting the training mean
    import numpy as np
    from physio.models.evaluate import regression_metrics
    from physio.models.train import log_run
    mean_pred = np.full(len(vas), trs.y_reg.mean())
    val = regression_metrics(vas.y_reg, mean_pred, vas.index["exercise"].to_numpy())
    log_run({"run_name": "mean_predictor", "task": "reg", "family": "trivial", "model": "train-mean",
             "n_train": len(trs), "n_val": len(vas), "val": val, "selection_metric": "mae", "selection_value": val["mae"]})
    print(f"{'mean_predictor':14s} val MAE={val['mae']:.2f} RMSE={val['rmse']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
