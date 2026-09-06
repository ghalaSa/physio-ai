"""Phases 6-7: temporal models for classification, regression and multi-task.

Usage:
    python scripts/train_temporal.py                       # all encoders x all tasks
    python scripts/train_temporal.py --encoders tcn --tasks cls --epochs 40
    python scripts/train_temporal.py --cv tcn:cls          # grouped 5-fold CV for one config
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio import config  # noqa: E402
from physio.data.manifest import load_manifest  # noqa: E402
from physio.data.splits import assert_no_leakage  # noqa: E402
from physio.models.train import TrainConfig, fit_temporal  # noqa: E402
from physio.pose.dataset import load_arrays  # noqa: E402


def run_one(arrays, encoder: str, task: str, epochs: int, seed: int = 0, name: str | None = None) -> dict:
    tr, va = arrays.for_split("train"), arrays.for_split("val")
    if task == "reg":
        tr, va = tr.scored(), va.scored()
    cfg = TrainConfig(run_name=name or f"{encoder}_{task}", task=task, encoder=encoder, epochs=epochs, seed=seed)
    rec = fit_temporal(cfg, tr.X_seq, tr.y_cls, tr.y_reg, va.X_seq, va.y_cls, va.y_reg, arrays.classes,
                       ex_va=va.index["exercise"].to_numpy())
    v = rec["val"]
    msg = f"{cfg.run_name:22s} epochs={rec['epochs_run']} best={rec['best_epoch']}"
    if "cls" in v:
        msg += f" | macroF1={v['cls']['macro_f1']:.3f} acc={v['cls']['accuracy']:.3f}"
    if "reg" in v:
        msg += f" | MAE={v['reg']['mae']:.2f} RMSE={v['reg']['rmse']:.2f} R2={v['reg']['r2']:.3f}"
    print(msg, flush=True)
    return rec


def run_cv(arrays, encoder: str, task: str, epochs: int) -> dict:
    """Grouped K-fold over development participants (test participants never touched)."""
    dev = arrays.subset((arrays.index["split"] != "test").to_numpy())
    if task == "reg":
        dev = dev.scored()
    folds = dev.index["cv_fold"].to_numpy()
    scores = []
    for k in sorted(set(folds)):
        trm, vam = folds != k, folds == k
        tr, va = dev.subset(trm), dev.subset(vam)
        assert not set(tr.groups) & set(va.groups)
        cfg = TrainConfig(run_name=f"cv_{encoder}_{task}_fold{k}", task=task, encoder=encoder, epochs=epochs, seed=k)
        rec = fit_temporal(cfg, tr.X_seq, tr.y_cls, tr.y_reg, va.X_seq, va.y_cls, va.y_reg, arrays.classes,
                           ex_va=va.index["exercise"].to_numpy())
        v = rec["val"]
        scores.append(v["cls"]["macro_f1"] if task == "cls" else v["reg"]["mae"])
        print(f"fold {k}: {scores[-1]:.3f}", flush=True)
    out = {"encoder": encoder, "task": task, "metric": "macro_f1" if task == "cls" else "mae",
           "folds": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
    (config.METRICS_DIR / f"cv_{encoder}_{task}.json").write_text(json.dumps(out, indent=2))
    print(f"CV {encoder}/{task}: {out['mean']:.3f} ± {out['std']:.3f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=["bilstm", "tcn", "transformer"])
    ap.add_argument("--tasks", nargs="*", default=["cls", "reg", "multi"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--cv", default=None, help="encoder:task to cross-validate")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip runs already logged in runs.jsonl with an existing artifact (resume after a pause)")
    args = ap.parse_args()

    man = load_manifest()
    assert_no_leakage(man)
    arrays = load_arrays(man)
    done = completed_runs() if args.skip_existing else set()
    if args.cv:
        enc, task = args.cv.split(":")
        cv_file = config.METRICS_DIR / f"cv_{enc}_{task}.json"
        if args.skip_existing and cv_file.exists():
            print(f"skip existing CV result {cv_file.name}")
            return 0
        run_cv(arrays, enc, task, args.epochs)
        return 0
    for enc in args.encoders:
        for task in args.tasks:
            name = f"{enc}_{task}"
            if name in done:
                print(f"skip completed run {name}")
                continue
            run_one(arrays, enc, task, args.epochs)
    return 0


def completed_runs() -> set[str]:
    from physio.models.train import RUNS_LOG
    names = set()
    if RUNS_LOG.exists():
        for line in RUNS_LOG.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                art = config.MODELS_DIR / r.get("artifact", "")
                if r.get("family") == "torch" and art.exists():
                    names.add(r["run_name"])
    return names


if __name__ == "__main__":
    raise SystemExit(main())
