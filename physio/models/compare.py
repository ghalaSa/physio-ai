"""Phase 8: model comparison and selection from the experiment log.

Selection is by measured validation performance only: macro F1 for classification,
MAE for movement quality. Writes outputs/metrics/model_comparison.md and
outputs/models/best_models.json, which the inference pipeline reads.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from physio import config
from physio.models.train import RUNS_LOG

BEST_PATH = config.MODELS_DIR / "best_models.json"


def load_runs(path: Path = RUNS_LOG) -> pd.DataFrame:
    rows = []
    if not Path(path).exists():
        return pd.DataFrame()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        val = r.get("val", {})
        cls = val.get("cls", val if r["task"] == "cls" else {})
        reg = val.get("reg", val if r["task"] == "reg" else {})
        rows.append({
            "run_name": r["run_name"], "task": r["task"], "family": r["family"], "model": r["model"],
            "params": r.get("params"), "n_train": r.get("n_train"), "n_val": r.get("n_val"),
            "val_macro_f1": cls.get("macro_f1"), "val_accuracy": cls.get("accuracy"),
            "val_mae": reg.get("mae"), "val_rmse": reg.get("rmse"), "val_r2": reg.get("r2"),
            "train_s": r.get("train_seconds"), "artifact": r.get("artifact", f"{r['run_name']}.joblib"),
        })
    df = pd.DataFrame(rows)
    return df.drop_duplicates("run_name", keep="last").reset_index(drop=True)


def select_best(df: pd.DataFrame) -> dict:
    best = {}
    cls = df[df["val_macro_f1"].notna()].sort_values("val_macro_f1", ascending=False)
    reg = df[df["val_mae"].notna()].sort_values("val_mae", ascending=True)
    if len(cls):
        r = cls.iloc[0]
        best["classification"] = {"run_name": r.run_name, "family": r.family, "artifact": r.artifact,
                                  "val_macro_f1": float(r.val_macro_f1), "task": r.task}
    if len(reg):
        r = reg.iloc[0]
        best["regression"] = {"run_name": r.run_name, "family": r.family, "artifact": r.artifact,
                              "val_mae": float(r.val_mae), "task": r.task}
    return best


def write_report(df: pd.DataFrame, best: dict, path: Path = config.METRICS_DIR / "model_comparison.md") -> Path:
    lines = ["# Model comparison (validation split, participant-independent)", ""]
    cls = df[df["val_macro_f1"].notna()].sort_values("val_macro_f1", ascending=False)
    if len(cls):
        lines += ["## Exercise classification", "", "| run | model | params | macro F1 | accuracy | train s |", "|---|---|---|---|---|---|"]
        for r in cls.itertuples():
            lines.append(f"| {r.run_name} | {r.model} | {r.params or '-'} | {r.val_macro_f1:.3f} | {r.val_accuracy:.3f} | {r.train_s} |")
        lines.append("")
    reg = df[df["val_mae"].notna()].sort_values("val_mae")
    if len(reg):
        lines += ["## Movement quality (0-100 physiotherapist average)", "", "| run | model | params | MAE | RMSE | R² | train s |", "|---|---|---|---|---|---|---|"]
        for r in reg.itertuples():
            lines.append(f"| {r.run_name} | {r.model} | {r.params or '-'} | {r.val_mae:.2f} | {r.val_rmse:.2f} | {r.val_r2:.3f} | {r.train_s} |")
        lines.append("")
    lines += ["## Selected models", "", "```json", json.dumps(best, indent=2), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def save_best(best: dict, path: Path = BEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best, indent=2))
    return path
