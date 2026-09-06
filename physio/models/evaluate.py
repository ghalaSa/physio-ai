"""Metrics and plots for both tasks (proposal sections 6 and 12)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_precision": float(p.mean()), "macro_recall": float(r.mean()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(), "classes": classes, "n": int(len(y_true)),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, exercises: np.ndarray | None = None) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    out = {"mae": float(mean_absolute_error(y_true, y_pred)),
           "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
           "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
           "n": int(len(y_true)),
           "target_mean": float(y_true.mean()), "target_std": float(y_true.std())}
    if exercises is not None:
        per = {}
        for ex in sorted(set(exercises)):
            m = exercises == ex
            if m.sum() >= 2:
                per[str(ex)] = {"mae": float(mean_absolute_error(y_true[m], y_pred[m])),
                                "rmse": float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))),
                                "n": int(m.sum())}
        out["per_exercise"] = per
    return out


def plot_confusion(cm: list[list[int]], classes: list[str], path: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, path: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5)
    lo, hi = 0, 100
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Physiotherapist average score (0-100)"); ax.set_ylabel("Predicted score")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
