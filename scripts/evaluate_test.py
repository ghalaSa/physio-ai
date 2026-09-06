"""Phase 14: ONE final evaluation of the selected models on the untouched test participants.

Also reports performance per exercise, camera angle and recording variation, as the
dataset README recommends, and runs the full pipeline (measurements, observations,
feedback) over the test clips to check the unable-to-analyze rate.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physio import config  # noqa: E402
from physio.data.manifest import load_manifest  # noqa: E402
from physio.data.splits import assert_no_leakage  # noqa: E402
from physio.llm.template_provider import TemplateProvider  # noqa: E402
from physio.models.evaluate import classification_metrics, plot_confusion, plot_pred_vs_true, regression_metrics  # noqa: E402
from physio.pipeline import Analyzer  # noqa: E402
from physio.pose.dataset import load_arrays  # noqa: E402
from physio.pose.extractor import PoseSequence, pose_cache_path  # noqa: E402


def breakdown(index, y_true, y_pred, col, kind):
    out = {}
    for v in sorted(index[col].unique()):
        m = (index[col] == v).to_numpy()
        if m.sum() < 3:
            continue
        if kind == "cls":
            out[str(v)] = {"accuracy": float((y_true[m] == y_pred[m]).mean()), "n": int(m.sum())}
        else:
            out[str(v)] = {"mae": float(np.abs(y_true[m] - y_pred[m]).mean()), "n": int(m.sum())}
    return out


def main() -> int:
    man = load_manifest()
    assert_no_leakage(man)
    an = Analyzer(provider=TemplateProvider())
    arrays = load_arrays(man)
    te = arrays.for_split("test")
    print(f"test clips: {len(te)} from {te.index['participant'].nunique()} participants")

    # classification
    probs = np.stack([an.classifier.predict(x)["probs"] for x in te.X_seq])
    pred = probs.argmax(1)
    cls = classification_metrics(te.y_cls, pred, arrays.classes)
    cls["by_angle"] = breakdown(te.index, te.y_cls, pred, "angle_name", "cls")
    cls["by_variation"] = breakdown(te.index, te.y_cls, pred, "variation_name", "cls")
    cls["mean_confidence_correct"] = float(probs.max(1)[pred == te.y_cls].mean())
    cls["mean_confidence_wrong"] = float(probs.max(1)[pred != te.y_cls].mean()) if (pred != te.y_cls).any() else None
    plot_confusion(cls["confusion_matrix"], arrays.classes, config.FIGURES_DIR / "test_confusion_matrix.png",
                   f"Test confusion matrix (macro F1 {cls['macro_f1']:.3f})")

    # regression
    reg = None
    if an.regressor is not None:
        tes = te.scored()
        sp = np.array([an.regressor.predict(x)["score"] for x in tes.X_seq])
        reg = regression_metrics(tes.y_reg, sp, tes.index["exercise"].to_numpy())
        reg["by_angle"] = breakdown(tes.index, tes.y_reg, sp, "angle_name", "reg")
        reg["by_variation"] = breakdown(tes.index, tes.y_reg, sp, "variation_name", "reg")
        mean_pred = np.full(len(tes), float(np.nanmean(arrays.for_split("train").y_reg)))
        reg["mean_predictor_mae"] = float(np.abs(tes.y_reg - mean_pred).mean())
        plot_pred_vs_true(tes.y_reg, sp, config.FIGURES_DIR / "test_pred_vs_true.png",
                          f"Test: MAE {reg['mae']:.2f}, R² {reg['r2']:.3f}")

    # full pipeline over test clips (incl. gated ones) for the unable-to-analyze rate & observation stats
    status, obs_counter, feedback_ok = Counter(), Counter(), 0
    test_ids = man[man["split"] == "test"]["video_id"]
    n_pipeline = 0
    for vid in test_ids:
        p = pose_cache_path(vid)
        if not p.exists():
            continue
        res = an.analyze_pose_sequence(PoseSequence.load(p))
        n_pipeline += 1
        status[res.status] += 1
        for o in res.observations:
            obs_counter[o["code"]] += 1
        feedback_ok += bool(res.feedback)
    pipeline = {"n": n_pipeline, "status": dict(status), "observation_counts": dict(obs_counter),
                "feedback_generated": feedback_ok}

    out = {"selected_models": an.best, "n_test_clips": int(len(te)), "n_test_participants": int(te.index["participant"].nunique()),
           "classification": cls, "regression": reg, "pipeline": pipeline}
    (config.METRICS_DIR / "test_evaluation.json").write_text(json.dumps(out, indent=2, default=float))

    lines = ["# Final test-set evaluation (held-out participants)", "",
             f"Test clips: {len(te)} from {te.index['participant'].nunique()} participants "
             f"({sorted(te.index['participant'].unique())})", "",
             "## Exercise classification", "",
             f"- accuracy: **{cls['accuracy']:.3f}**", f"- macro F1: **{cls['macro_f1']:.3f}**",
             f"- macro precision / recall: {cls['macro_precision']:.3f} / {cls['macro_recall']:.3f}", "",
             "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for c, m in cls["per_class"].items():
        lines.append(f"| {c} {an.codebook.name(c)} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    lines += ["", "By camera angle: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['n']})" for k, v in cls["by_angle"].items()),
              "", "By variation: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['n']})" for k, v in cls["by_variation"].items()),
              "", "![confusion](../figures/test_confusion_matrix.png)", ""]
    if reg:
        lines += ["## Movement quality (0-100)", "",
                  f"- MAE: **{reg['mae']:.2f}** (mean-predictor baseline {reg['mean_predictor_mae']:.2f})",
                  f"- RMSE: **{reg['rmse']:.2f}**", f"- R²: **{reg['r2']:.3f}**", f"- n scored test clips: {reg['n']}", "",
                  "| exercise | MAE | RMSE | n |", "|---|---|---|---|"]
        for c, m in reg["per_exercise"].items():
            lines.append(f"| {c} {an.codebook.name(c)} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['n']} |")
        lines += ["", "By camera angle: " + ", ".join(f"{k} {v['mae']:.2f} (n={v['n']})" for k, v in reg["by_angle"].items()),
                  "", "By variation: " + ", ".join(f"{k} {v['mae']:.2f} (n={v['n']})" for k, v in reg["by_variation"].items()),
                  "", "![pred vs true](../figures/test_pred_vs_true.png)", ""]
    lines += ["## End-to-end pipeline on test clips", "", f"```json\n{json.dumps(pipeline, indent=2)}\n```", ""]
    (config.METRICS_DIR / "test_evaluation.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
