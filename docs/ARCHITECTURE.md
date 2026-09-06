# Architecture

## Data flow

```
video (mp4)
  │  physio/pose/extractor.py      MediaPipe PoseLandmarker, 10 fps, ≤640 px
  ▼
PoseSequence (T×33×4 landmarks, detected mask, timestamps, meta)   ── cached as data/poses/<video_id>.npz
  │  quality gate: ≥15 detected frames, ≥60 % detection, mean visibility ≥0.5
  │        └─ fails → status = "unable_to_analyze" (no predictions are fabricated)
  ▼
physio/pose/normalize.py           pixel space → hip-centred → torso-scaled → resample to 64 frames
  │                                features per frame: xy(66) + visibility(33) + velocity(66) = 165
  ├──────────────► physio/models  exercise classifier  → code, confidence, class probabilities
  ├──────────────► physio/models  quality regressor    → 0-100 score
  ▼
physio/analysis/angles.py          10 joint angles + trunk lean from the un-resampled sequence
physio/analysis/features.py        ROM, SPARC smoothness, tempo, repetitions, rep-duration CV,
                                   symmetry ratio, trajectory consistency  (each with a `reliable` flag)
physio/analysis/observations.py    rule-based flags vs. training-split reference percentiles
  ▼
physio/llm/*                       SessionSummary (JSON) + prior sessions → feedback text
  ▼
physio/db/*                        ExerciseSession row (all of the above, JSON columns) + review flag
  ▼
physio/api/main.py  ⇄  physio/dashboard/app.py
```

## Four layers that are never mixed

| Layer | Produced by | Labelled as |
|---|---|---|
| Model predictions | classifier / regressor | `kind = "model"` observations, `exercise_confidence`, `quality_score` |
| Deterministic measurements | `analysis/features.py` | `measurements` with `unit`, `reliable`, `note` |
| Heuristic observations | `analysis/observations.py` | `kind = "heuristic"`, thresholds in `evidence` |
| Clinical interpretation | **nobody** | the physiotherapist's `clinician_note` only |

The LLM system prompt requires the three machine layers to stay distinguishable in
the text, forbids diagnosis and treatment advice, and the output is screened
(`llm/interface.py: FORBIDDEN_TERMS`) with a deterministic fallback.

## Leakage prevention

* `data/splits.py` assigns participants, not videos. Every exercise must be present in
  every split; the seed is searched until it is.
* Test participants get `cv_fold = -1` and are excluded from cross-validation.
* `assert_no_leakage` is executed by the tests and at the start of every training and
  evaluation script.
* Feature standardization statistics, regression-target statistics, class weights and
  the reference percentiles for heuristics are computed from the training split only.

## Models

* Baselines (`models/baselines.py`): logistic regression and random forest on per-clip
  summary statistics (mean/std/min/max of the 165 features → 660 dims); ridge and random
  forest regressors; a train-mean predictor as the trivial regression reference.
* Temporal (`models/temporal.py`): BiLSTM, dilated temporal CNN (TCN), lightweight
  Transformer encoder. Each has a classification head, a regression head, or both
  (shared encoder = multi-task). Trained with AdamW, cosine schedule, label smoothing,
  class weights, mirror / jitter / time-shift augmentation, early stopping on the
  validation metric (`models/train.py`).
* Selection (`models/compare.py`): validation macro-F1 for classification, validation
  MAE for quality; the result is `outputs/models/best_models.json`.
* Inference (`pipeline.py`) loads artefacts by that file, so the API never imports the
  training scripts.

## Storage

`patients(id, name, created_at)` and `sessions(...)` with JSON columns for class
probabilities, measurements (including angle time-series), observations, pose
quality, review reasons and model versions. SQLite by default; set `PHYSIO_DB_URL`
to a PostgreSQL URL to migrate (SQLAlchemy 2.0, no SQLite-specific SQL).

Progress features (`db/repo.py`): per-exercise quality trend, repeated observations
over the last N sessions, sessions requiring review (low classifier confidence or low
predicted quality), and a per-patient overview for the physiotherapist.

## Configuration

`physio/config.py` holds paths and tunable thresholds. Dataset facts (exercise names,
score scale) are parsed from `data/raw/MobiPhysio.md` and the scoring CSV at runtime.

## Known limitations

* 2D angles are image-plane approximations; rotations about the limb axis (E03/E04)
  are only partially observable, which is why the primary angle for those exercises is
  chosen by the largest observed range.
* 58 participants is a small population; all reported numbers are participant-level
  hold-out or grouped-CV estimates and should be read with their variance.
* The EAAQ average compresses seven binary/partial criteria into one score; the
  regression model predicts that average, not the individual criteria.
