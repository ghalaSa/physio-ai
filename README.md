# AI-Powered Home Physiotherapy Assistant

Capstone project: computer vision + movement-quality assessment + LLM feedback for
home physiotherapy exercise videos, built on the **MobiPhysio** dataset
(Harvard Dataverse, `doi:10.7910/DVN/XSI0QN`, CC0).

A patient uploads a video. The system extracts body pose, recognizes the exercise,
predicts a movement-quality score, computes interpretable movement measurements,
turns the validated results into plain-language feedback, stores the session, and
shows progress over time to the patient and the physiotherapist.

> **Scope and safety boundary.** This is a monitoring and educational support tool
> for a capstone project. It is not a medical device, does not diagnose, and does not
> replace assessment, prescription or supervision by a qualified physiotherapist.
> Model predictions, deterministic measurements and heuristic observations are kept
> separate and labelled as such everywhere; no clinical interpretation is generated.

## Results

Final numbers come from a single evaluation on 12 held-out participants (815 clips,
691 with a physiotherapist score) that were never used for training or model selection.

| Task | Selected model (by validation) | Held-out test result |
|---|---|---|
| Exercise classification (9 classes) | BiLSTM on pose sequences | accuracy 0.944, macro F1 0.944 |
| Movement quality (0-100 physiotherapist average) | Temporal CNN (TCN) on pose sequences | MAE 9.11, RMSE 11.65, R² 0.25 (mean-predictor MAE 11.88) |

Grouped 5-fold cross-validation over the development participants (TCN): macro F1
0.967 ± 0.017; quality MAE 9.94 ± 1.80. Full report with per-class, per-exercise,
per-angle and per-variation breakdowns and the confusion matrix:
[`outputs/metrics/test_evaluation.md`](outputs/metrics/test_evaluation.md); validation-based
model selection: [`outputs/metrics/model_comparison.md`](outputs/metrics/model_comparison.md).

## Pipeline

| Step | Component | Where |
|---|---|---|
| 1 | Video upload | `physio/api/main.py` (`POST /sessions/analyze`), `physio/dashboard/app.py` |
| 2 | Pose estimation (MediaPipe PoseLandmarker, 33 landmarks, 10 fps) | `physio/pose/extractor.py` |
| 3 | Exercise classification (9 classes) | `physio/models/` + `physio/pipeline.py` |
| 4 | Quality assessment (regression on the 0-100 physiotherapist average) | `physio/models/` + `physio/pipeline.py` |
| 5 | Movement analysis (angles, ROM, smoothness, tempo, symmetry, reps, consistency) | `physio/analysis/` |
| 6 | LLM feedback (structured results only, non-diagnostic, provider-independent) | `physio/llm/` |
| 7 | Session tracking (SQLite via SQLAlchemy; PostgreSQL by `PHYSIO_DB_URL`) | `physio/db/` |
| 8 | Dashboards (patient + physiotherapist) | `physio/dashboard/app.py` |

## Dataset facts (read from the downloaded files, not assumed)

* 3,686 videos, 58 participants, 9 active range-of-motion exercises (`E01` Abduction …
  `E09` Back Extension; names parsed from `data/raw/MobiPhysio.md`).
* Filenames encode exercise, participant, camera angle (front/left/right), recording
  variation (lighting, jitter, occlusion, low resolution) and gender.
* 3,010 videos carry an EAAQ score from three physiotherapists; the target is their
  average on the 0-100 scale (observed range 0-100, mean 82.3).
* Total size 158.7 GB, so videos are streamed one at a time and only pose sequences
  are cached (`data/poses/*.npz`).

## Participant-independent evaluation

`scripts/build_manifest.py` assigns whole participants to train / val / test
(40 / 6 / 12 people), stratified by recording phase, and checks every exercise is
present in each split. `physio.data.splits.assert_no_leakage` is run by the tests and
by every training script. Grouped 5-fold cross-validation folds over the development
participants are stored in the same file. The test participants are touched exactly
once, by `scripts/evaluate_test.py`.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The training pipeline is designed to run on a cloud VM (CPU is enough; the pose
models are compact). Helper scripts for Google Cloud are in `scripts/gcp/`:

```bash
export PROJECT=<gcp-project-id>
./scripts/gcp/create_vm.sh                      # n2-standard-8, 300 GB disk
./scripts/gcp/sync.sh push                      # copy the code
gcloud compute ssh physio-vm --zone us-central1-a -- 'cd physio-ai && bash scripts/gcp/bootstrap.sh'
gcloud compute ssh physio-vm --zone us-central1-a -- 'cd physio-ai && tmux new -d -s run "bash scripts/gcp/run_pipeline.sh"'
./scripts/gcp/sync.sh pull                      # fetch outputs/ when done
```

## Running the pipeline

```bash
python scripts/inspect_dataset.py data/raw          # Phase 1: what is actually on disk
python scripts/build_manifest.py                     # Phase 3: manifest + participant splits + leakage test
python scripts/extract_poses.py --workers 8          # Phase 4: stream download -> pose -> cache (resumable)
python scripts/train_baselines.py                    # Phase 5: sklearn baselines on summary statistics
python scripts/train_temporal.py                     # Phases 6-7: BiLSTM / TCN / Transformer, cls / reg / multi-task
python scripts/train_temporal.py --cv tcn:cls        # grouped CV estimate for one configuration
python scripts/compare_models.py                     # Phase 8: select by validation macro-F1 / MAE
python scripts/build_reference_stats.py              # Phase 9: per-exercise reference percentiles for heuristics
python scripts/evaluate_test.py                      # Phase 14: one final run on the held-out participants
```

Every training run is appended to `outputs/metrics/runs.jsonl` (experiment tracking)
and the selected artefacts are recorded in `outputs/models/best_models.json`, which
the inference pipeline reads.

## Running the application

```bash
uvicorn physio.api.main:app --port 8000              # backend, docs at /docs
streamlit run physio/dashboard/app.py                # frontend (PHYSIO_API_URL defaults to localhost:8000)
```

Everything the application needs at runtime is in the repository after training:
`outputs/models/` (selected model artefacts, `best_models.json`, `reference_stats.json`,
the MediaPipe pose model), `outputs/physio.db` (session database, seeded with demo
patients by `scripts/seed_demo.py`) and `data/raw/MobiPhysio.md` (exercise names).
The cached pose sequences in `data/poses/` are only needed for training, for
`scripts/seed_demo.py`, and for the `/sessions/analyze-cached` demo endpoint. A real
sample clip for the upload path is kept at `data/samples/E01_P01_AL_VFL_GM.mp4`.

macOS note: MediaPipe 1.0.x aborts on macOS when the pose graph starts (Metal
"Service is unavailable"); `requirements.txt` pins `mediapipe<1.0` on macOS, where
0.10.35 works.

Key endpoints: `POST /sessions/analyze` (multipart video + `patient_name`),
`POST /sessions/analyze-cached` (analyze a cached pose sequence, used for demos),
`GET /patients`, `GET /patients/{id}/sessions`, `GET /patients/{id}/progress`,
`GET /sessions/{id}`, `POST /sessions/{id}/note`, `GET /review`, `GET /health`, `GET /models`.

### LLM feedback

`PHYSIO_LLM_PROVIDER=auto|anthropic|template`. With an Anthropic credential
(`ANTHROPIC_API_KEY`) the Anthropic provider is used (`claude-opus-5` by default,
override with `PHYSIO_LLM_MODEL`); otherwise a deterministic template provider produces
the feedback. The LLM receives only the JSON summary (predictions, measurements with
reliability flags, heuristic observations, prior sessions), never video. Its output
is screened for diagnostic language and replaced by the template if it drifts.

## Tests

```bash
python -m pytest -q tests
```

Covers: README codebook parsing, filename decoding, score parsing, manifest shape,
participant-level split invariants and the leakage detector, normalization invariances,
angle geometry, movement features on synthetic motion, heuristic observations,
feedback providers and constraints, database + progress queries, the end-to-end
pipeline on synthetic pose sequences, the FastAPI endpoints, and the temporal models
(train, save, reload).

## Project layout

```
physio/            package: config, data, pose, models, analysis, llm, db, api, dashboard, pipeline
scripts/           phase scripts (see above) and scripts/gcp/
tests/             pytest suite
data/raw/          dataset README + scoring CSV (videos are never kept)
data/manifest/     Dataverse listing + manifest.csv (with split columns)
data/poses/        cached pose sequences
outputs/metrics/   runs.jsonl, per-run JSON, model_comparison.md, test_evaluation.md
outputs/models/    model artefacts, best_models.json, reference_stats.json
outputs/figures/   confusion matrix, predicted-vs-true plots
docs/ARCHITECTURE.md
```
