"""Central configuration: paths, constants, and dataset-derived facts.

Design rule (from the proposal, section 5): dataset-specific facts such as exercise
names, participant IDs and the score scale are READ from the downloaded files
(`data/raw/MobiPhysio.md` and the scoring CSV), never hardcoded here. This module
only holds paths, tunable pipeline constants, and helpers that load those facts.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PHYSIO_ROOT", Path(__file__).resolve().parents[1]))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
POSES_DIR = DATA_DIR / "poses"          # cached pose sequences (.npz), one per video
MANIFEST_DIR = DATA_DIR / "manifest"
TMP_DIR = DATA_DIR / "tmp"              # transient video downloads during pose extraction
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
SPLITS_DIR = OUTPUTS_DIR / "splits"
FIGURES_DIR = OUTPUTS_DIR / "figures"

README_PATH = RAW_DIR / "MobiPhysio.md"
SCORES_CSV_PATH = RAW_DIR / "Physiotherapist_Exercise_Marking.csv"
DATAVERSE_LISTING_PATH = MANIFEST_DIR / "dataverse_listing.json"
MANIFEST_PATH = MANIFEST_DIR / "manifest.csv"
SPLITS_PATH = SPLITS_DIR / "splits.json"

DATAVERSE_DOI = "doi:10.7910/DVN/XSI0QN"
DATAVERSE_API = "https://dataverse.harvard.edu/api"

# ---- Pose pipeline constants -------------------------------------------------
TARGET_FPS = 10.0            # frames sampled per second of video for pose extraction
MAX_FRAME_SIDE = 640         # frames are downscaled so the longer side is <= this before pose estimation
SEQ_LEN = 64                 # fixed temporal length after resampling for the temporal models
MIN_VALID_FRAMES = 15        # fewer detected-pose frames than this => "unable to analyze"
MIN_DETECTION_RATE = 0.6     # fraction of sampled frames with a detected pose required
MIN_MEAN_VISIBILITY = 0.5    # mean landmark visibility required over the sequence
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

# ---- Splitting -----------------------------------------------------------------
TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
CV_FOLDS = 5
SPLIT_SEED = 42

# ---- Database ----------------------------------------------------------------
DEFAULT_DB_URL = os.environ.get("PHYSIO_DB_URL", f"sqlite:///{(OUTPUTS_DIR / 'physio.db').as_posix()}")

# ---- Filename codebook --------------------------------------------------------
# Pattern documented in data/raw/MobiPhysio.md ("File naming convention").
# 'VP' is not in the README codebook but appears once in the listing; it is kept
# as an unknown variation rather than dropped.
FILENAME_RE = re.compile(
    r"^(?P<exercise>E\d{2})_(?P<participant>P\d{2})_A(?P<angle>[FLR])_V(?P<variation>[A-Z]{1,2})_G(?P<gender>[MF])$"
)
ANGLE_NAMES = {"F": "front", "L": "left", "R": "right"}
VARIATION_NAMES = {
    "FL": "full_light", "ML": "medium_light", "LL": "low_light",
    "LJ": "low_jitter", "HJ": "high_jitter", "O": "occlusion", "LR": "low_resolution",
}


@dataclass(frozen=True)
class Codebook:
    """Exercise code -> human name, read from the dataset README."""
    exercises: dict[str, str]

    @property
    def codes(self) -> list[str]:
        return sorted(self.exercises)

    def name(self, code: str) -> str:
        return self.exercises.get(code, code)


@lru_cache(maxsize=1)
def load_codebook(readme_path: Path = README_PATH) -> Codebook:
    """Parse '- **E01** Abduction' lines from the dataset README.

    Raises FileNotFoundError / ValueError if the README is missing or has no
    exercise table, because guessing exercise names is explicitly out of scope.
    """
    text = Path(readme_path).read_text(encoding="utf-8")
    pairs = re.findall(r"^\s*-\s+\*\*(E\d{2})\*\*\s+(.+?)\s*$", text, flags=re.MULTILINE)
    if not pairs:
        raise ValueError(f"No exercise codebook found in {readme_path}")
    return Codebook(exercises={code: name.strip() for code, name in pairs})


def parse_video_id(video_id: str) -> dict[str, str]:
    """Decode a MobiPhysio video stem such as 'E01_P01_AF_VFL_GM'."""
    m = FILENAME_RE.match(video_id.strip())
    if not m:
        raise ValueError(f"Filename does not follow the MobiPhysio convention: {video_id!r}")
    d = m.groupdict()
    d["angle_name"] = ANGLE_NAMES[d["angle"]]
    d["variation_name"] = VARIATION_NAMES.get(d["variation"], "unknown")
    return d


def ensure_dirs() -> None:
    for p in (RAW_DIR, POSES_DIR, MANIFEST_DIR, TMP_DIR, METRICS_DIR, MODELS_DIR, SPLITS_DIR, FIGURES_DIR):
        p.mkdir(parents=True, exist_ok=True)
