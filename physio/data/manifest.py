"""Build the dataset manifest: one row per video with decoded metadata and scores.

Sources (all read from disk, nothing assumed):
  * data/manifest/dataverse_listing.json  - Harvard Dataverse file listing (file id, name, size)
  * data/raw/Physiotherapist_Exercise_Marking.csv - EAAQ scores from three physiotherapists
  * data/raw/MobiPhysio.md - exercise code -> name

Score handling: the CSV stores three judge rows per video. The row that carries the
video name also carries the judges' average on the 0-100 scale ("Average Score").
That average is the movement-quality regression target. Videos absent from the CSV
have quality_score = NaN and are usable for exercise classification only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from physio import config
from physio.config import parse_video_id

SCORE_COLUMNS = ["video", "judge", "pf1", "pf2", "pf3", "pf4", "cf1", "cf2", "cf3",
                 "total", "scale100", "avg", "avg2"]


def load_listing(path: Path = config.DATAVERSE_LISTING_PATH) -> pd.DataFrame:
    """Flatten the Dataverse JSON listing into a DataFrame of video files."""
    payload = json.loads(Path(path).read_text())
    files = payload["data"]["latestVersion"]["files"]
    rows = []
    for f in files:
        df = f["dataFile"]
        name = df["filename"]
        if not name.lower().endswith(".mp4"):
            continue
        rows.append({"file_id": int(df["id"]), "filename": name,
                     "video_id": name[:-4], "size_bytes": int(df["filesize"])})
    return pd.DataFrame(rows)


def load_scores(path: Path = config.SCORES_CSV_PATH) -> pd.DataFrame:
    """Parse the three-judge scoring CSV into one row per named video.

    Returns columns: video_id, quality_score (0-100 average), judge_scores (list of
    three per-judge 0-100 scores), n_judges.
    """
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, names=SCORE_COLUMNS)
    body = raw.iloc[2:].copy()                        # first two rows are a two-line header
    body["video"] = body["video"].str.strip()
    body["group"] = body["video"].replace("", np.nan).ffill()
    named = body[body["video"] != ""].copy()

    judge_scores = (
        body[body["group"].notna()]
        .assign(s=lambda d: pd.to_numeric(d["scale100"], errors="coerce"))
        .groupby("group")["s"].apply(lambda s: [round(float(x), 4) for x in s.dropna().tolist()])
    )
    named["quality_score"] = pd.to_numeric(named["avg"], errors="coerce")
    named["judge_scores"] = named["video"].map(judge_scores)
    named["n_judges"] = named["judge_scores"].map(lambda v: len(v) if isinstance(v, list) else 0)
    out = named.rename(columns={"video": "video_id"})[["video_id", "quality_score", "judge_scores", "n_judges"]]
    out = out.drop_duplicates("video_id", keep="first").reset_index(drop=True)
    return out


def build_manifest(listing_path: Path = config.DATAVERSE_LISTING_PATH,
                   scores_path: Path = config.SCORES_CSV_PATH,
                   readme_path: Path = config.README_PATH) -> pd.DataFrame:
    listing = load_listing(listing_path)
    meta = pd.DataFrame([parse_video_id(v) for v in listing["video_id"]])
    df = pd.concat([listing, meta], axis=1)
    codebook = config.load_codebook(readme_path)
    df["exercise_name"] = df["exercise"].map(codebook.name)

    scores = load_scores(scores_path)
    df = df.merge(scores, on="video_id", how="left")
    df["has_score"] = df["quality_score"].notna()
    df["n_judges"] = df["n_judges"].fillna(0).astype(int)
    df["judge_scores"] = df["judge_scores"].map(lambda v: json.dumps(v) if isinstance(v, list) else "[]")
    # phase 1 participants (P01-P23) recorded all/most exercises at high bitrate; later
    # participants recorded a few exercises with much smaller files. Recorded from the
    # listing itself so the split can stratify on it.
    df["participant_num"] = df["participant"].str[1:].astype(int)
    return df.sort_values("video_id").reset_index(drop=True)


def save_manifest(df: pd.DataFrame, path: Path = config.MANIFEST_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_manifest(path: Path = config.MANIFEST_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Manifest not found at {path}; run scripts/build_manifest.py first")
    return pd.read_csv(path)


def summarize(df: pd.DataFrame) -> dict:
    """Plain-dict summary used by the build script and the README."""
    return {
        "n_videos": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "n_exercises": int(df["exercise"].nunique()),
        "videos_per_exercise": df["exercise"].value_counts().sort_index().to_dict(),
        "participants_per_exercise": df.groupby("exercise")["participant"].nunique().to_dict(),
        "videos_with_quality_score": int(df["has_score"].sum()),
        "quality_score_range": [float(df["quality_score"].min()), float(df["quality_score"].max())],
        "quality_score_mean": float(df["quality_score"].mean()),
        "angles": df["angle_name"].value_counts().to_dict(),
        "variations": df["variation_name"].value_counts().to_dict(),
        "total_size_gb": round(float(df["size_bytes"].sum()) / 1e9, 2),
    }
