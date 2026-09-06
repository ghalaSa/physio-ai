"""Participant-independent data splitting (proposal section 7.3).

Rules enforced here:
  * A participant's videos are never split across train / val / test.
  * The test participants are chosen once, saved, and never used for model selection.
  * Every exercise must appear in every split; seeds are searched until this holds.
  * Grouped K-fold assignments over the non-test participants are stored alongside so
    cross-validation is also participant-independent.
  * `assert_no_leakage` is the automated leakage test used by the test-suite and by
    every training script before it touches data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from physio import config


@dataclass
class Splits:
    seed: int
    train: list[str]
    val: list[str]
    test: list[str]
    cv_folds: dict[str, int] = field(default_factory=dict)   # participant -> fold index (non-test only)
    notes: str = ""

    def split_of(self, participant: str) -> str:
        if participant in self.test:
            return "test"
        if participant in self.val:
            return "val"
        if participant in self.train:
            return "train"
        raise KeyError(f"Participant {participant} not assigned to any split")

    def to_json(self, path: Path = config.SPLITS_PATH) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def from_json(cls, path: Path = config.SPLITS_PATH) -> "Splits":
        return cls(**json.loads(Path(path).read_text()))


def _stratum(participant_num: int) -> str:
    # Recording phase is a strong covariate (file size, exercise coverage, resolution),
    # so both phases are represented in each split. Boundary read from the manifest
    # statistics: participants <= 23 recorded most exercises at large file sizes.
    return "phase1" if participant_num <= 23 else "phase2"


def _assign(participants: pd.DataFrame, rng: np.random.Generator) -> tuple[list[str], list[str], list[str]]:
    train, val, test = [], [], []
    for _, grp in participants.groupby("stratum"):
        ids = grp["participant"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_test = max(1, round(n * config.TEST_FRACTION))
        n_val = max(1, round(n * config.VAL_FRACTION))
        test += ids[:n_test]
        val += ids[n_test:n_test + n_val]
        train += ids[n_test + n_val:]
    return sorted(train), sorted(val), sorted(test)


def _covers_all_exercises(manifest: pd.DataFrame, ids: list[str], exercises: list[str], min_videos: int = 1) -> bool:
    sub = manifest[manifest["participant"].isin(ids)]
    counts = sub["exercise"].value_counts()
    return all(counts.get(e, 0) >= min_videos for e in exercises)


def make_splits(manifest: pd.DataFrame, seed: int = config.SPLIT_SEED, max_tries: int = 500) -> Splits:
    exercises = sorted(manifest["exercise"].unique())
    participants = (manifest.groupby("participant")["participant_num"].first().reset_index())
    participants["stratum"] = participants["participant_num"].map(_stratum)

    for attempt in range(max_tries):
        rng = np.random.default_rng(seed + attempt)
        train, val, test = _assign(participants, rng)
        if all(_covers_all_exercises(manifest, ids, exercises) for ids in (train, val, test)):
            break
    else:
        raise RuntimeError("Could not find a participant split covering all exercises in every subset")

    # Grouped K-fold over the development (train + val) participants, per stratum.
    dev = participants[participants["participant"].isin(train + val)].copy()
    folds: dict[str, int] = {}
    for _, grp in dev.groupby("stratum"):
        ids = grp["participant"].tolist()
        rng.shuffle(ids)
        for i, pid in enumerate(ids):
            folds[pid] = i % config.CV_FOLDS

    return Splits(seed=seed + attempt, train=train, val=val, test=test, cv_folds=folds,
                  notes=f"participant-level split; strata by recording phase; found on attempt {attempt}")


def attach_split(manifest: pd.DataFrame, splits: Splits) -> pd.DataFrame:
    df = manifest.copy()
    df["split"] = df["participant"].map(splits.split_of)
    df["cv_fold"] = df["participant"].map(lambda p: splits.cv_folds.get(p, -1))
    return df


def assert_no_leakage(df: pd.DataFrame) -> None:
    """Raise AssertionError if any participant appears in more than one split."""
    per_participant = df.groupby("participant")["split"].nunique()
    leaked = per_participant[per_participant > 1]
    assert leaked.empty, f"Participant leakage across splits: {leaked.index.tolist()}"
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        pa = set(df.loc[df["split"] == a, "participant"])
        pb = set(df.loc[df["split"] == b, "participant"])
        assert not (pa & pb), f"Participants shared between {a} and {b}: {sorted(pa & pb)}"
    # folds: a participant has exactly one fold id
    dev = df[df["split"] != "test"]
    assert (dev.groupby("participant")["cv_fold"].nunique() == 1).all(), "Participant spans multiple CV folds"
    assert (df.loc[df["split"] == "test", "cv_fold"] == -1).all(), "Test participants must not be in CV folds"


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Videos and participants per split and exercise, for reporting."""
    videos = df.pivot_table(index="exercise", columns="split", values="video_id", aggfunc="count", fill_value=0)
    parts = df.groupby("split")["participant"].nunique().rename("participants")
    videos.loc["participants"] = parts.reindex(videos.columns).fillna(0).astype(int)
    return videos
