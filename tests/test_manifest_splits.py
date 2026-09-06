import numpy as np
import pytest

from physio import config
from physio.config import load_codebook, parse_video_id
from physio.data.manifest import load_scores
from physio.data.splits import assert_no_leakage, attach_split, make_splits


def test_codebook_has_nine_exercises_from_readme():
    cb = load_codebook()
    assert len(cb.codes) == 9
    assert cb.codes[0] == "E01" and cb.codes[-1] == "E09"
    assert all(name for name in cb.exercises.values())


def test_parse_video_id():
    d = parse_video_id("E03_P12_AL_VHJ_GF")
    assert d == {"exercise": "E03", "participant": "P12", "angle": "L", "variation": "HJ", "gender": "F",
                 "angle_name": "left", "variation_name": "high_jitter"}
    with pytest.raises(ValueError):
        parse_video_id("not_a_video")


def test_scores_parse_three_judges_and_scale():
    s = load_scores()
    assert len(s) > 2900
    assert s["quality_score"].between(0, 100).all()
    assert (s["n_judges"] == 3).mean() > 0.99
    row = s.set_index("video_id").loc["E01_P01_AF_VFL_GM"]
    assert abs(row["quality_score"] - 79.52) < 0.01
    assert len(row["judge_scores"]) == 3


def test_manifest_shape(manifest_df):
    df = manifest_df
    assert len(df) == 3686
    assert df["participant"].nunique() == 58
    assert df["exercise"].nunique() == 9
    assert df["has_score"].sum() == 3010
    assert df["quality_score"].dropna().between(0, 100).all()


def test_splits_are_participant_independent(manifest_df):
    sp = make_splits(manifest_df, seed=config.SPLIT_SEED)
    df = attach_split(manifest_df, sp)
    assert_no_leakage(df)
    assert set(sp.train) | set(sp.val) | set(sp.test) == set(df["participant"])
    for split in ("train", "val", "test"):
        assert df.loc[df["split"] == split, "exercise"].nunique() == 9
    assert set(sp.cv_folds) == set(sp.train) | set(sp.val)
    assert set(sp.cv_folds.values()) == set(range(config.CV_FOLDS))


def test_leakage_test_catches_a_leak(manifest_df):
    sp = make_splits(manifest_df)
    df = attach_split(manifest_df, sp)
    leaked = df.copy()
    pid = sp.test[0]
    idx = leaked.index[leaked["participant"] == pid][:3]
    leaked.loc[idx, "split"] = "train"
    with pytest.raises(AssertionError):
        assert_no_leakage(leaked)


def test_splits_deterministic(manifest_df):
    a = make_splits(manifest_df, seed=7)
    b = make_splits(manifest_df, seed=7)
    assert a.test == b.test and a.train == b.train and np.all(list(a.cv_folds.values()) == list(b.cv_folds.values()))
