import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physio import config  # noqa: E402
from physio.models.temporal import ENCODERS, MultiTaskNet, build_encoder, count_params  # noqa: E402
from physio.models.train import TrainConfig, fit_temporal, load_temporal  # noqa: E402
from physio.pose.normalize import N_FEATURES, normalize_sequence  # noqa: E402
from physio.pose.synthetic import synthetic_sequence  # noqa: E402


@pytest.mark.parametrize("name", list(ENCODERS))
def test_encoders_shapes(name):
    enc = build_encoder(name, N_FEATURES)
    net = MultiTaskNet(enc, n_classes=9, regression=True)
    out = net(torch.randn(4, config.SEQ_LEN, N_FEATURES))
    assert out["logits"].shape == (4, 9) and out["score"].shape == (4,)
    assert count_params(net) < 2_000_000


def test_train_and_reload_multitask(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(config, "METRICS_DIR", tmp_path)
    import physio.models.train as T
    monkeypatch.setattr(T, "RUNS_LOG", tmp_path / "runs.jsonl")
    X, yc, yr = [], [], []
    for side, label in (("right", 0), ("left", 1)):
        for seed in range(8):
            ps = synthetic_sequence(n_frames=80, side=side, seed=seed, amplitude_deg=50 + 4 * seed)
            X.append(normalize_sequence(ps.landmarks, ps.detected, 720, 1280)["features"])
            yc.append(label); yr.append(50.0 + 4 * seed)
    X, yc, yr = np.stack(X), np.array(yc), np.array(yr, float)
    # classes here are defined by body side, so mirror augmentation must be off for this test
    cfg = TrainConfig(run_name="t", task="multi", encoder="tcn", epochs=30, batch_size=8, patience=30, augment=False)
    rec = fit_temporal(cfg, X, yc, yr, X, yc, yr, ["E01", "E02"], ex_va=np.array(["E01"] * 8 + ["E02"] * 8))
    assert rec["val"]["cls"]["macro_f1"] > 0.6
    assert (tmp_path / "t.pt").exists() and (tmp_path / "runs.jsonl").exists()
    net, ck = load_temporal(tmp_path / "t.pt")
    x = torch.tensor((X[:2] - ck["feat_mu"]) / ck["feat_sd"], dtype=torch.float32)
    out = net(x)
    assert out["logits"].shape == (2, 2)
