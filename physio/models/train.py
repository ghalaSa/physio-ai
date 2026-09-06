"""Training loops and experiment tracking.

* `fit_sklearn` handles the baselines.
* `fit_temporal` trains a MultiTaskNet with early stopping on the validation split.
* Every run is appended to outputs/metrics/runs.jsonl (experiment tracking) and its
  full metrics go to outputs/metrics/<run_name>.json.
* Model artefacts (weights + feature normalization + label mapping) are saved under
  outputs/models/ so the inference pipeline can reload them without the training code.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from physio import config
from physio.models.evaluate import classification_metrics, regression_metrics
from physio.pose.normalize import FLIP_INDEX

RUNS_LOG = config.METRICS_DIR / "runs.jsonl"


@dataclass
class TrainConfig:
    run_name: str
    task: str                      # "cls" | "reg" | "multi"
    encoder: str = "tcn"           # bilstm | tcn | transformer
    epochs: int = 60
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-3
    dropout: float = 0.3
    patience: int = 12
    seed: int = 0
    augment: bool = True
    reg_weight: float = 1.0        # multitask loss weight for regression
    encoder_kwargs: dict = field(default_factory=dict)
    notes: str = ""


def log_run(record: dict) -> None:
    RUNS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNS_LOG, "a") as fh:
        fh.write(json.dumps(record, default=float) + "\n")
    (config.METRICS_DIR / f"{record['run_name']}.json").write_text(json.dumps(record, indent=2, default=float))


# --------------------------------------------------------------------------- sklearn
def fit_sklearn(name: str, model, task: str, Xtr, ytr, Xva, yva, classes, ex_va=None, extra: dict | None = None) -> dict:
    import joblib
    t0 = time.time()
    model.fit(Xtr, ytr)
    pred = model.predict(Xva)
    if task == "cls":
        val = classification_metrics(yva, pred, classes)
        key = "macro_f1"
    else:
        pred = np.clip(pred, 0, 100)
        val = regression_metrics(yva, pred, ex_va)
        key = "mae"
    rec = {"run_name": name, "task": task, "family": "sklearn", "model": type(model[-1]).__name__,
           "n_train": int(len(ytr)), "n_val": int(len(yva)), "val": val, "selection_metric": key,
           "selection_value": val[key], "train_seconds": round(time.time() - t0, 1), **(extra or {})}
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "classes": classes, "task": task}, config.MODELS_DIR / f"{name}.joblib")
    log_run(rec)
    return rec


# --------------------------------------------------------------------------- torch
def _augment(x, rng):
    """x: (B, L, 165) tensor. Random mirror + gaussian jitter + small time-shift."""
    import torch
    B, L, F = x.shape
    if rng.random() < 0.5:
        xy = x[:, :, :66].reshape(B, L, 33, 2)[:, :, FLIP_INDEX].clone()
        xy[..., 0] *= -1
        vis = x[:, :, 66:99][:, :, FLIP_INDEX]
        vel = torch.zeros_like(xy)
        vel[:, 1:] = xy[:, 1:] - xy[:, :-1]
        x = torch.cat([xy.reshape(B, L, 66), vis, vel.reshape(B, L, 66)], dim=2)
    x = x + 0.01 * torch.randn_like(x)
    shift = int(rng.integers(-L // 8, L // 8 + 1))
    if shift:
        x = torch.roll(x, shifts=shift, dims=1)
    return x


def fit_temporal(cfg: TrainConfig, Xtr, ytr_cls, ytr_reg, Xva, yva_cls, yva_reg, classes, ex_va=None, device: str | None = None) -> dict:
    import torch
    import torch.nn.functional as Fn
    from physio.models.temporal import MultiTaskNet, build_encoder, count_params

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    do_cls = cfg.task in ("cls", "multi")
    do_reg = cfg.task in ("reg", "multi")

    # feature standardization from the training split only
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    reg_mu = float(np.nanmean(ytr_reg)) if do_reg else 0.0
    reg_sd = float(np.nanstd(ytr_reg) + 1e-6) if do_reg else 1.0

    def prep(X):
        return torch.tensor((X - mu) / sd, dtype=torch.float32)

    Xt, Xv = prep(Xtr), prep(Xva).to(device)
    yc = torch.tensor(ytr_cls, dtype=torch.long)
    yr = torch.tensor(np.nan_to_num((ytr_reg - reg_mu) / reg_sd, nan=0.0), dtype=torch.float32)
    mr = torch.tensor(~np.isnan(ytr_reg), dtype=torch.float32)

    enc = build_encoder(cfg.encoder, Xtr.shape[-1], dropout=cfg.dropout, **cfg.encoder_kwargs)
    net = MultiTaskNet(enc, len(classes) if do_cls else None, do_reg, dropout=cfg.dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    class_w = None
    if do_cls:
        counts = np.bincount(ytr_cls, minlength=len(classes)).astype(float)
        class_w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32, device=device)

    def evaluate():
        net.eval()
        with torch.no_grad():
            out = net(Xv)
        res = {}
        if do_cls:
            pred = out["logits"].argmax(1).cpu().numpy()
            res["cls"] = classification_metrics(yva_cls, pred, classes)
        if do_reg:
            m = ~np.isnan(yva_reg)
            pred = np.clip(out["score"].cpu().numpy() * reg_sd + reg_mu, 0, 100)
            res["reg"] = regression_metrics(yva_reg[m], pred[m], None if ex_va is None else ex_va[m])
        return res

    def selection(res):
        if cfg.task == "cls":
            return res["cls"]["macro_f1"], "macro_f1", True
        if cfg.task == "reg":
            return -res["reg"]["mae"], "mae", False
        return res["cls"]["macro_f1"] - res["reg"]["mae"] / 100.0, "macro_f1 - mae/100", True

    best, best_state, best_epoch, bad, history = -1e9, None, 0, 0, []
    n = len(Xt)
    t0 = time.time()
    for epoch in range(cfg.epochs):
        net.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = Xt[idx]
            if cfg.augment:
                xb = _augment(xb, rng)
            xb = xb.to(device)
            out = net(xb)
            loss = 0.0
            if do_cls:
                loss = loss + Fn.cross_entropy(out["logits"], yc[idx].to(device), weight=class_w, label_smoothing=0.05)
            if do_reg:
                mb = mr[idx].to(device)
                l1 = Fn.smooth_l1_loss(out["score"], yr[idx].to(device), reduction="none")
                loss = loss + cfg.reg_weight * (l1 * mb).sum() / mb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
            opt.step()
            tot += float(loss) * len(idx)
        sched.step()
        res = evaluate()
        score, key, _ = selection(res)
        history.append({"epoch": epoch, "train_loss": tot / n, "val_selection": score})
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    net.load_state_dict(best_state)
    val = evaluate()
    score, key, _ = selection(val)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    art = config.MODELS_DIR / f"{cfg.run_name}.pt"
    torch.save({"state_dict": best_state, "cfg": asdict(cfg), "classes": classes, "in_dim": int(Xtr.shape[-1]),
                "feat_mu": mu.astype(np.float32), "feat_sd": sd.astype(np.float32),
                "reg_mu": reg_mu, "reg_sd": reg_sd, "seq_len": int(Xtr.shape[1])}, art)
    rec = {"run_name": cfg.run_name, "task": cfg.task, "family": "torch", "model": cfg.encoder,
           "params": count_params(net), "n_train": int(n), "n_val": int(len(Xv)), "epochs_run": len(history),
           "best_epoch": best_epoch, "val": val, "selection_metric": key,
           "selection_value": val["cls"]["macro_f1"] if cfg.task == "cls" else (val["reg"]["mae"] if cfg.task == "reg" else score),
           "train_seconds": round(time.time() - t0, 1), "config": asdict(cfg), "history": history, "artifact": art.name}
    log_run(rec)
    return rec


def load_temporal(path: Path, device: str = "cpu"):
    """Reload a trained temporal model for inference. Returns (net, meta)."""
    import torch
    from physio.models.temporal import MultiTaskNet, build_encoder
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    do_cls = cfg["task"] in ("cls", "multi")
    do_reg = cfg["task"] in ("reg", "multi")
    enc = build_encoder(cfg["encoder"], ck["in_dim"], dropout=cfg["dropout"], **cfg.get("encoder_kwargs", {}))
    net = MultiTaskNet(enc, len(ck["classes"]) if do_cls else None, do_reg, dropout=cfg["dropout"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    return net, ck
