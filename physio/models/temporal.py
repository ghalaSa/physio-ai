"""Compact temporal architectures for pose sequences (proposal section 7.2).

All encoders map (B, L, F) -> (B, D) embeddings. Heads:
  * classification: (B, n_classes) logits
  * regression:     (B,) standardized quality score
`MultiTaskNet` shares one encoder between the two heads.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 96, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.out_dim = hidden * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)                       # (B, L, 2H)
        return h.mean(1)


class TCNEncoder(nn.Module):
    """Temporal 1D CNN with dilated residual blocks."""

    def __init__(self, in_dim: int, channels: int = 96, blocks: int = 4, kernel: int = 5, dropout: float = 0.3):
        super().__init__()
        self.inp = nn.Conv1d(in_dim, channels, 1)
        layers = []
        for i in range(blocks):
            d = 2 ** i
            layers.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel, padding=d * (kernel - 1) // 2, dilation=d),
                nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel, padding=d * (kernel - 1) // 2, dilation=d),
                nn.BatchNorm1d(channels)))
        self.blocks = nn.ModuleList(layers)
        self.act = nn.GELU()
        self.out_dim = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x.transpose(1, 2))           # (B, C, L)
        for blk in self.blocks:
            h = self.act(h + blk(h))
        return h.mean(2)


class TransformerEncoder(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 96, heads: int = 4, layers: int = 2, dropout: float = 0.3, max_len: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=d_model * 2, dropout=dropout,
                                           batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out_dim = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x) + self.pos[:, : x.shape[1]]
        return self.enc(h).mean(1)


ENCODERS = {"bilstm": BiLSTMEncoder, "tcn": TCNEncoder, "transformer": TransformerEncoder}


def build_encoder(name: str, in_dim: int, **kw) -> nn.Module:
    if name not in ENCODERS:
        raise ValueError(f"unknown encoder {name!r}; choose from {list(ENCODERS)}")
    return ENCODERS[name](in_dim, **kw)


class MultiTaskNet(nn.Module):
    """Shared encoder with optional classification and regression heads."""

    def __init__(self, encoder: nn.Module, n_classes: int | None, regression: bool, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        d = encoder.out_dim
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, n_classes)) if n_classes else None
        self.reg_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, 1)) if regression else None

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        out = {}
        if self.cls_head is not None:
            out["logits"] = self.cls_head(z)
        if self.reg_head is not None:
            out["score"] = self.reg_head(z).squeeze(-1)
        return out


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
