"""Linear probing: freeze the 1D-ResNet backbone and retrain the classification head."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ..data.datasets import ECGDataset
from ..models.resnet1d import ResNet1dWang
from ..train import collect_logits, make_loader, resolve_device


def fit_linear_probe(
    model: ResNet1dWang,
    dataset: ECGDataset,
    adapt_idx: np.ndarray,
    cfg: dict[str, Any],
) -> ResNet1dWang:
    device = resolve_device(cfg.get("device", "auto"))
    probed = deepcopy(model).to(device)
    probed.freeze_backbone()
    for param in probed.head.parameters():
        param.requires_grad = True
    probed.head.train()

    adapt_cfg = cfg["adaptation"]
    batch_size = int(cfg["train"]["batch_size"])
    loader = make_loader(
        dataset,
        adapt_idx,
        cfg,
        shuffle=True,
        drop_last=len(adapt_idx) > batch_size,
    )
    y_adapt = dataset.labels[adapt_idx]
    pos = y_adapt.sum(axis=0)
    neg = len(y_adapt) - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 1.0, 20.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, probed.parameters()),
        lr=adapt_cfg["linear_probe_lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    epochs = adapt_cfg["linear_probe_epochs"]
    probed.train()
    probed.freeze_backbone()
    for _ in range(epochs):
        for x, y in loader:
            if x.size(0) < 2:
                continue
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probed(x), y)
            loss.backward()
            optimizer.step()
    probed.eval()
    return probed


class _HeadOnFeatures(nn.Module):
    """Same BN + dropout + linear stack as ClassifierHead, on already-pooled features."""

    def __init__(self, in_features: int, num_classes: int, dropout: float):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(self.bn(x)))


def fit_linear_probe_from_features(
    features: np.ndarray,
    labels: np.ndarray,
    cfg: dict[str, Any],
    init_head: nn.Module | None = None,
) -> _HeadOnFeatures:
    """Retrain the classification head on cached frozen features (no ResNet forwards)."""
    device = resolve_device(cfg.get("device", "auto"))
    feats = torch.as_tensor(features, dtype=torch.float32)
    y = torch.as_tensor(labels, dtype=torch.float32)
    head = _HeadOnFeatures(feats.shape[1], y.shape[1], cfg["model"].get("dropout", 0.5)).to(device)
    if init_head is not None:
        head.bn.load_state_dict(init_head.bn.state_dict())
        head.fc.load_state_dict(init_head.fc.state_dict())

    adapt_cfg = cfg["adaptation"]
    batch_size = max(2, min(int(cfg["train"]["batch_size"]), len(feats)))
    pos = labels.sum(axis=0)
    neg = len(labels) - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 1.0, 20.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=adapt_cfg["linear_probe_lr"], weight_decay=cfg["train"]["weight_decay"])

    head.train()
    n = len(feats)
    for _ in range(adapt_cfg["linear_probe_epochs"]):
        order = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            if idx.numel() < 2:
                continue
            xb = feats[idx].to(device)
            yb = y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(head(xb), yb)
            loss.backward()
            optimizer.step()
    head.eval()
    return head


def predict_head(head: nn.Module, features: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    device = resolve_device(cfg.get("device", "auto"))
    head.eval()
    with torch.no_grad():
        return head(torch.as_tensor(features, dtype=torch.float32, device=device)).cpu().numpy()


def probe_logits(model: ResNet1dWang, dataset: ECGDataset, indices: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    device = resolve_device(cfg.get("device", "auto"))
    loader = make_loader(dataset, indices, cfg, shuffle=False)
    _, logits = collect_logits(model, loader, device)
    return logits
