"""Linear probing: freeze the 1D-ResNet backbone and retrain the classification head."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from ..data.datasets import ECGDataset
from ..models.resnet1d import ResNet1dWang
from ..train import collect_logits, resolve_device


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
    loader = DataLoader(
        Subset(dataset, adapt_idx.tolist()),
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"].get("num_workers", 0),
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
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(probed(x), y)
            loss.backward()
            optimizer.step()
    probed.eval()
    return probed


def probe_logits(model: ResNet1dWang, dataset: ECGDataset, indices: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    device = resolve_device(cfg.get("device", "auto"))
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    _, logits = collect_logits(model, loader, device)
    return logits
