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


def probe_logits(model: ResNet1dWang, dataset: ECGDataset, indices: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    device = resolve_device(cfg.get("device", "auto"))
    loader = make_loader(dataset, indices, cfg, shuffle=False)
    _, logits = collect_logits(model, loader, device)
    return logits
