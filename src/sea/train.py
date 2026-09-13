from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from .data.datasets import ECGDataset
from .metrics import metric_bundle
from .models.resnet1d import resnet1d_wang


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits_all.append(model(x).cpu().numpy())
            y_all.append(y.numpy())
    return np.concatenate(y_all), np.concatenate(logits_all)


def train_one_model(
    dataset: ECGDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_classes: int,
    cfg: dict[str, Any],
    out_dir: Path,
    tag: str,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(cfg.get("device", "auto"))
    train_cfg = cfg["train"]
    model = resnet1d_wang(
        num_classes=num_classes,
        input_channels=cfg["signal"]["n_leads"],
        inplanes=cfg["model"]["inplanes"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    train_loader = DataLoader(
        Subset(dataset, train_idx.tolist()),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx.tolist()),
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
    )

    # pos_weight stabilizes rare diagnostic / SNOMED labels
    y_train = dataset.labels[train_idx]
    pos = y_train.sum(axis=0)
    neg = len(y_train) - pos
    pos_weight = torch.tensor(np.clip(neg / np.clip(pos, 1, None), 1.0, 20.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=train_cfg.get("amp", True) and device.type == "cuda")

    best_auroc = -1.0
    best_path = out_dir / f"{tag}.pt"
    history = []
    patience = 0

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        running = 0.0
        n_seen = 0
        progress = tqdm(train_loader, desc=f"{tag} epoch {epoch}", leave=False)
        for x, y in progress:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if train_cfg.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.item()) * len(x)
            n_seen += len(x)
            progress.set_postfix(loss=f"{running / max(n_seen, 1):.4f}")
        scheduler.step()

        y_val, logits_val = collect_logits(model, val_loader, device)
        metrics = metric_bundle(y_val, logits_val, n_bins=cfg["eval"]["ece_bins"])
        metrics["epoch"] = epoch
        metrics["train_loss"] = running / max(n_seen, 1)
        history.append(metrics)
        print(f"{tag} epoch {epoch}: val AUROC={metrics['macro_auroc']:.4f} brier={metrics['brier']:.4f} ece={metrics['ece']:.4f}")

        if metrics["macro_auroc"] > best_auroc:
            best_auroc = metrics["macro_auroc"]
            patience = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "num_classes": num_classes,
                    "config": cfg,
                    "val_metrics": metrics,
                    "tag": tag,
                },
                best_path,
            )
        else:
            patience += 1
            if patience >= train_cfg.get("early_stop_patience", 8):
                print(f"{tag}: early stop at epoch {epoch}")
                break

    summary = {"best_val_auroc": best_auroc, "checkpoint": str(best_path), "history": history}
    (out_dir / f"{tag}_train.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_trained_model(checkpoint: Path, cfg: dict[str, Any], device: torch.device | None = None) -> nn.Module:
    device = device or resolve_device(cfg.get("device", "auto"))
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = resnet1d_wang(
        num_classes=payload["num_classes"],
        input_channels=cfg["signal"]["n_leads"],
        inplanes=cfg["model"]["inplanes"],
        dropout=cfg["model"]["dropout"],
    )
    model.load_state_dict(payload["model"])
    return model.to(device)
