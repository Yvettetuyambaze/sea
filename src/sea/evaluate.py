from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Subset

from .data.datasets import ECGDataset
from .metrics import bootstrap_metrics, chance_level_test
from .train import collect_logits, resolve_device


def evaluate_loader(model, loader, cfg: dict[str, Any], seed: int = 0) -> dict[str, Any]:
    device = resolve_device(cfg.get("device", "auto"))
    y_true, logits = collect_logits(model, loader, device)
    boot = bootstrap_metrics(
        y_true,
        logits,
        n_boot=cfg["eval"]["bootstrap_n"],
        seed=seed,
        n_bins=cfg["eval"]["ece_bins"],
    )
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = []
    for _ in range(cfg["eval"]["bootstrap_n"]):
        idx = rng.integers(0, n, size=n)
        from .metrics import macro_auroc

        prob = 1.0 / (1.0 + np.exp(-logits[idx]))
        draws.append(macro_auroc(y_true[idx], prob))
    chance = chance_level_test(np.asarray(draws))
    return {
        "n": int(n),
        "metrics": boot,
        "chance_test": chance,
        "y_true": y_true,
        "logits": logits,
    }


def evaluate_indices(
    model,
    dataset: ECGDataset,
    indices: np.ndarray,
    cfg: dict[str, Any],
    seed: int = 0,
) -> dict[str, Any]:
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"].get("num_workers", 0),
    )
    return evaluate_loader(model, loader, cfg, seed=seed)


def save_eval(result: dict[str, Any], path: Path, drop_arrays: bool = True) -> None:
    payload = {k: v for k, v in result.items() if not (drop_arrays and k in {"y_true", "logits"})}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def save_arrays(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, y_true=result["y_true"], logits=result["logits"])


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    raise TypeError(type(obj))
