"""Materialize resampled 12-lead windows so Colab epochs are not WFDB-bound."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .datasets import ECGBundle, ECGDataset


class ArrayECGDataset(Dataset):
    """In-memory or memmap dataset with the same (x, y) contract as ECGDataset."""

    def __init__(self, signals: np.ndarray, labels: np.ndarray):
        self.signals = signals
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.asarray(self.signals[idx])), torch.from_numpy(self.labels[idx])


def cache_path(bundle: ECGBundle, cfg: dict[str, Any]) -> Path:
    fs = int(cfg["signal"]["target_fs"])
    duration = int(cfg["signal"]["duration_sec"])
    root = Path(cfg.get("output_dir", "results")) / "cache" / "signals"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{bundle.name}_fs{fs}_t{duration}.npy"


def materialize_dataset(bundle: ECGBundle, cfg: dict[str, Any]) -> ArrayECGDataset | ECGDataset:
    """Load-or-build a float32 array of shape (n, 12, T). Falls back to lazy WFDB."""
    if not cfg.get("cache", {}).get("signals", True):
        return ECGDataset(
            bundle.records,
            bundle.labels,
            cfg["signal"]["target_fs"],
            cfg["signal"]["duration_sec"],
            pn_dir=bundle.meta.get("pn_dir"),
            rel_paths=bundle.meta.get("rel_paths"),
            cache_root=bundle.meta.get("cache_root"),
        )

    path = cache_path(bundle, cfg)
    n = len(bundle.records)
    channels = int(cfg["signal"]["n_leads"])
    length = int(cfg["signal"]["target_fs"] * cfg["signal"]["duration_sec"])
    meta = path.with_suffix(".meta.npz")
    if path.exists() and meta.exists():
        payload = np.load(meta)
        if int(payload["n"][0]) == n and int(payload["length"][0]) == length:
            signals = np.load(path, mmap_mode="r")
            print(f"signal cache hit {path} shape={signals.shape}")
            return ArrayECGDataset(signals, bundle.labels)

    lazy = ECGDataset(
        bundle.records,
        bundle.labels,
        cfg["signal"]["target_fs"],
        cfg["signal"]["duration_sec"],
        pn_dir=bundle.meta.get("pn_dir"),
        rel_paths=bundle.meta.get("rel_paths"),
        cache_root=bundle.meta.get("cache_root"),
    )
    print(f"Building signal cache {path} (n={n}). First pass only.")
    array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, channels, length))
    failed = 0
    for i in tqdm(range(n), desc=f"cache {bundle.name}"):
        try:
            x, _ = lazy[i]
            array[i] = x.numpy()
        except Exception:
            failed += 1
            array[i] = 0.0
    array.flush()
    np.savez(meta, n=np.asarray([n]), length=np.asarray([length]), failed=np.asarray([failed]))
    if failed:
        print(f"signal cache: {failed}/{n} records failed and were zero-filled")
    return ArrayECGDataset(np.load(path, mmap_mode="r"), bundle.labels)
