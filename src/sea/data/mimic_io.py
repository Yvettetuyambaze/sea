"""MIMIC-IV-ECG tables + optional subset. Never log report text or identifiers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from .download import MIMIC_BASE, _download_with_progress, _download_many, physionet_credentials
from .labels import codes_to_multihot, report_text_to_snomed, shared_snomed_table


def optional_credentials() -> tuple[str | None, str | None]:
    user = os.environ.get("PHYSIONET_USER", "").strip() or None
    password = os.environ.get("PHYSIONET_PASSWORD", "").strip() or None
    return user, password


def download_mimic_tables(root: Path) -> None:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    user, password = optional_credentials()
    for name in ("record_list.csv", "machine_measurements.csv"):
        dest = root / name
        if dest.exists() and dest.stat().st_size > 1_000_000:
            print(f"MIMIC table present: {name}")
            continue
        print(f"Downloading MIMIC table {name} (CSV only; not the 34 GB zip)")
        try:
            _download_with_progress(f"{MIMIC_BASE}/{name}", dest, user, password)
        except Exception:
            user, password = physionet_credentials()
            _download_with_progress(f"{MIMIC_BASE}/{name}", dest, user, password)


def _local_waveform(root: Path, rel: str) -> bool:
    stem = root / rel
    return any(Path(str(stem) + ext).exists() for ext in (".hea", ".dat"))


def build_mimic_index(root: Path, cfg: dict) -> list[str]:
    root = Path(root)
    if not (root / "record_list.csv").exists():
        from .discover import find_mimic_root

        found = find_mimic_root(root)
        if found is None:
            raise FileNotFoundError(f"record_list.csv not found under {root}")
        root = found
    mimic_cfg = (cfg.get("datasets") or {}).get("mimic") or {}
    max_records = int(mimic_cfg.get("max_records", 8000))
    seed = int(mimic_cfg.get("subsample_seed", 42))
    records = pd.read_csv(root / "record_list.csv", usecols=["study_id", "path"])
    report_cols = [f"report_{i}" for i in range(18)]
    measures = pd.read_csv(
        root / "machine_measurements.csv",
        usecols=["study_id"] + report_cols,
        dtype=str,
        keep_default_na=False,
    )
    merged = records.merge(measures, on="study_id", how="inner")
    rows = shared_snomed_table(cfg)
    texts = merged[report_cols].fillna("").astype(str).agg(" ".join, axis=1)
    y = np.asarray([codes_to_multihot(report_text_to_snomed(t), rows) for t in texts], dtype=np.float32)
    keep = y.sum(axis=1) > 0
    merged = merged.loc[keep].reset_index(drop=True)
    y = y[keep]
    n_labelled = len(merged)
    if n_labelled == 0:
        raise RuntimeError("No MIMIC records mapped to the shared SNOMED set")
    rng = np.random.default_rng(seed)
    rel_all = [str(p).replace("\\", "/").lstrip("/") for p in merged["path"].tolist()]
    local_idx = np.array([i for i, rel in enumerate(rel_all) if _local_waveform(root, rel)], dtype=int)
    pool = local_idx if len(local_idx) >= min(256, max_records) else np.arange(n_labelled)
    take = min(max_records, len(pool))
    chosen = np.sort(rng.choice(pool, size=take, replace=False))
    merged = merged.iloc[chosen].reset_index(drop=True)
    y = y[chosen]
    rel_paths = [str(p).replace("\\", "/").lstrip("/") for p in merged["path"].tolist()]
    np.savez_compressed(
        root / "index.npz",
        paths=np.asarray(rel_paths),
        labels=y,
        label_names=np.asarray([r["name"] for r in rows]),
        n_labelled_pool=np.asarray([n_labelled]),
        max_records=np.asarray([take]),
        seed=np.asarray([seed]),
    )
    print(f"Wrote MIMIC stream index n={take} from labelled pool={n_labelled}")
    print("MIMIC mapped-label counts:", y.sum(axis=0).astype(int).tolist())
    return rel_paths


def download_mimic_iv_ecg(root: Path, cfg: dict, tables_only: bool = True) -> Path:
    root = Path(root)
    download_mimic_tables(root)
    rel_paths = build_mimic_index(root, cfg)
    mimic_cfg = (cfg.get("datasets") or {}).get("mimic") or {}
    stream = bool(mimic_cfg.get("stream", True))
    if tables_only or stream:
        print("Not downloading the 34 GB zip. Waveforms stream from PhysioNet and cache under data/mimic-iv-ecg.")
        return root
    user, password = optional_credentials()
    if not user:
        user, password = physionet_credentials()
    jobs = []
    for rel in rel_paths:
        for ext in (".hea", ".dat"):
            jobs.append((f"{MIMIC_BASE}/{rel}{ext}", root / f"{rel}{ext}"))
    _download_many(jobs, desc="MIMIC-IV-ECG subset", workers=8, user=user, password=password)
    return root


def extract_mimic_tables_from_zip(zip_path: Path, dest: Path) -> Path:
    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {"record_list.csv", "machine_measurements.csv"}
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            base = Path(name).name.lower()
            if base not in wanted:
                continue
            target = dest / Path(name).name
            if target.exists() and target.stat().st_size > 1_000_000:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as incoming, target.open("wb") as outgoing:
                outgoing.write(incoming.read())
    return dest


def extract_mimic_subset_from_zip(zip_path: Path, dest: Path, rel_paths: list[str]) -> int:
    """Pull only indexed waveforms out of the 34 GB zip. Never extracts the whole archive."""
    import zipfile

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {f"{rel}{ext}" for rel in rel_paths for ext in (".hea", ".dat")}
    extracted = 0
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            norm = name.replace("\\", "/")
            rel = None
            if "/files/" in norm:
                rel = "files/" + norm.split("/files/", 1)[1]
            elif norm.startswith("files/"):
                rel = norm
            if rel not in wanted:
                continue
            target = dest / rel
            if target.exists() and target.stat().st_size > 0:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as incoming, target.open("wb") as outgoing:
                outgoing.write(incoming.read())
            extracted += 1
    print(f"Extracted {extracted} MIMIC waveform files from zip (subset only)")
    return extracted


def prefetch_mimic_waveforms(root: Path, cfg: dict) -> int:
    """Download indexed .hea/.dat files so Colab training is not per-record HTTP."""
    root = Path(root)
    index_path = root / "index.npz"
    if not index_path.exists():
        build_mimic_index(root, cfg)
    rel_paths = [str(p).replace("\\", "/").lstrip("/") for p in np.load(index_path, allow_pickle=True)["paths"].tolist()]
    user, password = optional_credentials()
    jobs = []
    for rel in rel_paths:
        for ext in (".hea", ".dat"):
            dest = root / f"{rel}{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            jobs.append((f"{MIMIC_BASE}/{rel}{ext}", dest))
    if not jobs:
        print("MIMIC waveforms already cached")
        return 0
    try:
        _download_many(jobs, desc="MIMIC prefetch", workers=8, user=user, password=password)
    except Exception:
        user, password = physionet_credentials()
        _download_many(jobs, desc="MIMIC prefetch", workers=8, user=user, password=password)
    return len(jobs)
