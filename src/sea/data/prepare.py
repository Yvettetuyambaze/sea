"""Unpack uploaded PhysioNet zips into the layout the loaders expect."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

import yaml

from .discover import discover_roots, find_zip, find_ptbxl_root, find_chapman_root, find_mimic_root
from .download import extract_ptbxl_from_zip


def _safe_extract_zip(zip_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(dest)
    return dest


def _safe_extract_tar(path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path) as archive:
        archive.extractall(dest)
    return dest


def prepare_ptbxl(src: Path, dest: Path) -> Path | None:
    existing = find_ptbxl_root(dest) or find_ptbxl_root(src)
    if existing is not None and any(existing.glob("records100/**/*.dat")):
        return existing
    zip_path = find_zip(src, ("ptb-xl", "ptbxl")) or find_zip(dest, ("ptb-xl", "ptbxl"))
    if zip_path is None:
        return existing
    extract_ptbxl_from_zip(zip_path, dest)
    return find_ptbxl_root(dest)


def prepare_chapman(src: Path, dest: Path) -> Path | None:
    existing = find_chapman_root(dest) or find_chapman_root(src)
    if existing is not None:
        return existing
    archive = find_zip(src, ("chapman", "shaoxing")) or find_zip(dest, ("chapman", "shaoxing"))
    if archive is None:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        _safe_extract_zip(archive, dest)
    else:
        _safe_extract_tar(archive, dest)
    return find_chapman_root(dest)


def prepare_mimic(src: Path, dest: Path, cfg: dict | None = None, prefetch: bool = False) -> Path | None:
    from .mimic_io import (
        build_mimic_index,
        extract_mimic_subset_from_zip,
        extract_mimic_tables_from_zip,
        prefetch_mimic_waveforms,
    )

    existing = find_mimic_root(dest) or find_mimic_root(src)
    zip_path = find_zip(src, ("mimic-iv-ecg", "mimic_iv_ecg", "mimic-iv-ecg-diagnostic")) or find_zip(
        dest, ("mimic-iv-ecg",)
    )
    dest.mkdir(parents=True, exist_ok=True)
    if existing is None and zip_path is not None:
        extract_mimic_tables_from_zip(zip_path, dest)
        existing = dest
    if existing is None:
        return None
    if cfg is not None and not (existing / "index.npz").exists():
        rel_paths = build_mimic_index(existing, cfg)
    else:
        rel_paths = []
        index_path = existing / "index.npz"
        if index_path.exists():
            import numpy as np

            rel_paths = [str(p) for p in np.load(index_path, allow_pickle=True)["paths"].tolist()]
    if zip_path is not None and rel_paths:
        extract_mimic_subset_from_zip(zip_path, existing, rel_paths)
    if prefetch and cfg is not None:
        prefetch_mimic_waveforms(existing, cfg)
    return existing


def prepare_uploads(
    src: Path,
    dest: Path,
    cfg: dict,
    prefetch_mimic: bool = False,
) -> dict[str, str]:
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    roots = {}
    ptbxl = prepare_ptbxl(src, dest / "ptbxl")
    chapman = prepare_chapman(src, dest / "chapman")
    mimic = prepare_mimic(src, dest / "mimic-iv-ecg", cfg=cfg, prefetch=prefetch_mimic)
    if ptbxl is None:
        ptbxl = discover_roots(src).get("ptbxl")
    if chapman is None:
        chapman = discover_roots(src).get("chapman")
    if mimic is None:
        mimic = discover_roots(src).get("mimic")
    if ptbxl is not None:
        roots["ptbxl"] = str(ptbxl)
        cfg["datasets"]["ptbxl"]["root"] = str(ptbxl)
    if chapman is not None:
        roots["chapman"] = str(chapman)
        cfg["datasets"]["chapman"]["root"] = str(chapman)
    if mimic is not None:
        roots["mimic"] = str(mimic)
        cfg["datasets"]["mimic"]["root"] = str(mimic)
        local = next(Path(mimic).glob("files/**/*.dat"), None) is not None
        cfg["datasets"]["mimic"]["stream"] = not local
    cfg["data_root"] = str(dest)
    print("Prepared dataset roots:")
    for name, path in roots.items():
        print(f"  {name}: {path}")
    missing = [name for name in ("ptbxl", "chapman", "mimic") if name not in roots]
    if missing:
        print("Still missing:", ", ".join(missing))
    return roots


def write_runtime_yaml(cfg: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path
