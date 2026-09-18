from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import wfdb

from .discover import find_chapman_root, find_mimic_root, find_ptbxl_root
from .labels import (
    SUPERCLASS_LABELS,
    CHAPMAN_RHYTHM_TO_SNOMED,
    codes_to_multihot,
    scp_dict_to_snomed,
    scp_dict_to_superclasses,
    shared_snomed_table,
)
from .preprocess import STANDARD_LEADS, prepare_record


@dataclass
class ECGBundle:
    name: str
    task: str
    records: list[str]
    labels: np.ndarray
    label_names: list[str]
    folds: np.ndarray | None
    meta: dict[str, Any]


class ECGDataset(Dataset):
    def __init__(
        self,
        paths: list[str],
        labels: np.ndarray,
        target_fs: float = 100.0,
        duration: float = 10.0,
        pn_dir: str | None = None,
        rel_paths: list[str] | None = None,
        cache_root: str | Path | None = None,
    ):
        self.paths = paths
        self.rel_paths = rel_paths or [""] * len(paths)
        self.pn_dir = pn_dir
        self.cache_root = Path(cache_root) if cache_root else None
        self.labels = labels.astype(np.float32)
        self.target_fs = target_fs
        self.duration = duration

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal, fields = _read_waveform(
            self.paths[idx],
            self.rel_paths[idx],
            self.pn_dir,
            self.cache_root,
        )
        x = prepare_record(signal, fields["sig_name"], float(fields["fs"]), self.target_fs, self.duration)
        return torch.from_numpy(x), torch.from_numpy(self.labels[idx])


def make_lazy_dataset(bundle: ECGBundle, cfg: dict[str, Any]) -> ECGDataset:
    return ECGDataset(
        bundle.records,
        bundle.labels,
        cfg["signal"]["target_fs"],
        cfg["signal"]["duration_sec"],
        pn_dir=bundle.meta.get("pn_dir"),
        rel_paths=bundle.meta.get("rel_paths"),
        cache_root=bundle.meta.get("cache_root"),
    )


def make_dataset(bundle: ECGBundle, cfg: dict[str, Any]):
    if cfg.get("cache", {}).get("signals", True):
        from .cache import materialize_dataset

        return materialize_dataset(bundle, cfg)
    return make_lazy_dataset(bundle, cfg)


def _local_record_stem(path: str) -> Path | None:
    raw = Path(path)
    candidates = [
        Path(str(path) + ".hea"),
        raw.with_suffix(".hea"),
        raw if raw.suffix == ".hea" else None,
        Path(str(path) + ".csv"),
        raw.with_suffix(".csv"),
    ]
    for header in candidates:
        if header is not None and header.exists():
            return header.with_suffix("") if header.suffix == ".hea" else header
    return None


def _read_csv_ecg(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    frame = pd.read_csv(path, header=None)
    if frame.shape[1] == 1:
        frame = pd.read_csv(path)
    values = frame.to_numpy(dtype=np.float32)
    if values.shape[0] < values.shape[1] and values.shape[0] <= 15:
        values = values.T
    names = list(frame.columns) if frame.shape[1] == 12 and not all(str(c).isdigit() for c in frame.columns) else STANDARD_LEADS
    return values, {"sig_name": [str(n) for n in names[: values.shape[1]]], "fs": 500.0}


def _read_waveform(path: str, rel: str, pn_dir: str | None, cache_root: Path | None):
    local = _local_record_stem(path)
    if local is not None:
        if local.suffix == ".csv" or Path(str(local) + ".csv").exists():
            csv_path = local if local.suffix == ".csv" else Path(str(local) + ".csv")
            if csv_path.exists():
                return _read_csv_ecg(csv_path)
        return wfdb.rdsamp(str(local))
    if not pn_dir or not rel:
        raise FileNotFoundError(f"ECG record is not local: {path}")
    rel = rel.replace("\\", "/").lstrip("/")
    rec_name = Path(rel).name
    parent = str(Path(rel).parent).replace("\\", "/")
    if cache_root is not None:
        try:
            wfdb.dl_files(pn_dir, str(cache_root), [f"{rel}.hea", f"{rel}.dat"], keep_subdirs=True)
            cached = cache_root / rel
            if _local_record_stem(str(cached)) is not None:
                return wfdb.rdsamp(str(cached))
        except Exception:
            pass
    return wfdb.rdsamp(rec_name, pn_dir=f"{pn_dir}/{parent}")


def load_bundle(name: str, cfg: dict[str, Any], task: str) -> ECGBundle:
    name = name.lower()
    if name == "ptbxl":
        return load_ptbxl(_resolve_ptbxl(Path(cfg["datasets"]["ptbxl"]["root"])), cfg, task)
    if name == "mimic":
        return load_mimic(_resolve_mimic(Path(cfg["datasets"]["mimic"]["root"])), cfg, task)
    if name not in cfg["datasets"]:
        raise ValueError(f"Unknown dataset {name}")
    return load_challenge_dir(name, _resolve_chapman(Path(cfg["datasets"][name]["root"])), cfg, task)


def _resolve_ptbxl(root: Path) -> Path:
    found = find_ptbxl_root(root) if root.exists() else None
    if found is None and root.exists():
        found = find_ptbxl_root(root.parent)
    if found is None:
        raise FileNotFoundError(
            f"PTB-XL tables not found under {root}. Upload/extract the zip so "
            "ptbxl_database.csv is visible, then re-run scripts/prepare_uploads.py"
        )
    return found


def _resolve_mimic(root: Path) -> Path:
    found = find_mimic_root(root) if root.exists() else None
    if found is None and root.exists():
        found = find_mimic_root(root.parent)
    if found is None:
        raise FileNotFoundError(
            f"MIMIC tables not found under {root}. Upload record_list.csv and "
            "machine_measurements.csv (not the 34 GB zip) or extract them first."
        )
    return found


def _resolve_chapman(root: Path) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Chapman folder missing: {root}")
    found = find_chapman_root(root) or find_chapman_root(root.parent)
    return found or root


def _record_has_waveform(path: str) -> bool:
    stem = Path(path)
    return any(
        candidate.exists()
        for candidate in (
            Path(str(path) + ".hea"),
            stem.with_suffix(".hea"),
            Path(str(path) + ".dat"),
            Path(str(path) + ".mat"),
            stem.with_suffix(".mat"),
            Path(str(path) + ".csv"),
            stem.with_suffix(".csv"),
        )
    )


def load_ptbxl(root: Path, cfg: dict[str, Any], task: str) -> ECGBundle:
    root = Path(root)
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    sampling_rate = int(cfg["datasets"]["ptbxl"].get("sampling_rate", 100))
    file_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    paths = [str(root / p) for p in db[file_col].tolist()]
    present = np.array([_record_has_waveform(p) for p in paths], dtype=bool)
    if not present.any():
        raise FileNotFoundError(f"No PTB-XL waveforms under {root}/records100. Extract the 100 Hz records.")
    if not present.all():
        print(f"PTB-XL: keeping {int(present.sum())}/{len(paths)} records with local waveforms")
        db = db.loc[present].copy()
        paths = [p for p, keep in zip(paths, present) if keep]
    folds = db["strat_fold"].to_numpy()

    if task == "superclass":
        agg = pd.read_csv(root / "scp_statements.csv", index_col=0)
        agg = agg[agg.diagnostic == 1]
        agg_map = agg["diagnostic_class"].to_dict()
        y = np.zeros((len(db), len(SUPERCLASS_LABELS)), dtype=np.float32)
        for i, codes in enumerate(db.scp_codes):
            classes = scp_dict_to_superclasses(codes, agg_map)
            for cls in classes:
                if cls in SUPERCLASS_LABELS:
                    y[i, SUPERCLASS_LABELS.index(cls)] = 1.0
        return ECGBundle("ptbxl", task, paths, y, SUPERCLASS_LABELS, folds, {"n": len(paths), "root": str(root)})

    if task == "snomed_shared":
        rows = shared_snomed_table(cfg)
        y = np.zeros((len(db), len(rows)), dtype=np.float32)
        for i, codes in enumerate(db.scp_codes):
            y[i] = codes_to_multihot(scp_dict_to_snomed(codes), rows)
        return ECGBundle("ptbxl", task, paths, y, [r["name"] for r in rows], folds, {"n": len(paths), "root": str(root)})

    raise ValueError(f"Unsupported PTB-XL task: {task}")


def load_challenge_dir(name: str, root: Path, cfg: dict[str, Any], task: str) -> ECGBundle:
    if task != "snomed_shared":
        raise ValueError(f"{name} currently supports task=snomed_shared only")
    root = Path(root)
    zheng = _diagnostics_table(root)
    if zheng is not None and not any(root.rglob("JS*.hea")):
        return load_chapman_zheng(name, root, cfg, zheng, task)

    rows = shared_snomed_table(cfg)
    heas = sorted(root.rglob("*.hea"))
    if not heas:
        raise FileNotFoundError(f"No WFDB headers in {root}. Upload Chapman (CinC 2021 or Zheng files).")
    paths, labels = [], []
    skipped = 0
    for hea in heas:
        record = str(hea.with_suffix(""))
        if not (Path(record + ".mat").exists() or Path(record + ".dat").exists() or hea.with_suffix(".mat").exists()):
            continue
        header = _safe_rdheader(record)
        if header is None:
            skipped += 1
            continue
        codes = _header_snomed(header)
        paths.append(record)
        labels.append(codes_to_multihot(codes, rows))
    if skipped:
        print(f"{name}: skipped {skipped} unreadable WFDB headers", flush=True)
    if not paths:
        raise FileNotFoundError(f"Found headers in {root} but no matching .mat/.dat waveforms")
    y = np.asarray(labels, dtype=np.float32)
    print(f"{name}: {len(paths)} WFDB records, label counts={y.sum(axis=0).astype(int).tolist()}")
    return ECGBundle(name, task, paths, y, [r["name"] for r in rows], None, {"n": len(paths), "root": str(root)})


def _diagnostics_table(root: Path) -> pd.DataFrame | None:
    for name in ("Diagnostics.xlsx", "Diagnostics.csv", "diagnostics.csv", "Diagnostics.xls"):
        path = root / name
        if path.exists():
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            return pd.read_excel(path)
    return None


def load_chapman_zheng(name: str, root: Path, cfg: dict[str, Any], diag: pd.DataFrame, task: str) -> ECGBundle:
    rows = shared_snomed_table(cfg)
    file_col = next((c for c in diag.columns if str(c).lower() in {"filename", "file_name", "file"}), None)
    rhythm_col = next((c for c in diag.columns if str(c).lower() in {"rhythm", "rhythms"}), None)
    if file_col is None:
        raise ValueError(f"Chapman Diagnostics table at {root} has no FileName column")
    search_dirs = [root, root / "ECGData", root / "ECGDataDenoised", root / "ecg_data"]
    paths, labels = [], []
    for _, row in diag.iterrows():
        filename = str(row[file_col]).strip()
        stem = Path(filename).stem
        csv_path = None
        for folder in search_dirs:
            for candidate in (folder / filename, folder / f"{filename}.csv", folder / f"{stem}.csv"):
                if candidate.exists() and candidate.is_file():
                    csv_path = candidate
                    break
            if csv_path is not None:
                break
        if csv_path is None:
            continue
        codes = []
        if rhythm_col is not None:
            rhythm = str(row[rhythm_col]).strip().upper()
            if rhythm in CHAPMAN_RHYTHM_TO_SNOMED:
                codes.append(CHAPMAN_RHYTHM_TO_SNOMED[rhythm])
        paths.append(str(csv_path.with_suffix("")))
        labels.append(codes_to_multihot(codes, rows))
    if not paths:
        raise FileNotFoundError(f"Chapman Diagnostics found at {root} but no ECG CSV files")
    y = np.asarray(labels, dtype=np.float32)
    print(f"{name}: {len(paths)} Zheng CSV records, label counts={y.sum(axis=0).astype(int).tolist()}")
    return ECGBundle(name, task, paths, y, [r["name"] for r in rows], None, {"n": len(paths), "root": str(root), "format": "zheng_csv"})


def load_mimic(root: Path, cfg: dict[str, Any], task: str) -> ECGBundle:
    """Load a MIMIC-IV-ECG subset. Waveforms may be local or streamed from PhysioNet."""
    if task != "snomed_shared":
        raise ValueError("MIMIC currently supports task=snomed_shared only")
    root = Path(root)
    index_path = root / "index.npz"
    mimic_cfg = cfg.get("datasets", {}).get("mimic", {})
    pn_dir = mimic_cfg.get("pn_dir", "mimic-iv-ecg/1.0")
    if not index_path.exists():
        from .mimic_io import build_mimic_index

        print("MIMIC index missing; building it locally from CSV tables (report text is not logged).")
        build_mimic_index(root, cfg)
    payload = np.load(index_path, allow_pickle=True)
    rel_paths = [str(p).replace("\\", "/").lstrip("/") for p in payload["paths"].tolist()]
    labels = np.asarray(payload["labels"], dtype=np.float32)
    names = [str(n) for n in payload["label_names"].tolist()]
    abs_paths = [str(root / p) for p in rel_paths]
    local_hits = sum(1 for p in abs_paths if _record_has_waveform(p))
    stream = bool(mimic_cfg.get("stream", True)) and local_hits < max(8, int(0.5 * len(abs_paths)))
    meta = {
        "n": len(abs_paths),
        "n_labelled_pool": int(payload["n_labelled_pool"][0]) if "n_labelled_pool" in payload.files else len(abs_paths),
        "subsample_seed": int(payload["seed"][0]) if "seed" in payload.files else None,
        "rel_paths": rel_paths,
        "cache_root": str(root),
        "pn_dir": pn_dir if stream else None,
        "stream": stream,
        "local_waveforms": local_hits,
    }
    print(f"MIMIC n={len(abs_paths)} local_waveforms={local_hits} stream={stream}")
    return ECGBundle("mimic", task, abs_paths, labels, names, None, meta)


_SKIPPED_HEADER_PRINTS = 0


def _safe_rdheader(record: str):
    """Return a WFDB header, or None when the file is truncated or not WFDB."""
    global _SKIPPED_HEADER_PRINTS
    try:
        return wfdb.rdheader(record)
    except Exception as exc:
        if _SKIPPED_HEADER_PRINTS < 8:
            print(f"skipping unreadable header {Path(record).name}: {exc}", flush=True)
            _SKIPPED_HEADER_PRINTS += 1
        return None


def _header_snomed(header) -> list[str]:
    codes = []
    comments = list(getattr(header, "comments", []) or [])
    for line in comments:
        text = line.strip()
        if text.startswith("Dx:") or text.startswith("Dx "):
            payload = text.split(":", 1)[-1]
            for token in payload.replace(" ", "").split(","):
                if token.isdigit():
                    codes.append(token)
    return codes


def official_ptbxl_splits(bundle: ECGBundle) -> dict[str, np.ndarray]:
    """Strodthoff folds: 1-8 train, 9 val, 10 test."""
    if bundle.folds is None:
        raise ValueError("Official PTB-XL splits require strat_fold")
    folds = bundle.folds
    return {
        "train": np.where(folds <= 8)[0],
        "val": np.where(folds == 9)[0],
        "test": np.where(folds == 10)[0],
    }


def random_splits(n: int, seed: int, val_frac: float = 0.1, test_frac: float = 0.2) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    return {
        "test": idx[:n_test],
        "val": idx[n_test : n_test + n_val],
        "train": idx[n_test + n_val :],
    }


def target_adapt_eval_split(
    n: int, seed: int, eval_fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_eval = max(1, int(n * eval_fraction))
    return idx[n_eval:], idx[:n_eval]


def maybe_quick_splits(splits: dict[str, np.ndarray], cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    n = int((cfg.get("debug") or {}).get("max_split", 0) or 0)
    if n <= 0:
        return splits
    return {key: value[: min(len(value), n)] for key, value in splits.items()}
