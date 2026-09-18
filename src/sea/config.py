from __future__ import annotations

import argparse
import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"])

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["output_dir"])


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw = apply_env_overrides(raw)
    return Config(raw=raw, path=path)


def apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    data_root = os.environ.get("SEA_DATA_ROOT", "").strip()
    output_dir = os.environ.get("SEA_OUTPUT_DIR", "").strip()
    if data_root:
        rebase_dataset_roots(raw, Path(data_root))
    if output_dir:
        raw["output_dir"] = output_dir
    return raw


def rebase_dataset_roots(raw: dict[str, Any], data_root: Path) -> dict[str, Any]:
    data_root = Path(data_root)
    raw["data_root"] = str(data_root)
    defaults = {
        "ptbxl": data_root / "ptbxl",
        "chapman": data_root / "chapman",
        "georgia": data_root / "georgia",
        "cpsc": data_root / "cpsc",
        "mimic": data_root / "mimic-iv-ecg",
    }
    for name, folder in defaults.items():
        if name in raw.get("datasets", {}):
            current = Path(raw["datasets"][name]["root"])
            if not current.is_absolute():
                raw["datasets"][name]["root"] = str(folder)
    return raw


def apply_quick(raw: dict[str, Any]) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    raw.setdefault("debug", {})
    raw["debug"]["max_split"] = 128
    raw["train"]["epochs"] = min(int(raw["train"]["epochs"]), 2)
    raw["adaptation"]["monte_carlo_reps"] = 1
    raw["adaptation"]["linear_probe_epochs"] = 2
    raw["eval"]["bootstrap_n"] = 32
    if "mimic" in raw.get("datasets", {}):
        raw["datasets"]["mimic"]["max_records"] = min(int(raw["datasets"]["mimic"].get("max_records", 8000)), 128)
    return raw


def apply_runtime_args(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    raw = copy.deepcopy(raw)
    if getattr(args, "data_root", None):
        rebase_dataset_roots(raw, Path(args.data_root))
    if getattr(args, "output_dir", None):
        raw["output_dir"] = args.output_dir
    if getattr(args, "epochs", None):
        raw["train"]["epochs"] = args.epochs
    if getattr(args, "batch_size", None):
        raw["train"]["batch_size"] = args.batch_size
    if getattr(args, "reps", None):
        raw["adaptation"]["monte_carlo_reps"] = args.reps
    if getattr(args, "quick", False):
        raw = apply_quick(raw)
    return raw


def add_runtime_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Tiny split/epochs for a Colab wiring check")
    return parser


def resolve_config_path(args: argparse.Namespace, repo_root: Path) -> Path:
    if getattr(args, "config", None):
        return Path(args.config)
    runtime = repo_root / "configs" / "runtime.yaml"
    if runtime.exists():
        return runtime
    colab = repo_root / "configs" / "colab.yaml"
    if colab.exists() and Path("/content").exists():
        return colab
    return repo_root / "configs" / "default.yaml"


def save_config(raw: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path
