#!/usr/bin/env python
"""Download ECG datasets. Default MIMIC mode is tables-only + PhysioNet streaming."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import load_config
from sea.data.download import download_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["ptbxl", "chapman", "mimic"],
        help="ptbxl, chapman, georgia, cpsc, mimic",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        default=True,
        help="MIMIC: download CSVs only and stream waveforms (skip the 34 GB zip)",
    )
    parser.add_argument(
        "--waveforms",
        action="store_true",
        help="MIMIC: also prefetch the subset waveforms instead of streaming",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    tables_only = not args.waveforms
    for name in args.datasets:
        dest = Path(cfg.raw["datasets"][name]["root"])
        print(f"Downloading {name} -> {dest}")
        download_dataset(name, dest, cfg.raw, tables_only=tables_only)
        print(f"Done: {name}")


if __name__ == "__main__":
    main()
