#!/usr/bin/env python
"""Download the public archives used in the current runs: PTB-XL, Chapman, Georgia."""

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
        default=["ptbxl", "chapman", "georgia"],
        help="ptbxl, chapman, georgia (public archives used in this project)",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        default=True,
        help="If a future credentialed archive is requested: tables only, no zip",
    )
    parser.add_argument(
        "--waveforms",
        action="store_true",
        help="Prefetch waveforms instead of streaming (unused in the current public runs)",
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
