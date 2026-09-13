#!/usr/bin/env python
"""Download public ECG datasets. MIMIC-IV-ECG is credentialed and is skipped."""

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
        help="Public datasets to download. Use georgia as the US stand-in for MIMIC-IV-ECG.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    for name in args.datasets:
        dest = Path(cfg.raw["datasets"][name]["root"])
        print(f"Downloading {name} -> {dest}")
        download_dataset(name, dest)
        print(f"Done: {name}")


if __name__ == "__main__":
    main()
