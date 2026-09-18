#!/usr/bin/env python
"""Unpack Drive/Colab uploads into data/ and write configs/runtime.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import load_config, save_config
from sea.data.prepare import prepare_uploads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder with zips or extracted PhysioNet datasets")
    parser.add_argument("--dst", default=str(ROOT / "data"))
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--prefetch-mimic", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    cfg["data_root"] = args.dst
    roots = prepare_uploads(Path(args.src), Path(args.dst), cfg, prefetch_mimic=args.prefetch_mimic)
    runtime = ROOT / "configs" / "runtime.yaml"
    save_config(cfg, runtime)
    print(f"Wrote {runtime}")
    if "ptbxl" not in roots:
        print("WARNING: PTB-XL not found. Milestone 2 cannot train until the zip is uploaded.")
    if "chapman" not in roots:
        print("WARNING: Chapman not found. Milestone 3 needs at least one OOD target.")
    if "mimic" not in roots:
        print("WARNING: MIMIC tables not found. OOD can still run on Chapman only.")


if __name__ == "__main__":
    main()
