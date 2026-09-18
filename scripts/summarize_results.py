#!/usr/bin/env python
"""Print M2 Table-2 check and M3 adaptation Pmin tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    root = Path(args.output_dir)

    m2 = root / "eval" / "ptbxl_superclass_id.json"
    if m2.exists():
        payload = json.loads(m2.read_text(encoding="utf-8"))
        auroc = payload["metrics"]["macro_auroc"]
        print("=== Milestone 2 (PTB-XL diagnostic superclass) ===")
        print(f"macro-AUROC {auroc['point']:.4f}  95% CI [{auroc['ci_low']:.4f}, {auroc['ci_high']:.4f}]")
        print(f"n={payload['n']}  Table 2 target band: 0.92–0.93")
        print()

    eval_dir = root / "eval"
    if eval_dir.exists():
        print("=== Unadapted OOD ===")
        for path in sorted(eval_dir.glob("*_ood_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            auroc = payload["metrics"]["macro_auroc"]
            print(f"{path.stem}: AUROC={auroc['point']:.4f} "
                  f"CI [{auroc['ci_low']:.4f}, {auroc['ci_high']:.4f}] "
                  f"chance_p={payload['chance_test'].get('p_boot')}")
        print()

    adapt_dir = root / "adaptation"
    if adapt_dir.exists():
        print("=== Adaptation Pmin (proposal Eqs. 2–3) ===")
        for path in sorted(adapt_dir.glob("*_pmin.csv")):
            print(path.name)
            print(pd.read_csv(path).to_string(index=False))
            print()


if __name__ == "__main__":
    main()
