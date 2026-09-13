#!/usr/bin/env python
"""Unadapted OOD evaluation with bootstrap CIs and a chance-level test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import load_config
from sea.data.datasets import ECGDataset, load_bundle
from sea.evaluate import evaluate_indices, save_arrays, save_eval
from sea.train import load_trained_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", default="snomed_shared")
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    bundle = load_bundle(args.target, cfg, args.task)
    dataset = ECGDataset(bundle.records, bundle.labels, cfg["signal"]["target_fs"], cfg["signal"]["duration_sec"])
    model = load_trained_model(Path(args.checkpoint), cfg)
    idx = __import__("numpy").arange(len(dataset))
    result = evaluate_indices(model, dataset, idx, cfg, seed=cfg["seed"])
    stem = Path(args.checkpoint).stem
    out = Path(cfg["output_dir"]) / "eval" / f"{stem}_ood_{args.target}.json"
    save_eval(result, out)
    save_arrays(result, out.with_suffix(".npz"))
    auroc = result["metrics"]["macro_auroc"]
    print(f"OOD {stem} -> {args.target}: AUROC={auroc['point']:.4f} "
          f"95% CI [{auroc['ci_low']:.4f}, {auroc['ci_high']:.4f}]")
    print(f"chance test: {result['chance_test']}")
    print(f"n={result['n']} labels={bundle.label_names}")


if __name__ == "__main__":
    main()
