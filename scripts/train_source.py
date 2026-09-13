#!/usr/bin/env python
"""Train a source 1D-ResNet (M2: PTB-XL diagnostic superclass)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import load_config
from sea.data.datasets import ECGDataset, load_bundle, official_ptbxl_splits, random_splits
from sea.evaluate import evaluate_indices, save_arrays, save_eval
from sea.train import load_trained_model, train_one_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--dataset", default="ptbxl")
    parser.add_argument("--task", default="superclass", choices=["superclass", "snomed_shared"])
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    bundle = load_bundle(args.dataset, cfg, args.task)
    dataset = ECGDataset(bundle.records, bundle.labels, cfg["signal"]["target_fs"], cfg["signal"]["duration_sec"])
    splits = official_ptbxl_splits(bundle) if bundle.folds is not None else random_splits(len(bundle.records), cfg["seed"])

    tag = f"{args.dataset}_{args.task}"
    out_dir = Path(cfg["output_dir"]) / "checkpoints"
    summary = train_one_model(dataset, splits["train"], splits["val"], bundle.labels.shape[1], cfg, out_dir, tag)
    model = load_trained_model(Path(summary["checkpoint"]), cfg)
    result = evaluate_indices(model, dataset, splits["test"], cfg, seed=cfg["seed"])
    eval_dir = Path(cfg["output_dir"]) / "eval"
    save_eval(result, eval_dir / f"{tag}_id.json")
    save_arrays(result, eval_dir / f"{tag}_id.npz")
    print(f"ID test macro-AUROC={result['metrics']['macro_auroc']['point']:.4f} "
          f"95% CI [{result['metrics']['macro_auroc']['ci_low']:.4f}, {result['metrics']['macro_auroc']['ci_high']:.4f}]")
    print(f"Table 2 reference for diagnostic superclass: 0.92-0.93")
    print(f"n_train={len(splits['train'])} n_val={len(splits['val'])} n_test={len(splits['test'])}")
    print(f"label prevalence: {dict(zip(bundle.label_names, bundle.labels.mean(0).round(3)))}")


if __name__ == "__main__":
    main()
