#!/usr/bin/env python
"""Milestone 2: PTB-XL 1D-ResNet diagnostic superclass, official folds 1–8/9/10."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import add_runtime_args, apply_runtime_args, load_config, resolve_config_path
from sea.data.datasets import load_bundle, make_dataset, maybe_quick_splits, official_ptbxl_splits
from sea.evaluate import evaluate_indices, save_arrays, save_eval
from sea.train import load_trained_model, train_one_model
from sea.viz import eval_per_class_figure, figure_dir, plot_example_ecg, plot_label_prevalence, plot_m2_vs_table2, write_html_index


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    args = parser.parse_args()
    cfg = apply_runtime_args(load_config(resolve_config_path(args, ROOT)).raw, args)

    bundle = load_bundle("ptbxl", cfg, "superclass")
    figs = figure_dir(cfg["output_dir"])
    plot_label_prevalence(
        bundle.label_names,
        bundle.labels.mean(0),
        "M2 PTB-XL diagnostic superclass prevalence",
        figs / "01_ptbxl_superclass_prevalence.png",
    )
    dataset = make_dataset(bundle, cfg)
    splits = maybe_quick_splits(official_ptbxl_splits(bundle), cfg)
    x0, _ = dataset[0]
    plot_example_ecg(x0.numpy(), cfg["signal"]["target_fs"], figs / "02_example_12lead.png")

    out_dir = Path(cfg["output_dir"]) / "checkpoints"
    summary = train_one_model(dataset, splits["train"], splits["val"], bundle.labels.shape[1], cfg, out_dir, "ptbxl_superclass")
    model = load_trained_model(Path(summary["checkpoint"]), cfg)
    result = evaluate_indices(model, dataset, splits["test"], cfg, seed=cfg["seed"])
    eval_dir = Path(cfg["output_dir"]) / "eval"
    save_eval(result, eval_dir / "ptbxl_superclass_id.json")
    save_arrays(result, eval_dir / "ptbxl_superclass_id.npz")
    auroc = result["metrics"]["macro_auroc"]
    print(f"M2 ID test macro-AUROC={auroc['point']:.4f} 95% CI [{auroc['ci_low']:.4f}, {auroc['ci_high']:.4f}]")
    print("Table 2 reference for diagnostic superclass: 0.92-0.93")
    print(f"n_train={len(splits['train'])} n_val={len(splits['val'])} n_test={len(splits['test'])}")
    print(f"label prevalence: {dict(zip(bundle.label_names, bundle.labels.mean(0).round(3)))}")
    plot_m2_vs_table2(auroc["point"], auroc["ci_low"], auroc["ci_high"], figs / "03_m2_vs_table2.png")
    eval_per_class_figure(
        eval_dir / "ptbxl_superclass_id.npz",
        bundle.label_names,
        figs / "04_m2_per_class_auroc.png",
        "M2 test per-class AUROC (PTB-XL superclasses)",
    )
    write_html_index(cfg["output_dir"])
    if auroc["point"] < 0.85 and not cfg.get("debug", {}).get("max_split"):
        print("WARNING: AUROC is far below Table 2; check waveforms, leads, and that records100 was extracted.")


if __name__ == "__main__":
    main()
