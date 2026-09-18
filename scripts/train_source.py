#!/usr/bin/env python
"""Train a source 1D-ResNet (M2 superclass or M3 shared SNOMED)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import add_runtime_args, apply_runtime_args, load_config, resolve_config_path
from sea.data.datasets import load_bundle, make_dataset, maybe_quick_splits, official_ptbxl_splits, random_splits
from sea.evaluate import evaluate_indices, save_arrays, save_eval
from sea.train import load_trained_model, train_one_model
from sea.viz import eval_per_class_figure, figure_dir, plot_label_prevalence, write_html_index


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument("--dataset", default="ptbxl")
    parser.add_argument("--task", default="superclass", choices=["superclass", "snomed_shared"])
    args = parser.parse_args()
    cfg = apply_runtime_args(load_config(resolve_config_path(args, ROOT)).raw, args)

    bundle = load_bundle(args.dataset, cfg, args.task)
    dataset = make_dataset(bundle, cfg)
    splits = official_ptbxl_splits(bundle) if bundle.folds is not None else random_splits(len(bundle.records), cfg["seed"])
    splits = maybe_quick_splits(splits, cfg)

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
    if args.task == "superclass":
        print("Table 2 reference for diagnostic superclass: 0.92-0.93")
    print(f"n_train={len(splits['train'])} n_val={len(splits['val'])} n_test={len(splits['test'])}")
    print(f"label prevalence: {dict(zip(bundle.label_names, bundle.labels.mean(0).round(3)))}")
    figs = figure_dir(cfg["output_dir"])
    plot_label_prevalence(
        bundle.label_names,
        bundle.labels.mean(0),
        f"{tag} label prevalence",
        figs / f"{tag}_prevalence.png",
    )
    eval_per_class_figure(
        eval_dir / f"{tag}_id.npz",
        bundle.label_names,
        figs / f"{tag}_per_class_auroc.png",
        f"{tag} test per-class AUROC",
    )
    write_html_index(cfg["output_dir"])


if __name__ == "__main__":
    main()
