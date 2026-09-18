#!/usr/bin/env python
"""Unadapted OOD evaluation with bootstrap CIs and a chance-level test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import add_runtime_args, apply_runtime_args, load_config, resolve_config_path
from sea.data.datasets import load_bundle, make_dataset
from sea.evaluate import evaluate_indices, save_arrays, save_eval
from sea.train import load_trained_model
from sea.viz import eval_per_class_figure, figure_dir, plot_id_vs_ood, write_html_index


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", default="snomed_shared")
    args = parser.parse_args()
    cfg = apply_runtime_args(load_config(resolve_config_path(args, ROOT)).raw, args)

    bundle = load_bundle(args.target, cfg, args.task)
    dataset = make_dataset(bundle, cfg)
    model = load_trained_model(Path(args.checkpoint), cfg)
    idx = np.arange(len(dataset))
    cap = int((cfg.get("debug") or {}).get("max_split", 0) or 0)
    if cap > 0:
        idx = idx[:cap]
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
    figs = figure_dir(cfg["output_dir"])
    eval_per_class_figure(
        out.with_suffix(".npz"),
        bundle.label_names,
        figs / f"06_ood_{args.target}_per_class.png",
        f"Unadapted OOD per-class AUROC ({stem} → {args.target})",
    )
    id_json = Path(cfg["output_dir"]) / "eval" / "ptbxl_snomed_shared_id.json"
    rows = []
    if id_json.exists():
        import json

        id_payload = json.loads(id_json.read_text(encoding="utf-8"))
        rows.append({"label": "PTB-XL ID", "auroc": id_payload["metrics"]["macro_auroc"]["point"]})
    rows.append({"label": f"OOD {args.target}", "auroc": auroc["point"]})
    plot_id_vs_ood(rows, figs / f"05_id_vs_ood_{args.target}.png")
    write_html_index(cfg["output_dir"])


if __name__ == "__main__":
    main()
