#!/usr/bin/env python
"""Temperature Scaling and Linear Probing across P in {5,10,20,30,50}%."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.adapt.linear_probe import fit_linear_probe
from sea.adapt.temperature import apply_temperature, fit_temperature
from sea.audit import summarize_adaptation
from sea.config import add_runtime_args, apply_runtime_args, load_config, resolve_config_path
from sea.data.datasets import load_bundle, make_dataset, target_adapt_eval_split
from sea.evaluate import evaluate_indices
from sea.metrics import metric_bundle
from sea.train import collect_logits, load_trained_model, make_loader, resolve_device
from sea.viz import figure_dir, plot_adaptation_curves, write_html_index


def _rows(source, target, method, P, metrics, rep):
    out = []
    for metric, value in metrics.items():
        out.append(
            {
                "source": source,
                "target": target,
                "method": method,
                "P": P,
                "rep": rep,
                "metric": metric,
                "value": value,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    add_runtime_args(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", default="ptbxl")
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", default="snomed_shared")
    args = parser.parse_args()
    cfg = apply_runtime_args(load_config(resolve_config_path(args, ROOT)).raw, args)

    adapt_cfg = cfg["adaptation"]
    reps = args.reps or adapt_cfg["monte_carlo_reps"]
    bundle = load_bundle(args.target, cfg, args.task)
    dataset = make_dataset(bundle, cfg)
    model = load_trained_model(Path(args.checkpoint), cfg)
    device = resolve_device(cfg.get("device", "auto"))
    n = len(dataset)
    cap = int((cfg.get("debug") or {}).get("max_split", 0) or 0)
    if cap > 0:
        n = min(n, cap * 4)

    rows = []
    for rep in range(reps):
        seed = cfg["seed"] + 1000 * rep
        adapt_idx, eval_idx = target_adapt_eval_split(n, seed, adapt_cfg["eval_fraction"])
        baseline = evaluate_indices(model, dataset, eval_idx, cfg, seed=seed)
        rows.extend(_rows(args.source, args.target, "unadapted", 0, {k: v["point"] for k, v in baseline["metrics"].items()}, rep))

        y_eval, logits_eval_base = collect_logits(model, make_loader(dataset, eval_idx, cfg, shuffle=False), device)

        for P in adapt_cfg["percentages"]:
            n_adapt = max(8, int(round(len(adapt_idx) * P / 100.0)))
            rng = np.random.default_rng(seed + P)
            take = rng.choice(adapt_idx, size=min(n_adapt, len(adapt_idx)), replace=False)

            y_adapt, logits_adapt = collect_logits(model, make_loader(dataset, take, cfg, shuffle=False), device)
            temperature = fit_temperature(logits_adapt, y_adapt)
            ts_metrics = metric_bundle(y_eval, apply_temperature(logits_eval_base, temperature), cfg["eval"]["ece_bins"])
            rows.extend(_rows(args.source, args.target, "temperature_scaling", P, ts_metrics, rep))

            probed = fit_linear_probe(model, dataset, take, cfg)
            lp = evaluate_indices(probed, dataset, eval_idx, cfg, seed=seed)
            rows.extend(_rows(args.source, args.target, "linear_probe", P, {k: v["point"] for k, v in lp["metrics"].items()}, rep))
            print(f"rep={rep} P={P}% TS AUROC={ts_metrics['macro_auroc']:.3f} "
                  f"LP AUROC={lp['metrics']['macro_auroc']['point']:.3f} n_adapt={len(take)}")

    out_dir = Path(cfg["output_dir"]) / "adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.source}_to_{args.target}.csv"
    frame.to_csv(csv_path, index=False)
    taus = {
        "macro_auroc": adapt_cfg["tau_auroc"],
        "brier": adapt_cfg["tau_brier"],
        "ece": adapt_cfg["tau_ece"],
    }
    summary = summarize_adaptation(rows, taus)
    summary.to_csv(out_dir / f"{args.source}_to_{args.target}_pmin.csv", index=False)
    (out_dir / f"{args.source}_to_{args.target}_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    print(f"wrote {csv_path}")
    print(summary.to_string(index=False))
    plot_adaptation_curves(csv_path, figure_dir(cfg["output_dir"]), args.source, args.target)
    write_html_index(cfg["output_dir"])


if __name__ == "__main__":
    main()
