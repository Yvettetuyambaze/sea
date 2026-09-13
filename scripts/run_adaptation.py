#!/usr/bin/env python
"""Temperature Scaling and Linear Probing across P in {5,10,20,30,50}%."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.adapt.linear_probe import fit_linear_probe
from sea.adapt.temperature import apply_temperature, fit_temperature
from sea.audit import summarize_adaptation
from sea.config import load_config
from sea.data.datasets import ECGDataset, load_bundle, target_adapt_eval_split
from sea.evaluate import evaluate_indices
from sea.metrics import metric_bundle
from sea.train import collect_logits, load_trained_model, resolve_device


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
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", default="ptbxl")
    parser.add_argument("--target", required=True)
    parser.add_argument("--task", default="snomed_shared")
    parser.add_argument("--reps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config).raw
    adapt_cfg = cfg["adaptation"]
    reps = args.reps or adapt_cfg["monte_carlo_reps"]
    bundle = load_bundle(args.target, cfg, args.task)
    dataset = ECGDataset(bundle.records, bundle.labels, cfg["signal"]["target_fs"], cfg["signal"]["duration_sec"])
    model = load_trained_model(Path(args.checkpoint), cfg)
    device = resolve_device(cfg.get("device", "auto"))

    rows = []
    for rep in range(reps):
        seed = cfg["seed"] + 1000 * rep
        adapt_idx, eval_idx = target_adapt_eval_split(len(dataset), seed, adapt_cfg["eval_fraction"])
        baseline = evaluate_indices(model, dataset, eval_idx, cfg, seed=seed)
        rows.extend(_rows(args.source, args.target, "unadapted", 0, {k: v["point"] for k, v in baseline["metrics"].items()}, rep))

        eval_loader = DataLoader(
            Subset(dataset, eval_idx.tolist()),
            batch_size=cfg["train"]["batch_size"],
            shuffle=False,
            num_workers=cfg["train"].get("num_workers", 0),
        )
        y_eval, logits_eval_base = collect_logits(model, eval_loader, device)

        for P in adapt_cfg["percentages"]:
            n_adapt = max(8, int(round(len(adapt_idx) * P / 100.0)))
            rng = np.random.default_rng(seed + P)
            take = rng.choice(adapt_idx, size=min(n_adapt, len(adapt_idx)), replace=False)

            adapt_loader = DataLoader(
                Subset(dataset, take.tolist()),
                batch_size=cfg["train"]["batch_size"],
                shuffle=False,
                num_workers=cfg["train"].get("num_workers", 0),
            )
            y_adapt, logits_adapt = collect_logits(model, adapt_loader, device)
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


if __name__ == "__main__":
    main()
