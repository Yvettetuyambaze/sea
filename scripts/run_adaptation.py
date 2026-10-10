#!/usr/bin/env python
"""Temperature Scaling and Linear Probing across P in {5,10,20,30,50}%.

Extracts frozen Chapman features once, then resumes from the CSV if present so a
killed CPU job does not start over.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.adapt.linear_probe import fit_linear_probe_from_features, predict_head
from sea.adapt.temperature import apply_temperature, fit_temperature
from sea.audit import summarize_adaptation
from sea.config import add_runtime_args, apply_runtime_args, load_config, resolve_config_path
from sea.data.datasets import load_bundle, make_dataset, target_adapt_eval_split
from sea.metrics import metric_bundle
from sea.train import collect_logits_and_pooled_features, load_trained_model, make_loader, resolve_device
from sea.viz import figure_dir, plot_adaptation_curves, write_html_index


def _rows(source, target, method, P, metrics, rep):
    return [
        {
            "source": source,
            "target": target,
            "method": method,
            "P": P,
            "rep": rep,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def _done_keys(frame: pd.DataFrame) -> set[tuple]:
    if frame.empty:
        return set()
    return set(zip(frame["rep"].astype(int), frame["method"], frame["P"].astype(int)))


def _append(csv_path: Path, rows: list[dict]) -> pd.DataFrame:
    new = pd.DataFrame(rows)
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        frame = pd.concat([old, new], ignore_index=True)
    else:
        frame = new
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    return frame


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

    out_dir = Path(cfg["output_dir"]) / "adaptation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{args.source}_to_{args.target}.csv"
    existing = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    done = _done_keys(existing)
    print(f"resume: {len(done)} metric-rows already on disk at {csv_path}", flush=True)

    infer_cfg = {**cfg, "train": {**cfg["train"], "batch_size": max(32, int(cfg["train"]["batch_size"]))}}
    print(f"extracting frozen features for {n} {args.target} records on {device}...", flush=True)
    y_all, logits_all, feats_all = collect_logits_and_pooled_features(
        model,
        make_loader(dataset, np.arange(n), infer_cfg, shuffle=False),
        device,
    )
    print(f"cached features {feats_all.shape} logits {logits_all.shape}", flush=True)

    for rep in range(reps):
        seed = cfg["seed"] + 1000 * rep
        adapt_idx, eval_idx = target_adapt_eval_split(n, seed, adapt_cfg["eval_fraction"])
        y_eval, logits_eval = y_all[eval_idx], logits_all[eval_idx]
        feats_eval = feats_all[eval_idx]
        bins = cfg["eval"]["ece_bins"]

        if (rep, "unadapted", 0) not in done:
            baseline = metric_bundle(y_eval, logits_eval, bins)
            _append(csv_path, _rows(args.source, args.target, "unadapted", 0, baseline, rep))
            done.update((rep, "unadapted", 0) for _ in baseline)
            print(f"rep={rep} unadapted AUROC={baseline['macro_auroc']:.3f} n_eval={len(eval_idx)}", flush=True)

        for P in adapt_cfg["percentages"]:
            ts_done = (rep, "temperature_scaling", P) in done
            lp_done = (rep, "linear_probe", P) in done
            if ts_done and lp_done:
                print(f"rep={rep} P={P}% already done, skip", flush=True)
                continue

            n_adapt = max(8, int(round(len(adapt_idx) * P / 100.0)))
            rng = np.random.default_rng(seed + P)
            take = rng.choice(adapt_idx, size=min(n_adapt, len(adapt_idx)), replace=False)

            if not ts_done:
                temperature = fit_temperature(logits_all[take], y_all[take])
                ts_metrics = metric_bundle(y_eval, apply_temperature(logits_eval, temperature), bins)
                _append(csv_path, _rows(args.source, args.target, "temperature_scaling", P, ts_metrics, rep))
                done.update((rep, "temperature_scaling", P) for _ in ts_metrics)
            else:
                ts_metrics = {"macro_auroc": float("nan")}

            if not lp_done:
                probed = fit_linear_probe_from_features(
                    feats_all[take],
                    y_all[take],
                    cfg,
                    init_head=model.head,
                )
                lp_metrics = metric_bundle(y_eval, predict_head(probed, feats_eval, cfg), bins)
                _append(csv_path, _rows(args.source, args.target, "linear_probe", P, lp_metrics, rep))
                done.update((rep, "linear_probe", P) for _ in lp_metrics)
            else:
                lp_metrics = {"macro_auroc": float("nan")}

            print(
                f"rep={rep} P={P}% TS AUROC={ts_metrics['macro_auroc']:.3f} "
                f"LP AUROC={lp_metrics['macro_auroc']:.3f} n_adapt={len(take)}",
                flush=True,
            )

    frame = pd.read_csv(csv_path)
    taus = {
        "macro_auroc": adapt_cfg["tau_auroc"],
        "brier": adapt_cfg["tau_brier"],
        "ece": adapt_cfg["tau_ece"],
    }
    summary = summarize_adaptation(frame.to_dict("records"), taus)
    summary.to_csv(out_dir / f"{args.source}_to_{args.target}_pmin.csv", index=False)
    (out_dir / f"{args.source}_to_{args.target}_summary.json").write_text(
        summary.to_json(orient="records", indent=2), encoding="utf-8"
    )
    print(f"wrote {csv_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    plot_adaptation_curves(csv_path, figure_dir(cfg["output_dir"]), args.source, args.target)
    write_html_index(cfg["output_dir"])


if __name__ == "__main__":
    main()
