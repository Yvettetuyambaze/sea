"""Step-by-step figures for Milestones 2 and 3. No patient identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import per_class_auroc


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 10,
    }
)


def figure_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir) / "figures"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"figure: {path}")
    return path


def plot_pipeline_board(status: dict[str, str], output_dir: str | Path) -> Path:
    steps = [
        "1 Download / extract",
        "2 Label inventory",
        "3 Example 12-lead",
        "4 M2 train (superclass)",
        "5 M2 vs Table 2",
        "6 M3 source (SNOMED)",
        "7 Unadapted OOD",
        "8 TS + linear probe",
    ]
    colors = []
    for step in steps:
        key = step.split(" ", 1)[0]
        state = status.get(key, status.get(step, "pending"))
        if state == "done":
            colors.append("#1f7a4d")
        elif state in {"running", "in_progress"}:
            colors.append("#b26a00")
        elif state == "failed":
            colors.append("#a33b3b")
        else:
            colors.append("#7a7a7a")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.barh(list(reversed(steps)), [1] * len(steps), color=list(reversed(colors)))
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title("SEA local run: Milestone 2 then Milestone 3")
    return _save(fig, figure_dir(output_dir) / "00_pipeline_steps.png")


def plot_label_prevalence(names: Iterable[str], rates: np.ndarray, title: str, path: Path) -> Path:
    names = list(names)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(names, rates.tolist())
    ax.set_ylabel("Prevalence")
    ax.set_xlabel("Label")
    ax.set_title(title)
    ax.set_ylim(0, max(0.05, float(np.max(rates)) * 1.15))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, path)


def plot_example_ecg(x: np.ndarray, fs: float, path: Path, title: str = "Example 12-lead window (10 s, 100 Hz)") -> Path:
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    t = np.arange(x.shape[1]) / fs
    fig, axes = plt.subplots(6, 2, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, x[i], lw=0.8)
        ax.set_ylabel(leads[i], rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle(title)
    return _save(fig, path)


def plot_training_history(history: list[dict[str, Any]], path: Path, title: str) -> Path:
    if not history:
        return path
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    axes[0].plot(epochs, [h["macro_auroc"] for h in history], marker="o")
    axes[0].set_title("Validation macro-AUROC")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0.45, 1.0)
    axes[1].plot(epochs, [h["brier"] for h in history], marker="o", color="C1")
    axes[1].set_title("Validation Brier score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Brier")
    axes[2].plot(epochs, [h["ece"] for h in history], marker="o", color="C2")
    axes[2].set_title("Validation ECE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("ECE")
    fig.suptitle(title)
    return _save(fig, path)


def plot_m2_vs_table2(point: float, ci_low: float, ci_high: float, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(["This run"], [point], yerr=[[point - ci_low], [ci_high - point]], capsize=6, color="C0")
    ax.axhspan(0.92, 0.93, color="C2", alpha=0.25, label="Table 2 band 0.92–0.93")
    ax.axhline(0.925, color="C2", ls="--", lw=1)
    ax.set_ylabel("Macro-AUROC")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("M2 PTB-XL diagnostic superclass vs Table 2")
    ax.legend(loc="lower right")
    return _save(fig, path)


def plot_per_class_auroc(names: list[str], aucs: dict[str, float], path: Path, title: str) -> Path:
    vals = [aucs.get(n, float("nan")) for n in names]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(names, vals)
    ax.axhline(0.5, color="0.4", ls="--", lw=1, label="Chance")
    ax.set_ylabel("AUROC")
    ax.set_xlabel("Label")
    ax.set_ylim(0.4, 1.0)
    ax.set_title(title)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, path)


def plot_id_vs_ood(rows: list[dict[str, Any]], path: Path) -> Path:
    labels = [r["label"] for r in rows]
    vals = [r["auroc"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.bar(labels, vals)
    ax.axhline(0.5, color="0.4", ls="--", label="Chance")
    ax.set_ylabel("Macro-AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("In-distribution vs unadapted OOD macro-AUROC")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)


def plot_adaptation_curves(csv_path: Path, out_dir: Path, source: str, target: str) -> list[Path]:
    frame = pd.read_csv(csv_path)
    written = []
    metric_titles = {
        "macro_auroc": ("Macro-AUROC", True),
        "brier": ("Brier score (lower is better)", False),
        "ece": ("ECE (lower is better)", False),
    }
    for metric, (ylabel, is_auroc) in metric_titles.items():
        part = frame[frame["metric"] == metric]
        if part.empty:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4.0))
        for method, sub in part.groupby("method"):
            if method == "unadapted":
                continue
            stats = sub.groupby("P")["value"].agg(["mean", "std"]).sort_index()
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"].fillna(0), marker="o", label=method.replace("_", " "))
        base = part[part["method"] == "unadapted"]["value"].mean()
        if not np.isnan(base):
            ax.axhline(base, color="0.35", ls="--", label="Unadapted")
        if is_auroc:
            ax.axhline(0.5, color="0.6", ls=":", label="Chance")
            ax.set_ylim(0.45, 1.0)
        ax.set_xlabel("Target labels P (% of adaptation pool)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"M3 {source} → {target}: {ylabel}")
        ax.legend()
        written.append(_save(fig, out_dir / f"m3_{source}_to_{target}_{metric}.png"))
    return written


def eval_per_class_figure(npz_path: Path, names: list[str], png_path: Path, title: str) -> Path | None:
    if not npz_path.exists():
        return None
    payload = np.load(npz_path)
    y = payload["y_true"]
    logits = payload["logits"]
    prob = 1.0 / (1.0 + np.exp(-logits))
    aucs = per_class_auroc(y, prob, names)
    return plot_per_class_auroc(names, aucs, png_path, title)


def write_html_index(output_dir: str | Path) -> Path:
    fig_dir = figure_dir(output_dir)
    images = sorted(fig_dir.glob("*.png"))
    rows = []
    for img in images:
        rows.append(f'<section><h2>{img.stem.replace("_", " ")}</h2><p><img src="{img.name}" style="max-width:100%;border:1px solid #ddd"/></p></section>')
    html = "<html><head><meta charset='utf-8'><title>SEA M2/M3 figures</title></head><body>"
    html += "<h1>SEA Milestones 2 and 3 — step-by-step figures</h1>"
    html += "".join(rows) + "</body></html>"
    path = fig_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    print(f"figure index: {path}")
    return path


def maybe_plot_history_file(train_json: Path, png_path: Path, title: str) -> Path | None:
    if not train_json.exists():
        return None
    payload = json.loads(train_json.read_text(encoding="utf-8"))
    return plot_training_history(payload.get("history") or [], png_path, title)
