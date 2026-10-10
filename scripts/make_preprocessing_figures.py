#!/usr/bin/env python
"""Step-by-step preprocessing and dataset-count figures for slides."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ACCENT = "#1f4e5f"
GREEN = "#2f6b4f"
ORANGE = "#c47b2b"
RED = "#a33b3b"
GREY = "#7a7a7a"


def save(fig: plt.Figure, name: str) -> Path:
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path}")
    return path


def box(ax, x, y, w, h, text, color, fs=9):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222", lw=0.8, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="white", zorder=3)


def plot_waveform_steps() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    fig.suptitle("Waveform preprocessing — the same 7 steps on every dataset", x=0.03, ha="left", fontsize=13)
    steps = [
        (0.2, "1  Read\nWFDB / CSV"),
        (1.85, "2  Skip bad\nheaders"),
        (3.5, "3  Order\n12 leads"),
        (5.15, "4  Resample\nto 100 Hz"),
        (6.8, "5  Pad / crop\nto 10 s"),
        (8.45, "6  NaN / Inf\n→ 0"),
        (10.1, "7  Z-score\neach lead"),
    ]
    colors = [ACCENT, RED, ACCENT, ACCENT, ACCENT, ORANGE, GREEN]
    for (x, text), color in zip(steps, colors):
        box(ax, x, 2.2, 1.5, 1.7, text, color, fs=9)
        if x < 10:
            ax.annotate("", xy=(x + 1.58, 3.05), xytext=(x + 1.5, 3.05), arrowprops=dict(arrowstyle="->", color="#222", lw=1.3))
    ax.text(
        6,
        0.85,
        "Output tensor for every ECG:  12 leads  ×  1,000 samples  (10 seconds at 100 Hz)\n"
        "PTB-XL 21,799    ·    Chapman 10,237    ·    Georgia 10,334",
        ha="center",
        va="center",
        fontsize=11,
    )
    save(fig, "17_waveform_preprocess_steps.png")


def plot_how_many_used() -> None:
    names = ["PTB-XL", "Chapman", "Georgia"]
    used = np.array([21799, 10237, 10334])
    skipped = np.array([0, 10, 10])
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    x = np.arange(len(names))
    ax.bar(x, used, color=GREEN, label="Used in training / eval")
    ax.bar(x, skipped, bottom=used, color=RED, label="Unreadable header (skipped)")
    for i, (u, s) in enumerate(zip(used, skipped)):
        ax.text(i, u / 2, f"{u:,}", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        if s:
            ax.text(i, u + s + 250, f"+{s} skipped", ha="center", fontsize=9, color=RED)
    ax.set_xticks(x, names)
    ax.set_ylabel("ECG records")
    ax.set_title("How many recordings were used")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 25000)
    save(fig, "18_how_many_records_used.png")


def plot_label_steps() -> None:
    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    fig.suptitle("Label preprocessing — two tasks, drop never invent", x=0.03, ha="left", fontsize=13)
    box(ax, 0.3, 5.4, 3.5, 1.4, "Raw codes in the file\nPTB-XL: SCP  ·  others: Dx:", ACCENT)
    box(ax, 4.3, 5.4, 3.4, 1.4, "Map to a published basis\nsuperclasses  or  CinC SNOMED", "#6b4f2f")
    box(ax, 8.2, 5.4, 3.5, 1.4, "Unmapped code DROPPED\nrecord stays; that class = 0", RED)
    ax.annotate("", xy=(4.2, 6.1), xytext=(3.9, 6.1), arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))
    ax.annotate("", xy=(8.1, 6.1), xytext=(7.8, 6.1), arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))
    box(ax, 0.8, 2.6, 4.6, 1.8, "Task A  ·  PTB-XL only\n44 diagnostic SCP → 5 superclasses\nNORM  MI  STTC  CD  HYP\nused n = 21,799", GREEN)
    box(ax, 6.6, 2.6, 4.6, 1.8, "Task B  ·  all three archives\n10 shared SNOMED-CT heads\nAF IAVB LBBB RBBB NSR\nPAC SB STach TAb LAD", ORANGE)
    ax.text(
        6,
        0.9,
        "Chapman: 12,475 code hits kept, 6,357 dropped   ·   Georgia: 10,707 kept, 12,745 dropped\n"
        "Never fill NORM or AF. Incomplete RBBB is not counted as RBBB.",
        ha="center",
        fontsize=10,
    )
    save(fig, "19_label_preprocess_steps.png")


def plot_splits() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    labs = ["Train\nfolds 1–8", "Val\nfold 9", "Test\nfold 10"]
    n = [17418, 2183, 2198]
    bars = ax.bar(labs, n, color=[ACCENT, GREY, GREEN])
    ax.set_title("PTB-XL official folds  (n = 21,799)")
    ax.set_ylabel("Records")
    for b, v in zip(bars, n):
        ax.text(b.get_x() + b.get_width() / 2, v + 200, f"{v:,}", ha="center", fontsize=9)
    ax.set_ylim(0, 20000)

    ax = axes[1]
    cats = ["Usable", "Frozen\ntest 30%", "Adapt\npool 70%", "P=5%", "P=20%", "P=50%"]
    chap = [10237, 3071, 7166, 358, 1433, 3583]
    geo = [10334, 3100, 7234, 362, 1447, 3617]
    x = np.arange(len(cats))
    w = 0.38
    ax.bar(x - w / 2, chap, w, color=ORANGE, label="Chapman")
    ax.bar(x + w / 2, geo, w, color=GREEN, label="Georgia")
    ax.set_xticks(x, cats)
    ax.set_title("Chapman / Georgia adaptation splits")
    ax.legend()
    save(fig, "20_splits_how_many.png")


def plot_kept_vs_dropped() -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    names = ["Chapman\nDx codes", "Georgia\nDx codes"]
    kept = [12475, 10707]
    dropped = [6357, 12745]
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, kept, w, color=GREEN, label="Mapped to the 10 SNOMED heads")
    ax.bar(x + w / 2, dropped, w, color=RED, label="Dropped (unmapped)")
    ax.set_xticks(x, names)
    ax.set_ylabel("Code occurrences (not records)")
    ax.set_title("Label codes: kept vs dropped")
    ax.legend()
    save(fig, "21_codes_kept_vs_dropped.png")


def plot_nan_audit(payload: dict) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.axis("off")
    ax.set_title("NaN / missing values after the rule", loc="left", fontsize=13)
    header = ["Dataset", "Cache shape", "Sampled recs", "NaN left", "Inf left", "All-zero leads"]
    rows = []
    for key, title in (("ptbxl", "PTB-XL"), ("chapman", "Chapman"), ("georgia", "Georgia")):
        w = payload["waveforms"][key]
        rows.append(
            [
                title,
                str(tuple(w["shape"])),
                str(w["sampled_records"]),
                str(w["nan_values"]),
                str(w["inf_values"]),
                str(w["all_zero_leads"]),
            ]
        )
    table = ax.table(cellText=rows, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#ddd")
    ax.text(
        0.5,
        0.08,
        "Rule: NaN/Inf → 0, then z-score the lead. Diagnoses are never filled in. Sampled caches have 0 NaN left.",
        ha="center",
        fontsize=10,
        transform=ax.transAxes,
    )
    save(fig, "22_nan_audit_table.png")


def plot_example_from_cache() -> None:
    path = ROOT / "results" / "cache" / "signals" / "ptbxl_fs100_t10.npy"
    if not path.exists():
        return
    arr = np.load(path, mmap_mode="r")
    x = np.array(arr[0], dtype=np.float32)
    leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    t = np.arange(x.shape[1]) / 100.0
    fig, axes = plt.subplots(6, 2, figsize=(10, 8), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.plot(t, x[i], lw=0.8, color=ACCENT)
        ax.set_ylabel(leads[i], rotation=0, ha="right", va="center")
        ax.set_yticks([])
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("After preprocessing: one PTB-XL record  (12 × 1000, z-scored)")
    save(fig, "23_processed_example_ecg.png")


def plot_js_bars() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    labs = ["PTB-XL vs\nChapman", "PTB-XL vs\nGeorgia", "Chapman vs\nGeorgia"]
    vals = [0.267, 0.257, 0.048]
    colors = [ORANGE, ORANGE, GREEN]
    bars = ax.bar(labs, vals, color=colors)
    ax.set_ylabel("Jensen–Shannon divergence")
    ax.set_title("Are the 10-label distributions the same?  (0 = identical)")
    ax.set_ylim(0, 0.35)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
    save(fig, "24_js_divergence_labels.png")


def main() -> None:
    payload = json.loads((ROOT / "results" / "eval" / "dataset_label_audit.json").read_text(encoding="utf-8"))
    plot_waveform_steps()
    plot_how_many_used()
    plot_label_steps()
    plot_splits()
    plot_kept_vs_dropped()
    plot_nan_audit(payload)
    plot_example_from_cache()
    plot_js_bars()
    print("done")


if __name__ == "__main__":
    main()
