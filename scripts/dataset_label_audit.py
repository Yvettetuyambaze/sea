#!/usr/bin/env python
"""Label counts, prevalence tests, and missing-value audit for midterm feedback."""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, entropy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sea.config import load_config
from sea.data.datasets import (
    _header_snomed,
    _safe_rdheader,
    load_bundle,
    official_ptbxl_splits,
)
from sea.data.labels import EQUIVALENT_SNOMED, SCP_TO_SNOMED, SUPERCLASS_LABELS, canonical_snomed
from sea.viz import figure_dir, write_html_index

SNOMED_NAMES = ["AF", "IAVB", "LBBB", "RBBB", "NSR", "PAC", "SB", "STach", "TAb", "LAD"]
ACCENT = "#1f4e5f"
COLORS = ["#1f4e5f", "#c47b2b", "#2f6b4f"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = (z / den) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * entropy(p, m) + 0.5 * entropy(q, m))


def summarize_multihot(name: str, y: np.ndarray, label_names: list[str]) -> dict:
    n, c = y.shape
    counts = y.sum(axis=0).astype(int)
    prev = counts / max(n, 1)
    card = y.sum(axis=1)
    rows = []
    for i, lab in enumerate(label_names):
        lo, hi = wilson_ci(int(counts[i]), n)
        rows.append(
            {
                "label": lab,
                "positives": int(counts[i]),
                "negatives": int(n - counts[i]),
                "prevalence": float(prev[i]),
                "ci_low": lo,
                "ci_high": hi,
                "scorable": bool(counts[i] > 0 and counts[i] < n),
            }
        )
    return {
        "dataset": name,
        "n": int(n),
        "n_labels": int(c),
        "labels": rows,
        "mean_labels_per_record": float(card.mean()),
        "median_labels_per_record": float(np.median(card)),
        "records_with_zero_mapped_labels": int((card == 0).sum()),
        "records_with_zero_mapped_frac": float((card == 0).mean()),
        "cardinality_hist": {str(int(k)): int(v) for k, v in zip(*np.unique(card, return_counts=True))},
    }


def pair_tests(a: dict, b: dict) -> list[dict]:
    out = []
    na, nb = a["n"], b["n"]
    for ra, rb in zip(a["labels"], b["labels"]):
        table = np.array(
            [
                [ra["positives"], ra["negatives"]],
                [rb["positives"], rb["negatives"]],
            ],
            dtype=float,
        )
        if table.min() == 0:
            chi2, p, cramer = float("nan"), float("nan"), float("nan")
        else:
            chi2, p, _, _ = chi2_contingency(table, correction=True)
            cramer = float(np.sqrt(chi2 / (table.sum() * (min(table.shape) - 1))))
        out.append(
            {
                "label": ra["label"],
                "left": a["dataset"],
                "right": b["dataset"],
                "prev_left": ra["prevalence"],
                "prev_right": rb["prevalence"],
                "abs_diff": abs(ra["prevalence"] - rb["prevalence"]),
                "chi2": float(chi2) if chi2 == chi2 else None,
                "p_value": float(p) if p == p else None,
                "cramers_v": cramer if cramer == cramer else None,
            }
        )
    return out


def ptbxl_superclass_and_scp(root: Path) -> dict:
    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    db.scp_codes = db.scp_codes.apply(ast.literal_eval)
    agg = pd.read_csv(root / "scp_statements.csv", index_col=0)
    diagnostic = agg[agg.diagnostic == 1]
    agg_map = diagnostic["diagnostic_class"].to_dict()
    y = np.zeros((len(db), len(SUPERCLASS_LABELS)), dtype=np.float32)
    scp_counter = Counter()
    mapped_diag = Counter()
    unmapped_diag = Counter()
    for i, codes in enumerate(db.scp_codes):
        for key in codes:
            scp_counter[key] += 1
            if key in agg_map:
                mapped_diag[key] += 1
                cls = agg_map[key]
                if cls in SUPERCLASS_LABELS:
                    y[i, SUPERCLASS_LABELS.index(cls)] = 1.0
            else:
                unmapped_diag[key] += 1
    per_class_scp = {cls: [] for cls in SUPERCLASS_LABELS}
    for key, cls in agg_map.items():
        if cls in per_class_scp:
            per_class_scp[cls].append(key)

    demo = {}
    for col in ("age", "sex"):
        if col in db.columns:
            s = pd.to_numeric(db[col], errors="coerce") if col == "age" else db[col]
            demo[col] = {
                "n": int(len(db)),
                "missing": int(pd.isna(s).sum()),
                "missing_frac": float(pd.isna(s).mean()),
            }
            if col == "age":
                valid = s.dropna()
                demo[col].update(
                    {
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "median": float(valid.median()),
                        "min": float(valid.min()),
                        "max": float(valid.max()),
                    }
                )
    if "sex" in db.columns:
        demo["sex"]["counts"] = {str(k): int(v) for k, v in db["sex"].value_counts(dropna=False).items()}

    folds = db["strat_fold"].to_numpy()
    return {
        "n_records": int(len(db)),
        "y": y,
        "unique_scp_codes": int(len(scp_counter)),
        "diagnostic_scp_codes": int(len(agg_map)),
        "non_diagnostic_scp_occurrences": int(sum(unmapped_diag.values())),
        "scp_per_superclass": {k: sorted(v) for k, v in per_class_scp.items()},
        "n_scp_per_superclass": {k: len(v) for k, v in per_class_scp.items()},
        "demographics": demo,
        "fold_counts": {
            "train_1_8": int((folds <= 8).sum()),
            "val_9": int((folds == 9).sum()),
            "test_10": int((folds == 10).sum()),
        },
    }


def challenge_code_inventory(root: Path, shared_codes: set[str]) -> dict:
    heas = sorted(Path(root).rglob("*.hea"))
    raw = Counter()
    kept = Counter()
    dropped = Counter()
    missing_dx = 0
    unreadable = 0
    n = 0
    for hea in heas:
        record = str(hea.with_suffix(""))
        if not (
            Path(record + ".mat").exists()
            or Path(record + ".dat").exists()
            or hea.with_suffix(".mat").exists()
        ):
            continue
        header = _safe_rdheader(record)
        if header is None:
            unreadable += 1
            continue
        n += 1
        codes = _header_snomed(header)
        if not codes:
            missing_dx += 1
        for code in codes:
            raw[code] += 1
            canon = canonical_snomed(code)
            if canon in shared_codes:
                kept[canon] += 1
            else:
                dropped[code] += 1
    return {
        "n_readable": n,
        "unreadable_headers": unreadable,
        "records_missing_dx_field": missing_dx,
        "unique_raw_codes": int(len(raw)),
        "kept_code_occurrences": int(sum(kept.values())),
        "dropped_code_occurrences": int(sum(dropped.values())),
        "top_dropped": dropped.most_common(15),
    }


def waveform_missing(cache_path: Path, sample: int = 400) -> dict:
    if not cache_path.exists():
        return {"exists": False, "path": str(cache_path)}
    arr = np.load(cache_path, mmap_mode="r")
    n = min(sample, arr.shape[0])
    idx = np.linspace(0, arr.shape[0] - 1, n, dtype=int)
    chunk = np.array(arr[idx], dtype=np.float32)
    nan_n = int(np.isnan(chunk).sum())
    inf_n = int(np.isinf(chunk).sum())
    zero_leads = int((np.abs(chunk).sum(axis=2) == 0).sum())
    return {
        "exists": True,
        "path": str(cache_path),
        "shape": list(arr.shape),
        "sampled_records": n,
        "sampled_values": int(chunk.size),
        "nan_values": nan_n,
        "inf_values": inf_n,
        "all_zero_leads": zero_leads,
        "all_zero_lead_frac": float(zero_leads / max(n * arr.shape[1], 1)),
    }


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure: {path}")
    return path


def plot_two_systems(path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.suptitle("Two label systems — they are not conversions of each other", x=0.03, ha="left", fontsize=14)

    def box(x, y, w, h, text, color):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#222", lw=0.8, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.5, color="white", zorder=3)

    box(0.3, 5.2, 3.6, 1.4, "PTB-XL SCP statements\n71 codes in this table", ACCENT)
    box(4.6, 5.2, 2.6, 1.4, "Official basis\ndiagnostic_class\nWagner / Strodthoff", "#6b4f2f")
    box(8.0, 5.2, 3.7, 1.4, "Task A — 5 superclasses\nNORM  MI  STTC  CD  HYP\nMilestone 2 only", "#2f6b4f")
    ax.annotate("", xy=(4.55, 5.9), xytext=(4.0, 5.9), arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))
    ax.annotate("", xy=(7.95, 5.9), xytext=(7.3, 5.9), arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))

    box(0.3, 2.4, 3.6, 1.8, "PTB-XL SCP codes\n+ Chapman / Georgia\nDx: SNOMED numbers", ACCENT)
    box(4.6, 2.4, 2.6, 1.8, "CinC 2021 basis\nequivalents kept\nall other codes dropped", "#6b4f2f")
    box(8.0, 2.4, 3.7, 1.8, "Task B — 10 shared SNOMED\nAF IAVB LBBB RBBB NSR\nPAC SB STach TAb LAD\nused on every archive", "#c47b2b")
    ax.annotate("", xy=(4.55, 3.3), xytext=(4.0, 3.3), arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))
    ax.annotate("", xy=(7.95, 3.3), xytext=(7.3, 3.3), arrowprops=dict(arrowstyle="->", color="#222", lw=1.4))

    ax.text(
        6.0,
        0.85,
        "The 5 superclasses exist only on PTB-XL. The 10 SNOMED labels exist on PTB-XL, Chapman and Georgia.\n"
        "10 classes do not become 10 superclasses. They are two parallel tasks with two published bases.",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    return _save(fig, path)


def plot_counts_single(summary: dict, title: str, path: Path, color: str = ACCENT) -> Path:
    labels = [r["label"] for r in summary["labels"]]
    counts = [r["positives"] for r in summary["labels"]]
    n = summary["n"]
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    bars = ax.bar(labels, counts, color=color)
    ax.set_ylabel("Records with the label (multi-label)")
    ax.set_title(f"{title}   n = {n:,}")
    ymax = max(counts) * 1.22 if counts else 1
    ax.set_ylim(0, ymax)
    for bar, row in zip(bars, summary["labels"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{row['positives']:,}\n{100 * row['prevalence']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.text(0.99, 0.97, "Bars need not sum to n", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#555")
    return _save(fig, path)


def plot_snomed_grouped(summaries: list[dict], path: Path) -> Path:
    labels = [r["label"] for r in summaries[0]["labels"]]
    x = np.arange(len(labels))
    width = 0.27
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 7.2), sharex=True)
    for i, summary in enumerate(summaries):
        counts = [r["positives"] for r in summary["labels"]]
        prev = [100 * r["prevalence"] for r in summary["labels"]]
        axes[0].bar(x + (i - 1) * width, counts, width, label=f"{summary['dataset']} (n={summary['n']:,})", color=COLORS[i])
        axes[1].bar(x + (i - 1) * width, prev, width, color=COLORS[i])
    axes[0].set_ylabel("Positive records")
    axes[0].set_title("Ten shared SNOMED-CT labels on every dataset used so far")
    axes[0].legend(loc="upper right")
    axes[1].set_ylabel("Prevalence (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Shared SNOMED-CT label")
    return _save(fig, path)


def plot_prevalence_heatmap(summaries: list[dict], path: Path) -> Path:
    labels = [r["label"] for r in summaries[0]["labels"]]
    mat = np.array([[r["prevalence"] for r in s["labels"]] for s in summaries])
    fig, ax = plt.subplots(figsize=(10.6, 3.6))
    im = ax.imshow(mat, cmap="YlOrBr", vmin=0, vmax=max(0.35, float(mat.max())))
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(summaries)), [s["dataset"] for s in summaries])
    ax.set_title("Label prevalence (share of records). Darker = more common.")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{100 * mat[i, j]:.1f}%", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Prevalence")
    return _save(fig, path)


def plot_cardinality(summaries: list[dict], path: Path) -> Path:
    fig, axes = plt.subplots(1, len(summaries), figsize=(11.2, 3.6), sharey=True)
    if len(summaries) == 1:
        axes = [axes]
    for ax, summary in zip(axes, summaries):
        hist = {int(k): v for k, v in summary["cardinality_hist"].items()}
        xs = list(range(0, max(hist) + 1))
        ys = [hist.get(k, 0) for k in xs]
        ax.bar(xs, ys, color=ACCENT)
        ax.set_title(f"{summary['dataset']}\nmean {summary['mean_labels_per_record']:.2f}")
        ax.set_xlabel("Mapped labels / ECG")
        ax.set_xticks(xs)
    axes[0].set_ylabel("Records")
    fig.suptitle("How many of the 10 SNOMED labels sit on one ECG")
    return _save(fig, path)


def plot_shift_table(tests: list[dict], path: Path) -> Path:
    labels = sorted({t["label"] for t in tests})
    pairs = sorted({(t["left"], t["right"]) for t in tests})
    fig, ax = plt.subplots(figsize=(10.8, 4.6))
    ax.axis("off")
    ax.set_title("Prevalence shift between datasets (absolute difference, Cramér's V)", loc="left")
    header = ["Label"] + [f"{a} vs {b}\n|Δp| / V" for a, b in pairs]
    cell = []
    for lab in labels:
        row = [lab]
        for a, b in pairs:
            t = next(x for x in tests if x["label"] == lab and x["left"] == a and x["right"] == b)
            v = t["cramers_v"]
            vtxt = "—" if v is None else f"{t['abs_diff']:.2f} / {v:.2f}"
            row.append(vtxt)
        cell.append(row)
    table = ax.table(cellText=cell, colLabels=header, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    return _save(fig, path)


def plot_missing_policy(stats: dict, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.4, 5.6))
    ax.axis("off")
    ax.set_title("Missing values: what is imputed, and what is dropped", loc="left", fontsize=13)
    header = ["Hole", "What I do", "What I do not do"]
    cells = [
        ["Unmapped diagnosis code", "Drop that code. Record stays; class = 0", "Invent AF / NORM / etc."],
        ["Unreadable WFDB header", "Skip the record (10 + 10 skipped)", "Repair the header"],
        ["NaN / Inf sample in waveform", "Replace with 0, then z-score the lead", "Interpolate neighbouring beats"],
        ["Missing lead", "Leave a zero row (or Einthoven if I/II exist)", "Copy another lead"],
        ["Short / long recording", "Zero-pad or centre-crop to 10 s at 100 Hz", "Change the diagnosis"],
        ["Failed waveform read", "Write a 10 s all-zero window", "Drop the label silently"],
        ["Age coded 300 on PTB-XL", "Leave it (90+ privacy code). Not a model input", "Mean-impute age"],
    ]
    table = ax.table(cellText=cells, colLabels=header, loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color="white", weight="bold")
        cell.set_edgecolor("#ddd")
    boxes = stats.get("waveforms", {})
    bits = []
    for name, info in boxes.items():
        if info.get("exists"):
            bits.append(
                f"{name}: cache {tuple(info['shape'])}; sample {info['sampled_records']} recs; "
                f"NaN={info['nan_values']}, Inf={info['inf_values']}, zero-leads={info['all_zero_leads']}"
            )
    ax.text(0.01, 0.08, "After the rule, sampled caches have no remaining NaN/Inf.\n" + "\n".join(bits), fontsize=8, va="top", family="Segoe UI")
    return _save(fig, path)


def main() -> None:
    cfg = load_config(ROOT / "configs" / "local.yaml").raw
    fig_dir = figure_dir(cfg.get("output_dir", "results"))
    shared_codes = {canonical_snomed(row["code"]) for row in cfg["snomed_shared"]}
    for row in cfg["snomed_shared"]:
        for extra in row.get("equivalents", []):
            shared_codes.add(canonical_snomed(extra))

    print("Loading PTB-XL / Chapman / Georgia …", flush=True)
    ptb_super = load_bundle("ptbxl", cfg, "superclass")
    ptb_snomed = load_bundle("ptbxl", cfg, "snomed_shared")
    chapman = load_bundle("chapman", cfg, "snomed_shared")
    georgia = load_bundle("georgia", cfg, "snomed_shared")

    super_sum = summarize_multihot("PTB-XL superclass", ptb_super.labels, SUPERCLASS_LABELS)
    ptb = summarize_multihot("PTB-XL", ptb_snomed.labels, SNOMED_NAMES)
    ch = summarize_multihot("Chapman", chapman.labels, SNOMED_NAMES)
    ga = summarize_multihot("Georgia", georgia.labels, SNOMED_NAMES)
    splits = official_ptbxl_splits(ptb_super)

    ptb_extra = ptbxl_superclass_and_scp(Path(ptb_super.meta["root"]))
    # y already computed; replace with loader labels to stay consistent
    super_sum = summarize_multihot("PTB-XL superclass", ptb_super.labels, SUPERCLASS_LABELS)
    super_sum["fold_counts"] = {k: int(len(v)) for k, v in splits.items()}
    super_sum["scp_basis"] = {
        "n_records_in_table": ptb_extra["n_records"],
        "unique_scp_codes": ptb_extra["unique_scp_codes"],
        "diagnostic_scp_codes": ptb_extra["diagnostic_scp_codes"],
        "n_scp_per_superclass": ptb_extra["n_scp_per_superclass"],
        "scp_per_superclass": ptb_extra["scp_per_superclass"],
        "demographics": ptb_extra["demographics"],
    }

    print("Inventory of dropped CinC codes …", flush=True)
    inventories = {
        "chapman": challenge_code_inventory(Path(chapman.meta["root"]), shared_codes),
        "georgia": challenge_code_inventory(Path(georgia.meta["root"]), shared_codes),
    }

    tests = []
    tests += pair_tests(ptb, ch)
    tests += pair_tests(ptb, ga)
    tests += pair_tests(ch, ga)
    prev_mat = {
        s["dataset"]: [r["prevalence"] for r in s["labels"]]
        for s in (ptb, ch, ga)
    }
    js = {
        "PTB-XL vs Chapman": js_divergence(np.array(prev_mat["PTB-XL"]), np.array(prev_mat["Chapman"])),
        "PTB-XL vs Georgia": js_divergence(np.array(prev_mat["PTB-XL"]), np.array(prev_mat["Georgia"])),
        "Chapman vs Georgia": js_divergence(np.array(prev_mat["Chapman"]), np.array(prev_mat["Georgia"])),
    }

    cache_root = Path(cfg.get("output_dir", "results")) / "cache" / "signals"
    wave = {
        "ptbxl": waveform_missing(cache_root / "ptbxl_fs100_t10.npy"),
        "chapman": waveform_missing(cache_root / "chapman_fs100_t10.npy"),
        "georgia": waveform_missing(cache_root / "georgia_fs100_t10.npy"),
    }

    payload = {
        "superclass": super_sum,
        "snomed": {"PTB-XL": ptb, "Chapman": ch, "Georgia": ga},
        "prevalence_shift_tests": tests,
        "js_divergence_prevalence": js,
        "challenge_code_inventory": inventories,
        "waveforms": wave,
        "policy": {
            "labels": "drop unmapped; never impute a class",
            "waveforms": "NaN/Inf -> 0; missing lead -> 0; failed read -> zeros; pad/crop to 10 s @ 100 Hz; per-lead z-score",
            "snomed_basis": "PhysioNet/CinC 2021 equivalents (Reyna/Alday); intersection present on PTB-XL, Chapman, Georgia",
            "superclass_basis": "PTB-XL scp_statements.csv diagnostic_class (Wagner 2020; Strodthoff 2021)",
        },
    }
    out_json = Path(cfg.get("output_dir", "results")) / "eval" / "dataset_label_audit.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    serialisable = json.loads(json.dumps(payload, default=str))
    out_json.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    print(f"wrote {out_json}", flush=True)

    plot_two_systems(fig_dir / "07_two_label_systems.png")
    plot_counts_single(super_sum, "PTB-XL diagnostic superclasses (Task A / Milestone 2 only)", fig_dir / "08_ptbxl_superclass_counts.png", "#2f6b4f")
    plot_counts_single(ptb, "PTB-XL — 10 shared SNOMED-CT labels", fig_dir / "09_ptbxl_snomed_counts.png", COLORS[0])
    plot_counts_single(ch, "Chapman — 10 shared SNOMED-CT labels", fig_dir / "10_chapman_snomed_counts.png", COLORS[1])
    plot_counts_single(ga, "Georgia — 10 shared SNOMED-CT labels", fig_dir / "11_georgia_snomed_counts.png", COLORS[2])
    plot_snomed_grouped([ptb, ch, ga], fig_dir / "12_snomed_counts_all_datasets.png")
    plot_prevalence_heatmap([ptb, ch, ga], fig_dir / "13_snomed_prevalence_heatmap.png")
    plot_cardinality([ptb, ch, ga], fig_dir / "14_snomed_multilabel_cardinality.png")
    plot_shift_table(tests, fig_dir / "15_prevalence_shift_stats.png")
    plot_missing_policy({"waveforms": wave}, fig_dir / "16_missing_value_policy.png")
    write_html_index(cfg.get("output_dir", "results"))
    print("audit complete", flush=True)


if __name__ == "__main__":
    main()
