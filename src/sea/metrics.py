from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if y_true.max() == y_true.min():
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def macro_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    aucs = []
    for k in range(y_true.shape[1]):
        value = _safe_auroc(y_true[:, k], y_score[:, k])
        if not np.isnan(value):
            aucs.append(value)
    return float(np.mean(aucs)) if aucs else float("nan")


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Multi-label ECE: flatten labels and average |acc - conf| over confidence bins."""
    y = y_true.reshape(-1)
    p = np.clip(y_prob.reshape(-1), 1e-7, 1 - 1e-7)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if not np.any(mask):
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def metric_bundle(y_true: np.ndarray, logits: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    prob = 1.0 / (1.0 + np.exp(-logits))
    return {
        "macro_auroc": macro_auroc(y_true, prob),
        "brier": brier_score(y_true, prob),
        "ece": expected_calibration_error(y_true, prob, n_bins=n_bins),
    }


def bootstrap_metrics(
    y_true: np.ndarray,
    logits: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    n_bins: int = 15,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = {key: [] for key in ("macro_auroc", "brier", "ece")}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bundle = metric_bundle(y_true[idx], logits[idx], n_bins=n_bins)
        for key, value in bundle.items():
            if not np.isnan(value):
                samples[key].append(value)
    out = {}
    point = metric_bundle(y_true, logits, n_bins=n_bins)
    for key, values in samples.items():
        arr = np.asarray(values)
        out[key] = {
            "point": point[key],
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "ci_low": float(np.quantile(arr, 0.025)) if len(arr) else float("nan"),
            "ci_high": float(np.quantile(arr, 0.975)) if len(arr) else float("nan"),
        }
    return out


def chance_level_test(auroc_samples: np.ndarray, chance: float = 0.5) -> dict[str, float]:
    """One-sided bootstrap test that OOD AUROC exceeds chance."""
    arr = np.asarray(auroc_samples, dtype=float)
    arr = arr[~np.isnan(arr)]
    p_value = float(np.mean(arr <= chance)) if len(arr) else float("nan")
    return {
        "chance": chance,
        "p_boot": p_value,
        "significant": bool(p_value < 0.05) if not np.isnan(p_value) else False,
        "ci_excludes_chance": bool(np.quantile(arr, 0.025) > chance) if len(arr) else False,
    }
