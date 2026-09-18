from __future__ import annotations

import numpy as np
from scipy.signal import resample


STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_ALIASES = {
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "I": "I",
    "II": "II",
    "III": "III",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "V4": "V4",
    "V5": "V5",
    "V6": "V6",
}


def normalize_lead_name(name: str) -> str:
    cleaned = name.strip().replace(" ", "")
    return LEAD_ALIASES.get(cleaned.upper() if cleaned.upper() in LEAD_ALIASES else cleaned, cleaned)


def reorder_leads(signal: np.ndarray, sig_names: list[str]) -> np.ndarray:
    """Return array of shape (12, T) in STANDARD_LEADS order."""
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D signal, got {signal.shape}")
    if signal.shape[0] < signal.shape[1] and signal.shape[1] <= 15:
        signal = signal.T
    name_to_idx = {normalize_lead_name(n): i for i, n in enumerate(sig_names)}
    n_samples = signal.shape[0] if signal.shape[1] <= 15 else signal.shape[1]
    if signal.shape[1] <= 15:
        src = signal
    else:
        src = signal.T
        n_samples = src.shape[0]
    out = np.zeros((12, n_samples), dtype=np.float32)
    for i, lead in enumerate(STANDARD_LEADS):
        if lead in name_to_idx:
            out[i] = src[:, name_to_idx[lead]]
        elif lead == "III" and "I" in name_to_idx and "II" in name_to_idx:
            out[i] = src[:, name_to_idx["II"]] - src[:, name_to_idx["I"]]
        elif lead == "aVR" and "I" in name_to_idx and "II" in name_to_idx:
            out[i] = -0.5 * (src[:, name_to_idx["I"]] + src[:, name_to_idx["II"]])
        elif lead == "aVL" and "I" in name_to_idx and "III" in name_to_idx:
            out[i] = 0.5 * (src[:, name_to_idx["I"]] - src[:, name_to_idx["III"]])
        elif lead == "aVF" and "II" in name_to_idx and "III" in name_to_idx:
            out[i] = 0.5 * (src[:, name_to_idx["II"]] + src[:, name_to_idx["III"]])
    return out


def resample_signal(x: np.ndarray, src_fs: float, target_fs: float) -> np.ndarray:
    if abs(src_fs - target_fs) < 1e-6:
        return x.astype(np.float32)
    n_out = int(round(x.shape[1] * target_fs / src_fs))
    return resample(x, n_out, axis=1).astype(np.float32)


def pad_or_crop(x: np.ndarray, length: int) -> np.ndarray:
    t = x.shape[1]
    if t == length:
        return x
    if t > length:
        start = (t - length) // 2
        return x[:, start : start + length]
    out = np.zeros((x.shape[0], length), dtype=np.float32)
    start = (length - t) // 2
    out[:, start : start + t] = x
    return out


def standardize_per_lead(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    return ((x - mean) / (std + eps)).astype(np.float32)


def prepare_record(
    signal: np.ndarray,
    sig_names: list[str],
    src_fs: float,
    target_fs: float = 100.0,
    duration_sec: float = 10.0,
) -> np.ndarray:
    x = reorder_leads(np.asarray(signal, dtype=np.float32), sig_names)
    x = resample_signal(x, src_fs, target_fs)
    x = pad_or_crop(x, int(target_fs * duration_sec))
    x = standardize_per_lead(x)
    return x
