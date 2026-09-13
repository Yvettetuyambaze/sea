"""Post-hoc temperature scaling (Guo et al., ICML 2017).

Fits a single scalar T > 0 on frozen logits. Ranking is unchanged, so AUROC
is a manipulation check and is not an outcome for this method.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def fit_temperature(logits: np.ndarray, y_true: np.ndarray, max_iter: int = 200) -> float:
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(y_true, dtype=torch.float32)
    log_t = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_t)
        loss = nn.functional.binary_cross_entropy_with_logits(z / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_t).detach().clamp(min=1e-3, max=100.0).item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)
