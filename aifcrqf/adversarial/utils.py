"""Shared utilities for adversarial attack modules."""

from __future__ import annotations

import numpy as np


def l2_perturbation(X_orig: np.ndarray, X_adv: np.ndarray) -> np.ndarray:
    """Compute per-sample L2 norm of adversarial perturbation."""
    diff = X_adv - X_orig
    return np.linalg.norm(diff, axis=1)


def clip_to_valid_range(
    X_adv: np.ndarray,
    X_orig: np.ndarray,
    epsilon: float,
) -> np.ndarray:
    """Project adversarial examples back into the epsilon-ball around X_orig."""
    return np.clip(X_adv, X_orig - epsilon, X_orig + epsilon)
