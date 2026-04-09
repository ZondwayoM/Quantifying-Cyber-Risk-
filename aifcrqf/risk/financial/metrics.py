"""Financial risk metrics: EL, RRI, VaR, CVaR, and supporting utilities."""

from __future__ import annotations

import numpy as np

from config.settings import CVAR_CONFIDENCE, VAR_CONFIDENCE


def tail_mean(losses: np.ndarray, confidence: float) -> float:
    """Mean of losses exceeding the VaR at ``confidence``."""
    var_threshold = np.percentile(losses, confidence * 100)
    tail = losses[losses > var_threshold]
    return float(tail.mean()) if len(tail) > 0 else float(var_threshold)


def beta_fit(successes: int, trials: int) -> tuple[float, float]:
    """Bayesian Beta posterior: α = successes+1, β = failures+1 (Laplace smoothing). Returns (alpha, beta)."""
    failures = max(trials - successes, 0)
    alpha = successes + 1
    beta = failures + 1
    return float(alpha), float(beta)


def expected_loss(p_success: float, impact: float) -> float:
    """EL = P_success × Impact."""
    return p_success * impact


def residual_risk_index(el: float | np.ndarray, iso_maturity: float) -> float | np.ndarray:
    """RRI = EL × (1 − M), where M is ISO 27001 maturity ∈ [0, 1]."""
    m = float(np.clip(iso_maturity, 0.0, 1.0))
    return el * (1.0 - m)


def value_at_risk(loss_distribution: np.ndarray, confidence: float = VAR_CONFIDENCE) -> float:
    """VaR at the given confidence level."""
    return float(np.percentile(loss_distribution, confidence * 100))


def conditional_value_at_risk(loss_distribution: np.ndarray, confidence: float = CVAR_CONFIDENCE) -> float:
    """CVaR (Expected Shortfall) — mean loss beyond VaR."""
    return float(tail_mean(loss_distribution, confidence))
