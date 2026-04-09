"""Governance compliance and ISO 27001 maturity metrics for the AIFCRQF dashboard."""

from __future__ import annotations

from typing import Dict

from config.settings import ISO_MATURITY_LEVELS
from risk.financial.metrics import residual_risk_index


DOMAIN_WEIGHTS: dict[str, float] = {
    "Access Control":           0.20,
    "Asset Management":         0.10,
    "Incident Response":        0.25,
    "Cryptography":             0.15,
    "Operations Security":      0.15,
    "Communications Security":  0.10,
    "Supplier Relationships":   0.05,
}


def compute_weighted_maturity(domain_scores: dict[str, float]) -> float:
    """
    Compute the weighted ISO 27001 maturity score M ∈ [0, 1].

    Parameters
    ----------
    domain_scores : dict mapping ISO domain name → score ∈ [0, 1]

    Returns
    -------
    float  — weighted overall maturity score
    """
    total_weight = 0.0
    weighted_sum = 0.0

    for domain, weight in DOMAIN_WEIGHTS.items():
        score = domain_scores.get(domain, 0.0)
        weighted_sum += weight * score
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0


def label_to_maturity(label: str) -> float:
    """Convert 'weak' / 'medium' / 'strong' label to numeric maturity."""
    return ISO_MATURITY_LEVELS.get(label.lower(), 0.60)


def governance_dashboard_metrics(
    domain_scores: Dict[str, float],
    el: float,
) -> Dict[str, float]:
    """
    Compute all governance-layer metrics for dashboard export.

    Parameters
    ----------
    domain_scores : dict — ISO 27001 domain maturity scores
    el            : float — Expected Loss from Monte Carlo engine

    Returns
    -------
    dict with maturity, rri, compliance_pct
    """
    maturity = compute_weighted_maturity(domain_scores)
    rri = residual_risk_index(el, maturity)
    compliance_pct = maturity * 100.0

    return {
        "iso_maturity": maturity,
        "compliance_pct": round(compliance_pct, 1),
        "rri": round(rri, 2),
        "el": round(el, 2),
    }
