"""Core Matplotlib plots for AIFCRQF — all functions save to VIZ_DIR and return the path."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config.settings import VIZ_DIR

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="Blues_d")


def plot_mc_loss_distribution(
    rri_distribution: np.ndarray,
    var_95: float,
    var_99: float,
    cvar_99: float,
    filename: str = "mc_loss_distribution.png",
) -> Path:
    """Plot Monte Carlo loss distribution with VaR and CVaR thresholds."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rri_distribution, bins=50, color="steelblue", alpha=0.7, label="Loss Distribution")
    ax.axvline(rri_distribution.mean(), color="green", lw=2, label=f"Mean = ${rri_distribution.mean():.2f}")
    ax.axvline(var_95, color="orange", lw=2, linestyle="--", label=f"VaR 95% = ${var_95:.2f}")
    ax.axvline(var_99, color="red", lw=2, linestyle="--", label=f"VaR 99% = ${var_99:.2f}")
    ax.axvline(cvar_99, color="darkred", lw=2, linestyle="--", label=f"CVaR 99% = ${cvar_99:.2f}")
    ax.set_xlabel("Risk-Reduced Impact (USD per fraud)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Loss Distribution — Adversarial Risk Quantification")
    ax.legend(fontsize=9)
    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_bn_mc_integration(
    bn_mc_scenarios: dict,
    filename: str = "bn_mc_integration.png",
) -> Path:
    """Overlay RRI distributions for all BN+MC scenarios (empirical, baseline, worst-case, mitigated)."""
    _COLORS = {
        "empirical":  "#2166ac",
        "baseline":   "#4dac26",
        "worst_case": "#d01c8b",
        "mitigated":  "#f1a340",
    }
    _DEFAULT_COLOR = "#888888"

    fig, ax = plt.subplots(figsize=(10, 5))

    for key, scenario in bn_mc_scenarios.items():
        dist = scenario.get("_rri_distribution")
        if dist is None or len(dist) == 0:
            continue
        color = _COLORS.get(key, _DEFAULT_COLOR)
        label_str = (
            f"{scenario.get('label', key)}\n"
            f"P={scenario['bn_p_success']:.3f} | "
            f"EL=${scenario.get('el_mean', 0):.0f} | "
            f"CVaR99=${scenario.get('cvar_99', 0):.0f}"
        )
        ax.hist(
            dist, bins=60, density=True, alpha=0.45,
            color=color, label=label_str,
        )
        ax.axvline(
            scenario.get("cvar_99", 0), color=color,
            lw=1.8, linestyle="--", alpha=0.9,
        )

    ax.set_xlabel("Risk-Reduced Impact (USD)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(
        "Bayesian Network — Monte Carlo Integration\n"
        "How threat-environment scenarios translate to financial loss distributions",
        fontsize=11,
    )
    ax.legend(fontsize=7.5, loc="upper right")

    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


_DOMAIN_METRIC_KEYS = {
    "fraud":   ["fraud_leakage_rate", "chargeback_ratio", "recall", "precision"],
    "credit":  ["default_miss_rate", "approval_error_rate", "recall", "precision"],
    "aml":     ["suspicious_activity_miss_rate", "detection_coverage", "recall", "f1"],
    "trading": ["execution_error_rate", "signal_precision", "recall", "f1"],
}
_DOMAIN_METRIC_LABELS = {
    "fraud_leakage_rate":           "Fraud Leakage Rate",
    "chargeback_ratio":             "Chargeback Ratio",
    "default_miss_rate":            "Default Miss Rate",
    "approval_error_rate":          "Approval Error Rate",
    "suspicious_activity_miss_rate":"SAR Miss Rate",
    "detection_coverage":           "Detection Coverage",
    "execution_error_rate":         "Execution Error Rate",
    "signal_precision":             "Signal Precision",
    "recall":                       "Recall",
    "precision":                    "Precision",
    "f1":                           "F1 Score",
}


def plot_domain_metrics_bar(
    domain: str,
    metrics: dict,
    filename: str = "domain_metrics.png",
) -> Path:
    """Horizontal bar chart of domain-specific and generic classifier metrics."""
    keys = _DOMAIN_METRIC_KEYS.get(domain, ["recall", "precision", "f1"])
    values = [metrics.get(k, 0.0) for k in keys]
    labels = [_DOMAIN_METRIC_LABELS.get(k, k) for k in keys]

    generic = {"recall", "precision", "f1", "roc_auc"}
    colors = ["#4292c6" if k in generic else "#d6604d" for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Score / Rate", fontsize=11)
    ax.set_title(
        f"Domain Impact Metrics — {domain.upper()} Model\n"
        "Red = domain-specific  |  Blue = standard classifier",
        fontsize=10,
    )
    for bar, val in zip(bars, values):
        ax.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9,
        )
    ax.invert_yaxis()

    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_disclosure_comparison(
    comparison: dict,
    filename: str = "disclosure_comparison.png",
) -> Path:
    """Side-by-side bar comparing reputational loss under immediate vs delayed disclosure."""
    immediate = comparison.get("immediate_disclosure", {})
    delayed   = comparison.get("delayed_disclosure", {})
    r_f       = comparison.get("reputational_amplification_factor")

    labels = ["Immediate Disclosure", "Delayed Disclosure"]
    rep_vals = [
        immediate.get("reputational_mean", 0),
        delayed.get("reputational_mean", 0),
    ]
    total_vals = [
        immediate.get("total_cascade_mean", 0),
        delayed.get("total_cascade_mean", 0),
    ]

    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    b1 = ax.bar(x - width / 2, rep_vals,   width, label="Reputational Loss",  color="#4393c3")
    b2 = ax.bar(x + width / 2, total_vals, width, label="Total Cascade Loss", color="#d6604d", alpha=0.75)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Loss (USD)")
    ax.set_title(
        "Disclosure Timing — Reputational Loss Amplification\n"
        "(Wu et al., 2022 — R_f = 1.5 for delayed disclosure)",
        fontsize=10,
    )
    ax.legend()

    if r_f is not None:
        ax.annotate(
            f"R_f = {r_f:.2f}×",
            xy=(0.5, max(rep_vals) * 1.05),
            ha="center", fontsize=11, color="#d01c8b",
            fontweight="bold",
        )

    for bar in b1:
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"${bar.get_height():.0f}", ha="center", va="bottom", fontsize=9,
        )

    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", path)
    return path


def plot_cascading_waterfall(
    cascade_summary: dict,
    filename: str = "cascading_waterfall.png",
) -> Path:
    """Waterfall chart of cascading loss components."""
    components = ["direct", "regulatory", "reputational", "churn", "operational"]
    values = [cascade_summary.get(k, 0) for k in components]
    labels = ["Direct\nLoss", "Regulatory\nFines", "Reputational\nDamage",
              "Customer\nChurn", "Operational\nCost"]

    running = 0
    bottoms = []
    for v in values:
        bottoms.append(running)
        running += v

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, bottom=bottoms, color="steelblue", edgecolor="white")
    ax.set_ylabel("Cumulative Loss (USD)")
    ax.set_title("Cascading Impact Build-Up")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_y() + bar.get_height() / 2,
                f"${val:.0f}", ha="center", va="center", fontsize=9, color="white")
    path = VIZ_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
