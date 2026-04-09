"""Single domain evaluator with per-domain derived metrics."""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from models.utils import classification_report_dict

logger = logging.getLogger(__name__)

# Domain-specific metric lambdas: (metrics_dict, total_count) → extra keys
_DOMAIN_EXTRAS: Dict[str, object] = {
    "fraud": lambda m, t: {
        "fraud_leakage_rate": m["false_negative_rate"],
        "chargeback_ratio": m["false_positives"] / t if t > 0 else 0.0,
    },
    "credit": lambda m, t: {
        "approval_error_rate": (m["false_positives"] + m["false_negatives"]) / t if t > 0 else 0.0,
        "default_miss_rate": m["false_negative_rate"],
    },
    "aml": lambda m, t: {
        "suspicious_activity_miss_rate": m["false_negative_rate"],
        "detection_coverage": m["recall"],
    },
    "trading": lambda m, t: {
        "execution_error_rate": (m["false_positives"] + m["false_negatives"]) / t if t > 0 else 0.0,
        "signal_precision": m["precision"],
    },
}


class DomainEvaluator:
    """Evaluate any domain model and return a flat metrics dict."""

    def __init__(self, model, domain: str) -> None:
        self.model = model
        self.domain = domain

    def evaluate(self, X: np.ndarray, y: np.ndarray, split_name: str = "test") -> Dict[str, float]:
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        metrics = classification_report_dict(y, y_pred, y_proba)
        total = sum(metrics[k] for k in (
            "true_positives", "false_positives", "false_negatives", "true_negatives"
        ))
        extras_fn = _DOMAIN_EXTRAS.get(self.domain)
        if extras_fn:
            metrics.update(extras_fn(metrics, total))
        logger.info("=== %s Model (%s) ===", self.domain.capitalize(), split_name)
        for k, v in metrics.items():
            logger.info("  %-25s: %.4f", k, v)
        return metrics
