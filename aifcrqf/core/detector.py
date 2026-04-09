"""IntelligentDetector — auto-infers domain, problem type, and target column from DataFrame metadata."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from config.settings import (
    IMBALANCE_THRESHOLD,
    PROBLEM_CLASSIFICATION,
    PROBLEM_KEYWORDS,
    PROBLEM_REGRESSION,
    SUPPORTED_DOMAINS,
)

logger = logging.getLogger(__name__)


class IntelligentDetector:
    """Auto-infer domain, problem type, and target column from column names and class distribution."""

    def detect_target_column(
        self, df: pd.DataFrame, provided_target: Optional[str] = None
    ) -> str:
        """Return provided_target if valid; else scan column names against domain keywords; fallback = last column."""
        if provided_target and provided_target in df.columns:
            logger.info("Using provided target column: '%s'", provided_target)
            return provided_target

        # Skip high-cardinality columns (target should have ≤20 unique values or ≤1% of rows)
        _max_card = min(20, max(2, int(len(df) * 0.01)))
        cols_lower = {c: c.lower() for c in df.columns}
        for domain, keywords in PROBLEM_KEYWORDS.items():
            for col, col_low in cols_lower.items():
                if any(kw in col_low for kw in keywords):
                    n_unique = df[col].nunique()
                    if n_unique <= _max_card:
                        logger.info(
                            "Auto-detected target column '%s' (domain hint: %s, "
                            "%d unique values)",
                            col, domain, n_unique,
                        )
                        return col
                    logger.debug(
                        "Skipping '%s' as target — %d unique values (too high).",
                        col, n_unique,
                    )

        fallback = df.columns[-1]
        logger.warning(
            "Could not auto-detect target; using last column '%s'", fallback
        )
        return fallback

    def detect_problem_type(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[str, str]:
        """Return (problem_type, domain) from target cardinality and column keyword scan."""
        target = df[target_col]
        n_unique = target.nunique()

        if n_unique <= 20 or target.dtype in ("object", "bool", "category"):
            problem_type = PROBLEM_CLASSIFICATION
        else:
            problem_type = PROBLEM_REGRESSION

        all_text = " ".join(df.columns.tolist()).lower()
        domain = "unknown"
        for d in SUPPORTED_DOMAINS:
            if any(kw in all_text for kw in PROBLEM_KEYWORDS.get(d, [])):
                domain = d
                break

        logger.info(
            "Problem type: %s | Domain: %s | Target unique values: %d",
            problem_type, domain, n_unique,
        )
        return problem_type, domain

    def analyze_dataset(self, df: pd.DataFrame, target_col: str) -> Dict:
        """Return dataset stats: n_rows, n_features, n_classes, imbalance_ratio, missing_pct, domain."""
        problem_type, domain = self.detect_problem_type(df, target_col)

        feature_cols = [c for c in df.columns if c != target_col]
        n_features = len(feature_cols)
        n_rows = len(df)

        missing_pct = float(df[feature_cols].isnull().mean().mean())

        analysis: Dict = {
            "n_rows": n_rows,
            "n_features": n_features,
            "missing_pct": round(missing_pct * 100, 2),
            "problem_type": problem_type,
            "domain": domain,
            "n_classes": None,
            "imbalance_ratio": None,
            "is_imbalanced": False,
        }

        if problem_type == PROBLEM_CLASSIFICATION:
            value_counts = df[target_col].value_counts(normalize=True)
            analysis["n_classes"] = len(value_counts)
            minority_freq = float(value_counts.min())
            analysis["imbalance_ratio"] = round(minority_freq, 4)
            analysis["is_imbalanced"] = minority_freq < IMBALANCE_THRESHOLD

        logger.info("Dataset analysis: %s", analysis)
        return analysis
