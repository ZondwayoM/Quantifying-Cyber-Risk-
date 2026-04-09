"""Shared model utilities: evaluation metrics and data splitting."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_SEED, SPLIT_RATIOS

logger = logging.getLogger(__name__)


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Return accuracy, ROC-AUC, PR-AUC, precision, recall, F1, FPR, FNR, and confusion matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = SPLIT_RATIOS["train"],
    val_ratio: float = SPLIT_RATIOS["val"],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60/20/20 split; falls back to non-stratified if any class has <2 samples."""
    test_ratio = 1.0 - train_ratio - val_ratio

    def _safe_stratify(arr: np.ndarray) -> np.ndarray | None:
        counts = np.bincount(arr.astype(int)) if arr.dtype.kind in "ui" else np.array([])
        if len(counts) > 1 and counts.min() >= 2:
            return arr
        return None

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y,
        test_size=(1.0 - train_ratio),
        random_state=RANDOM_SEED,
        stratify=_safe_stratify(y),
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp,
        test_size=(1.0 - relative_val),
        random_state=RANDOM_SEED,
        stratify=_safe_stratify(y_tmp),
    )

    logger.info(
        "Split: train=%d | val=%d | test=%d",
        len(y_train), len(y_val), len(y_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
