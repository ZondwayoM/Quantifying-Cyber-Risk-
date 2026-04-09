"""Data preprocessing: cleaning, feature/target splitting, and StandardScaler fitting."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Clean, split, and scale fintech datasets."""

    def __init__(self) -> None:
        self._scaler: Optional[StandardScaler] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicates; fill numeric NaN with median, categorical NaN with mode."""
        original_len = len(df)
        df = df.drop_duplicates()
        logger.info("Dropped %d duplicate rows.", original_len - len(df))

        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_val = df[col].median()
                else:
                    fill_val = df[col].mode().iloc[0]
                df[col] = df[col].fillna(fill_val)
                logger.debug("Filled NaN in '%s' with %s", col, fill_val)

        return df

    def split_features_target(
        self, df: pd.DataFrame, target_col: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Split DataFrame into feature matrix X and target y; encode string targets."""
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not in DataFrame.")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        if not pd.api.types.is_numeric_dtype(y):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index, name=target_col)
            logger.info(
                "Target '%s' encoded: %s → %s",
                target_col, list(le.classes_), list(range(len(le.classes_))),
            )

        non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            X = self._encode_non_numeric(X, non_numeric)

        logger.info(
            "Features: %d columns | Target: '%s' (%d rows)",
            X.shape[1], target_col, len(y),
        )
        return X, y

    def scale_features(
        self, X: pd.DataFrame, fit: bool = True
    ) -> np.ndarray:
        """StandardScaler fit+transform (fit=True) or transform only (fit=False)."""
        if fit or self._scaler is None:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            logger.info("Scaler fitted and applied.")
        else:
            X_scaled = self._scaler.transform(X)
            logger.info("Scaler applied (pre-fitted).")

        return X_scaled

    def _encode_non_numeric(self, X: pd.DataFrame, cols: list) -> pd.DataFrame:
        """Encode non-numeric columns rather than dropping them.

        - Timestamp-like columns: extract hour and day-of-week, drop raw string.
        - High-cardinality strings (>100 unique values): frequency encode.
        - Low-cardinality strings (≤100 unique values): label encode.
        """
        X = X.copy()
        for col in cols:
            # --- Timestamp columns ---
            if pd.api.types.is_object_dtype(X[col]):
                try:
                    parsed = pd.to_datetime(X[col], infer_datetime_format=True)
                    X[f"{col}_hour"] = parsed.dt.hour.astype(float)
                    X[f"{col}_dow"]  = parsed.dt.dayofweek.astype(float)
                    X = X.drop(columns=[col])
                    logger.info("Timestamp '%s' → hour + day-of-week features.", col)
                    continue
                except Exception:
                    pass

            n_unique = X[col].nunique()
            if n_unique > 100:
                # Frequency encode: replace each value with its count proportion
                freq = X[col].value_counts(normalize=True)
                X[col] = X[col].map(freq).astype(float)
                logger.info("Frequency-encoded '%s' (%d unique values).", col, n_unique)
            else:
                # Label encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str)).astype(float)
                logger.info("Label-encoded '%s' (%d unique values).", col, n_unique)
        return X

    @property
    def scaler(self) -> Optional[StandardScaler]:
        """Fitted scaler (None before first call to scale_features)."""
        return self._scaler
