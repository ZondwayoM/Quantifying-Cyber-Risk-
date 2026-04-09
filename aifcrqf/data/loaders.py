"""Data loading utilities for AIFCRQF."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load raw datasets from disk into pandas DataFrames."""

    def load_csv(self, filepath: str | Path, **kwargs: Any) -> pd.DataFrame:
        """
        Load a CSV file.

        Parameters
        ----------
        filepath : path to the CSV file
        **kwargs : passed directly to pd.read_csv

        Returns
        -------
        pd.DataFrame
        """
        path = Path(filepath)
        logger.info("Loading CSV: %s", path)
        df = pd.read_csv(path, **kwargs)
        self._log_stats(df, path.name)
        return df

    @staticmethod
    def _log_stats(df: pd.DataFrame, name: str) -> None:
        mem_mb = df.memory_usage(deep=True).sum() / 1e6
        logger.info(
            "Loaded '%s': %d rows × %d cols | %.1f MB",
            name, len(df), df.shape[1], mem_mb,
        )
