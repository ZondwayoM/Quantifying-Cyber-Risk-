"""Generic domain trainer — all four domain trainers share this logic."""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from config.settings import MODELS_DIR
from models.utils import stratified_split

logger = logging.getLogger(__name__)


class DomainTrainer:
    """Split data, train a domain model, and optionally persist it."""

    def __init__(self, model_class, domain: str, config: Optional[dict] = None) -> None:
        self.model = model_class(config)
        self._domain = domain

    def train(self, X: np.ndarray, y: np.ndarray, save: bool = True) -> Dict[str, np.ndarray]:
        X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(X, y)
        self.model.train(X_train, y_train, X_val, y_val)
        logger.info("%s model training complete.", self._domain.capitalize())
        if save:
            self.model.save(MODELS_DIR / f"{self._domain}_model.pkl")
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
        }
