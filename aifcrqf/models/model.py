"""Generic financial model — shared logic for all four domain models."""
from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from core.bases import BaseModel

logger = logging.getLogger(__name__)


class FinancialModel(BaseModel):
    """
    XGBoost or LightGBM classifier for fintech risk detection.

    Selects the backend from config["model_type"] ("xgboost" | "lightgbm").
    Libraries are imported lazily inside train() so unused backends are never loaded.
    """

    def __init__(self, config: dict, threshold: float = 0.50) -> None:
        super().__init__(config)
        self.threshold = threshold

    def train(self, X_train, y_train, X_val=None, y_val=None) -> None:
        params = {k: v for k, v in self.config.items() if k != "model_type"}
        eval_set = [(X_val, y_val)] if X_val is not None else None
        if self.config.get("model_type") == "lightgbm":
            import lightgbm as lgb
            self._model = lgb.LGBMClassifier(**params)
            self._model.fit(X_train, y_train, eval_set=eval_set)
        else:
            import xgboost as xgb
            self._model = xgb.XGBClassifier(**params)
            self._model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        logger.info("%s trained on %d samples.", self.__class__.__name__, len(y_train))

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._assert_trained()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return (self._model.predict_proba(X)[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._assert_trained()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return self._model.predict_proba(X)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._model, f)

    def load(self, path: Path) -> None:
        with open(Path(path), "rb") as f:
            self._model = pickle.load(f)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        return self._model.feature_importances_ if self._model is not None else None
