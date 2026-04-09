"""Abstract base classes for models, attacks, and risk components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """All domain models (fraud, credit, AML, trading) inherit from this."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._model: Optional[Any] = None

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> None:
        """Fit the model on training data."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard predictions."""

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialise the trained model."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Deserialise a model from path."""

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def _assert_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError(f"{self.__class__.__name__} has not been trained yet.")


class BaseAttack(ABC):
    """Every attack (FGSM, PGD, C&W, poisoning) inherits from this."""

    def __init__(self, model: BaseModel, config: dict) -> None:
        self.model = model
        self.config = config

    @abstractmethod
    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Produce adversarial examples from clean inputs."""

    @abstractmethod
    def evaluate(self, X_orig: np.ndarray, X_adv: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Measure attack effectiveness."""

    def _evasion_success_rate(self, X_orig: np.ndarray, X_adv: np.ndarray, y: np.ndarray) -> float:
        orig_pred = self.model.predict(X_orig)
        adv_pred = self.model.predict(X_adv)
        correct_mask = orig_pred == y
        if correct_mask.sum() == 0:
            return 0.0
        evaded = (adv_pred[correct_mask] != y[correct_mask]).sum()
        return float(evaded / correct_mask.sum())


class BaseRiskModel(ABC):
    """All risk-quantification components inherit from this."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._loss_distribution: Optional[np.ndarray] = None

    @abstractmethod
    def calculate_expected_loss(self, **kwargs) -> float:
        """EL = P_success × Impact."""

    @abstractmethod
    def calculate_var(self, confidence: float = 0.95) -> float:
        """VaR at the given confidence level."""

    @abstractmethod
    def calculate_cvar(self, confidence: float = 0.99) -> float:
        """CVaR = mean of losses exceeding VaR."""

    @property
    def loss_distribution(self) -> Optional[np.ndarray]:
        return self._loss_distribution
