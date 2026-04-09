"""FGSM attack (Goodfellow et al., 2015) — numerical gradient via central finite differences (h=0.5)."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from adversarial.utils import l2_perturbation
from config.settings import ATTACK_CONFIGS
from core.bases import BaseAttack
from core.bases import BaseModel

logger = logging.getLogger(__name__)

_CFG = ATTACK_CONFIGS["fgsm"]


class FGSMAttack(BaseAttack):
    """Gradient-sign perturbation using numerical finite-difference gradient (h=0.5)."""

    def __init__(
        self,
        model: BaseModel,
        epsilon: float = 0.10,
        config: dict | None = None,
    ) -> None:
        super().__init__(model, config or _CFG)
        self.epsilon = epsilon

    def _numerical_gradient(
        self, X: np.ndarray, y: np.ndarray, h: float = 0.5
    ) -> np.ndarray:
        """Central finite-difference gradient (h=0.5); h crosses tree split thresholds on standardised features."""
        n, d = X.shape
        grad = np.zeros_like(X)

        for i in range(d):
            X_fwd = X.copy()
            X_bwd = X.copy()
            X_fwd[:, i] += h
            X_bwd[:, i] -= h

            proba_fwd = np.clip(self.model.predict_proba(X_fwd), 1e-9, 1.0)
            proba_bwd = np.clip(self.model.predict_proba(X_bwd), 1e-9, 1.0)

            loss_fwd = -np.log(proba_fwd[np.arange(n), y.astype(int)])
            loss_bwd = -np.log(proba_bwd[np.arange(n), y.astype(int)])

            grad[:, i] = (loss_fwd - loss_bwd) / (2 * h)

        return grad

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return FGSM-perturbed examples at the configured epsilon."""
        grad = self._numerical_gradient(X, y)
        X_adv = X + self.epsilon * np.sign(grad)
        logger.info(
            "FGSM generated %d adversarial samples (ε=%.3f).",
            len(X), self.epsilon,
        )
        return X_adv

    def evaluate(
        self,
        X_orig: np.ndarray,
        X_adv: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        success_rate = self._evasion_success_rate(X_orig, X_adv, y)
        mean_l2 = float(l2_perturbation(X_orig, X_adv).mean())
        logger.info(
            "FGSM (ε=%.3f) | success_rate=%.4f | mean_L2=%.4f",
            self.epsilon, success_rate, mean_l2,
        )
        return {
            "success_rate": success_rate,
            "mean_perturbation": mean_l2,
            "epsilon": self.epsilon,
        }

    def sweep(
        self, X: np.ndarray, y: np.ndarray
    ) -> list[Dict[str, float]]:
        """Run evaluate across all epsilon values in the config."""
        results = []
        for eps in self.config.get("epsilon_range", [self.epsilon]):
            self.epsilon = eps
            X_adv = self.generate(X, y)
            results.append(self.evaluate(X, X_adv, y))
        return results
