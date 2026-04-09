"""C&W L2 attack (Carlini & Wagner, 2017) — coordinate-descent adaptation for tabular tree models, lr=0.10."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from adversarial.utils import l2_perturbation
from config.settings import ATTACK_CONFIGS
from core.bases import BaseAttack
from core.bases import BaseModel

logger = logging.getLogger(__name__)

_CFG = ATTACK_CONFIGS["carlini_wagner"]


class CarliniWagnerAttack(BaseAttack):
    """Coordinate-descent C&W variant: per-step ±lr perturbation that maximally reduces true-class confidence."""

    def __init__(
        self,
        model: BaseModel,
        max_iterations: int = _CFG["max_iterations"],
        learning_rate: float = _CFG["learning_rate"],
        kappa: float = _CFG["kappa"],
        config: dict | None = None,
    ) -> None:
        super().__init__(model, config or _CFG)
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.kappa = kappa

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate minimally perturbed adversarial examples."""
        X_adv = X.copy().astype(float)
        n, d = X.shape

        for _ in range(self.max_iterations):
            proba = self.model.predict_proba(X_adv)
            preds = self.model.predict(X_adv)

            still_correct = preds == y

            if not still_correct.any():
                break

            # Test ±lr per feature; pick direction that drops true-class confidence more
            delta = np.zeros_like(X_adv)
            for i in range(d):
                X_pos = X_adv.copy(); X_pos[:, i] += self.learning_rate
                X_neg = X_adv.copy(); X_neg[:, i] -= self.learning_rate
                p_pos = self.model.predict_proba(X_pos)
                p_neg = self.model.predict_proba(X_neg)
                for idx in np.where(still_correct)[0]:
                    true_cls  = int(y[idx])
                    drop_pos  = proba[idx, true_cls] - p_pos[idx, true_cls]
                    drop_neg  = proba[idx, true_cls] - p_neg[idx, true_cls]
                    if drop_pos > 0 or drop_neg > 0:
                        delta[idx, i] = (
                            self.learning_rate if drop_pos >= drop_neg
                            else -self.learning_rate
                        )

            X_adv += delta

        logger.info("C&W generated %d adversarial samples.", n)
        return X_adv

    def evaluate(self, X_orig, X_adv, y) -> Dict[str, float]:
        success_rate = self._evasion_success_rate(X_orig, X_adv, y)
        mean_l2 = float(l2_perturbation(X_orig, X_adv).mean())
        logger.info(
            "C&W | success_rate=%.4f | mean_L2=%.4f", success_rate, mean_l2
        )
        return {
            "success_rate": success_rate,
            "mean_perturbation": mean_l2,
            "max_iterations": self.max_iterations,
        }
