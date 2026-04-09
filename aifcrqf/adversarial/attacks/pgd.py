"""PGD attack (Madry et al., 2017) — iterative FGSM with epsilon-ball projection after each step."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from adversarial.attacks.fgsm import FGSMAttack
from adversarial.utils import clip_to_valid_range, l2_perturbation
from config.settings import ATTACK_CONFIGS
from core.bases import BaseModel

logger = logging.getLogger(__name__)

_CFG = ATTACK_CONFIGS["pgd"]


class PGDAttack(FGSMAttack):
    """Iterative PGD for tabular models — same numerical gradient as FGSM, iterated num_steps times."""

    def __init__(
        self,
        model: BaseModel,
        epsilon: float = _CFG["epsilon"],
        alpha: float = _CFG["alpha"],
        num_steps: int = _CFG["num_steps"],
        config: dict | None = None,
    ) -> None:
        super().__init__(model, epsilon=epsilon, config=config or _CFG)
        self.alpha = alpha
        self.num_steps = num_steps

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Run PGD iterations from a random start within the epsilon-ball."""
        X_adv = X + np.random.uniform(-self.epsilon, self.epsilon, X.shape)

        for step in range(self.num_steps):
            grad = self._numerical_gradient(X_adv, y)
            X_adv = X_adv + self.alpha * np.sign(grad)
            X_adv = clip_to_valid_range(X_adv, X, self.epsilon)

        logger.info(
            "PGD generated %d adversarial samples (%d steps, ε=%.3f).",
            len(X), self.num_steps, self.epsilon,
        )
        return X_adv

    def evaluate(self, X_orig, X_adv, y) -> Dict[str, float]:
        success_rate = self._evasion_success_rate(X_orig, X_adv, y)
        mean_l2 = float(l2_perturbation(X_orig, X_adv).mean())
        logger.info(
            "PGD | success_rate=%.4f | mean_L2=%.4f", success_rate, mean_l2
        )
        return {
            "success_rate": success_rate,
            "mean_perturbation": mean_l2,
            "epsilon": self.epsilon,
            "num_steps": self.num_steps,
        }
