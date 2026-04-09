"""Centroid Evasion Attack — black-box, gradient-free: iteratively moves positives toward the negative-class centroid."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from adversarial.utils import l2_perturbation
from config.settings import ATTACK_CONFIGS
from core.bases import BaseAttack, BaseModel

logger = logging.getLogger(__name__)

_CFG            = ATTACK_CONFIGS.get("centroid_evasion", {})
_DEFAULT_ALPHA  = _CFG.get("alpha", 0.05)
_DEFAULT_STEPS  = _CFG.get("max_steps", 20)
_MAX_SAMPLES    = 1_000


class CentroidEvasionAttack(BaseAttack):
    """Black-box directional evasion: steps each positive toward the negative centroid until prediction flips."""

    def __init__(
        self,
        model: BaseModel,
        alpha: float = _DEFAULT_ALPHA,
        max_steps: int = _DEFAULT_STEPS,
        config: dict | None = None,
    ) -> None:
        super().__init__(model, config or _CFG)
        self.alpha     = alpha
        self.max_steps = max_steps

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perturb positive-class samples toward the negative centroid; negative-class samples unchanged."""
        n_total = len(X)
        if n_total > _MAX_SAMPLES:
            rng = np.random.default_rng(42)
            work_idx = rng.choice(n_total, _MAX_SAMPLES, replace=False)
            X_work, y_work = X[work_idx], y[work_idx]
        else:
            work_idx = np.arange(n_total)
            X_work, y_work = X, y

        neg_mask = y_work == 0
        if neg_mask.sum() == 0:
            return X.copy()
        centroid_neg = X_work[neg_mask].mean(axis=0)

        X_adv = X.copy().astype(float)
        pos_local = np.where(y_work == 1)[0]

        for local_i in pos_local:
            global_i = work_idx[local_i]
            x = X_adv[global_i].copy()
            diff = centroid_neg - x
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                continue
            direction = diff / dist
            step_size = self.alpha * dist

            for _ in range(self.max_steps):
                x = x + step_size * direction
                pred = self.model.predict(x.reshape(1, -1))[0]
                if pred != 1:
                    X_adv[global_i] = x
                    break

        logger.info(
            "CentroidEvasion: %d positive samples attacked | alpha=%.3f | max_steps=%d",
            len(pos_local), self.alpha, self.max_steps,
        )
        return X_adv

    def evaluate(self, X_orig: np.ndarray, X_adv: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        success_rate = self._evasion_success_rate(X_orig, X_adv, y)
        mean_l2      = float(l2_perturbation(X_orig, X_adv).mean())
        return {
            "success_rate":      success_rate,
            "mean_perturbation": mean_l2,
            "alpha":             self.alpha,
            "max_steps":         self.max_steps,
        }
