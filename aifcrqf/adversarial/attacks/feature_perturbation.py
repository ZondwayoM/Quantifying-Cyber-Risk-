"""Feature perturbation attack — Gaussian noise on top-k Gain-ranked features only."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

from adversarial.utils import l2_perturbation
from config.settings import ATTACK_CONFIGS
from core.bases import BaseAttack
from core.bases import BaseModel

logger = logging.getLogger(__name__)

_CFG = ATTACK_CONFIGS["feature_perturbation"]


class FeaturePerturbationAttack(BaseAttack):
    """Perturb only the most risk-relevant features."""

    def __init__(
        self,
        model: BaseModel,
        top_k: int = _CFG["top_k_features"],
        scale: float = _CFG["perturbation_scale"],
        important_feature_indices: Optional[List[int]] = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(model, config or _CFG)
        self.top_k = top_k
        self.scale = scale
        self.important_feature_indices = important_feature_indices

    def _get_top_features(self, X: np.ndarray) -> List[int]:
        """Return indices of the top-k features by model importance."""
        if self.important_feature_indices:
            return self.important_feature_indices[: self.top_k]

        if hasattr(self.model, "feature_importances") and self.model.feature_importances is not None:
            importances = self.model.feature_importances
            return list(np.argsort(importances)[::-1][: self.top_k])

        # Fallback: highest-variance features
        return list(np.argsort(X.var(axis=0))[::-1][: self.top_k])

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        top_idx = self._get_top_features(X)
        X_adv = X.copy().astype(float)
        stds = X[:, top_idx].std(axis=0) * self.scale
        X_adv[:, top_idx] += np.random.normal(0, stds, (len(X), len(top_idx)))
        logger.info("Feature perturbation applied to features: %s", top_idx)
        return X_adv

    def evaluate(self, X_orig, X_adv, y) -> Dict[str, float]:
        success_rate = self._evasion_success_rate(X_orig, X_adv, y)
        mean_l2 = float(l2_perturbation(X_orig, X_adv).mean())
        return {"success_rate": success_rate, "mean_perturbation": mean_l2, "top_k": self.top_k}
