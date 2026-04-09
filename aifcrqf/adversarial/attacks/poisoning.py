"""Training-data poisoning: label_flip, targeted_flip, feature_perturb, gain_guided, clean_label, backdoor."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from config.settings import ATTACK_CONFIGS, RANDOM_SEED
from core.bases import BaseAttack, BaseModel

logger = logging.getLogger(__name__)

_CFG = ATTACK_CONFIGS["poisoning"]

_GAIN_TOP_K: int = 5  # top features targeted by gain-guided attack


class PoisoningAttack(BaseAttack):
    """Corrupts training data before fitting to degrade model recall across fintech domains."""

    def __init__(
        self,
        model: BaseModel,
        corruption_rate: float = 0.03,
        attack_type: str = "label_flip",
        config: dict | None = None,
    ) -> None:
        super().__init__(model, config or _CFG)
        self.corruption_rate = corruption_rate
        self.attack_type = attack_type
        self._rng = np.random.default_rng(RANDOM_SEED)

    def generate(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return poisoned X (labels modified separately via poison_labels for label-based attacks)."""
        if self.attack_type == "feature_perturb":
            return self._perturb_features(X)
        if self.attack_type == "gain_guided":
            return self._gain_guided(X)
        if self.attack_type == "clean_label":
            return self._clean_label(X, y)
        # label_flip, targeted_flip, backdoor: handled in generate_with_labels
        return X.copy()

    def generate_with_labels(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (X_poisoned, y_poisoned).  Used for attack types that modify
        both features and labels jointly.
        """
        if self.attack_type == "backdoor":
            return self._backdoor(X, y)
        if self.attack_type == "targeted_flip":
            return self._targeted_flip(X, y)
        return self.generate(X, y), self.poison_labels(y)  # label_flip / feature_perturb etc.

    def evaluate(
        self,
        X_orig: np.ndarray,
        X_adv: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "corruption_rate": self.corruption_rate,
            "attack_type": self.attack_type,
        }

    def poison_labels(self, y: np.ndarray) -> np.ndarray:
        """Flip a fraction of binary labels uniformly at random."""
        y_poisoned = y.copy()
        n_flip = int(len(y) * self.corruption_rate)
        flip_idx = self._rng.choice(len(y), size=n_flip, replace=False)
        y_poisoned[flip_idx] = 1 - y_poisoned[flip_idx]
        logger.info("Poisoned %d/%d labels (%.1f%%).", n_flip, len(y), self.corruption_rate * 100)
        return y_poisoned

    def _perturb_features(self, X: np.ndarray) -> np.ndarray:
        """Add Gaussian noise uniformly across all features."""
        X_p = X.copy().astype(float)
        n_corrupt = int(len(X) * self.corruption_rate)
        corrupt_idx = self._rng.choice(len(X), size=n_corrupt, replace=False)
        noise = self._rng.normal(0, X.std(axis=0), (n_corrupt, X.shape[1]))
        X_p[corrupt_idx] += noise
        logger.info("Feature-poisoned %d/%d samples.", n_corrupt, len(X))
        return X_p

    def _gain_guided(self, X: np.ndarray) -> np.ndarray:
        """Lognormal multiplicative noise on top-K Gain-ranked features; preserves non-negative right-skewed distributions."""
        X_p = X.copy().astype(float)
        n_corrupt = int(len(X) * self.corruption_rate)
        corrupt_idx = self._rng.choice(len(X), size=n_corrupt, replace=False)

        try:
            importances = self.model._model.feature_importances_
            top_features = np.argsort(importances)[-_GAIN_TOP_K:]
        except AttributeError:
            top_features = np.arange(min(_GAIN_TOP_K, X.shape[1]))

        for feat_idx in top_features:
            col = X_p[corrupt_idx, feat_idx]
            mean_f = np.abs(col.mean()) + 1e-9
            std_f = col.std() + 1e-9
            sigma = np.sqrt(np.log(1 + (std_f / mean_f) ** 2))
            X_p[corrupt_idx, feat_idx] = col * self._rng.lognormal(0, sigma, n_corrupt)

        logger.info(
            "Gain-guided: perturbed %d samples on top-%d features %s.",
            n_corrupt, len(top_features), top_features.tolist(),
        )
        return X_p

    def _backdoor(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Insert trigger (feature[0] = mean+3σ) into positives and relabel as negative (Chen et al., 2017)."""
        X_p = X.copy().astype(float)
        y_p = y.copy()

        pos_idx = np.where(y == 1)[0]
        n_inject = max(1, int(len(pos_idx) * self.corruption_rate))
        inject_idx = self._rng.choice(pos_idx, size=min(n_inject, len(pos_idx)), replace=False)

        trigger_val = float(X[:, 0].mean() + 3.0 * X[:, 0].std())
        X_p[inject_idx, 0] = trigger_val
        y_p[inject_idx] = 0

        logger.info(
            "Backdoor: trigger=%.4f injected into %d/%d positive samples.",
            trigger_val, len(inject_idx), len(pos_idx),
        )
        return X_p, y_p

    def _targeted_flip(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Flip labels of top-quartile L1-magnitude samples at 2× base rate — targets high-value transactions."""
        X_p = X.copy()
        y_p = y.copy()
        feature_magnitude = np.abs(X).sum(axis=1)
        threshold = np.percentile(feature_magnitude, 75)
        high_val_idx = np.where(feature_magnitude >= threshold)[0]
        n_flip = int(len(high_val_idx) * min(self.corruption_rate * 2, 1.0))
        if n_flip > 0 and len(high_val_idx) > 0:
            flip_idx = self._rng.choice(
                high_val_idx, size=min(n_flip, len(high_val_idx)), replace=False
            )
            y_p[flip_idx] = 1 - y_p[flip_idx]
        logger.info(
            "Targeted-flip: %.1f%% budget → flipped %d/%d high-value samples.",
            self.corruption_rate * 100, n_flip, len(high_val_idx),
        )
        return X_p, y_p

    def _clean_label(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Perturb positive-class features (0.5σ noise) without changing labels — shifts decision boundary silently."""
        X_p = X.copy().astype(float)
        pos_idx = np.where(y == 1)[0]
        if len(pos_idx) == 0:
            return X_p
        n_corrupt = max(1, int(len(pos_idx) * self.corruption_rate))
        corrupt_idx = self._rng.choice(
            pos_idx, size=min(n_corrupt, len(pos_idx)), replace=False
        )
        noise = self._rng.normal(
            0,
            0.5 * X.std(axis=0),
            (len(corrupt_idx), X.shape[1]),
        )
        X_p[corrupt_idx] += noise
        logger.info(
            "Clean-label: corrupted %d/%d positive-class samples (features only).",
            len(corrupt_idx), len(pos_idx),
        )
        return X_p

    def get_trigger_value(self, X: np.ndarray) -> float:
        """Return the trigger value used by the backdoor attack."""
        return float(X[:, 0].mean() + 3.0 * X[:, 0].std())

    def run_sweep(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[float, np.ndarray, np.ndarray]]:
        """Yield (rate, X_poisoned, y_poisoned) for each configured corruption rate."""
        results = []
        for rate in self.config.get("corruption_rates", [self.corruption_rate]):
            self.corruption_rate = rate
            X_p, y_p = self.generate_with_labels(X, y)
            results.append((rate, X_p, y_p))
        return results
