"""Cascading loss model: 5 components (direct + regulatory + reputational + churn + operational).
Delayed disclosure applies R_f=1.5 reputational multiplier (Wu et al., 2022)."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from config.settings import MONTE_CARLO_ITERATIONS, RANDOM_SEED

logger = logging.getLogger(__name__)


class CascadingImpactModel:
    """Simulate 5-component cascading loss on top of the MC EL distribution."""

    def __init__(
        self,
        regulatory_multiplier: float = 0.20,
        reputational_multiplier: float = 0.05,
        churn_multiplier: float = 0.03,
        operational_fixed: float = 50_000,
        operational_std: float = 10_000,
        disclosure_delay: bool = False,
        n_iter: int = MONTE_CARLO_ITERATIONS,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.regulatory_multiplier = regulatory_multiplier
        self.reputational_multiplier = reputational_multiplier
        self.churn_multiplier = churn_multiplier
        self.operational_fixed = operational_fixed
        self.operational_std = operational_std
        self.disclosure_multiplier: float = 1.5 if disclosure_delay else 1.0  # R_f (Wu et al., 2022)
        self.n_iter = n_iter
        self._rng = np.random.default_rng(seed)
        self._results: Dict[str, np.ndarray] = {}

    def run(self, direct_losses: np.ndarray) -> Dict[str, np.ndarray]:
        """Apply 5-component cascade to the MC EL distribution; return per-component arrays."""
        n = len(direct_losses)

        regulatory = direct_losses * self._rng.uniform(
            self.regulatory_multiplier * 0.5,
            self.regulatory_multiplier * 1.5,
            n,
        )
        effective_rep = self.reputational_multiplier * self.disclosure_multiplier  # R_f applied here
        reputational = direct_losses * self._rng.uniform(
            effective_rep * 0.5,
            effective_rep * 1.5,
            n,
        )
        churn = direct_losses * self._rng.uniform(
            self.churn_multiplier * 0.5,
            self.churn_multiplier * 1.5,
            n,
        )
        op_sigma = np.sqrt(np.log(1 + (self.operational_std / self.operational_fixed) ** 2))
        op_mu = np.log(self.operational_fixed) - 0.5 * op_sigma ** 2
        operational = self._rng.lognormal(op_mu, op_sigma, n)

        total = direct_losses + regulatory + reputational + churn + operational

        self._results = {
            "direct": direct_losses,
            "regulatory": regulatory,
            "reputational": reputational,
            "churn": churn,
            "operational": operational,
            "total_cascade": total,
        }

        logger.info(
            "Cascading impact | total mean=%.2f | max=%.2f",
            total.mean(), total.max(),
        )
        return self._results

    def get_summary(self) -> Dict[str, float]:
        """Return mean values per loss component."""
        return {k: float(v.mean()) for k, v in self._results.items()}
