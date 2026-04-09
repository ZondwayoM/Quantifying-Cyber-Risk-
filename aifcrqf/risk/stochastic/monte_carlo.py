"""Monte Carlo engine: EL = P_success × Impact, RRI = EL × (1−M), CVaR = mean of losses exceeding VaR."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from config.settings import (
    CVAR_CONFIDENCE,
    MONTE_CARLO_ITERATIONS,
    RANDOM_SEED,
    VAR_CONFIDENCE,
)
from core.bases import BaseRiskModel
from risk.financial.metrics import beta_fit, tail_mean


def sample_p_success(alpha, beta, n, rng):
    return rng.beta(alpha, beta, size=n)


def sample_impact(mean, std, n, rng):
    """Lognormal MLE (method-of-moments): mu/sigma from mean and std, then rng.lognormal."""
    sigma = np.sqrt(np.log(1 + (std / mean) ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return rng.lognormal(mu, sigma, size=n)


def sample_iso_maturity(maturity_level, n, rng, uncertainty=0.05):
    return np.clip(rng.normal(maturity_level, uncertainty, size=n), 0.0, 1.0)

logger = logging.getLogger(__name__)


class MonteCarloEngine(BaseRiskModel):
    """50,000-iteration MC: Beta(p_success), Lognormal(impact), Normal(M, 0.05) maturity noise."""

    def __init__(self, config: dict | None = None) -> None:
        super().__init__(config or {})
        self._n_iter = self.config.get("iterations", MONTE_CARLO_ITERATIONS)
        self._seed = self.config.get("random_seed", RANDOM_SEED)
        self._rng = np.random.default_rng(self._seed)
        self._el_distribution: np.ndarray | None = None
        self._rri_distribution: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Main simulation
    # ------------------------------------------------------------------

    def run(
        self,
        p_success_observed: float,
        n_trials: int,
        impact_mean: float,
        impact_std: float,
        iso_maturity: float,
    ) -> None:
        """Run 50,000 MC iterations: Beta p_success, Lognormal impact, Normal(M, 0.05) maturity."""
        if p_success_observed == 0.0:
            zeros = np.zeros(self._n_iter)
            self._el_distribution = zeros.copy()
            self._rri_distribution = zeros.copy()
            self._loss_distribution = zeros.copy()
            logger.info("MC simulation complete | EL mean=0.00 | RRI mean=0.00 (p=0 case)")
            return

        alpha, beta_param = beta_fit(
            successes=int(p_success_observed * n_trials),
            trials=n_trials,
        )

        p_samples = sample_p_success(alpha, beta_param, self._n_iter, self._rng)
        i_samples = sample_impact(impact_mean, impact_std, self._n_iter, self._rng)

        if iso_maturity >= 1.0:
            m_samples = np.ones(self._n_iter)
        else:
            m_samples = sample_iso_maturity(iso_maturity, n=self._n_iter, rng=self._rng)

        self._el_distribution = p_samples * i_samples
        self._rri_distribution = self._el_distribution * (1.0 - m_samples)
        self._loss_distribution = self._el_distribution

        logger.info(
            "MC simulation complete | EL mean=%.2f | RRI mean=%.2f",
            self._el_distribution.mean(), self._rri_distribution.mean(),
        )

    def calculate_expected_loss(self, **kwargs) -> float:
        self._require_simulation()
        return float(self._el_distribution.mean())

    def calculate_var(self, confidence: float = VAR_CONFIDENCE) -> float:
        self._require_simulation()
        return float(np.percentile(self._rri_distribution, confidence * 100))

    def calculate_cvar(self, confidence: float = CVAR_CONFIDENCE) -> float:
        self._require_simulation()
        return float(tail_mean(self._rri_distribution, confidence))

    def get_metrics(self) -> Dict[str, float]:
        """Return all core risk metrics as a flat dict."""
        el_mean = self.calculate_expected_loss()
        threshold = 5.0 * max(el_mean, 1e-9)  # p_extreme = P(RRI > 5 × EL_mean)
        p_extreme = float(np.mean(self._rri_distribution > threshold))
        return {
            "el_mean":   el_mean,
            "el_std":    float(self._el_distribution.std()),
            "el_median": float(np.median(self._el_distribution)),
            "el_max":    float(self._el_distribution.max()),
            "rri_mean":  float(self._rri_distribution.mean()),
            "rri_max":   float(self._rri_distribution.max()),
            "var_95":    self.calculate_var(0.95),
            "var_99":    self.calculate_var(0.99),
            "cvar_99":   self.calculate_cvar(0.99),
            "p_extreme": p_extreme,
        }

    def _require_simulation(self) -> None:
        if self._el_distribution is None:
            raise RuntimeError("Call .run() before requesting metrics.")
