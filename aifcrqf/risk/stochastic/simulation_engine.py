"""SimulationEngine — combines MonteCarloEngine and CascadingImpactModel in one call."""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from config.settings import ISO_MATURITY_LEVELS
from risk.stochastic.cascading import CascadingImpactModel
from risk.stochastic.monte_carlo import MonteCarloEngine

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Run the full stochastic risk pipeline (MC + cascading) in one call."""

    def __init__(self, mc_config: dict | None = None) -> None:
        self.mc = MonteCarloEngine(mc_config)
        self.cascade: CascadingImpactModel | None = None

    def run(
        self,
        p_success: float,
        n_trials: int,
        impact_mean: float,
        impact_std: float,
        iso_maturity_label: str = "medium",
        operational_mean: float = 50_000,
        operational_std: float = 10_000,
    ) -> Dict[str, float]:
        """Run MC + cascading; return merged metrics dict including cascade_* keys."""
        maturity = ISO_MATURITY_LEVELS.get(iso_maturity_label, 0.60)

        self.mc.run(
            p_success_observed=p_success,
            n_trials=n_trials,
            impact_mean=impact_mean,
            impact_std=impact_std,
            iso_maturity=maturity,
        )

        direct = self.mc._el_distribution
        self.cascade = CascadingImpactModel(
            operational_fixed=operational_mean,
            operational_std=operational_std,
        )
        cascade_results = self.cascade.run(direct)

        mc_metrics = self.mc.get_metrics()
        cascade_summary = self.cascade.get_summary()

        combined = {**mc_metrics, **{f"cascade_{k}": v for k, v in cascade_summary.items()}}
        combined["iso_maturity"] = maturity
        combined["iso_maturity_label"] = iso_maturity_label

        return combined
