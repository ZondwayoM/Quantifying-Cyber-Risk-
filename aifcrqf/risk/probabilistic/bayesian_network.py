"""5-node Bayesian Network: ThreatLevel → AttackSuccess ← ControlStrength → DirectLoss → CriticalLoss.
AttackSuccess CPD is calibrated from empirical FGSM/C&W p_success; other cells scaled by relative-risk multipliers."""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

try:
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork   # pgmpy < 1.0
    _PGMPY_AVAILABLE = True
except ImportError:
    _PGMPY_AVAILABLE = False
    logger.warning("pgmpy not installed -- Bayesian Network unavailable.")


# Relative-risk multipliers per (ThreatLevel, ControlStrength) cell, baseline (TL=1,CS=1) = 1.0
_RELATIVE_MULTIPLIERS = [
    0.920, 0.491, 0.184,   # TL=0, CS=0/1/2
    2.147, 1.000, 0.429,   # TL=1, CS=0/1/2
    5.092, 3.374, 2.049,   # TL=2, CS=0/1/2
]

_DEFAULT_EMPIRICAL_P = 0.163   # default when no live adversarial result supplied

_P_DL_GIVEN_NO_ATTACK  = 0.04
_P_DL_GIVEN_ATTACK     = 0.88

_P_CL_GIVEN_NO_LOSS    = 0.02
_P_CL_GIVEN_LOSS       = 0.55


class AIFCRQFBayesianNetwork:
    """5-node BN calibrated to empirical p_success from adversarial evaluator (default 0.163)."""

    def __init__(self, empirical_p_success: Optional[float] = None) -> None:
        if not _PGMPY_AVAILABLE:
            raise ImportError("Install pgmpy: pip install pgmpy")
        self._empirical_p = (
            float(empirical_p_success)
            if empirical_p_success is not None
            else _DEFAULT_EMPIRICAL_P
        )
        self._model = None
        self._inference: VariableElimination | None = None
        self._build()

    # ------------------------------------------------------------------
    # Internal build
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Build 5-node network and compute CPDs from empirical baseline."""
        self._model = BayesianNetwork([
            ("ThreatLevel",     "AttackSuccess"),
            ("ControlStrength", "AttackSuccess"),
            ("AttackSuccess",   "DirectLoss"),
            ("DirectLoss",      "CriticalLoss"),
        ])

        # Equal priors; always overridden by full evidence in queries
        cpd_threat = TabularCPD(
            variable="ThreatLevel", variable_card=3,
            values=[[0.33], [0.34], [0.33]],
        )
        cpd_control = TabularCPD(
            variable="ControlStrength", variable_card=3,
            values=[[0.33], [0.34], [0.33]],
        )

        # Scale each cell from empirical baseline using relative-risk multipliers; clamp to [0.001, 0.999]
        p_success_vals = [
            min(0.999, max(0.001, self._empirical_p * m))
            for m in _RELATIVE_MULTIPLIERS
        ]
        p_no_success_vals = [1.0 - p for p in p_success_vals]

        cpd_attack = TabularCPD(
            variable="AttackSuccess", variable_card=2,
            values=[p_no_success_vals, p_success_vals],
            evidence=["ThreatLevel", "ControlStrength"],
            evidence_card=[3, 3],
        )

        cpd_loss = TabularCPD(
            variable="DirectLoss", variable_card=2,
            values=[
                [1 - _P_DL_GIVEN_NO_ATTACK, 1 - _P_DL_GIVEN_ATTACK],
                [    _P_DL_GIVEN_NO_ATTACK,     _P_DL_GIVEN_ATTACK],
            ],
            evidence=["AttackSuccess"], evidence_card=[2],
        )

        cpd_critical = TabularCPD(
            variable="CriticalLoss", variable_card=2,
            values=[
                [1 - _P_CL_GIVEN_NO_LOSS, 1 - _P_CL_GIVEN_LOSS],
                [    _P_CL_GIVEN_NO_LOSS,     _P_CL_GIVEN_LOSS],
            ],
            evidence=["DirectLoss"], evidence_card=[2],
        )

        self._model.add_cpds(
            cpd_threat, cpd_control, cpd_attack, cpd_loss, cpd_critical
        )
        assert self._model.check_model(), "Bayesian Network is invalid."
        self._inference = VariableElimination(self._model)
        logger.info(
            "BN built | empirical P_success=%.4f | baseline P(AS=1|Med,Med)=%.4f",
            self._empirical_p, p_success_vals[4],
        )

    def query(
        self,
        target: str,
        evidence: Dict[str, int] | None = None,
    ) -> Dict[str, float]:
        """Variable elimination for target given evidence; returns {state_index: probability}."""
        result = self._inference.query(
            [target], evidence=evidence or {}, show_progress=False
        )
        return {i: float(v) for i, v in enumerate(result.values.tolist())}

    def attack_success_probability(self, threat: int = 1, control: int = 1) -> float:
        """P(AttackSuccess=1 | ThreatLevel=threat, ControlStrength=control). States: 0=Low, 1=Med, 2=High."""
        return self.query(
            "AttackSuccess",
            {"ThreatLevel": threat, "ControlStrength": control},
        )[1]

    def critical_loss_probability(self, threat: int = 1, control: int = 1) -> float:
        """P(CriticalLoss=1) propagated through AttackSuccess → DirectLoss → CriticalLoss."""
        return self.query(
            "CriticalLoss",
            {"ThreatLevel": threat, "ControlStrength": control},
        )[1]
