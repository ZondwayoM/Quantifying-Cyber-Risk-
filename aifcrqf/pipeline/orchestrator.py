"""Pipeline Orchestrator — top-level controller that runs all stages: data → model → attack → risk → governance → export."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from config.settings import FINANCIAL_IMPACT, ISO_MATURITY_LEVELS, MONTE_CARLO_ITERATIONS, RANDOM_SEED
from core.detector import IntelligentDetector
from core.preprocessor import DataPreprocessor
from data.loaders import DataLoader
from models.aml.model import AMLDetectionModel
from models.credit.model import CreditScoringModel
from models.fraud.model import FraudDetectionModel
from models.trading.model import TradingSignalModel
from models.trainer import DomainTrainer
from models.evaluator import DomainEvaluator

logger = logging.getLogger(__name__)

_MODEL_MAP = {
    "fraud":   FraudDetectionModel,
    "credit":  CreditScoringModel,
    "aml":     AMLDetectionModel,
    "trading": TradingSignalModel,
}


class Orchestrator:
    """Runs the full AIFCRQF pipeline end-to-end for a single domain."""

    def __init__(
        self,
        data_path: str,
        target_col: Optional[str] = None,
        output_dir: str = "outputs/",
        iso_maturity: str = "medium",
        mc_iterations: int = MONTE_CARLO_ITERATIONS,
    ) -> None:
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = Path(output_dir)
        self.iso_maturity = iso_maturity
        self.mc_iterations = mc_iterations
        self.state: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute all pipeline stages sequentially."""
        self._stage_load_data()
        self._stage_detect_and_preprocess()
        self._stage_train_model()
        self._stage_adversarial()
        self._stage_risk_quantification()
        self._stage_governance()
        self._stage_export()
        logger.info("Pipeline finished. Results in: %s", self.output_dir)
        return self.state

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage_load_data(self) -> None:
        t0 = time.perf_counter()
        self.state["df"] = DataLoader().load_csv(self.data_path)
        logger.info("Stage 'load_data' completed in %.2fs.", time.perf_counter() - t0)

    _MAX_TRAIN_SAMPLES = 1_000_000   # stratified cap for large datasets

    def _stage_detect_and_preprocess(self) -> None:
        df = self.state["df"]
        detector = IntelligentDetector()
        target = detector.detect_target_column(df, self.target_col)
        analysis = detector.analyze_dataset(df, target)
        self.state["target_col"] = target
        self.state["analysis"] = analysis
        self.state["domain"] = analysis["domain"]

        if len(df) > self._MAX_TRAIN_SAMPLES:
            logger.info(
                "Dataset has %d rows — sampling %d for pipeline (stratified).",
                len(df), self._MAX_TRAIN_SAMPLES,
            )
            try:
                df = df.groupby(target, group_keys=False).apply(
                    lambda g: g.sample(
                        frac=self._MAX_TRAIN_SAMPLES / len(df),
                        random_state=RANDOM_SEED,
                    )
                ).reset_index(drop=True)
            except Exception:  # noqa: BLE001
                df = df.sample(self._MAX_TRAIN_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)

        prep = DataPreprocessor()
        df_clean = prep.clean_data(df)
        X, y = prep.split_features_target(df_clean, target)
        X_scaled = prep.scale_features(X, fit=True)
        self.state.update({"X": X_scaled, "y": y.values, "preprocessor": prep})

        # Lognormal fit on raw dataset amounts; falls back to config if insufficient data
        self.state["impact_cfg"] = self._compute_impact_from_data(
            self.state["df"], self.state["domain"], target
        )

    def _stage_train_model(self) -> None:
        domain = self.state["domain"]
        X, y = self.state["X"], self.state["y"]

        model_cls = _MODEL_MAP.get(domain, FraudDetectionModel)
        trainer = DomainTrainer(model_cls, domain)
        splits = trainer.train(X, y, save=True)
        self.state.update({"trainer": trainer, "splits": splits})

        evaluator = DomainEvaluator(trainer.model, domain)
        self.state["baseline_metrics"] = evaluator.evaluate(splits["X_test"], splits["y_test"])

    def _stage_adversarial(self) -> None:
        from adversarial.evaluator import AdversarialEvaluator
        splits = self.state["splits"]
        model = self.state["trainer"].model
        evaluator = AdversarialEvaluator(model)

        attack_summary = evaluator.run_all(splits["X_test"], splits["y_test"])

        poisoning_results = evaluator.run_poisoning_with_retraining(
            splits["X_train"], splits["y_train"],
            splits["X_test"], splits["y_test"],
            domain=self.state.get("domain", "unknown"),
            output_dir=self.output_dir / "exports",
        )
        attack_summary["poisoning"] = poisoning_results

        self.state["attack_summary"] = attack_summary
        self.state["adversarial_evaluator"] = evaluator

    def _stage_risk_quantification(self) -> None:
        from risk.stochastic.simulation_engine import SimulationEngine
        domain = self.state["domain"]
        # Use data-derived impact if available, otherwise fall back to config defaults
        impact_cfg = self.state.get("impact_cfg") or FINANCIAL_IMPACT.get(domain, FINANCIAL_IMPACT["fraud"])
        logger.info(
            "Impact parameters [%s]: mean=$%.2f  std=$%.2f  (source=%s)",
            domain, impact_cfg["mean"], impact_cfg["std"],
            "data-derived" if self.state.get("impact_cfg") else "config-default",
        )
        evasion_p = float(self.state["attack_summary"].get("p_success_mean", 0.0))
        # Independence model: P_combined = 1 − (1 − p_evasion)(1 − p_poison)
        poisoning_data = self.state["attack_summary"].get("poisoning", {})
        max_poison_deg = max(
            (v.get("recall_degradation", 0.0) for v in poisoning_data.get("rates", {}).values()),
            default=0.0,
        )
        p_success = 1.0 - (1.0 - evasion_p) * (1.0 - max(max_poison_deg, 0.0))
        p_success = float(np.clip(p_success, 0.0, 1.0))
        self.state["combined_p_success"] = p_success  # expose to dashboard builder
        logger.info(
            "Combined p_success [%s]: evasion_mean=%.6f  max_poison_degradation=%.6f  p_success=%.6f",
            domain, evasion_p, max_poison_deg, p_success,
        )
        n_trials = len(self.state["splits"]["y_test"])
        op_mean = float(impact_cfg.get("operational_mean", 50_000))
        op_std  = float(impact_cfg.get("operational_std",  10_000))

        # Primary simulation at the configured maturity level
        engine = SimulationEngine({"iterations": self.mc_iterations})
        metrics = engine.run(
            p_success=p_success,
            n_trials=n_trials,
            impact_mean=impact_cfg["mean"],
            impact_std=impact_cfg["std"],
            iso_maturity_label=self.iso_maturity,
            operational_mean=op_mean,
            operational_std=op_std,
        )
        self.state["risk_metrics"] = metrics
        self.state["simulation_engine"] = engine

        # Run MC at weak/medium/strong; EL stays constant, only RRI = EL × (1−M) varies
        maturity_comparison: Dict[str, Any] = {}
        for label in ("weak", "medium", "strong"):
            mat_level = ISO_MATURITY_LEVELS[label]
            eng = SimulationEngine({"iterations": self.mc_iterations})
            m = eng.run(
                p_success=p_success,
                n_trials=n_trials,
                impact_mean=impact_cfg["mean"],
                impact_std=impact_cfg["std"],
                iso_maturity_label=label,
                operational_mean=op_mean,
                operational_std=op_std,
            )
            tail_prem = m["cvar_99"] - m.get("var_99", m["cvar_99"])
            maturity_comparison[label] = {
                "maturity_value": mat_level,
                "el_mean":      m["el_mean"],
                "el_std":       m.get("el_std", 0.0),
                "rri_mean":     m["rri_mean"],
                "var_95":       m["var_95"],
                "var_99":       m.get("var_99", 0.0),
                "cvar_99":      m["cvar_99"],
                "tail_premium": max(tail_prem, 0.0),
                "p_extreme":    m.get("p_extreme", 0.0),
            }
            logger.info(
                "Maturity [%s | M=%.1f]: EL=$%.2f  RRI=$%.2f  VaR95=$%.2f  CVaR99=$%.2f",
                label, mat_level,
                m["el_mean"], m["rri_mean"], m["var_95"], m["cvar_99"],
            )
        self.state["maturity_comparison"] = maturity_comparison

        # CVaR(99) stability: 5 replications, max deviation ≤ 5% criteria
        try:
            from risk.stochastic.monte_carlo import MonteCarloEngine
            _cvar_reps: list[float] = []
            for _seed_offset in range(5):
                _mc = MonteCarloEngine({
                    "iterations": self.mc_iterations,
                    "random_seed": RANDOM_SEED + _seed_offset * 137,
                })
                _mc.run(
                    p_success_observed=p_success,
                    n_trials=n_trials,
                    impact_mean=impact_cfg["mean"],
                    impact_std=impact_cfg["std"],
                    iso_maturity=ISO_MATURITY_LEVELS[self.iso_maturity],
                )
                _cvar_reps.append(_mc.calculate_cvar(0.99))
            _cvar_arr  = np.array(_cvar_reps)
            _cvar_mean = float(_cvar_arr.mean())
            _max_dev_pct = (
                float(np.abs(_cvar_arr - _cvar_mean).max() / _cvar_mean * 100)
                if _cvar_mean > 0 else 0.0
            )
            self.state["cvar_stability"] = {
                "replication_cvars": _cvar_reps,
                "mean": _cvar_mean,
                "max_deviation_pct": round(_max_dev_pct, 4),
                "stable": _max_dev_pct <= 5.0,
            }
            logger.info(
                "CVaR stability [%s]: mean=$%.2f  max_dev=%.2f%%  stable=%s",
                domain, _cvar_mean, _max_dev_pct, _max_dev_pct <= 5.0,
            )
        except Exception as _exc:  # noqa: BLE001
            logger.warning("CVaR stability check failed: %s", _exc)
            self.state["cvar_stability"] = {"stable": None, "max_deviation_pct": None}

        # Disclosure delay: R_f=1.5 amplifies reputational loss (Wu et al., 2022)
        try:
            from risk.stochastic.cascading import CascadingImpactModel
            direct = engine.mc._el_distribution
            _no_delay = CascadingImpactModel(disclosure_delay=False, seed=RANDOM_SEED)
            _no_delay.run(direct)
            _with_delay = CascadingImpactModel(disclosure_delay=True, seed=RANDOM_SEED)
            _with_delay.run(direct)
            nd = _no_delay.get_summary()
            wd = _with_delay.get_summary()
            self.state["disclosure_comparison"] = {
                "immediate_disclosure": {
                    "reputational_mean": nd["reputational"],
                    "total_cascade_mean": nd["total_cascade"],
                },
                "delayed_disclosure": {
                    "reputational_mean": wd["reputational"],
                    "total_cascade_mean": wd["total_cascade"],
                },
                "reputational_amplification_factor": (
                    wd["reputational"] / nd["reputational"]
                    if nd["reputational"] > 0 else None
                ),
            }
            rf = self.state["disclosure_comparison"]["reputational_amplification_factor"]
            logger.info(
                "Disclosure delay comparison | R_f=%s | immediate_rep=%.2f | delayed_rep=%.2f",
                f"{rf:.2f}" if rf is not None else "N/A",
                nd["reputational"],
                wd["reputational"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Disclosure comparison failed: %s", exc)
            self.state["disclosure_comparison"] = {}

        # 5-node Bayesian Network inference — baseline, worst-case, mitigated scenarios
        try:
            from risk.probabilistic.bayesian_network import AIFCRQFBayesianNetwork
            empirical_p = float(self.state["attack_summary"].get("p_success_mean", p_success))
            bn = AIFCRQFBayesianNetwork(empirical_p_success=empirical_p)
            self.state["bn_results"] = {
                "baseline": {
                    "label": "Baseline (Medium Threat, Medium Control)",
                    "attack_success_prob": bn.attack_success_probability(threat=1, control=1),
                    "critical_loss_prob": bn.critical_loss_probability(threat=1, control=1),
                },
                "worst_case": {
                    "label": "Worst Case (High Threat, Weak Control)",
                    "attack_success_prob": bn.attack_success_probability(threat=2, control=0),
                    "critical_loss_prob": bn.critical_loss_probability(threat=2, control=0),
                },
                "mitigated": {
                    "label": "Mitigated (High Threat, Strong Control)",
                    "attack_success_prob": bn.attack_success_probability(threat=2, control=2),
                    "critical_loss_prob": bn.critical_loss_probability(threat=2, control=2),
                },
            }
            self.state["bayesian_network"] = bn
            logger.info("Bayesian Network inference complete: %s", self.state["bn_results"])

            bn_mc_scenarios: dict = {}
            for scenario_key, bn_result in self.state["bn_results"].items():
                bn_p = bn_result["attack_success_prob"]
                _eng = SimulationEngine({"iterations": self.mc_iterations})
                _m = _eng.run(
                    p_success=bn_p,
                    n_trials=n_trials,
                    impact_mean=impact_cfg["mean"],
                    impact_std=impact_cfg["std"],
                    iso_maturity_label=self.iso_maturity,
                    operational_mean=op_mean,
                    operational_std=op_std,
                )
                bn_mc_scenarios[scenario_key] = {
                    "label": bn_result["label"],
                    "bn_p_success": bn_p,
                    **_m,
                    "_rri_distribution": _eng.mc._rri_distribution,
                }

            bn_mc_scenarios["empirical"] = {
                "label": f"Empirical (FGSM/C&W, P={empirical_p:.3f})",
                "bn_p_success": empirical_p,
                **self.state["risk_metrics"],
                "_rri_distribution": engine.mc._rri_distribution,
            }
            self.state["bn_mc_scenarios"] = bn_mc_scenarios
            logger.info(
                "BN-MC bridge complete | scenarios: %s",
                list(bn_mc_scenarios.keys()),
            )
        except ImportError:
            logger.warning("pgmpy not installed — Bayesian Network skipped.")
            self.state["bn_results"] = {}
            self.state["bn_mc_scenarios"] = {}

    _AMOUNT_COLS: Dict[str, str] = {
        "fraud":   "Amount",          # creditcard.csv
        "credit":  "Credit amount",   # german_credit.csv
        "aml":     "Amount Received", # aml_ibm.csv
        "trading": None,              # no direct amount column
    }

    def _compute_impact_from_data(
        self, df: Any, domain: str, target_col: str
    ) -> Dict[str, float]:
        """Fit lognormal MLE (mu, sigma) on positive-class amounts; return (mean, std) for MC."""
        import numpy as _np
        col = self._AMOUNT_COLS.get(domain)
        fallback = FINANCIAL_IMPACT.get(domain, FINANCIAL_IMPACT["fraud"])

        if col is None or col not in df.columns or target_col not in df.columns:
            return fallback

        pos_amounts = df.loc[df[target_col] == 1, col].dropna()
        pos_amounts = pos_amounts[pos_amounts > 0]
        if len(pos_amounts) < 10:
            return fallback

        # Winsorise at 95th pct to remove AML currency-scale outliers
        cap = pos_amounts.quantile(0.95)
        clipped = pos_amounts.clip(upper=cap)

        # Lognormal MLE: mu/sigma from log-scale moments
        log_x = _np.log(clipped.values.astype(float))
        mu    = float(log_x.mean())
        sigma = float(log_x.std())

        mean_val = float(_np.exp(mu + 0.5 * sigma ** 2))
        std_val  = float(mean_val * _np.sqrt(_np.exp(sigma ** 2) - 1))

        logger.info(
            "Lognormal fit [%s]: col='%s' | n=%d | mu=%.4f | sigma=%.4f | "
            "E[loss]=$%.2f | SD=$%.2f | 95th-pct-cap=$%.2f",
            domain, col, len(pos_amounts), mu, sigma, mean_val, std_val, cap,
        )
        return {"mean": mean_val, "std": std_val, "ln_mu": mu, "ln_sigma": sigma}

    def _stage_governance(self) -> None:
        from governance.metrics import governance_dashboard_metrics
        el = self.state["risk_metrics"]["el_mean"]
        maturity = ISO_MATURITY_LEVELS[self.iso_maturity]

        domain_scores = {d: maturity for d in [
            "Access Control", "Asset Management", "Incident Response",
            "Cryptography", "Operations Security", "Communications Security",
            "Supplier Relationships",
        ]}
        gov_metrics = governance_dashboard_metrics(domain_scores, el)
        self.state["governance_metrics"] = gov_metrics

    _DOMAIN_DISPLAY = {
        "fraud":   "Fraud Detection",
        "credit":  "Credit Scoring",
        "aml":     "AML Detection",
        "trading": "Algorithmic Trading",
    }

    def _stage_export(self) -> None:
        from visualization.dashboard_builder import export_all
        from visualization.dashboard_builder import DashboardBuilder
        export_all(self.state, self.output_dir)
        DashboardBuilder(self.output_dir / "exports").build(self.state)
        self._write_run_config()
        self._generate_plots()

    def _write_run_config(self) -> None:
        """Write/update last_run_config.json with the maturity level used this run."""
        domain_key = self._DOMAIN_DISPLAY.get(
            self.state.get("domain", ""), self.state.get("domain", "unknown")
        )
        config_path = self.output_dir / "exports" / "powerbi" / "last_run_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                existing = {}
        existing[domain_key] = {
            "iso_maturity": self.iso_maturity,
            "timestamp": datetime.now().isoformat(),
        }
        config_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
        logger.info("Run config written: %s → maturity=%s", domain_key, self.iso_maturity)

    def _generate_plots(self) -> None:
        """Generate and save all Matplotlib plots to VIZ_DIR."""
        from visualization import plots
        try:
            if self.state.get("bn_mc_scenarios"):
                plots.plot_bn_mc_integration(
                    self.state["bn_mc_scenarios"],
                    filename=f"bn_mc_integration_{self.state.get('domain','')}.png",
                )
            if self.state.get("baseline_metrics"):
                plots.plot_domain_metrics_bar(
                    domain=self.state.get("domain", "unknown"),
                    metrics=self.state["baseline_metrics"],
                    filename=f"domain_metrics_{self.state.get('domain','')}.png",
                )
            if self.state.get("disclosure_comparison"):
                plots.plot_disclosure_comparison(
                    self.state["disclosure_comparison"],
                    filename="disclosure_comparison.png",
                )
            mc = self.state.get("simulation_engine")
            if mc and mc.mc._rri_distribution is not None:
                rm = self.state["risk_metrics"]
                plots.plot_mc_loss_distribution(
                    mc.mc._rri_distribution,
                    rm["var_95"], rm["var_99"], rm["cvar_99"],
                )
            if mc and mc.cascade:
                plots.plot_cascading_waterfall(mc.cascade.get_summary())
        except Exception as exc:  # noqa: BLE001
            logger.warning("Plot generation error (non-fatal): %s", exc)
