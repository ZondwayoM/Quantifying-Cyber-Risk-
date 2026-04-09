"""Dashboard builder — assembles pbi_*.csv feeds for the Streamlit dashboard from pipeline state."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from config.settings import EXPORTS_DIR

logger = logging.getLogger(__name__)


def _write_csv(data: Dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([data]).to_csv(path, index=False)
    return path


def export_domain_metrics(
    domain: str,
    baseline_metrics: Dict[str, Any],
    adversarial_metrics: Dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> Path:
    """Export domain-specific model metrics, accumulating across pipeline runs."""
    out = (output_dir or EXPORTS_DIR) / "domain_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    row: Dict[str, Any] = {"domain": domain}
    for k, v in baseline_metrics.items():
        row[f"baseline_{k}"] = v
    if adversarial_metrics:
        for k, v in adversarial_metrics.items():
            row[f"adversarial_{k}"] = v
        domain_keys = {
            "fraud":   ("fraud_leakage_rate", "chargeback_ratio"),
            "credit":  ("approval_error_rate", "default_miss_rate"),
            "aml":     ("suspicious_activity_miss_rate", "detection_coverage"),
            "trading": ("execution_error_rate", "signal_precision"),
        }
        for key in domain_keys.get(domain, ()):
            base = baseline_metrics.get(key)
            adv = adversarial_metrics.get(key)
            if base is not None and adv is not None:
                row[f"delta_{key}"] = adv - base

    new_df = pd.DataFrame([row])
    if out.exists():
        existing = pd.read_csv(out)
        existing = existing[existing["domain"] != domain]
        new_df = pd.concat([existing, new_df], ignore_index=True)
    new_df.to_csv(out, index=False)
    logger.info("Domain metrics exported: %s", out)
    return out


def export_disclosure_comparison(comparison: Dict[str, Any], output_dir: Path | None = None) -> Path:
    """Export the immediate vs delayed disclosure scenario comparison."""
    out = (output_dir or EXPORTS_DIR) / "disclosure_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario, values in comparison.items():
        if isinstance(values, dict):
            rows.append({"scenario": scenario, **values})
    if "reputational_amplification_factor" in comparison:
        rows.append({
            "scenario": "amplification_factor",
            "reputational_amplification_factor": comparison["reputational_amplification_factor"],
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("Disclosure comparison exported: %s", out)
    return out


def export_all(state: Dict[str, Any], output_dir: Path) -> None:
    """Export all pipeline result CSVs to output_dir/exports/."""
    output_dir = Path(output_dir) / "exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    if "risk_metrics" in state:
        _write_csv(state["risk_metrics"], output_dir / "risk_metrics.csv")
        logger.info("Risk metrics exported")
    if "attack_summary" in state:
        _write_csv(state["attack_summary"], output_dir / "attack_summary.csv")
        logger.info("Attack summary exported")
    if "governance_metrics" in state:
        _write_csv(state["governance_metrics"], output_dir / "governance_metrics.csv")
        logger.info("Governance metrics exported")
    if "domain" in state and state.get("splits") and state.get("baseline_metrics"):
        export_domain_metrics(
            state["domain"],
            state["baseline_metrics"],
            state.get("adversarial_baseline_metrics"),
            output_dir,
        )
    if state.get("disclosure_comparison"):
        export_disclosure_comparison(state["disclosure_comparison"], output_dir)

    logger.info("All exports written to: %s", output_dir)


class DashboardBuilder:
    """Assemble all pbi_*.csv dashboard feeds from pipeline state."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(self, state: Dict[str, Any]) -> Dict[str, Path]:
        """Write all dashboard CSVs; return {name: path}."""
        paths: Dict[str, Path] = {}

        if "risk_metrics" in state:
            paths["risk_metrics"] = self._write(state["risk_metrics"], "risk_metrics.csv")
        if "attack_summary" in state:
            paths["attack_summary"] = self._write(state["attack_summary"], "attack_summary.csv")
        if "governance_metrics" in state:
            paths["governance_metrics"] = self._write(state["governance_metrics"], "governance_metrics.csv")
        if "simulation_engine" in state and state["simulation_engine"].cascade:
            cascade = state["simulation_engine"].cascade.get_summary()
            paths["cascade_components"] = self._write(cascade, "cascade_components.csv")

        if state.get("bn_mc_scenarios"):
            rows = []
            for key, scenario in state["bn_mc_scenarios"].items():
                row = {"scenario": key}
                for k, v in scenario.items():
                    if not k.startswith("_"):
                        row[k] = v
                rows.append(row)
            paths["bn_mc_scenarios"] = self._write_df(rows, "bn_mc_scenarios.csv")

        if state.get("baseline_metrics") and state.get("domain"):
            record = {"domain": state["domain"], **state["baseline_metrics"]}
            dm_path = self.output_dir / "domain_metrics.csv"
            lock_path = dm_path.with_suffix(".writing")
            new_row = pd.DataFrame([record])
            deadline = time.monotonic() + 10
            while lock_path.exists() and time.monotonic() < deadline:
                time.sleep(0.2)
            try:
                lock_path.touch()
                if dm_path.exists():
                    existing = pd.read_csv(dm_path)
                    existing = existing[existing["domain"] != state["domain"]]
                    combined = pd.concat([existing, new_row], ignore_index=True)
                else:
                    combined = new_row
                combined.to_csv(dm_path, index=False)
            finally:
                lock_path.unlink(missing_ok=True)
            paths["domain_metrics"] = dm_path

        if state.get("disclosure_comparison"):
            rows = []
            for scenario, values in state["disclosure_comparison"].items():
                if isinstance(values, dict):
                    rows.append({"scenario": scenario, **values})
                else:
                    rows.append({"scenario": scenario, "value": values})
            paths["disclosure_comparison"] = self._write_df(rows, "disclosure_comparison.csv")

        if state.get("maturity_comparison"):
            rows = [
                {"maturity_level": label, **vals}
                for label, vals in state["maturity_comparison"].items()
            ]
            paths["iso_maturity_comparison"] = self._write_df(rows, "iso_maturity_comparison.csv")

        # Generate pbi_*.csv files consumed by the Streamlit dashboard
        if state.get("domain"):
            self._write_pbi_files(state)

        logger.info("Dashboard feeds written: %s", list(paths.keys()))
        return paths

    def _pbi(self, filename: str) -> Path:
        """Return path inside powerbi/ subdirectory (created on demand)."""
        p = self.output_dir / "powerbi"
        p.mkdir(parents=True, exist_ok=True)
        return p / filename

    def _pbi_accumulate(self, filename: str, new_rows: list, domain: str) -> None:
        """Append rows to a pbi CSV, replacing existing rows for this domain."""
        path = self._pbi(filename)
        new_df = pd.DataFrame(new_rows)
        if path.exists():
            existing = pd.read_csv(path)
            if "domain" in existing.columns:
                existing = existing[existing["domain"] != domain]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

    _DOMAIN_DISPLAY = {
        "fraud":   "Fraud Detection",
        "credit":  "Credit Scoring",
        "aml":     "AML Detection",
        "trading": "Algorithmic Trading",
    }

    def _write_pbi_files(self, state: Dict) -> None:
        domain = state["domain"]
        display = self._DOMAIN_DISPLAY.get(domain, domain)
        attack = state.get("attack_summary", {})
        mat = state.get("maturity_comparison", {})
        gov = state.get("governance_metrics", {})
        sim = state.get("simulation_engine")

        from config.settings import FINANCIAL_IMPACT
        _impact_cfg = state.get("impact_cfg") or FINANCIAL_IMPACT.get(domain, {})
        impact_mean_usd = float(_impact_cfg.get("mean", 0.0))

        p_success_pct = float(state.get("combined_p_success", attack.get("p_success_mean", 0.0))) * 100
        cvar_medium = float(mat.get("medium", {}).get("cvar_99", 0.0))
        cvar_weak = float(mat.get("weak", {}).get("cvar_99", 0.0))
        cvar_strong = float(mat.get("strong", {}).get("cvar_99", 0.0))
        reduction = ((cvar_weak - cvar_strong) / cvar_weak * 100) if cvar_weak > 0 else 0.0

        self._pbi_accumulate("pbi_domain_summary.csv", [{
            "domain":                           display,
            "p_success_pct":                    round(p_success_pct, 4),
            "cvar_99_weak_usd":                 round(cvar_weak, 2),
            "cvar_99_medium_usd":               round(cvar_medium, 2),
            "cvar_99_strong_usd":               round(cvar_strong, 2),
            "cvar_reduction_weak_to_strong_pct": round(reduction, 2),
            "impact_mean_usd":                  impact_mean_usd,
            "p_success_mean":                   float(state.get("combined_p_success", attack.get("p_success_mean", 0.0))),
            "p_success_max":                    float(attack.get("p_success_max", 0.0)),
        }], display)

        risk_rows = []
        for label, vals in mat.items():
            cvar = float(vals.get("cvar_99", 0.0))
            var99 = float(vals.get("var_99", 0.0))
            risk_rows.append({
                "domain":        display,
                "maturity_label": label,
                "iso_maturity":  vals.get("maturity_value", 0.0),
                "el_mean":       round(float(vals.get("el_mean", 0.0)), 4),
                "rri_mean":      round(float(vals.get("rri_mean", 0.0)), 4),
                "var_95":        round(float(vals.get("var_95", 0.0)), 4),
                "var_99":        round(var99, 4),
                "cvar_99":       round(cvar, 4),
                "tail_premium":  round(max(cvar - var99, 0.0), 4),
                "el_std":        round(float(vals.get("el_std", 0.0)), 4),
                "p_extreme":     round(float(vals.get("p_extreme", 0.0)), 6),
            })
        path = self._pbi("pbi_risk_metrics.csv")
        new_df = pd.DataFrame(risk_rows)
        if path.exists():
            existing = pd.read_csv(path)
            existing = existing[existing["domain"] != display]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

        # Evasion success_rate = flipped fraction; poisoning success_rate = max recall degradation
        attack_rows = []
        for family, key in [
            ("FGSM",              "fgsm_max_success_rate"),
            ("PGD",               "pgd_success_rate"),
            ("C&W",               "cw_success_rate"),
            ("Feature Perturb",   "feature_perturb_success_rate"),
            ("Precision Attack",  "precision_attack_success_rate"),
            ("Centroid Evasion",  "centroid_evasion_success_rate"),
        ]:
            rate = float(attack.get(key, 0.0))
            attack_rows.append({
                "domain": display, "family": family, "attack_class": "Evasion",
                "max_success_rate": round(rate, 6),
                "success_rate":     round(rate, 6),
            })

        from config.settings import EXPORTS_DIR
        sweep_path = EXPORTS_DIR / "pbi_poisoning_sweep.csv"
        if sweep_path.exists():
            sweep = pd.read_csv(sweep_path)
            sweep_dom = sweep[sweep["domain"] == domain]  # short name in sweep
            for p_family, p_label in [
                ("label_flip",     "Label Flip"),
                ("targeted_flip",  "Targeted Flip"),
                ("gain_guided",    "Gain-Guided"),
                ("clean_label",    "Clean Label"),
                ("backdoor",       "Backdoor"),
            ]:
                sub = sweep_dom[sweep_dom["attack_type"] == p_family]
                if not sub.empty:
                    max_deg = float(sub["recall_degradation"].clip(lower=0).max())
                    attack_rows.append({
                        "domain": display, "family": p_label, "attack_class": "Poisoning",
                        "max_success_rate": round(max_deg, 6),
                        "success_rate":     round(max_deg, 6),
                    })

        path = self._pbi("pbi_attack_profile.csv")
        new_df = pd.DataFrame(attack_rows)
        if path.exists():
            existing = pd.read_csv(path)
            existing = existing[existing["domain"] != display]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

        iso_rows = [
            {
                "domain": display,
                "maturity_label": label,
                "iso_maturity": vals.get("maturity_value", 0.0),
                "cvar_99":  round(float(vals.get("cvar_99", 0.0)), 4),
                "el_mean":  round(float(vals.get("el_mean", 0.0)), 4),
                "rri_mean": round(float(vals.get("rri_mean", 0.0)), 4),
            }
            for label, vals in mat.items()
        ]
        path = self._pbi("pbi_iso_sensitivity.csv")
        new_df = pd.DataFrame(iso_rows)
        if path.exists():
            existing = pd.read_csv(path)
            existing = existing[existing["domain"] != display]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

        if sim and sim.cascade:
            cas = sim.cascade.get_summary()
            self._pbi_accumulate("pbi_cascade_components.csv", [{
                "domain": display, **{k: round(float(v), 4) for k, v in cas.items()}
            }], display)

        if gov:
            gov_rows = [
                {"domain": display, "control_domain": k, "maturity_score": round(float(v), 4)}
                for k, v in gov.items()
                if isinstance(v, (int, float))
            ]
            if gov_rows:
                path = self._pbi("pbi_governance_scores.csv")
                new_df = pd.DataFrame(gov_rows)
                if path.exists():
                    existing = pd.read_csv(path)
                    existing = existing[existing["domain"] != display]
                    new_df = pd.concat([existing, new_df], ignore_index=True)
                new_df.to_csv(path, index=False)

        rm = state.get("risk_metrics", {})
        cvar99  = rm.get("cvar_99", 0);  var99 = rm.get("var_99", 0)
        var95   = rm.get("var_95", 0);   el   = rm.get("el_mean", 0)
        rri     = rm.get("rri_mean", 0)
        rri_strong = mat.get("strong", {}).get("rri_mean", 1)
        rri_weak   = mat.get("weak",   {}).get("rri_mean", 1)
        checks = [
            ("CVaR >= VaR",
             f">= {var99:,.2f}",   round(cvar99, 4),  cvar99 >= var99),
            ("VaR99 >= VaR95",
             f">= {var95:,.2f}",   round(var99, 4),   var99  >= var95),
            ("EL > 0 or p=0",
             "> 0",                round(el, 4),       el >= 0),
            ("RRI <= EL",
             f"<= {el:,.2f}",      round(rri, 4),     rri <= el + 1e-6),
            ("Maturity reduces RRI",
             f"< {rri_weak:,.4f}", round(rri_strong, 4), rri_strong <= rri_weak + 1e-6),
        ]

        # FPR check only for fraud/AML where extreme imbalance makes low FPR operationally critical
        baseline = state.get("baseline_metrics", {})
        fpr = baseline.get("false_positive_rate", None)
        if fpr is not None and domain in ("fraud", "aml"):
            fpr_pct = round(fpr * 100, 4)
            checks.append((
                "FPR <= 0.05%",
                "<= 0.05%",
                fpr_pct,
                fpr <= 0.0005,
            ))

        # CVaR stability check only when p_success ≥ 5% (sparse tail otherwise)
        p_success_used = float(state.get("combined_p_success", 0.0) or 0.0)
        stab = state.get("cvar_stability", {})
        dev = stab.get("max_deviation_pct")
        if dev is not None and p_success_used >= 0.05:
            checks.append((
                "CVaR99 stable ±5%",
                "<= 5.00%",
                round(float(dev), 4),
                float(dev) <= 5.0,
            ))
        cons_rows = [
            {"domain": display, "check": c, "expected": e, "actual": a, "passed": bool(p)}
            for c, e, a, p in checks
        ]
        path = self._pbi("pbi_consistency_validation.csv")
        new_df = pd.DataFrame(cons_rows)
        if path.exists():
            existing = pd.read_csv(path)
            existing = existing[existing["domain"] != display]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(path, index=False)

        src = self.output_dir.parent / "pbi_poisoning_sweep.csv"
        dst = self._pbi("pbi_poisoning_sweep.csv")
        if src.exists():
            import shutil
            shutil.copy2(src, dst)

    def _write(self, data: Dict, filename: str) -> Path:
        path = self.output_dir / filename
        pd.DataFrame([data]).to_csv(path, index=False)
        return path

    def _write_df(self, rows: list, filename: str) -> Path:
        path = self.output_dir / filename
        pd.DataFrame(rows).to_csv(path, index=False)
        return path
