"""Adversarial evaluation orchestrator — runs all attack families and aggregates P_success metrics."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, recall_score

from adversarial.attacks.carlini_wagner import CarliniWagnerAttack
from adversarial.attacks.centroid_evasion import CentroidEvasionAttack
from adversarial.attacks.feature_perturbation import FeaturePerturbationAttack
from adversarial.attacks.fgsm import FGSMAttack
from adversarial.attacks.pgd import PGDAttack
from adversarial.attacks.poisoning import PoisoningAttack
from config.settings import ATTACK_CONFIGS, EXPORTS_DIR
from core.bases import BaseModel

logger = logging.getLogger(__name__)

_MAX_ADV_SAMPLES = 2_000


def _kl_divergence(p_clean: np.ndarray, p_poisoned: np.ndarray, bins: int = 20) -> float:
    """D_KL(clean || poisoned) over histogram bins; detects calibration shift even when recall is stable."""
    range_ = (0.0, 1.0)
    p, _ = np.histogram(p_clean,    bins=bins, range=range_, density=True)
    q, _ = np.histogram(p_poisoned, bins=bins, range=range_, density=True)
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def _n_missed_per_1000(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """False negatives per 1,000 transactions — operationalises FNR for risk reporting."""
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    base_rate = y_true.mean()
    fnr = fn / max(fn + tp, 1)
    return round(fnr * base_rate * 1000, 4)


class AdversarialEvaluator:
    """Run all configured attacks; output feeds the Beta distribution for P_success in the MC engine."""

    def __init__(self, model: BaseModel) -> None:
        self.model = model
        self.results: Dict[str, List[Dict]] = {}

    def run_all(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, float]:
        """Run FGSM sweep, PGD, C&W, feature perturbation, centroid evasion; return success rates."""
        if len(X_test) > _MAX_ADV_SAMPLES:
            rng = np.random.default_rng(42)
            pos_idx = np.where(y_test == 1)[0]
            neg_idx = np.where(y_test == 0)[0]
            # At least 200 positives for statistically meaningful evasion rates on imbalanced domains
            n_pos = min(len(pos_idx), max(200, int(_MAX_ADV_SAMPLES * 0.10)))
            n_neg = min(len(neg_idx), _MAX_ADV_SAMPLES - n_pos)
            idx = np.concatenate([
                rng.choice(pos_idx, n_pos, replace=(len(pos_idx) < n_pos)),
                rng.choice(neg_idx, n_neg, replace=False),
            ])
            rng.shuffle(idx)
            X_test, y_test = X_test[idx], y_test[idx]
            logger.info(
                "Adversarial eval: stratified cap — %d pos + %d neg = %d total.",
                n_pos, n_neg, len(idx),
            )

        summary: Dict[str, float] = {}

        fgsm = FGSMAttack(self.model)
        fgsm_results = fgsm.sweep(X_test, y_test)
        self.results["fgsm"] = fgsm_results
        summary["fgsm_max_success_rate"] = max(r["success_rate"] for r in fgsm_results)

        pgd = PGDAttack(self.model)
        X_adv_pgd = pgd.generate(X_test, y_test)
        pgd_result = pgd.evaluate(X_test, X_adv_pgd, y_test)
        self.results["pgd"] = [pgd_result]
        summary["pgd_success_rate"] = pgd_result["success_rate"]

        cw = CarliniWagnerAttack(self.model)
        X_adv_cw = cw.generate(X_test, y_test)
        cw_result = cw.evaluate(X_test, X_adv_cw, y_test)
        self.results["cw"] = [cw_result]
        summary["cw_success_rate"] = cw_result["success_rate"]

        fp = FeaturePerturbationAttack(self.model)
        X_adv_fp = fp.generate(X_test, y_test)
        fp_result = fp.evaluate(X_test, X_adv_fp, y_test)
        self.results["feature_perturb"] = [fp_result]
        summary["feature_perturb_success_rate"] = fp_result["success_rate"]

        # Precision attack: 2× perturbation on high-confidence positives (>70th pct)
        try:
            proba = self.model.predict_proba(X_test)[:, 1]
            conf_threshold = float(np.percentile(proba, 70))
            high_conf_idx = np.where(proba >= conf_threshold)[0]
            if len(high_conf_idx) >= 10:
                X_hc = X_test[high_conf_idx]
                y_hc = y_test[high_conf_idx]
                pa = FeaturePerturbationAttack(
                    self.model,
                    scale=ATTACK_CONFIGS["feature_perturbation"]["perturbation_scale"] * 2.0,
                )
                X_adv_pa = pa.generate(X_hc, y_hc)
                pa_result = pa.evaluate(X_hc, X_adv_pa, y_hc)
            else:
                pa_result = {"success_rate": 0.0, "mean_perturbation": 0.0}
        except Exception:   # noqa: BLE001
            pa_result = {"success_rate": 0.0, "mean_perturbation": 0.0}
        self.results["precision_attack"] = [pa_result]
        summary["precision_attack_success_rate"] = pa_result["success_rate"]

        # Centroid evasion: black-box, gradient-free, moves positives toward negative centroid
        try:
            ce = CentroidEvasionAttack(self.model)
            X_adv_ce = ce.generate(X_test, y_test)
            ce_result = ce.evaluate(X_test, X_adv_ce, y_test)
        except Exception:   # noqa: BLE001
            ce_result = {"success_rate": 0.0, "mean_perturbation": 0.0}
        self.results["centroid_evasion"] = [ce_result]
        summary["centroid_evasion_success_rate"] = ce_result["success_rate"]

        all_rates = list(summary.values())
        summary["p_success_mean"] = float(np.mean(all_rates))
        summary["p_success_max"] = float(np.max(all_rates))

        logger.info("Adversarial evaluation complete: %s", summary)
        return summary

    def run_poisoning_with_retraining(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        domain: str = "unknown",
        output_dir: Path | None = None,
    ) -> Dict[str, Any]:
        """Retrain on poisoned data for every (attack_type, rate) pair; record recall, PR-AUC, KL, missed/1k."""
        baseline_preds  = self.model.predict(X_test)
        baseline_proba  = self.model.predict_proba(X_test)[:, 1]
        baseline_recall = float(recall_score(y_test, baseline_preds, zero_division=0))
        baseline_prauc  = float(average_precision_score(y_test, baseline_proba))

        attack_types = ATTACK_CONFIGS["poisoning"].get(
            "attack_types", ["label_flip", "feature_perturb", "gain_guided", "backdoor"]
        )

        results: Dict[str, Any] = {
            "baseline_recall": baseline_recall,
            "baseline_prauc": baseline_prauc,
            "rates": {},
        }

        sweep_rows: list[dict] = []

        for attack_type in attack_types:
            poisoner = PoisoningAttack(self.model, attack_type=attack_type)
            for rate, X_p, y_p in poisoner.run_sweep(X_train, y_train):
                fresh_model = type(self.model)(self.model.config)
                fresh_model.train(X_p, y_p)

                poisoned_preds  = fresh_model.predict(X_test)
                poisoned_proba  = fresh_model.predict_proba(X_test)[:, 1]
                poisoned_recall = float(recall_score(y_test, poisoned_preds, zero_division=0))
                poisoned_prauc  = float(average_precision_score(y_test, poisoned_proba))
                kl_div          = _kl_divergence(baseline_proba, poisoned_proba)
                n_missed        = _n_missed_per_1000(y_test, poisoned_preds)
                degradation     = baseline_recall - poisoned_recall

                key = f"{attack_type}_{rate:.4f}"
                results["rates"][key] = {
                    "attack_type":        attack_type,
                    "poisoned_recall":    poisoned_recall,
                    "poisoned_prauc":     poisoned_prauc,
                    "kl_divergence":      kl_div,
                    "n_missed_per_1000":  n_missed,
                    "recall_degradation": degradation,
                    "degradation_pct":    degradation / max(baseline_recall, 1e-9) * 100,
                }

                sweep_rows.append({
                    "domain":            domain,
                    "attack_type":       attack_type,
                    "corruption_rate":   rate,
                    "baseline_recall":   baseline_recall,
                    "poisoned_recall":   poisoned_recall,
                    "baseline_prauc":    baseline_prauc,
                    "poisoned_prauc":    poisoned_prauc,
                    "kl_divergence":     kl_div,
                    "n_missed_per_1000": n_missed,
                    "recall_degradation":degradation,
                })

                logger.info(
                    "%s @ %.1f%% | recall %.4f→%.4f | PR-AUC %.4f→%.4f | KL %.4f | missed/1k %.2f",
                    attack_type, rate * 100,
                    baseline_recall, poisoned_recall,
                    baseline_prauc, poisoned_prauc,
                    kl_div, n_missed,
                )

        self.results["poisoning"] = results
        self._export_poisoning_sweep(sweep_rows, output_dir)
        return results

    def _export_poisoning_sweep(
        self,
        rows: list[dict],
        output_dir: Path | None,
    ) -> None:
        """Write domain poisoning sweep rows to pbi_poisoning_sweep.csv (domain-replaces existing rows)."""
        if not rows:
            return
        out = (output_dir or EXPORTS_DIR) / "pbi_poisoning_sweep.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        new_df = pd.DataFrame(rows)
        if out.exists():
            existing = pd.read_csv(out)
            domain = rows[0].get("domain", "")
            existing = existing[existing["domain"] != domain]
            new_df = pd.concat([existing, new_df], ignore_index=True)
        new_df.to_csv(out, index=False)
        logger.info("Poisoning sweep exported: %s", out)
