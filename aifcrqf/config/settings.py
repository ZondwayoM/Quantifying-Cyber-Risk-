"""Central settings for the AIFCRQF project — single source of truth for all constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # aifcrqf/
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
LOGS_DIR = OUTPUTS_DIR / "logs"
EXPORTS_DIR = OUTPUTS_DIR / "exports"
VIZ_DIR = OUTPUTS_DIR / "visualizations"
for _d in [MODELS_DIR, LOGS_DIR, EXPORTS_DIR, VIZ_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED: int = 42

PROBLEM_KEYWORDS: dict[str, list[str]] = {
    "fraud": [
        "fraud", "fraudulent", "is_fraud", "class", "label",
        "transaction", "isFraud", "fraud_flag",
    ],
    "credit": [
        "credit", "default", "creditworthy", "risk", "loan",
        "repay", "good_bad", "creditrisk",
    ],
    "aml": [
        "aml", "laundering", "suspicious", "sar", "flagged",
        "alert", "money_laundering",
    ],
    "trading": [
        "return", "signal", "direction", "price", "trade",
        "position", "pnl", "profit",
    ],
}

MODEL_CONFIGS: dict[str, dict] = {
    "fraud": {
        "model_type": "xgboost",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.1,
        "scale_pos_weight": 578.26,   # 1:578 imbalance ratio
        "eval_metric": "auc",
        "random_state": RANDOM_SEED,
    },
    "credit": {
        "model_type": "lightgbm",
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "verbose": -1,
    },
    "aml": {
        "model_type": "xgboost",
        "n_estimators": 400,
        "max_depth": 7,
        "learning_rate": 0.05,
        "scale_pos_weight": 980,   # 5,073,168 / 5,177 ≈ 980 (actual 1:980 imbalance)
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "random_state": RANDOM_SEED,
        "eval_metric": "aucpr",    # PR-AUC more informative than AUC for severe imbalance
    },
    "trading": {
        "model_type": "lightgbm",
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    },
}

SPLIT_RATIOS: dict[str, float] = {
    "train": 0.60,
    "val": 0.20,
    "test": 0.20,
}

MONTE_CARLO_ITERATIONS: int = 50_000

# Domain-calibrated loss parameters (USD per misclassified event).
# operational_mean/std fed to CascadingImpactModel; overridden by data-derived lognormal fit.
FINANCIAL_IMPACT: dict[str, dict] = {
    "fraud": {
        "mean": 10_000, "std": 3_000,
        "operational_mean": 5_000, "operational_std": 1_500,
    },
    "credit": {
        "mean": 50_000, "std": 15_000,
        "operational_mean": 25_000, "operational_std": 8_000,
    },
    "aml": {
        "mean": 200_000, "std": 60_000,
        "operational_mean": 250_000, "operational_std": 75_000,
    },
    "trading": {
        "mean": 100_000, "std": 25_000,
        "operational_mean": 100_000, "operational_std": 30_000,
    },
}

# ISO 27001 maturity bands used throughout (weak=0.30, medium=0.60, strong=0.80)
ISO_MATURITY_LEVELS: dict[str, float] = {
    "weak": 0.30,
    "medium": 0.60,
    "strong": 0.80,
}

ISO_CONTROL_DOMAINS: list[str] = [
    "Access Control",
    "Asset Management",
    "Incident Response",
    "Cryptography",
    "Operations Security",
    "Communications Security",
    "Supplier Relationships",
]

ATTACK_CONFIGS: dict[str, dict] = {
    "fgsm": {
        "epsilon_range": [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "epsilon_step": 0.05,
        "targeted": False,
    },
    "pgd": {
        "epsilon": 0.10,
        "alpha": 0.01,
        "num_steps": 40,
        "targeted": False,
    },
    "carlini_wagner": {
        "kappa": 0,
        "learning_rate": 0.10,   # lr=0.10 needed to cross tree split thresholds
        "max_iterations": 100,
        "norm": "l2",
        "c_init": 1e-3,
    },
    "centroid_evasion": {
        "alpha": 0.05,    # step size as fraction of distance to negative centroid
        "max_steps": 20,
    },
    "poisoning": {
        "corruption_rates": [
            0.001, 0.003, 0.005, 0.010, 0.015,
            0.020, 0.025, 0.030, 0.040, 0.050, 0.075, 0.100,
        ],
        "attack_types": [
            "label_flip",
            "targeted_flip",
            "feature_perturb",
            "gain_guided",
            "clean_label",
            "backdoor",
        ],
    },
    "feature_perturbation": {
        "top_k_features": 5,    # perturb top-k Gain-ranked features
        "perturbation_scale": 0.20,
    },
}

VAR_CONFIDENCE: float = 0.95
CVAR_CONFIDENCE: float = 0.99

SUPPORTED_DOMAINS: list[str] = ["fraud", "credit", "aml", "trading"]
PROBLEM_CLASSIFICATION: str = "classification"
PROBLEM_REGRESSION: str = "regression"
IMBALANCE_THRESHOLD: float = 0.10   # minority class freq below this → imbalanced

DECISION_THRESHOLDS: dict[str, float] = {
    "fraud":   0.50,
    "credit":  0.50,
    "aml":     0.25,   # low threshold to maximise SAR recall on severely imbalanced data
    "trading": 0.50,
}
