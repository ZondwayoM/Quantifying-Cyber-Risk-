# AIFCRQF — AI-Driven Fintech Cyber Risk Quantification Framework

## Live Dashboard

[View the live risk dashboard](https://your-app-name.streamlit.app) — shows pipeline results across all four fintech domains including Monte Carlo loss distributions, adversarial attack profiles, ISO 27001 maturity comparisons, and cascading impact breakdowns.

---

AIFCRQF runs a three-layer pipeline: adversarial ML attacks to measure empirical attack success rates, Monte Carlo and Bayesian Network risk quantification to convert those rates into financial loss metrics (EL, VaR, CVaR, RRI), and ISO 27001 governance maturity mapping to produce residual risk outputs.

The attacks cover five evasion families (FGSM, PGD, C&W, centroid evasion, feature perturbation) and six poisoning types (label_flip, targeted_flip, feature_perturb, gain_guided, clean_label, backdoor) across 12 corruption rates.

---

## Running the Pipeline Locally

### 1. Clone and install

```bash
git clone https://github.com/ZondwayoM/Quantifying-Cyber-Risk-
cd Quantifying-Cyber-Risk-/aifcrqf
pip install -r requirements.txt
```

### 2. Download the datasets

Place datasets in `data/raw/`:

| Domain | File | Source |
|---|---|---|
| Fraud | `creditcard.csv` | [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Credit | `german_credit.csv` | [Kaggle German Credit Risk](https://www.kaggle.com/datasets/uciml/german-credit) |
| AML | `aml_ibm.csv` | [IBM AML Transactions (Kaggle)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) |
| Trading | `trading_features.csv` | Auto-generated — run `python data/fetch_trading_data.py` |

### 3. Run the pipeline

The dashboard launches automatically at http://localhost:8501 after every run.

```bash
# All domains
python main.py all weak
python main.py all medium
python main.py all strong

# Fraud Detection
python main.py fraud weak
python main.py fraud medium
python main.py fraud strong

# Credit Scoring
python main.py credit weak
python main.py credit medium
python main.py credit strong

# AML Detection
python main.py aml weak
python main.py aml medium
python main.py aml strong

# Algorithmic Trading
python main.py trading weak
python main.py trading medium
python main.py trading strong
```

Any domain and maturity can be combined freely:
`python main.py [domain] [maturity]`
where `[domain]` is one of `fraud | credit | aml | trading | all`
and `[maturity]` is one of `weak | medium | strong`
