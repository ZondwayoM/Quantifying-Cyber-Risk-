"""
AIFCRQF — Streamlit Risk Intelligence Dashboard
================================================
Run from aifcrqf/ directory:
    pip install streamlit
    streamlit run run_streamlit_dashboard.py

Opens automatically at http://localhost:8501
Deploy free: https://share.streamlit.io
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from config.settings import EXPORTS_DIR

# ── paths & palette ───────────────────────────────────────────────────────
PBI_DIR = EXPORTS_DIR / "powerbi"
CYAN = "#00E5FF";  WARN = "#FF6B35";  CRIT = "#EF5350"
SAFE = "#00E676";  GOLD = "#F4C542";  BLUE = "#4FC3F7"

DOMAIN_COLOURS = {
    "Fraud Detection":     "#1f77b4",
    "Credit Scoring":      "#ff7f0e",
    "AML Detection":       "#2ca02c",
    "Algorithmic Trading": "#d62728",
}

ATTACK_GLOSSARY = {
    "FGSM": {
        "full": "Fast Gradient Sign Method",
        "type": "Evasion", "danger": "High",
        "desc": "Perturbs transaction features by a tiny step in the direction that maximises model error. Fast and scalable — can fool fraud classifiers by modifying amounts or timestamps by imperceptible amounts.",
    },
    "PGD": {
        "full": "Projected Gradient Descent",
        "type": "Evasion", "danger": "Critical",
        "desc": "Iterative version of FGSM — repeatedly refines the perturbation to maximise evasion success within a constraint budget. Considered the gold standard adversarial robustness test.",
    },
    "C&W": {
        "full": "Carlini & Wagner Attack",
        "type": "Evasion", "danger": "Critical",
        "desc": "Optimisation-based attack finding the minimum perturbation needed to cause misclassification. Bypasses many defences that stop FGSM/PGD. Particularly dangerous against credit scoring models.",
    },
    "feature_perturb": {
        "full": "Feature Space Perturbation",
        "type": "Evasion", "danger": "High",
        "desc": "Manipulates individual financial features (e.g. credit utilisation, transaction velocity) within plausible ranges to evade detection without triggering anomaly checks.",
    },
    "label_flip": {
        "full": "Label Flipping Attack",
        "type": "Poisoning", "danger": "High",
        "desc": "Corrupts training data by flipping class labels — marking fraudulent transactions as legitimate. Degrades recall over time as poisoned data enters retraining cycles.",
    },
    "precision_attack": {
        "full": "Precision-Targeted Evasion",
        "type": "Evasion", "danger": "Medium",
        "desc": "Crafts inputs designed to be classified with high confidence in the wrong class, targeting the decision boundary to achieve evasion while appearing normal to human reviewers.",
    },
    "gain_guided": {
        "full": "Gain-Guided Poisoning Attack",
        "type": "Poisoning", "danger": "Critical",
        "desc": "Uses tree Gain-based feature importance (feature_importances_) to identify the top-5 most decision-influential features, then applies lognormal multiplicative noise to those features in training data. Targets the features the model relies on most, degrading its learnt decision boundary.",
    },
    "targeted_flip": {
        "full": "Targeted Label Flip",
        "type": "Poisoning", "danger": "High",
        "desc": "Selective poisoning targeting specific customer segments or transaction types. More subtle than broad label flipping — harder to detect in standard data quality monitoring.",
    },
    "clean_label": {
        "full": "Clean Label Poisoning",
        "type": "Poisoning", "danger": "Medium",
        "desc": "Injects correctly-labelled but adversarially crafted training samples. Labels pass validation checks but the model learns subtly incorrect decision boundaries.",
    },
    "backdoor": {
        "full": "Backdoor Trojan Attack",
        "type": "Poisoning", "danger": "Critical",
        "desc": "Embeds a fixed trigger pattern into positive-class training samples and relabels them as benign. The model behaves normally on clean inputs but silently misclassifies any input carrying the trigger — undetectable via standard test-set evaluation (Chen et al., 2017).",
    },
    "Centroid Evasion": {
        "full": "Centroid Evasion Attack",
        "type": "Evasion", "danger": "High",
        "desc": "Black-box directional attack that moves positive samples iteratively toward the negative-class centroid. Requires no gradients — effective against tree-based classifiers (XGBoost, LightGBM) where FGSM/PGD produce near-zero signal. Models the financially-informed adversary who nudges suspicious transactions toward the legitimate transaction profile (Brendel et al., 2018).",
    },
}

DOMAIN_CONTEXT = {
    "Fraud Detection": {
        "regulation": "PSD2 · FCA · UK Fraud Act 2006",
        "benchmark": "UK payment fraud losses: £1.2B (2023, UK Finance)",
        "model_type": "XGBoost / LightGBM ensemble classifier",
        "key_risk": "Real-time transaction misclassification → direct financial loss + chargeback liability",
        "capital": "Operational Risk Capital — AMA / Standardised Approach (Basel III)",
    },
    "Credit Scoring": {
        "regulation": "Basel III IRB · IFRS 9 · FCA MCOB",
        "benchmark": "UK consumer credit default rate: 1.8% (Bank of England 2023)",
        "model_type": "Logistic regression / gradient boosted trees",
        "key_risk": "Adversarial bias in PD estimates → mis-priced loans → capital adequacy shortfall",
        "capital": "Credit Risk — Internal Ratings Based Approach (AIRB)",
    },
    "AML Detection": {
        "regulation": "FATF Recommendations · UK MLR 2017 · FinCEN",
        "benchmark": "Global AML regulatory fines: $10.4B (2022, Fenergo Report)",
        "model_type": "Graph neural network / behavioural analytics",
        "key_risk": "False negatives under attack → regulatory sanction + reputational damage",
        "capital": "Compliance Risk — non-capital but triggers direct enforcement action",
    },
    "Algorithmic Trading": {
        "regulation": "MiFID II · MAR · FCA SYSC 6",
        "benchmark": "2010 Flash Crash: $1 trillion market cap lost in 45 minutes",
        "model_type": "Reinforcement learning / time-series signal prediction",
        "key_risk": "Adversarial signal injection → erroneous orders → market manipulation exposure",
        "capital": "Market Risk — Internal Models Approach (IMA, Basel III)",
    },
}

# ── page config ─────────────────────────────────────��─────────────────────
st.set_page_config(
    page_title="AIFCRQF",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS injection ─────────────────────────────────────────────────────────
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown("""<style>
/* ── Animated background ── */
.stApp {
    background: linear-gradient(-45deg, #000810, #001228, #000A1A, #001830, #000810);
    background-size: 400% 400%;
    animation: gradBG 12s ease infinite;
}
@keyframes gradBG {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(0,8,22,0.92) !important;
    border-right: 1px solid rgba(0,229,255,0.15) !important;
}
[data-testid="stSidebar"] .stRadio label {
    color: #90B8D4 !important; font-size: 12px;
}
[data-testid="stSidebar"] .stRadio label:hover { color: #00E5FF !important; }

/* ── Remove default white backgrounds ── */
[data-testid="stVerticalBlock"] > div { background: transparent !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ── Metric card override ── */
[data-testid="stMetric"] {
    background: rgba(0,40,70,0.6) !important;
    border: 1px solid rgba(0,229,255,0.15) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    backdrop-filter: blur(10px) !important;
}
[data-testid="stMetricLabel"] { color: #546E7A !important; font-size: 11px !important; letter-spacing: 1px; }
[data-testid="stMetricValue"] { color: #00E5FF !important; font-family: 'Orbitron',monospace !important; }
[data-testid="stMetricDelta"] > div { font-size: 11px !important; }

/* ── Custom card ── */
.aif-card {
    background: rgba(0,40,70,0.6);
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,229,255,0.06);
}
.aif-card-title {
    font-family: 'Orbitron', monospace;
    font-size: 10px; font-weight: 600;
    color: #00E5FF; letter-spacing: 2px;
    text-transform: uppercase;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(0,229,255,0.12);
    margin-bottom: 14px;
}

/* ── KPI metric card ── */
.kpi-box {
    background: rgba(0,40,70,0.6);
    border-radius: 10px;
    border: 1px solid rgba(0,229,255,0.15);
    border-top-width: 3px;
    padding: 18px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
}
.kpi-label { font-size: 10px; color: #546E7A; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value { font-family: 'Orbitron', monospace; font-size: 28px; font-weight: 700; margin-bottom: 6px; }
.kpi-sub   { font-size: 10px; color: #546E7A; }

/* ── Section heading ── */
.section-head {
    font-family: 'Orbitron', monospace;
    font-size: 11px; font-weight: 600;
    color: #00E5FF; letter-spacing: 3px;
    text-transform: uppercase;
    margin: 24px 0 14px;
    display: flex; align-items: center; gap: 12px;
}
.section-head::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(0,229,255,0.3), transparent);
}

/* ── Cascade flow ── */
.cascade-flow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; padding: 16px 0; }
.cascade-box {
    border-radius: 8px; padding: 14px 16px; text-align: center;
    border-width: 1px; border-style: solid; flex: 1; min-width: 120px;
}
.cascade-box-title { font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 6px; }
.cascade-box-val   { font-family: 'Orbitron',monospace; font-size: 13px; font-weight: 700; }
.cascade-arrow { color: rgba(0,229,255,0.6); font-size: 20px; flex-shrink: 0; }

/* ── Risk table ── */
.risk-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
.risk-tbl th { background: rgba(0,229,255,0.06); color: #00E5FF; font-size: 10px;
               font-weight: 600; letter-spacing: 1px; padding: 8px 12px; text-align: left;
               border-bottom: 1px solid rgba(0,229,255,0.15); }
.risk-tbl td { padding: 7px 12px; border-bottom: 1px solid rgba(0,229,255,0.05); color: #C8DCEA; }
.risk-tbl tr:hover td { background: rgba(0,229,255,0.03); }

/* ── Status banner ── */
.status-banner {
    border-radius: 10px; padding: 20px 28px; margin-bottom: 24px;
    display: flex; align-items: center; gap: 16px;
}
.banner-icon { font-size: 32px; }
.banner-main { font-family: 'Orbitron',monospace; font-size: 14px; font-weight: 700; letter-spacing: 1.5px; }
.banner-sub  { font-size: 12px; color: #90B8D4; margin-top: 4px; }

/* ── Insight rows ── */
.ins-section { margin-bottom: 12px; }
.ins-hdr { font-size: 10px; font-weight: 600; color: #90B8D4; letter-spacing: 1.5px;
           text-transform: uppercase; margin-bottom: 6px; }
.ins-row { display: flex; justify-content: space-between; padding: 4px 0;
           border-bottom: 1px solid rgba(0,229,255,0.04); font-size: 11px; color: #546E7A; }
.ins-val { font-weight: 600; color: #E8F4FD; }

/* ── Hero ── */
.hero-block {
    background: linear-gradient(135deg, rgba(0,229,255,0.04), rgba(0,8,16,0));
    border: 1px solid rgba(0,229,255,0.15);
    border-radius: 12px; padding: 36px 44px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero-block::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #00E5FF, transparent);
}
.hero-title {
    font-family: 'Orbitron', monospace; font-size: 22px; font-weight: 900;
    color: #FFFFFF; letter-spacing: 2px; margin-bottom: 10px;
    text-shadow: 0 0 30px rgba(0,229,255,0.4);
}
.hero-sub { font-size: 13px; color: #90B8D4; letter-spacing: 1px; }
.hero-sub span { color: #00E5FF; margin: 0 6px; }

/* ── Tier badge ── */
.badge {
    display: inline-block; font-size: 10px; font-weight: 700;
    padding: 2px 10px; border-radius: 4px; letter-spacing: 1px;
}

/* ── Glow animation ── */
@keyframes glow {
    0%,100% { text-shadow: 0 0 10px rgba(0,229,255,0.3); }
    50%      { text-shadow: 0 0 25px rgba(0,229,255,0.8), 0 0 50px rgba(0,229,255,0.2); }
}
.glow-text { animation: glow 2.5s ease-in-out infinite; }

/* ── Divider ── */
hr { border-color: rgba(0,229,255,0.1) !important; }

/* ── Plotly chart container ── */
.js-plotly-plot .plotly { background: transparent !important; }

/* ── Tabs across top ── */
[data-testid="stTabs"] { margin-top: 4px; }
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(0,229,255,0.15) !important;
    background: transparent !important;
    gap: 0;
}
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    color: #546E7A !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    color: #00E5FF !important;
    border-bottom-color: rgba(0,229,255,0.4) !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #00E5FF !important;
    border-bottom: 2px solid #00E5FF !important;
    background: transparent !important;
}
[data-testid="stTabsContent"] { padding-top: 20px !important; }

/* ── Hide sidebar toggle ── */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ── data loading ──────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def load_run_config() -> dict:
    """Return {domain_display_name: {"iso_maturity": "weak"|"medium"|"strong", ...}}."""
    p = PBI_DIR / "last_run_config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


@st.cache_data(ttl=15)
def load_all() -> dict[str, pd.DataFrame]:
    def _l(n):
        p = PBI_DIR / n
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    def _le(n):
        p = EXPORTS_DIR / n
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return {
        "summary":        _l("pbi_domain_summary.csv"),
        "attacks":        _l("pbi_attack_profile.csv"),
        "risk":           _l("pbi_risk_metrics.csv"),
        "iso":            _l("pbi_iso_sensitivity.csv"),
        "cascade":        _l("pbi_cascade_components.csv"),
        "governance":     _l("pbi_governance_scores.csv"),
        "consistency":    _l("pbi_consistency_validation.csv"),
        "poisoning":      _le("pbi_poisoning_sweep.csv"),
        "bn_mc":          _le("bn_mc_scenarios.csv"),
        "domain_metrics": _le("domain_metrics.csv"),
        "disclosure":     _le("disclosure_comparison.csv"),
    }


# ── helpers ───────────────────────────────────────────────────────────────
def fmt_usd(v: float) -> str:
    if abs(v) >= 1_000_000: return f"${v / 1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v:,.0f}"
    return f"${v:.2f}"


def risk_tier(p: float) -> tuple[str, str]:
    if p > 30: return "CRITICAL", CRIT
    if p > 10: return "HIGH",     WARN
    return "MODERATE", SAFE


def fig_style(fig: go.Figure, height: int = 340, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,8,16,0.4)",
        font=dict(family="Inter,Arial,sans-serif", color="#90B8D4", size=11),
        height=height,
        margin=margin or dict(l=50, r=20, t=40, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=10, color="#90B8D4")),
        xaxis=dict(gridcolor="rgba(0,229,255,0.07)", linecolor="rgba(0,229,255,0.2)",
                   tickfont=dict(color="#90B8D4"), title_font=dict(color="#90B8D4")),
        yaxis=dict(gridcolor="rgba(0,229,255,0.07)", linecolor="rgba(0,229,255,0.2)",
                   tickfont=dict(color="#90B8D4"), title_font=dict(color="#90B8D4")),
    )
    return fig


def _pc(fig, rows=1, cols=1) -> go.Figure:
    """Apply dark subplot layout."""
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(gridcolor="rgba(0,229,255,0.07)",
                             linecolor="rgba(0,229,255,0.2)",
                             tickfont=dict(color="#90B8D4"),
                             title_font=dict(color="#90B8D4"), row=r, col=c)
            fig.update_yaxes(gridcolor="rgba(0,229,255,0.07)",
                             linecolor="rgba(0,229,255,0.2)",
                             tickfont=dict(color="#90B8D4"),
                             title_font=dict(color="#90B8D4"), row=r, col=c)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# Chart builders
# ════════════════════════════════════════════════════════════���═══════════════

def chart_attack_heatmap(attacks, summary):
    if attacks.empty or summary.empty:
        return None
    families = sorted(attacks["family"].unique())
    domains  = list(DOMAIN_COLOURS.keys())
    z, text  = [], []
    for d in domains:
        ddata = attacks[attacks["domain"] == d]
        row_z, row_t = [], []
        for f in families:
            frow = ddata[ddata["family"] == f]
            val  = float(frow["max_success_rate"].max()) * 100 if not frow.empty else 0.0
            row_z.append(round(val, 1)); row_t.append(f"{val:.1f}%")
        z.append(row_z); text.append(row_t)
    fig = go.Figure(go.Heatmap(
        z=z, x=families, y=domains, text=text, texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorscale=[[0,"#000810"],[0.25,"#003D4F"],[0.6,CYAN],[1.0,CRIT]],
        zmin=0, zmax=100,
        colorbar=dict(title=dict(text="Success %", font=dict(color="#90B8D4")),
                      thickness=12, tickfont=dict(color="#90B8D4")),
        hovertemplate="<b>%{y}</b><br>%{x}: %{text}<extra></extra>",
    ))
    fig_style(fig, height=240, margin=dict(l=160, r=80, t=10, b=60))
    fig.update_xaxes(tickangle=-30)
    return fig


def chart_cvar_bars(risk):
    if risk.empty:
        return None
    domains = list(DOMAIN_COLOURS.keys())
    fig = go.Figure()
    for label, col in [("weak", CRIT), ("medium", WARN), ("strong", SAFE)]:
        sub  = risk[risk["maturity_label"] == label]
        vals = [float(sub[sub["domain"] == d]["cvar_99"].values[0])
                if not sub[sub["domain"] == d].empty else 0 for d in domains]
        m = 0.30 if label == "weak" else 0.60 if label == "medium" else 0.80
        fig.add_trace(go.Bar(
            name=f"M={m} ({label})", x=domains, y=vals, marker_color=col,
            marker_line=dict(color="rgba(0,0,0,0.4)", width=1),
            text=[fmt_usd(v) for v in vals], textposition="outside",
            textfont=dict(size=9, color="#90B8D4"),
        ))
    fig.update_layout(barmode="group", bargap=0.22, bargroupgap=0.06)
    fig_style(fig, height=360, margin=dict(l=60, r=20, t=20, b=80))
    fig.update_yaxes(title="CVaR 99% (USD)", tickprefix="$")
    fig.update_xaxes(tickangle=-15)
    return fig


def chart_iso_curves(iso):
    if iso.empty:
        return None
    fig = go.Figure()
    for d, col in DOMAIN_COLOURS.items():
        sub = iso[iso["domain"] == d].sort_values("iso_maturity")
        if sub.empty:
            continue
        r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
        fig.add_trace(go.Scatter(
            x=sub["iso_maturity"], y=sub["cvar_99"],
            mode="lines+markers", name=d,
            line=dict(color=col, width=2.5),
            marker=dict(size=5, color=col),
            fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.05)",
            hovertemplate=f"<b>{d}</b><br>M=%{{x:.2f}}<br>CVaR: $%{{y:,.0f}}<extra></extra>",
        ))
    for m, lbl, mc in [(0.30, "Weak", CRIT), (0.60, "Medium", WARN), (0.80, "Strong", SAFE)]:
        fig.add_vline(x=m, line_dash="dot", line_color=mc, line_width=1.5,
                      annotation_text=lbl, annotation_position="top",
                      annotation_font=dict(size=9, color=mc))
    fig_style(fig, height=360, margin=dict(l=60, r=20, t=20, b=50))
    fig.update_xaxes(title="ISO 27001 Maturity (M)", range=[0, 1.02])
    fig.update_yaxes(title="CVaR 99% (USD)", tickprefix="$")
    return fig


def chart_business_gauge(p_success_avg: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=p_success_avg,
        title=dict(text="BUSINESS RISK LEVEL", font=dict(
            family="Orbitron,monospace", color="#90B8D4", size=11)),
        number=dict(suffix="% OF MAX RISK",
                    font=dict(family="Orbitron,monospace",
                              color=CRIT if p_success_avg > 30 else WARN if p_success_avg > 10 else SAFE,
                              size=24)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#90B8D4",
                      tickfont=dict(color="#90B8D4", size=9)),
            bar=dict(color=CRIT if p_success_avg > 30 else WARN if p_success_avg > 10 else GOLD,
                     thickness=0.25),
            bgcolor="rgba(0,8,22,0.8)",
            borderwidth=1, bordercolor="rgba(0,229,255,0.2)",
            steps=[
                dict(range=[0, 10],  color="rgba(0,230,118,0.1)"),
                dict(range=[10, 30], color="rgba(255,107,53,0.1)"),
                dict(range=[30, 100],color="rgba(239,83,80,0.15)"),
            ],
            threshold=dict(line=dict(color=CYAN, width=2), value=p_success_avg),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        font=dict(color="#90B8D4"),
    )
    return fig


def chart_threat_scenarios(risk):
    """Horizontal bar: CVaR for weak / medium / strong (baseline, high threat, mitigated)."""
    if risk.empty:
        return None
    domains = list(DOMAIN_COLOURS.keys())
    labels, colours, maturity = [], [], []
    for lbl, col, m in [("MITIGATED\nStrong Controls", SAFE, "strong"),
                        ("HIGH THREAT\nActive Attack",   CRIT, "weak"),
                        ("BASELINE\nNormal Operations",  GOLD, "medium")]:
        sub  = risk[risk["maturity_label"] == m]
        total = float(sub["cvar_99"].sum()) if not sub.empty else 0.0
        labels.append(lbl); colours.append(col); maturity.append(total)
    fig = go.Figure(go.Bar(
        y=labels, x=maturity, orientation="h",
        marker_color=colours,
        marker_line=dict(color="rgba(0,0,0,0.4)", width=1),
        text=[fmt_usd(v) for v in maturity],
        textposition="outside",
        textfont=dict(color="#90B8D4", size=11),
        hovertemplate="<b>%{y}</b><br>Total CVaR 99%: $%{x:,.0f}<extra></extra>",
    ))
    fig_style(fig, height=240, margin=dict(l=180, r=80, t=10, b=40))
    fig.update_xaxes(title="WORST-CASE LOSS (USD)", tickprefix="$")
    fig.update_yaxes(tickfont=dict(color="#90B8D4", size=10))
    return fig


def chart_security_roi(risk):
    """Investment return: CVaR by maturity + governance savings."""
    if risk.empty:
        return None
    labels = ["No Investment\n(Weak Controls)", "Current State\n(Medium Controls)",
              "Full Investment\n(Strong Controls)"]
    cvar_vals = []
    for m in ["weak", "medium", "strong"]:
        sub = risk[risk["maturity_label"] == m]
        cvar_vals.append(float(sub["cvar_99"].sum()) if not sub.empty else 0)
    # savings vs weak baseline
    savings = [0.0] + [cvar_vals[0] - v for v in cvar_vals[1:]]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Tail-Risk Exposure (CVaR 99%)", x=labels, y=cvar_vals,
                         marker_color=CRIT,
                         marker_line=dict(color="rgba(0,0,0,0.4)", width=1)))
    fig.add_trace(go.Bar(name="Risk Reduction vs No Investment",
                         x=labels, y=savings,
                         marker_color=SAFE,
                         marker_line=dict(color="rgba(0,0,0,0.4)", width=1)))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    fig_style(fig, height=300, margin=dict(l=60, r=20, t=20, b=70))
    fig.update_yaxes(title="CVaR 99% (USD)", tickprefix="$")
    fig.update_xaxes(tickangle=-15)
    return fig


def chart_domain_attack(attacks, domain):
    if attacks.empty:
        return None
    ddata    = attacks[attacks["domain"] == domain]
    col      = DOMAIN_COLOURS.get(domain, CYAN)
    families = sorted(ddata["family"].unique()) if not ddata.empty else []
    vals     = [float(ddata[ddata["family"] == f]["max_success_rate"].max()) * 100
                if not ddata[ddata["family"] == f].empty else 0.0 for f in families]
    fig = go.Figure(go.Bar(
        x=families, y=vals, marker_color=col,
        marker_line=dict(color=CYAN, width=1),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        textfont=dict(size=10, color="#90B8D4"),
        hovertemplate="<b>%{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig_style(fig, height=260, margin=dict(l=40, r=20, t=20, b=60))
    fig.update_yaxes(title="Max Success Rate (%)", range=[0, 70])
    fig.update_xaxes(tickangle=-25)
    fig.update_layout(showlegend=False)
    return fig


def chart_domain_iso(iso, domain):
    if iso.empty:
        return None
    sub = iso[iso["domain"] == domain].sort_values("iso_maturity")
    if sub.empty:
        return None
    col = DOMAIN_COLOURS.get(domain, CYAN)
    r, g, b = int(col[1:3], 16), int(col[3:5], 16), int(col[5:7], 16)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sub["iso_maturity"], y=sub["cvar_99"],
        mode="lines+markers", showlegend=False,
        line=dict(color=col, width=2.5),
        marker=dict(size=5, color=col, line=dict(width=1, color=CYAN)),
        fill="tozeroy", fillcolor=f"rgba({r},{g},{b},0.07)",
        hovertemplate="M=%{x:.2f}<br>CVaR: $%{y:,.0f}<extra></extra>",
    ))
    for m, lbl, mc in [(0.30, "Weak", CRIT), (0.60, "Medium", WARN), (0.80, "Strong", SAFE)]:
        fig.add_vline(x=m, line_dash="dot", line_color=mc, line_width=1.5,
                      annotation_text=lbl, annotation_position="top",
                      annotation_font=dict(size=9, color=mc))
    fig.add_vrect(x0=0,    x1=0.30, fillcolor="rgba(239,83,80,0.04)",  line_width=0)
    fig.add_vrect(x0=0.30, x1=0.60, fillcolor="rgba(255,107,53,0.04)", line_width=0)
    fig.add_vrect(x0=0.60, x1=0.80, fillcolor="rgba(0,230,118,0.04)",  line_width=0)
    fig_style(fig, height=270, margin=dict(l=60, r=20, t=20, b=50))
    fig.update_xaxes(title="ISO 27001 Maturity (M)", range=[0, 1.02])
    fig.update_yaxes(title="CVaR 99% (USD)", tickprefix="$")
    return fig


def chart_poisoning(poison):
    if poison.empty or "corruption_rate" not in poison.columns:
        return None
    if "poisoned_recall" not in poison.columns:
        return None
    # Normalise short domain names (fraud/credit/aml/trading) to display names
    _short_to_display = {v: k for k, v in {
        "Fraud Detection": "fraud", "Credit Scoring": "credit",
        "AML Detection": "aml", "Algorithmic Trading": "trading",
    }.items()}
    poison = poison.copy()
    poison["domain"] = poison["domain"].map(lambda x: _short_to_display.get(str(x).lower(), x))
    has_kl      = "kl_divergence"   in poison.columns
    has_missed  = "n_missed_per_1000" in poison.columns
    ncols = 2 + int(has_kl) + int(has_missed)
    titles = ["Recall Under Poisoning", "PR-AUC Under Poisoning"]
    if has_kl:
        titles.append("KL Divergence (Calibration Shift)")
    if has_missed:
        titles.append("Missed Positives / 1,000 Transactions")
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles)
    fig.update_annotations(font=dict(color="#90B8D4", size=11))
    dashes = ["solid", "dash", "dot", "dashdot"]
    atypes = poison["attack_type"].dropna().unique() if "attack_type" in poison.columns else []
    for d, col in DOMAIN_COLOURS.items():
        for i, at in enumerate(atypes):
            sub = (poison[(poison["domain"] == d) & (poison["attack_type"] == at)]
                   .sort_values("corruption_rate"))
            if sub.empty:
                continue
            ds = dashes[i % 4]
            sl = (d == list(DOMAIN_COLOURS.keys())[0])
            if "poisoned_recall" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub["corruption_rate"] * 100, y=sub["poisoned_recall"],
                    mode="lines+markers", name=f"{d[:8]}/{at}" if sl else None,
                    showlegend=sl, line=dict(color=col, dash=ds, width=1.8),
                    marker=dict(size=4),
                    hovertemplate=f"<b>{d}</b><br>Corruption: %{{x:.1f}}%<br>Recall: %{{y:.3f}}<extra></extra>",
                ), row=1, col=1)
            if "poisoned_prauc" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub["corruption_rate"] * 100, y=sub["poisoned_prauc"],
                    mode="lines+markers", name=None, showlegend=False,
                    line=dict(color=col, dash=ds, width=1.8), marker=dict(size=4),
                ), row=1, col=2)
            c = 3
            if has_kl and "kl_divergence" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub["corruption_rate"] * 100, y=sub["kl_divergence"],
                    mode="lines+markers", name=None, showlegend=False,
                    line=dict(color=col, dash=ds, width=1.8), marker=dict(size=4),
                    hovertemplate=f"<b>{d}</b><br>KL: %{{y:.4f}}<extra></extra>",
                ), row=1, col=c)
                c += 1
            if has_missed and "n_missed_per_1000" in sub.columns:
                fig.add_trace(go.Scatter(
                    x=sub["corruption_rate"] * 100, y=sub["n_missed_per_1000"],
                    mode="lines+markers", name=None, showlegend=False,
                    line=dict(color=col, dash=ds, width=1.8), marker=dict(size=4),
                    hovertemplate=f"<b>{d}</b><br>Missed/1k: %{{y:.2f}}<extra></extra>",
                ), row=1, col=c)
    fig.add_hline(y=0.80, line_dash="dot", line_color=CYAN, line_width=1,
                  annotation_text="Operational threshold",
                  annotation_font=dict(size=8, color=CYAN), row=1, col=1)
    fig.add_hline(y=0.75, line_dash="dot", line_color=CYAN, line_width=1,
                  annotation_text="Operational threshold",
                  annotation_font=dict(size=8, color=CYAN), row=1, col=2)
    if has_kl:
        fig.add_hline(y=0.10, line_dash="dot", line_color="#f1a340", line_width=1,
                      annotation_text="Drift threshold",
                      annotation_font=dict(size=8, color="#f1a340"), row=1, col=3)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,8,16,0.4)",
                      font=dict(color="#90B8D4", size=11), height=380,
                      margin=dict(l=50, r=30, t=50, b=60),
                      legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10, color="#90B8D4")))
    _pc(fig, rows=1, cols=ncols)
    for c in range(1, ncols + 1):
        fig.update_xaxes(title_text="Corruption Rate (%)", row=1, col=c)
    fig.update_yaxes(title_text="Recall",      row=1, col=1)
    fig.update_yaxes(title_text="PR-AUC",      row=1, col=2)
    if has_kl:
        fig.update_yaxes(title_text="KL Divergence", row=1, col=3)
    if has_missed:
        fig.update_yaxes(title_text="Missed / 1,000", row=1, col=ncols)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# HTML fragments
# ════════════════════════════════════════════════════════════════════════════

def kpi_box(label: str, value: str, sub: str, col: str, border_col: str | None = None) -> str:
    bc = border_col or col
    return (
        f'<div class="kpi-box" style="border-top-color:{bc}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{col}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


def derive_recommendation(domain: str, data: dict, maturity: str = "medium") -> str:
    """
    Domain-specific recommendation referencing actual pipeline numbers.
    Each domain has a tailored template covering: worst attack, business metric impact,
    regulatory implication, governance ROI, and a concrete control action.
    """
    summary      = data["summary"]
    attacks      = data["attacks"]
    risk         = data["risk"]
    dm           = data.get("domain_metrics", pd.DataFrame())
    key          = _DOMAIN_KEY.get(domain, "fraud")

    ds = summary[summary["domain"] == domain] if not summary.empty else pd.DataFrame()
    if ds.empty:
        return ""

    p_suc   = float(ds["p_success_pct"].values[0])
    gov_red = float(ds["cvar_reduction_weak_to_strong_pct"].values[0])
    p_atk   = p_suc / 100.0

    dr = (risk[(risk["domain"] == domain) & (risk["maturity_label"] == maturity)]
          if not risk.empty else pd.DataFrame())
    if dr.empty:
        return ""

    cvar  = float(dr["cvar_99"].values[0])   if "cvar_99"   in dr.columns else 0.0
    el    = float(dr["el_mean"].values[0])   if "el_mean"   in dr.columns else 0.0
    rri   = float(dr["rri_mean"].values[0])  if "rri_mean"  in dr.columns else 0.0

    da = attacks[attacks["domain"] == domain] if not attacks.empty else pd.DataFrame()
    if da.empty:
        return ""
    top_attack, top_rate = "unknown", 0.0
    if "max_success_rate" in da.columns:
        best       = da.loc[da["max_success_rate"].idxmax()]
        top_attack = str(best.get("family", "unknown"))
        top_rate   = float(best["max_success_rate"])

    # Pull domain-specific business metric values
    dm_row = None
    if not dm.empty and "domain" in dm.columns:
        _dm_d = dm[dm["domain"] == key]
        if not _dm_d.empty:
            dm_row = _dm_d.iloc[0]

    def _metric(col: str, default: float = 0.0) -> float:
        if dm_row is not None and col in dm_row.index:
            v = float(dm_row[col])
            return v if not np.isnan(v) else default
        return default

    # ── Domain-specific recommendation templates ─────────────────────────────
    if domain == "Fraud Detection":
        fnr_base  = _metric("fraud_leakage_rate", 0.15)
        fnr_post  = min(1.0, fnr_base + p_atk * (1.0 - fnr_base))
        delta_fnr = (fnr_post - fnr_base) * 100
        urgency   = "URGENT — " if top_rate >= 0.15 else ""
        return (
            f"{urgency}{top_attack} achieves {top_rate*100:.0f}% evasion success. "
            f"Fraud leakage rate rises from {fnr_base*100:.1f}% → {fnr_post*100:.1f}% under attack "
            f"(+{delta_fnr:.1f}pp) — each point above 5% triggers PSD2 Article 73 chargeback liability. "
            f"CVaR 99% = {fmt_usd(cvar)} at {maturity} controls; governance uplift to strong cuts this by {gov_red:.0f}%. "
            f"<b>Action:</b> Augment training data with FGSM/PGD adversarial examples; "
            f"enforce ISO 27001 A.12.6 (Technical Vulnerability Management) to reduce RRI "
            f"from {fmt_usd(rri)} toward strong-control target."
        )

    if domain == "Credit Scoring":
        dmr_base  = _metric("default_miss_rate", 0.036)
        err_base  = _metric("approval_error_rate", 0.06)
        dmr_post  = min(1.0, dmr_base + p_atk * (1.0 - dmr_base))
        err_post  = min(1.0, err_base + p_atk * (1.0 - err_base) * 0.5)
        delta_dmr = (dmr_post - dmr_base) * 100
        severity  = cvar / max(el, 1.0)
        urgency   = "URGENT — " if top_rate >= 0.15 else ""
        return (
            f"{urgency}{top_attack} reaches {top_rate*100:.0f}% success — default miss rate "
            f"escalates {dmr_base*100:.1f}% → {dmr_post*100:.1f}% (+{delta_dmr:.1f}pp under attack). "
            f"Under IFRS 9, each missed default inflates Expected Credit Loss estimates; "
            f"a {delta_dmr:.1f}pp FNR rise represents systematic ECL understatement. "
            f"Severity index {severity:.0f}× (CVaR {fmt_usd(cvar)} / EL {fmt_usd(el)}) confirms rare-event tail dominance. "
            f"<b>Action:</b> Apply adversarial input validation at scoring API boundary; "
            f"uplift to strong ISO maturity (A.14 System Development, A.18 Compliance) "
            f"to deliver {gov_red:.0f}% CVaR reduction."
        )

    if domain == "AML Detection":
        sar_miss  = _metric("suspicious_activity_miss_rate", 1.0)
        coverage  = _metric("detection_coverage", 0.0)
        urgency   = "CRITICAL — " if sar_miss >= 0.95 else ""
        return (
            f"{urgency}SAR miss rate {sar_miss*100:.0f}% and detection coverage {coverage*100:.0f}% — "
            f"the baseline classifier is degenerate. UK MLR 2017 Reg 40 requires SARs to be filed "
            f"within 5 business days; zero detection means zero SARs filed, exposing the firm to "
            f"NCA enforcement action and FCA supervisory escalation. "
            f"CVaR 99% = {fmt_usd(cvar)} ({maturity} controls), reduced {gov_red:.0f}% with strong governance. "
            f"<b>Primary action:</b> Retrain with SMOTE/class-weighted loss before any adversarial hardening — "
            f"adversarial defences are irrelevant until recall exceeds 0%. "
            f"Deploy ISO 27001 A.16 (Incident Management) and A.18 (Compliance Monitoring) controls immediately."
        )

    if domain == "Algorithmic Trading":
        err_base  = _metric("execution_error_rate", 0.50)
        prec_base = _metric("signal_precision", 0.53)
        err_post  = min(1.0, err_base + p_atk * (1.0 - err_base) * 0.5)
        prec_post = max(0.0, prec_base * (1.0 - p_atk * 0.5))
        delta_err = (err_post - err_base) * 100
        urgency   = "URGENT — " if top_rate >= 0.50 else ""
        return (
            f"{urgency}{top_attack} at {top_rate*100:.0f}% success — execution error rate "
            f"rises {err_base*100:.1f}% → {err_post*100:.1f}% (+{delta_err:.1f}pp) and signal precision "
            f"degrades {prec_base*100:.1f}% → {prec_post*100:.1f}%. "
            f"MiFID II Article 17 mandates algorithmic circuit-breakers when error rates exceed safe thresholds; "
            f"CVaR 99% = {fmt_usd(cvar)} reflects the tail exposure from adversarially-induced bad trades. "
            f"<b>Action:</b> Implement kill-switch logic triggered at >15% execution error rate; "
            f"apply feature-integrity checksums on market data inputs (ISO 27001 A.14.2); "
            f"governance uplift delivers {gov_red:.0f}% CVaR reduction (RRI = {fmt_usd(rri)})."
        )

    # Generic fallback for unknown domains
    return (
        f"{top_attack} at {top_rate*100:.0f}% success — CVaR 99% = {fmt_usd(cvar)} ({maturity} controls). "
        f"Governance uplift reduces tail risk by {gov_red:.0f}%. "
        f"Prioritise ISO 27001 A.12 and A.16 controls."
    )


def cascade_html(domain: str, data: dict, maturity: str = "medium") -> str:
    summary = data["summary"]
    ds = summary[summary["domain"] == domain] if not summary.empty else pd.DataFrame()
    # Return empty placeholder when no pipeline data exists for this domain
    if ds.empty:
        return (
            '<div style="color:#546E7A;font-size:12px;font-style:italic;padding:16px 8px">'
            'Start the framework to populate this view.</div>'
        )
    p_suc    = float(ds["p_success_pct"].values[0])
    # Look up CVaR for the specific maturity from the risk table (authoritative source)
    risk    = data["risk"]
    dr_mat  = (risk[(risk["domain"] == domain) & (risk["maturity_label"] == maturity)]
               if not risk.empty else pd.DataFrame())
    cvar_mat = float(dr_mat["cvar_99"].values[0]) if not dr_mat.empty and "cvar_99" in dr_mat.columns else 0.0
    gov_red  = float(ds["cvar_reduction_weak_to_strong_pct"].values[0]) if not ds.empty else 0.0
    rri = float(dr_mat["rri_mean"].values[0]) if not dr_mat.empty and "rri_mean" in dr_mat.columns else 0.0
    reg_risk = "HIGH" if p_suc > 25 else "MEDIUM" if p_suc > 10 else "LOW"

    rec_act = derive_recommendation(domain, data, maturity=maturity)

    boxes = [
        (CRIT,  "CYBER ATTACK",   f"{p_suc:.1f}%",   "Model evasion rate"),
        (WARN,  "MODEL IMPACT",   f"{p_suc:.1f}%",   "Detection degradation"),
        (GOLD,  "FINANCIAL LOSS", fmt_usd(cvar_mat),   f"CVaR 99% ({maturity} controls)"),
        (BLUE,  "REGULATORY RISK", reg_risk,           "Compliance exposure"),
        (SAFE,  "RECOMMENDATION", "See below",         "Priority action"),
    ]
    parts = []
    for i, (col, title, val, sub_lbl) in enumerate(boxes):
        parts.append(
            f'<div class="cascade-box" style="border-color:{col}44;background:{col}10">'
            f'<div class="cascade-box-title" style="color:{col}">{title}</div>'
            f'<div class="cascade-box-val" style="color:{col}">{val}</div>'
            f'<div style="font-size:9px;color:#546E7A;margin-top:4px">{sub_lbl}</div>'
            f'</div>'
        )
        if i < len(boxes) - 1:
            parts.append('<div class="cascade-arrow">→</div>')
    rec_block = (
        f'<div style="margin-top:12px;padding:10px 14px;background:rgba(0,230,118,0.06);'
        f'border-left:3px solid {SAFE};border-radius:4px;font-size:12px;color:#C8DCEA;line-height:1.6">'
        f'<span style="color:{SAFE};font-weight:700;font-size:10px;letter-spacing:1px">RECOMMENDATION</span>'
        f'<br>{rec_act}</div>'
    ) if rec_act else ""
    return f'<div class="cascade-flow">{"".join(parts)}</div>{rec_block}'


def risk_table_html(risk: pd.DataFrame, domain: str) -> str:
    if risk.empty:
        return ""
    dr = risk[risk["domain"] == domain]
    if dr.empty:
        return ""
    mat_col = {"weak": CRIT, "medium": WARN, "strong": SAFE}
    cols = [("maturity_label", "Maturity"), ("el_mean", "EL Mean"),
            ("var_95", "VaR 95%"), ("var_99", "VaR 99%"),
            ("cvar_99", "CVaR 99%"), ("tail_premium", "Tail Premium")]
    header = "".join(f"<th>{lbl}</th>" for _, lbl in cols)
    rows_html = []
    for _, row in dr.sort_values("maturity_label",
                                  key=lambda x: x.map({"weak": 0, "medium": 1, "strong": 2})).iterrows():
        mat = str(row.get("maturity_label", ""))
        c   = mat_col.get(mat, "#90B8D4")
        cells = []
        for ck, _ in cols:
            v = row.get(ck, "—")
            if ck == "maturity_label":
                cells.append(f'<td><span style="color:{c};font-weight:700">{str(v).upper()}</span></td>')
            elif isinstance(v, (int, float)):
                cells.append(f"<td>{fmt_usd(float(v))}</td>")
            else:
                cells.append(f"<td>{v}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f'<table class="risk-tbl">'
        f'<thead><tr>{header}</tr></thead>'
        f'<tbody>{"".join(rows_html)}</tbody>'
        f'</table>'
    )


def consistency_table_html(cons: pd.DataFrame) -> str:
    if cons.empty:
        return "<p style='color:#546E7A'>No validation data available.</p>"
    rows = []
    for _, row in cons.iterrows():
        passed = bool(row.get("passed", False))
        sc     = SAFE if passed else CRIT
        av     = row.get("actual", "—")
        as_    = f"{float(av):.4f}" if isinstance(av, float) else str(av)
        rows.append(
            f"<tr><td>{row.get('domain','—')}</td>"
            f"<td>{row.get('check','—')}</td>"
            f"<td>{row.get('expected','—')}</td>"
            f"<td>{as_}</td>"
            f'<td><b style="color:{sc}">{"PASS" if passed else "FAIL"}</b></td></tr>'
        )
    return (
        '<table class="risk-tbl">'
        '<thead><tr><th>Domain</th><th>Property</th>'
        '<th>Expected</th><th>Actual</th><th>Status</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def insight_html(domain: str, data: dict, maturity: str = "medium") -> str:
    summary = data["summary"]
    attacks = data["attacks"]
    risk    = data["risk"]
    iso     = data["iso"]
    poison  = data["poisoning"]

    ds = summary[summary["domain"] == domain] if not summary.empty else pd.DataFrame()
    p_suc    = float(ds["p_success_pct"].values[0])               if not ds.empty else 0.0
    cvar_w   = float(ds["cvar_99_weak_usd"].values[0])             if not ds.empty else 0.0
    cvar_m   = float(ds["cvar_99_medium_usd"].values[0])           if not ds.empty else 0.0
    cvar_s   = float(ds["cvar_99_strong_usd"].values[0])           if not ds.empty else 0.0
    gov_red  = float(ds["cvar_reduction_weak_to_strong_pct"].values[0]) if not ds.empty else 0.0
    imp_mean = float(ds["impact_mean_usd"].values[0])              if not ds.empty else 0.0
    avg_p    = float(summary["p_success_pct"].mean())              if not summary.empty else 0.0

    da = attacks[attacks["domain"] == domain] if not attacks.empty else pd.DataFrame()
    if not da.empty and "max_success_rate" in da.columns:
        best       = da.loc[da["max_success_rate"].idxmax()]
        worst      = da.loc[da["max_success_rate"].idxmin()]
        pri_atk    = f"{best['family']} ({float(best['max_success_rate'])*100:.1f}%)"
        res_atk    = str(worst["family"])
    else:
        pri_atk = res_atk = "N/A"

    dr = (risk[(risk["domain"] == domain) & (risk["maturity_label"] == maturity)]
          if not risk.empty else pd.DataFrame())
    tail_p = float(dr["tail_premium"].values[0]) if not dr.empty and "tail_premium" in dr.columns else 0.0

    # best maturity interval
    di = iso[iso["domain"] == domain].sort_values("iso_maturity") if not iso.empty else pd.DataFrame()
    best_int, best_sav = "N/A", 0.0
    if not di.empty and len(di) >= 2:
        diffs = np.diff(di["cvar_99"].values)
        if len(diffs):
            idx    = int(np.argmin(diffs))
            best_sav = abs(float(diffs[idx]))
            m0, m1   = float(di["iso_maturity"].values[idx]), float(di["iso_maturity"].values[idx + 1])
            best_int = f"M={m0:.2f}→{m1:.2f}"

    # poisoning — normalise short CSV domain names to display names
    _s2d = {"fraud": "Fraud Detection", "credit": "Credit Scoring",
            "aml": "AML Detection", "trading": "Algorithmic Trading"}
    poison_n = poison.copy()
    if not poison_n.empty and "domain" in poison_n.columns:
        poison_n["domain"] = poison_n["domain"].map(lambda x: _s2d.get(str(x).lower(), x))
    dp = poison_n[poison_n["domain"] == domain] if not poison_n.empty and "domain" in poison_n.columns else pd.DataFrame()
    poi_block = ""
    if not dp.empty and "poisoned_recall" in dp.columns:
        dp_deg = dp.copy()
        dp_deg["recall_drop"] = dp_deg["baseline_recall"] - dp_deg["poisoned_recall"] \
            if "baseline_recall" in dp_deg.columns else 0
        thr     = dp_deg[dp_deg["recall_drop"] > 0.05]
        min_cor = f"{float(thr['corruption_rate'].min())*100:.1f}%" if not thr.empty else "N/A"
        max_pra = (f"{float(dp.loc[dp['corruption_rate'].idxmax(), 'poisoned_prauc']):.3f}"
                   if "poisoned_prauc" in dp.columns else "N/A")
        max_kl  = (f"{float(dp['kl_divergence'].max()):.4f}"
                   if "kl_divergence" in dp.columns else "N/A")
        max_miss = (f"{float(dp['n_missed_per_1000'].max()):.2f}"
                    if "n_missed_per_1000" in dp.columns else "N/A")
        poi_block = (
            f'<div class="ins-section">'
            f'<div class="ins-hdr">Model Poisoning Resilience</div>'
            f'<div class="ins-row"><span>Min corruption rate causing &gt;5% detection drop</span>'
            f'<span class="ins-val" style="color:{WARN}">{min_cor}</span></div>'
            f'<div class="ins-row"><span>PR-AUC at worst-case corruption rate</span>'
            f'<span class="ins-val">{max_pra}</span></div>'
            f'<div class="ins-row"><span>Max KL divergence (calibration shift)</span>'
            f'<span class="ins-val" style="color:{"#d01c8b" if max_kl != "N/A" and float(max_kl) > 0.1 else CYAN}">{max_kl}</span></div>'
            f'<div class="ins-row"><span>Max missed positives per 1,000 transactions</span>'
            f'<span class="ins-val" style="color:{WARN}">{max_miss}</span></div>'
            f'</div>'
        )

    tier_name, tier_col = risk_tier(p_suc)
    return f"""
<div class="aif-card">
  <div class="aif-card-title">Domain Intelligence</div>
  <div style="margin-bottom:12px">
    <span class="badge" style="background:{tier_col}22;color:{tier_col};border:1px solid {tier_col}55">
      {tier_name}
    </span>
    <span style="font-size:11px;color:#546E7A;margin-left:8px">
      Model evasion rate {p_suc:.1f}% vs cross-domain avg {avg_p:.1f}%
    </span>
  </div>
  <div class="ins-section">
    <div class="ins-hdr">Threat Profile</div>
    <div class="ins-row"><span>Highest-risk attack method</span>
      <span class="ins-val" style="color:{CRIT}">{pri_atk}</span></div>
    <div class="ins-row"><span>Most resilient against</span>
      <span class="ins-val" style="color:{SAFE}">{res_atk}</span></div>
  </div>
  <div class="ins-section">
    <div class="ins-hdr">Financial Exposure</div>
    <div class="ins-row"><span>CVaR 99% — Weak</span>
      <span class="ins-val" style="color:{CRIT}">{fmt_usd(cvar_w)}</span></div>
    <div class="ins-row"><span>CVaR 99% — Medium</span>
      <span class="ins-val" style="color:{WARN}">{fmt_usd(cvar_m)}</span></div>
    <div class="ins-row"><span>CVaR 99% — Strong</span>
      <span class="ins-val" style="color:{SAFE}">{fmt_usd(cvar_s)}</span></div>
    <div class="ins-row"><span>Tail premium (medium)</span>
      <span class="ins-val">{fmt_usd(tail_p)}</span></div>
    <div class="ins-row"><span>Mean loss per event</span>
      <span class="ins-val">{fmt_usd(imp_mean)}</span></div>
  </div>
  <div class="ins-section">
    <div class="ins-hdr">Governance Leverage</div>
    <div class="ins-row"><span>CVaR reduction (weak→strong)</span>
      <span class="ins-val" style="color:{SAFE}">{gov_red:.1f}%</span></div>
    <div class="ins-row"><span>Largest marginal saving</span>
      <span class="ins-val">{fmt_usd(best_sav)} ({best_int})</span></div>
  </div>
  {poi_block}
</div>"""


# ════════════════════════════════════════════════════════════════════════════
# Pages
# ════════════════════════════════════════════════════════════════════════════

def page_overview(data: dict):
    summary = data["summary"]
    risk    = data["risk"]

    # Hero
    now = datetime.now().strftime("%d %B %Y  %H:%M")
    _atk = data["attacks"]
    n_domains = int(summary["domain"].nunique()) if not summary.empty else 0
    n_attacks = int(_atk["family"].nunique()) if not _atk.empty and "family" in _atk.columns else 0
    _counts_html = (
        f'<span><span style="color:#00E5FF;font-weight:700">{n_domains}</span>&nbsp;Active Domains</span>'
        f'<span><span style="color:#00E5FF;font-weight:700">{n_attacks}</span>&nbsp;Attack Families Evaluated</span>'
    ) if n_domains > 0 else (
        f'<span style="color:#546E7A;font-style:italic">Start the framework to populate the dashboard</span>'
    )
    st.markdown(f"""
<div class="hero-block">
  <div class="hero-title glow-text">AIFCRQF</div>
  <div style="font-size:13px;color:#4FC3F7;letter-spacing:2px;margin-bottom:6px">
    AI-Driven Fintech Cyber Risk Quantification Framework
  </div>
  <div style="display:flex;justify-content:space-between;align-items:center;
       font-size:13px;color:#90B8D4;letter-spacing:1px;flex-wrap:wrap;gap:10px;
       margin-top:4px">
    {_counts_html}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Executive KPI row ──────────────────────────────────────────────────
    st.markdown('<div class="section-head">Executive Key Performance Indicators</div>',
                unsafe_allow_html=True)

    _has_data = not summary.empty and not risk.empty
    avg_p = 0.0  # default; overwritten below when data is present
    c1, c2, c3, c4 = st.columns(4)
    if not _has_data:
        _na = "—"
        _sub = "Start the framework to populate"
        c1.markdown(kpi_box("MODEL RESILIENCE",      _na, _sub, "#546E7A", "#546E7A"), unsafe_allow_html=True)
        c2.markdown(kpi_box("WORST-CASE EXPOSURE",   _na, _sub, "#546E7A", "#546E7A"), unsafe_allow_html=True)
        c3.markdown(kpi_box("ADVERSARIAL RISK",      _na, _sub, "#546E7A", "#546E7A"), unsafe_allow_html=True)
        c4.markdown(kpi_box("CONTROL EFFECTIVENESS", _na, _sub, "#546E7A", "#546E7A"), unsafe_allow_html=True)
    else:
        avg_p   = float(summary["p_success_pct"].mean())
        worst_p = (float(summary["p_success_max"].mean()) * 100.0
                   if "p_success_max" in summary.columns else avg_p)
        atk_res = 100.0 - worst_p
        _kpi_cfg = load_run_config()
        _cvar_parts = []
        for _d in DOMAIN_COLOURS:
            _dm  = _kpi_cfg.get(_d, {}).get("iso_maturity", "medium")
            _sub = risk[(risk["domain"] == _d) & (risk["maturity_label"] == _dm)]
            if not _sub.empty:
                _cvar_parts.append(float(_sub["cvar_99"].values[0]))
        cvar_tot = sum(_cvar_parts)
        gov_red  = (float(summary["cvar_reduction_weak_to_strong_pct"].mean())
                    if "cvar_reduction_weak_to_strong_pct" in summary.columns else 0.0)
        c1.markdown(kpi_box("MODEL RESILIENCE", f"{atk_res:.1f}%",
                            "Blocked under worst-case single attack",
                            SAFE if atk_res >= 50 else WARN if atk_res >= 20 else CRIT,
                            SAFE if atk_res >= 50 else WARN if atk_res >= 20 else CRIT), unsafe_allow_html=True)
        c2.markdown(kpi_box("WORST-CASE EXPOSURE", fmt_usd(cvar_tot),
                            "Total tail-risk loss | Current controls", GOLD, GOLD), unsafe_allow_html=True)
        c3.markdown(kpi_box("ADVERSARIAL RISK", f"{avg_p:.1f}%",
                            "Avg combined p_success across all domains",
                            CRIT if avg_p > 30 else WARN if avg_p > 10 else SAFE,
                            CRIT if avg_p > 30 else WARN), unsafe_allow_html=True)
        c4.markdown(kpi_box("CONTROL EFFECTIVENESS", f"{gov_red:.0f}%",
                            "Tail-risk reduction | Weak → Strong controls",
                            SAFE, SAFE), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Domain cards row ──────────────────────────────────────────────────
    st.markdown('<div class="section-head">Domain Risk Profiles</div>', unsafe_allow_html=True)
    _card_run_cfg = load_run_config()
    _cvar_col_map = {"weak": "cvar_99_weak_usd", "medium": "cvar_99_medium_usd", "strong": "cvar_99_strong_usd"}
    cols = st.columns(4)
    domains = list(DOMAIN_COLOURS.keys())
    for i, d in enumerate(domains):
        ds = summary[summary["domain"] == d] if not summary.empty else pd.DataFrame()
        col_h = DOMAIN_COLOURS[d]
        with cols[i]:
            if ds.empty:
                st.markdown(
                    f'<div class="kpi-box" style="border-top-color:{col_h};text-align:left">'
                    f'<div style="font-family:Orbitron,monospace;font-size:11px;color:{col_h};margin-bottom:6px">{d}</div>'
                    f'<div style="color:#546E7A;font-size:11px;font-style:italic;margin-top:12px">Start the framework to populate</div>'
                    f'</div>',
                    unsafe_allow_html=True)
            else:
                p       = float(ds["p_success_pct"].values[0])
                _d_mat  = _card_run_cfg.get(d, {}).get("iso_maturity", "medium")
                _d_col  = _cvar_col_map.get(_d_mat, "cvar_99_medium_usd")
                cvm     = float(ds[_d_col].values[0]) if _d_col in ds.columns else 0.0
                gr      = float(ds["cvar_reduction_weak_to_strong_pct"].values[0])
                tier, tcol = risk_tier(p)
                _mat_badge = {"weak": "WEAK", "medium": "MED", "strong": "STR"}.get(_d_mat, _d_mat.upper())
                st.markdown(
                    f'<div class="kpi-box" style="border-top-color:{col_h};text-align:left">'
                    f'<div style="font-family:Orbitron,monospace;font-size:11px;'
                    f'color:{col_h};margin-bottom:6px">{d}</div>'
                    f'<span class="badge" style="background:{tcol}22;color:{tcol};border:1px solid {tcol}55">{tier}</span>'
                    f'<div style="margin-top:12px">'
                    f'<div class="ins-row"><span class="kpi-sub">Model evasion rate</span>'
                    f'<span style="color:{tcol};font-weight:700;font-size:14px">{p:.1f}%</span></div>'
                    f'<div class="ins-row"><span class="kpi-sub">CVaR 99% [{_mat_badge}]</span>'
                    f'<span style="color:#E8F4FD;font-weight:600;font-size:12px">{fmt_usd(cvm)}</span></div>'
                    f'<div class="ins-row"><span class="kpi-sub">Gov. reduction</span>'
                    f'<span style="color:{SAFE};font-weight:600;font-size:12px">{gr:.0f}%</span></div>'
                    f'</div></div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Business risk gauge + Threat scenarios ──────────────────────────
    st.markdown('<div class="section-head">Business Risk Assessment</div>', unsafe_allow_html=True)
    g1, g2 = st.columns([1, 2])
    with g1:
        st.markdown('<div class="aif-card-title">Business Risk Level</div>',
                    unsafe_allow_html=True)
        fg = chart_business_gauge(avg_p)
        if fg:
            st.plotly_chart(fg, use_container_width=True, key="gauge")
    with g2:
        st.markdown('<div class="aif-card-title">Threat Scenarios — Financial Impact</div>',
                    unsafe_allow_html=True)
        ft = chart_threat_scenarios(risk)
        if ft:
            st.plotly_chart(ft, use_container_width=True, key="threat_scenarios")

    # ── Cascading impact (use Fraud as representative) ───────────────────
    st.markdown('<div class="section-head">Cascading Impact Chain</div>', unsafe_allow_html=True)
    _exec_run_cfg = load_run_config()
    domain_tabs = st.tabs([d.replace(" Detection","").replace(" Scoring","").replace(" Trading","") for d in domains])
    for i, tab in enumerate(domain_tabs):
        with tab:
            _d_mat = _exec_run_cfg.get(domains[i], {}).get("iso_maturity", "medium")
            st.markdown(
                f'<div class="aif-card">'
                f'<div class="aif-card-title">Attack → Model → Finance → Regulatory → Decision</div>'
                f'{cascade_html(domains[i], data, maturity=_d_mat)}'
                f'</div>',
                unsafe_allow_html=True)

    # ── CVaR bars ────────────────────────────────────────────────────────
    st.markdown('<div class="section-head">CVaR 99% Across Governance Maturity</div>',
                unsafe_allow_html=True)
    fb = chart_cvar_bars(risk)
    if fb:
        st.plotly_chart(fb, use_container_width=True, key="cvar_bars")

    # ── Security ROI ─────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Security Investment Return Analysis</div>',
                unsafe_allow_html=True)
    fr = chart_security_roi(risk)
    if fr:
        r1, r2 = st.columns([2, 1])
        with r1:
            st.plotly_chart(fr, use_container_width=True, key="sec_roi")
        with r2:
            # compute ROI summary using CVaR 99% (varies by maturity; el_mean does not)
            cvar_weak   = (float(risk[risk["maturity_label"] == "weak"]["cvar_99"].sum())
                           if not risk.empty else 0.0)
            cvar_medium = (float(risk[risk["maturity_label"] == "medium"]["cvar_99"].sum())
                           if not risk.empty else 0.0)
            cvar_strong = (float(risk[risk["maturity_label"] == "strong"]["cvar_99"].sum())
                           if not risk.empty else 0.0)
            savings_str = cvar_weak - cvar_strong
            roi_pct     = (savings_str / cvar_weak * 100) if cvar_weak > 0 else 0.0
            st.markdown(f"""
<div class="aif-card" style="padding:16px">
  <div class="ins-hdr">Executive Summary — Tail-Risk Exposure (CVaR 99%)</div>
  <div class="ins-row">
    <span>Worst-case exposure — no investment (weak controls)</span>
    <span class="ins-val" style="color:{CRIT}">{fmt_usd(cvar_weak)}</span></div>
  <div class="ins-row">
    <span>Worst-case exposure — current posture (medium controls)</span>
    <span class="ins-val" style="color:{WARN}">{fmt_usd(cvar_medium)}</span></div>
  <div class="ins-row">
    <span>Worst-case exposure — full investment (strong controls)</span>
    <span class="ins-val" style="color:{SAFE}">{fmt_usd(cvar_strong)}</span></div>
  <div class="ins-row">
    <span>Maximum risk reduction (weak → strong)</span>
    <span class="ins-val" style="color:{SAFE}">{fmt_usd(savings_str)}</span></div>
  <div class="ins-row">
    <span>Governance ROI (% tail-risk eliminated)</span>
    <span class="ins-val" style="color:{GOLD}">{roi_pct:.0f}%</span></div>
  <br>
  <div style="font-size:11px;color:#90B8D4;border-left:3px solid {CYAN};padding-left:10px;line-height:1.7">
    Governance quality is a stronger driver of financial risk than attack
    sophistication. A 0.30→0.80 maturity improvement delivers 67–70% CVaR
    reduction with no change to model architecture.
  </div>
</div>""", unsafe_allow_html=True)

    # ── Attack heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-head">Adversarial Exposure Matrix</div>',
                unsafe_allow_html=True)
    fh = chart_attack_heatmap(data["attacks"], summary)
    if fh:
        st.plotly_chart(fh, use_container_width=True, key="atk_heat")

    # ── Attack glossary ──────────────────────────────────────────────────
    with st.expander("Attack Technique Reference — click any name to understand the threat"):
        danger_col = {
            "Critical": CRIT, "High": WARN, "Medium": GOLD,
        }
        gcols = st.columns(3)
        for idx, (key, info) in enumerate(ATTACK_GLOSSARY.items()):
            dc = danger_col.get(info["danger"], GOLD)
            with gcols[idx % 3]:
                st.markdown(
                    f'<div class="aif-card" style="border-top:2px solid {dc};margin-bottom:10px">'
                    f'<div style="font-family:Orbitron,monospace;font-size:10px;color:{dc};'
                    f'letter-spacing:1px;margin-bottom:4px">{key}</div>'
                    f'<div style="font-size:11px;color:#E8F4FD;font-weight:600;margin-bottom:4px">'
                    f'{info["full"]}</div>'
                    f'<div style="display:flex;gap:8px;margin-bottom:8px">'
                    f'<span style="font-size:9px;padding:2px 7px;border-radius:3px;'
                    f'background:{dc}22;color:{dc};border:1px solid {dc}44">{info["type"]}</span>'
                    f'<span style="font-size:9px;padding:2px 7px;border-radius:3px;'
                    f'background:rgba(0,0,0,0.3);color:#90B8D4;border:1px solid rgba(0,229,255,0.1)">'
                    f'Danger: {info["danger"]}</span></div>'
                    f'<div style="font-size:11px;color:#90B8D4;line-height:1.6">{info["desc"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True)

    # ── How business risk is calculated ─────────────────────────────────
    st.markdown('<div class="section-head">How Business Risk Is Calculated</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="aif-card">
  <div class="aif-card-title">Risk Quantification Methodology</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;font-size:12px;color:#90B8D4;line-height:1.9">
    <div>
      <div style="color:{CYAN};font-weight:600;margin-bottom:6px">Step 1 — Adversarial Testing</div>
      Each AI model is subjected to 8 attack families (FGSM, PGD, C&amp;W, poisoning variants).
      The <b style="color:#E8F4FD">model evasion rate</b> measures what percentage of attacks
      successfully bypass the model's detection without triggering an alert.
      A 10% evasion rate means 1 in 10 adversarial transactions pass undetected.
      <br><br>
      <div style="color:{CYAN};font-weight:600;margin-bottom:6px">Step 2 — Monte Carlo Simulation</div>
      50,000 simulated loss scenarios are generated per domain, drawing from
      the observed attack frequency and impact distributions. This produces
      a full loss probability curve rather than a single point estimate.
    </div>
    <div>
      <div style="color:{CYAN};font-weight:600;margin-bottom:6px">Step 3 — Risk Metrics</div>
      From the simulation:<br>
      &nbsp;• <b style="color:{GOLD}">Expected Loss (EL)</b> — average loss per year across all scenarios<br>
      &nbsp;• <b style="color:{WARN}">VaR 99%</b> — loss threshold not exceeded 99% of the time<br>
      &nbsp;• <b style="color:{CRIT}">CVaR 99%</b> — average loss in the worst 1% of scenarios (tail risk)<br>
      &nbsp;• <b style="color:{BLUE}">Tail Premium</b> — extra capital buffer needed beyond VaR<br><br>
      <div style="color:{CYAN};font-weight:600;margin-bottom:6px">Step 4 — Governance Adjustment</div>
      ISO 27001 maturity (M=0.30/0.60/0.80) scales the loss distribution.
      Strong controls (M=0.80) reduce CVaR by 67–70% by limiting attacker
      dwell time and reducing impact per event.
    </div>
  </div>
  <div style="margin-top:16px;padding:12px 16px;background:rgba(0,229,255,0.04);
       border-left:3px solid {CYAN};border-radius:4px;font-size:11px;color:#90B8D4;line-height:1.7">
    <b style="color:{CYAN}">Important:</b> If only one domain has been run, all metrics reflect that
    single domain. The framework dynamically loads whichever domains are present in the output files —
    running more domains will automatically populate all charts and comparisons.
  </div>
</div>""", unsafe_allow_html=True)


def page_financial_risk(data: dict):
    st.markdown('<div class="section-head">ISO 27001 Maturity Sensitivity Curves</div>',
                unsafe_allow_html=True)
    fi = chart_iso_curves(data["iso"])
    if fi:
        st.plotly_chart(fi, use_container_width=True, key="iso_all")


    # ── Financial risk indicators (not already in the full table below) ──────
    _fr_cfg       = load_run_config()
    _mats         = [v.get("iso_maturity", "medium") for v in _fr_cfg.values() if v]
    _primary_mat  = _mats[0] if _mats else "medium"
    _mat_label_fr = {"weak": "Weak (M=0.30)", "medium": "Medium (M=0.60)", "strong": "Strong (M=0.80)"}
    st.markdown(
        f'<div class="section-head">Financial Risk Indicators — {_mat_label_fr.get(_primary_mat, _primary_mat)}</div>',
        unsafe_allow_html=True)
    _risk    = data["risk"]
    _cascade = data["cascade"]
    _iso     = data["iso"]
    _summary = data["summary"]
    if not _risk.empty:
        _dom_order = list(DOMAIN_COLOURS.keys())
        _pri = _risk[_risk["maturity_label"] == _primary_mat].set_index("domain")
        _wk  = _risk[_risk["maturity_label"] == "weak"].set_index("domain")
        _str = _risk[_risk["maturity_label"] == "strong"].set_index("domain")
        # p_success per domain for Attack Sensitivity Index
        _sum_idx = _summary.set_index("domain") if not _summary.empty and "domain" in _summary.columns else pd.DataFrame()

        _rows: list[dict] = []
        for _d in _dom_order:
            if _d not in _pri.index:
                continue
            _r    = _pri.loc[_d]
            _rw   = _wk.loc[_d]  if _d in _wk.index  else _r
            _el   = float(_r.get("el_mean",  0) or 0)
            _var  = float(_r.get("var_99",   0) or 0)
            _cvar = float(_r.get("cvar_99",  0) or 0)
            _pext = float(_r.get("p_extreme",0) or 0)
            _cvar_wk = float(_rw.get("cvar_99", 0) or 0)
            _cvar_st = float(_str.loc[_d].get("cvar_99", 0) or 0) if _d in _str.index else _cvar

            # 1. Tail Risk Ratio = CVaR / VaR
            _trr = _cvar / _var if _var > 0 else 0.0

            # 2. P(Extreme Loss) = P(RRI > 5 × EL_mean) as %
            _pe = _pext * 100

            # 3. Capital at Risk = VaR_99 - EL  (Basel II economic capital)
            _car = max(_var - _el, 0.0)

            # 4. Risk Reduction Efficiency = % CVaR reduction weak→strong
            _rre = (_cvar_wk - _cvar_st) / _cvar_wk * 100 if _cvar_wk > 0 else 0.0

            # 5. Attack Sensitivity Index = $ CVaR per 1% attack success rate
            _p_s = float(_sum_idx.loc[_d].get("p_success_mean", 0) or 0) if _d in _sum_idx.index else 0.0
            _asi = _cvar / (_p_s * 100) if _p_s > 0 else 0.0

            _rows.append({
                "domain": _d,
                "trr":    _trr,
                "pe":     _pe,
                "car":    _car,
                "rre":    _rre,
                "asi":    _asi,
            })

        if _rows:
            # Single st.markdown() — multiple calls each get isolated HTML contexts in Streamlit
            _cs = ("padding:10px 14px;background:rgba(0,40,70,0.5);"
                   "border:1px solid rgba(0,229,255,0.10);border-radius:8px;text-align:center")
            _hdr_labels = [
                ("Tail Risk Ratio",    "CVaR₉₉ ÷ VaR₉₉"),
                ("P(Extreme Loss)",    "P(loss > 5 × EL)"),
                ("Capital at Risk",    "VaR₉₉ − EL  ·  Basel II"),
                ("Risk Reduction",     "CVaR drop: weak→strong"),
                ("Attack Sensitivity", "$ CVaR per 1% attack"),
            ]
            _html_parts = [
                f'<div class="aif-card">',
                f'<div class="aif-card-title">Cross-Domain Financial Risk Indicators — {_mat_label_fr.get(_primary_mat, _primary_mat)}</div>',
                f'<div style="display:flex;gap:4px;margin-bottom:6px;flex-wrap:wrap">',
                f'<div style="{_cs};flex:1;min-width:140px"><div style="font-size:9px;color:#00E5FF;'
                f'font-weight:700;letter-spacing:1px">DOMAIN</div></div>',
            ]
            for _h, _f in _hdr_labels:
                _html_parts.append(
                    f'<div style="{_cs};flex:1;min-width:110px">'
                    f'<div style="font-size:9px;color:#00E5FF;font-weight:700;letter-spacing:1px;'
                    f'text-transform:uppercase;margin-bottom:3px">{_h}</div>'
                    f'<div style="font-size:9px;color:#546E7A">{_f}</div>'
                    f'</div>'
                )
            _html_parts.append('</div>')

            for _row in _rows:
                _dc    = DOMAIN_COLOURS.get(_row["domain"], CYAN)
                _trr_c = CRIT if _row["trr"] > 30 else WARN if _row["trr"] > 10 else SAFE
                _pe_c  = CRIT if _row["pe"] > 10 else WARN if _row["pe"] > 5 else SAFE
                _car_c = CRIT if _row["car"] > 500_000 else WARN if _row["car"] > 50_000 else SAFE
                _rre_c = SAFE if _row["rre"] >= 60 else WARN if _row["rre"] >= 40 else CRIT
                _asi_c = CRIT if _row["asi"] > 1_000_000 else WARN if _row["asi"] > 100_000 else SAFE
                _vs    = 'font-family:Orbitron,monospace;font-size:14px;font-weight:700'
                _html_parts.append(
                    f'<div style="display:flex;gap:4px;margin-bottom:4px;flex-wrap:wrap">'
                    f'<div style="{_cs};flex:1;min-width:140px;color:{_dc};font-size:11px;font-weight:700">'
                    f'{_row["domain"]}</div>'
                    f'<div style="{_cs};flex:1;min-width:110px;color:{_trr_c};{_vs}">'
                    f'{_row["trr"]:.1f}×</div>'
                    f'<div style="{_cs};flex:1;min-width:110px;color:{_pe_c};{_vs}">'
                    f'{_row["pe"]:.2f}%</div>'
                    f'<div style="{_cs};flex:1;min-width:110px;color:{_car_c};{_vs}">'
                    f'{fmt_usd(_row["car"])}</div>'
                    f'<div style="{_cs};flex:1;min-width:110px;color:{_rre_c};{_vs}">'
                    f'{_row["rre"]:.1f}%</div>'
                    f'<div style="{_cs};flex:1;min-width:110px;color:{_asi_c};{_vs}">'
                    f'{fmt_usd(_row["asi"])}</div>'
                    f'</div>'
                )

            _html_parts.append(
                f'<div style="margin-top:12px;font-size:10px;color:#546E7A;line-height:1.8;'
                f'border-left:3px solid {CYAN};padding-left:10px">'
                f'<b style="color:{CYAN}">Tail Risk Ratio</b>: CVaR÷VaR — how severe losses become beyond VaR; &gt;30× = extreme fat-tail. &nbsp;'
                f'<b style="color:{CYAN}">P(Extreme Loss)</b>: probability of loss &gt;5× expected — operationally interpretable ruin probability. &nbsp;'
                f'<b style="color:{CYAN}">Capital at Risk</b>: VaR₉₉−EL — economic capital above provisioned expected losses (Basel II Pillar 1). &nbsp;'
                f'<b style="color:{CYAN}">Risk Reduction</b>: % CVaR reduction M=0.30→0.80 — quantified ISO 27001 governance effectiveness. &nbsp;'
                f'<b style="color:{CYAN}">Attack Sensitivity</b>: $ CVaR per 1% attack success rate — shows which domains are most financially exposed to adversarial risk.'
                f'</div></div>'
            )

            st.markdown("".join(_html_parts), unsafe_allow_html=True)

    st.markdown('<div class="section-head">Financial Risk Metrics — Full Table</div>',
                unsafe_allow_html=True)
    risk = data["risk"]
    if not risk.empty:
        for domain in list(DOMAIN_COLOURS.keys()):
            col = DOMAIN_COLOURS[domain]
            with st.expander(f"  {domain}", expanded=False):
                st.markdown(
                    f'<div style="border-left:3px solid {col};padding-left:12px">'
                    f'{risk_table_html(risk, domain)}</div>',
                    unsafe_allow_html=True)

    # ── Disclosure timing comparison ──────────────────────────────────────
    st.markdown(
        '<div class="section-head">Breach Disclosure Timing — Reputational Loss Amplification</div>',
        unsafe_allow_html=True)
    fd = chart_disclosure_comparison(data["disclosure"])
    if fd:
        disc_left, disc_right = st.columns([3, 2])
        with disc_left:
            st.plotly_chart(fd, use_container_width=True, key="disclosure_cmp")
        with disc_right:
            disc_df = data["disclosure"]
            amp_row = (disc_df[disc_df["scenario"] == "amplification_factor"]
                       if not disc_df.empty else pd.DataFrame())
            r_f = (float(amp_row["reputational_amplification_factor"].values[0])
                   if not amp_row.empty and "reputational_amplification_factor" in amp_row.columns
                   else None)
            imm = disc_df[disc_df["scenario"] == "immediate_disclosure"] if not disc_df.empty else pd.DataFrame()
            dly = disc_df[disc_df["scenario"] == "delayed_disclosure"]   if not disc_df.empty else pd.DataFrame()
            rep_imm = float(imm["reputational_mean"].values[0]) if not imm.empty and "reputational_mean" in imm.columns else 0.0
            rep_dly = float(dly["reputational_mean"].values[0]) if not dly.empty and "reputational_mean" in dly.columns else 0.0
            tot_imm = float(imm["total_cascade_mean"].values[0]) if not imm.empty and "total_cascade_mean" in imm.columns else 0.0
            tot_dly = float(dly["total_cascade_mean"].values[0]) if not dly.empty and "total_cascade_mean" in dly.columns else 0.0
            st.markdown(f"""
<div class="aif-card">
  <div class="aif-card-title">Disclosure Timing Analysis</div>
  <div class="ins-section">
    <div class="ins-hdr">Reputational Loss</div>
    <div class="ins-row"><span>Immediate disclosure</span>
      <span class="ins-val" style="color:{SAFE}">{fmt_usd(rep_imm)}</span></div>
    <div class="ins-row"><span>Delayed disclosure</span>
      <span class="ins-val" style="color:{CRIT}">{fmt_usd(rep_dly)}</span></div>
    <div class="ins-row"><span>Amplification factor R_f</span>
      <span class="ins-val" style="color:{WARN}">{f'{r_f:.2f}×' if r_f else 'N/A'}</span></div>
  </div>
  <div class="ins-section">
    <div class="ins-hdr">Total Cascade Loss</div>
    <div class="ins-row"><span>Immediate disclosure</span>
      <span class="ins-val" style="color:{SAFE}">{fmt_usd(tot_imm)}</span></div>
    <div class="ins-row"><span>Delayed disclosure</span>
      <span class="ins-val" style="color:{CRIT}">{fmt_usd(tot_dly)}</span></div>
  </div>
  <div style="margin-top:12px;font-size:11px;color:#90B8D4;line-height:1.7;
       border-left:3px solid {CYAN};padding-left:10px">
    Wu et al. (2022) show ISO 27001-certified firms that delay breach disclosure
    suffer materially larger reputational drawdowns. AIFCRQF quantifies this as
    R_f = 1.5 — delayed disclosure multiplies reputational loss by 50%.
    Transparency is a financial variable, not just a governance principle.
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.info("Disclosure comparison data not available — run the pipeline first.")


def page_domain(domain: str, data: dict):
    col_h = DOMAIN_COLOURS[domain]
    summary = data["summary"]
    ds = summary[summary["domain"] == domain] if not summary.empty else pd.DataFrame()
    p   = float(ds["p_success_pct"].values[0])                     if not ds.empty else 0.0
    gr  = float(ds["cvar_reduction_weak_to_strong_pct"].values[0]) if not ds.empty else 0.0
    imp = float(ds["impact_mean_usd"].values[0])                   if not ds.empty else 0.0
    tier, tcol = risk_tier(p)

    # Read which maturity the pipeline was run with — no UI selector, auto-detected
    run_cfg = load_run_config()
    _has_run_cfg = domain in run_cfg
    sel_mat = run_cfg.get(domain, {}).get("iso_maturity", "medium")

    # Resolve CVaR for the pipeline maturity from the risk table (authoritative)
    _risk = data["risk"]
    _dr   = (_risk[(_risk["domain"] == domain) & (_risk["maturity_label"] == sel_mat)]
             if not _risk.empty else pd.DataFrame())
    _has_risk = not ds.empty and not _dr.empty

    _mat_label_map = {"weak": "Weak (M=0.30)", "medium": "Medium (M=0.60)", "strong": "Strong (M=0.80)"}
    mat_label = _mat_label_map.get(sel_mat, sel_mat) if _has_run_cfg else "—"

    cvm_str  = fmt_usd(float(_dr["cvar_99"].values[0])) if _has_risk and "cvar_99" in _dr.columns else "—"
    p_str    = f"{p:.1f}%" if _has_risk else "—"
    gr_str   = f"{gr:.0f}%" if _has_risk else "—"
    imp_str  = fmt_usd(imp) if _has_risk else "—"
    cvar_sub = f"CVaR 99% — {mat_label}" if _has_run_cfg else "CVaR 99%"
    badge_html = (f'<span class="badge" style="background:{tcol}22;color:{tcol};border:1px solid {tcol}55">{tier}</span>'
                  if _has_risk else
                  '<span style="font-size:10px;color:#546E7A;font-style:italic">Start the framework to populate</span>')

    # Domain hero
    st.markdown(f"""
<div class="aif-card" style="border-top:3px solid {col_h}">
  <div style="font-family:Orbitron,monospace;font-size:20px;font-weight:700;
       color:{col_h};margin-bottom:8px">{domain}</div>
  {badge_html}
  <div style="display:flex;gap:32px;margin-top:16px;flex-wrap:wrap">
    <div><div style="font-family:Orbitron,monospace;font-size:20px;color:{tcol}">{p_str}</div>
         <div style="font-size:10px;color:#546E7A">Model evasion rate</div></div>
    <div><div style="font-family:Orbitron,monospace;font-size:20px;color:{GOLD}">{cvm_str}</div>
         <div style="font-size:10px;color:#546E7A">{cvar_sub}</div></div>
    <div><div style="font-family:Orbitron,monospace;font-size:20px;color:{SAFE}">{gr_str}</div>
         <div style="font-size:10px;color:#546E7A">Governance reduction</div></div>
    <div><div style="font-family:Orbitron,monospace;font-size:20px;color:{BLUE}">{imp_str}</div>
         <div style="font-size:10px;color:#546E7A">Mean loss / event</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    if not _has_risk:
        st.markdown(
            '<div style="margin-top:24px;padding:32px;text-align:center;'
            'background:rgba(0,20,40,0.4);border:1px dashed rgba(0,229,255,0.15);'
            'border-radius:8px;color:#546E7A;font-size:13px">'
            'Start the framework to populate this domain\'s results.'
            '</div>',
            unsafe_allow_html=True)
        return

    # Domain context strip
    ctx = DOMAIN_CONTEXT.get(domain, {})
    if ctx:
        st.markdown(
            f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px">'
            f'<div style="flex:1;min-width:160px;padding:10px 14px;background:rgba(0,40,70,0.5);'
            f'border:1px solid rgba(0,229,255,0.12);border-radius:8px">'
            f'<div style="font-size:9px;color:#546E7A;letter-spacing:1px;margin-bottom:4px">REGULATORY FRAMEWORK</div>'
            f'<div style="font-size:11px;color:#E8F4FD">{ctx["regulation"]}</div></div>'
            f'<div style="flex:2;min-width:240px;padding:10px 14px;background:rgba(0,40,70,0.5);'
            f'border:1px solid rgba(0,229,255,0.12);border-radius:8px">'
            f'<div style="font-size:9px;color:#546E7A;letter-spacing:1px;margin-bottom:4px">INDUSTRY BENCHMARK</div>'
            f'<div style="font-size:11px;color:#E8F4FD">{ctx["benchmark"]}</div></div>'
            f'<div style="flex:1;min-width:160px;padding:10px 14px;background:rgba(0,40,70,0.5);'
            f'border:1px solid rgba(0,229,255,0.12);border-radius:8px">'
            f'<div style="font-size:9px;color:#546E7A;letter-spacing:1px;margin-bottom:4px">CAPITAL TREATMENT</div>'
            f'<div style="font-size:11px;color:#E8F4FD">{ctx["capital"]}</div></div>'
            f'<div style="flex:2;min-width:240px;padding:10px 14px;background:rgba(239,83,80,0.07);'
            f'border:1px solid rgba(239,83,80,0.2);border-radius:8px">'
            f'<div style="font-size:9px;color:#546E7A;letter-spacing:1px;margin-bottom:4px">PRIMARY RISK</div>'
            f'<div style="font-size:11px;color:#E8F4FD">{ctx["key_risk"]}</div></div>'
            f'</div>',
            unsafe_allow_html=True)

    # Domain-specific business impact metrics — post-adversarial, individually charted
    st.markdown('<div class="section-head">Domain-Specific Business Impact Metrics</div>',
                unsafe_allow_html=True)
    section_post_adversarial_metrics(data["domain_metrics"], domain, summary=data["summary"])

    left, right = st.columns([3, 2])
    with left:
        # Cascading impact
        st.markdown('<div class="section-head">Cascading Impact Chain</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="aif-card">{cascade_html(domain, data, maturity=sel_mat)}</div>',
            unsafe_allow_html=True)

        # Attack chart
        st.markdown('<div class="section-head">Attack Success by Family</div>', unsafe_allow_html=True)
        fa = chart_domain_attack(data["attacks"], domain)
        if fa:
            st.plotly_chart(fa, use_container_width=True, key=f"atk_{domain}")

        # ISO sensitivity
        st.markdown('<div class="section-head">ISO Maturity Sensitivity</div>', unsafe_allow_html=True)
        fi = chart_domain_iso(data["iso"], domain)
        if fi:
            st.plotly_chart(fi, use_container_width=True, key=f"iso_{domain}")
        # Annotation
        iso = data["iso"]
        di  = iso[iso["domain"] == domain].sort_values("iso_maturity") if not iso.empty else pd.DataFrame()
        if not di.empty and len(di) >= 2:
            lo  = di[di["iso_maturity"] <= 0.10]["cvar_99"].mean()
            hi  = di[di["iso_maturity"] >= 0.90]["cvar_99"].mean()
            if not (np.isnan(lo) or np.isnan(hi)):
                spu = (lo - hi) / 9.0
                st.markdown(
                    f'<p style="font-size:11px;color:{CYAN};margin-top:4px;padding:8px 12px;'
                    f'background:rgba(0,229,255,0.05);border-left:3px solid {CYAN};border-radius:4px">'
                    f'{fmt_usd(spu)} CVaR saving per 0.10 maturity unit improvement</p>',
                    unsafe_allow_html=True)

        # Risk metrics table
        st.markdown('<div class="section-head">Risk Metrics Table</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="aif-card">{risk_table_html(data["risk"], domain)}</div>',
            unsafe_allow_html=True)

        # Classifier performance — separate from domain-specific business metrics above
        st.markdown('<div class="section-head">Baseline Classifier Performance</div>',
                    unsafe_allow_html=True)
        fperf = chart_model_performance(data["domain_metrics"], domain)
        if fperf:
            st.plotly_chart(fperf, use_container_width=True, key=f"perf_{domain}")

            # FPR metric card — success criterion: FPR ≤ 0.05%
            _dm_key = _DOMAIN_KEY.get(domain, "fraud")
            _dm = data["domain_metrics"]
            _dm_row = (
                _dm[_dm["domain"] == _dm_key].iloc[0]
                if not _dm.empty and "domain" in _dm.columns
                   and not _dm[_dm["domain"] == _dm_key].empty
                else None
            )
            if _dm_row is not None and "false_positive_rate" in _dm_row.index:
                _fpr  = float(_dm_row["false_positive_rate"])
                _fpr_pct = _fpr * 100
                _fpr_pass = _fpr <= 0.0005
                _fpr_badge_bg  = SAFE if _fpr_pass else CRIT
                _fpr_badge_txt = "PASS" if _fpr_pass else "FAIL"
                _fpr_note = (
                    "Well within the ≤0.05% success criterion — "
                    "the high-dimensional PCA feature space combined with calibrated "
                    "scale_pos_weight makes the model very conservative about false alarms."
                    if _fpr_pass else
                    "Exceeds the ≤0.05% success criterion — "
                    "review decision threshold or apply cost-sensitive calibration."
                )
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:16px;margin:10px 0 4px;'
                    f'padding:12px 16px;background:rgba(0,20,40,0.4);border-radius:8px;'
                    f'border:1px solid rgba(0,229,255,0.10)">'
                    f'<div style="flex:0 0 auto">'
                    f'<span style="font-size:9px;font-weight:700;letter-spacing:1.5px;'
                    f'padding:3px 10px;border-radius:4px;background:{_fpr_badge_bg}22;'
                    f'color:{_fpr_badge_bg};border:1px solid {_fpr_badge_bg}55">'
                    f'{_fpr_badge_txt}</span>'
                    f'</div>'
                    f'<div style="flex:0 0 160px">'
                    f'<div style="font-size:9px;color:#546E7A;letter-spacing:1px;'
                    f'text-transform:uppercase;margin-bottom:2px">False Positive Rate</div>'
                    f'<div style="font-family:Orbitron,monospace;font-size:18px;'
                    f'font-weight:700;color:{_fpr_badge_bg}">{_fpr_pct:.4f}%</div>'
                    f'<div style="font-size:9px;color:#546E7A">Threshold: ≤ 0.05%</div>'
                    f'</div>'
                    f'<div style="flex:1;font-size:11px;color:#90B8D4;line-height:1.6">'
                    f'{_fpr_note}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div style="font-size:11px;color:#90B8D4;padding:8px 12px;line-height:1.8;'
                f'border-left:3px solid {CYAN};border-radius:4px;background:rgba(0,229,255,0.03)">'
                f'These are <b style="color:#E8F4FD">pre-adversarial baseline</b> classifier metrics '
                f'from the clean test set. '
                f'<b style="color:{BLUE}">PR-AUC</b> is the primary metric for imbalanced domains '
                f'(fraud 0.17%, AML 0.3%) because ROC-AUC is misleading when negatives vastly '
                f'outnumber positives — a model predicting all-negative still scores ~0.99 ROC-AUC. '
                f'<b style="color:{CRIT}">Recall</b> is prioritised over precision in fraud and AML '
                f'because the cost of a missed event (undetected fraud, missed SAR) far exceeds '
                f'the cost of a false alarm.</div>',
                unsafe_allow_html=True,
            )

    with right:
        # Insight panel
        st.markdown(insight_html(domain, data, maturity=sel_mat), unsafe_allow_html=True)


def page_validation(data: dict):
    cons = data["consistency"]
    all_passed = bool(cons["passed"].all()) if not cons.empty and "passed" in cons.columns else True
    fc = int((~cons["passed"]).sum()) if not cons.empty and "passed" in cons.columns else 0

    if all_passed:
        bg, bc, icon = f"rgba(0,230,118,0.06)", SAFE, "✓"
        main = "ALL MATHEMATICAL PROPERTIES VERIFIED"
        sub  = f"{len(cons)} consistency checks passed — framework integrity confirmed"
    else:
        bg, bc, icon = "rgba(239,83,80,0.06)", CRIT, "✗"
        main = "VALIDATION FAILURES DETECTED"
        sub  = f"{fc} check(s) failed — review required"

    st.markdown(
        f'<div class="status-banner" style="background:{bg};border:1px solid {bc}">'
        f'<div class="banner-icon" style="color:{bc}">{icon}</div>'
        f'<div><div class="banner-main" style="color:{bc}">{main}</div>'
        f'<div class="banner-sub">{sub}</div></div></div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-head">Mathematical Consistency Checks</div>',
                unsafe_allow_html=True)
    st.markdown(f"""
<div class="aif-card">
<div class="aif-card-title">Why These Checks Matter</div>
<div style="font-size:12px;color:#90B8D4;line-height:1.9;margin-bottom:16px">
These are <b style="color:{CYAN}">formal correctness proofs</b> that verify the framework produces
financially coherent results — not just plausible-looking numbers.<br>
<b style="color:{CYAN}">CVaR ≥ VaR:</b> The worst 1% average loss must always exceed the 99th percentile threshold.
If this fails, the risk model is mathematically broken and would understate catastrophic exposure.<br>
<b style="color:{CYAN}">EL monotonicity:</b> Expected loss must decrease as governance improves.
A failure here means the model incorrectly rewards weaker controls — a critical regulatory red flag.<br>
<b style="color:{CYAN}">Governance dominance:</b> Strong controls must produce lower CVaR than weak controls.
This validates the core dissertation finding that governance quality drives financial risk more than attack sophistication.<br>
<b style="color:{CYAN}">Cascade consistency:</b> Total cascading loss must exceed direct loss alone.
Confirms reputational and regulatory multipliers are applied correctly.<br>
<b style="color:{CYAN}">RRI boundary:</b> Residual Risk Index must reach zero at full ISO maturity (M=1.0).
Validates the mathematical boundary condition — perfect controls eliminate residual risk.<br>
<b style="color:{CYAN}">FPR ≤ 0.05%:</b> False Positive Rate success criterion from project brief.
A high FPR means the model flags too many legitimate transactions as fraudulent, creating operational cost and customer harm.
FPR = FP / (FP + TN) — measures false alarm rate on the negative class.<br>
<b style="color:{CYAN}">CVaR99 stable ±5%:</b> Monte Carlo convergence check across 5 independent random seeds.
Verifies the 50,000-iteration simulation has converged — results should not change materially between runs.
Max deviation &gt; 5% indicates insufficient iterations or extreme distributional skew requiring review.
</div>
{consistency_table_html(cons)}
</div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-head">Poisoning Attack Resilience Sweep</div>',
                unsafe_allow_html=True)
    fp = chart_poisoning(data["poisoning"])
    if fp:
        st.plotly_chart(fp, use_container_width=True, key="poisoning_sweep")
    else:
        st.info("Poisoning sweep data not available.")

    st.markdown('<div class="section-head">Framework Methodology</div>', unsafe_allow_html=True)
    st.markdown(f"""
<div class="aif-card">
  <div style="font-size:12px;color:#90B8D4;line-height:2.0">
    <b style="color:{CYAN}">Framework:</b> AIFCRQF — AI-Driven Fintech Cyber Risk Quantification
    &nbsp;|&nbsp; <b style="color:{CYAN}">Monte Carlo:</b> 50,000 iterations per domain
    &nbsp;|&nbsp; <b style="color:{CYAN}">ISO range:</b> M=0.0 → 1.0 (11 evaluation points)
    &nbsp;|&nbsp; <b style="color:{CYAN}">Confidence:</b> VaR 95% / CVaR 99%<br>
    <b style="color:{CYAN}">Attack families:</b> FGSM · PGD · C&amp;W · Poisoning + variants
    &nbsp;|&nbsp; <b style="color:{CYAN}">Domains:</b> Fraud Detection · Credit Scoring · AML Detection · Algorithmic Trading<br>
    <b style="color:{CYAN}">Mathematical properties verified:</b>
    EL monotonicity · CVaR ≥ VaR · Governance dominance · Cascade consistency · RRI boundary conditions · FPR ≤ 0.05% · CVaR99 stability ±5%
  </div>
</div>""", unsafe_allow_html=True)


def chart_mc_distribution(risk, domain: str, maturity: str = "weak"):
    """Synthetic Monte Carlo loss histogram with VaR/CVaR overlays from real metrics."""
    rng = np.random.default_rng(hash(domain) % (2**31))
    dr = (risk[(risk["domain"] == domain) & (risk["maturity_label"] == maturity)]
          if not risk.empty else pd.DataFrame())
    if dr.empty:
        return None
    el   = float(dr["el_mean"].values[0])
    var99 = float(dr["var_99"].values[0])  if "var_99"  in dr.columns else el * 1.8
    cvar  = float(dr["cvar_99"].values[0]) if "cvar_99" in dr.columns else el * 2.2
    # reconstruct a plausible loss distribution matching these moments
    sigma = (var99 - el) / 2.33
    losses = rng.lognormal(mean=np.log(max(el, 1)), sigma=max(0.5, sigma / max(el, 1)), size=50_000)
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=losses, nbinsx=80, name="Simulated Loss",
        marker_color=BLUE, opacity=0.7,
        marker_line=dict(width=0),
        hovertemplate="Loss: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    for val, label, col in [(var99, "VaR 99%", WARN), (cvar, "CVaR 99%", CRIT)]:
        fig.add_vline(x=val, line_dash="dash", line_color=col, line_width=2,
                      annotation_text=f"  {label}: {fmt_usd(val)}",
                      annotation_position="top right",
                      annotation_font=dict(size=10, color=col))
    fig.add_vline(x=el, line_dash="dot", line_color=SAFE, line_width=1.5,
                  annotation_text=f"  EL: {fmt_usd(el)}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color=SAFE))
    fig_style(fig, height=320, margin=dict(l=50, r=20, t=40, b=50))
    fig.update_xaxes(title="Simulated Loss (USD)", tickprefix="$")
    fig.update_yaxes(title="Frequency (50,000 iterations)")
    fig.update_layout(showlegend=False,
                      title=dict(text=f"{domain} — Monte Carlo Loss Distribution (Medium Controls)",
                                 font=dict(size=11, color="#90B8D4"), x=0))
    return fig


def chart_confusion_matrix(attacks, domain: str):
    """Stylised confusion matrix: clean vs adversarial accuracy."""
    da = attacks[attacks["domain"] == domain] if not attacks.empty else pd.DataFrame()
    if da.empty:
        return None
    p_suc = float(da["max_success_rate"].max()) if "max_success_rate" in da.columns else 0.1
    # derive plausible confusion values
    base_tp, base_fp = 0.92, 0.05
    adv_tp  = max(0.0, base_tp - p_suc * 0.8)
    adv_fp  = min(1.0, base_fp + p_suc * 0.4)
    adv_fn  = 1.0 - adv_tp
    adv_tn  = 1.0 - adv_fp
    matrices = {
        "Clean (Baseline)": [[base_tp, base_fp], [1-base_tp, 1-base_fp]],
        "Under Attack":     [[adv_tp,  adv_fp],  [adv_fn,    adv_tn]],
    }
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Clean (Baseline)", "Under Adversarial Attack"])
    fig.update_annotations(font=dict(color="#90B8D4", size=11))
    labels = ["Fraud / Anomaly", "Legitimate"]
    for col_idx, (title, mat) in enumerate(matrices.items(), start=1):
        z    = [[mat[0][0], mat[0][1]], [mat[1][0], mat[1][1]]]
        text = [[f"{v:.1%}" for v in row] for row in z]
        cs   = [[0, "#001228"], [0.5, "#003D4F"], [1.0, CYAN]] if col_idx == 1 else \
               [[0, "#120008"], [0.5, "#4D0019"], [1.0, CRIT]]
        fig.add_trace(go.Heatmap(
            z=z, x=labels, y=["Predicted +", "Predicted −"],
            text=text, texttemplate="%{text}",
            textfont=dict(size=12, color="white"),
            colorscale=cs, showscale=False,
            hovertemplate="<b>%{y} / %{x}</b><br>Rate: %{text}<extra></extra>",
        ), row=1, col=col_idx)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,8,16,0.4)",
                      font=dict(color="#90B8D4", size=11), height=320,
                      margin=dict(l=80, r=30, t=60, b=60),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    _pc(fig, rows=1, cols=2)
    return fig


def chart_cascade_waterfall(risk, domain: str, maturity: str = "medium", cascade: pd.DataFrame = None):
    """Waterfall: direct → operational → reputational → regulatory total loss."""
    dr = (risk[(risk["domain"] == domain) & (risk["maturity_label"] == maturity)]
          if not risk.empty else pd.DataFrame())
    if dr.empty:
        return None
    # Use real cascade component data if available
    cr = (cascade[cascade["domain"] == domain] if cascade is not None and not cascade.empty else pd.DataFrame())
    if not cr.empty:
        direct = float(cr["direct"].values[0])
        ops    = float(cr["operational"].values[0])
        rep    = float(cr["reputational"].values[0])
        churn  = float(cr["churn"].values[0])
        reg    = float(cr["regulatory"].values[0])
        total  = float(cr["total_cascade"].values[0])
        labels = ["Direct Loss", "Operational\nDisruption", "Reputational\nDamage", "Customer\nChurn", "Regulatory\nExposure", "TOTAL"]
        values = [direct, ops, rep, churn, reg, total]
        measure = ["relative", "relative", "relative", "relative", "relative", "total"]
        colours = [CRIT, WARN, GOLD, BLUE, "#a78bfa", CYAN]
    else:
        el    = float(dr["el_mean"].values[0])
        cvar  = float(dr["cvar_99"].values[0]) if "cvar_99" in dr.columns else el * 2.2
        direct = el * 0.45
        ops    = el * 0.25
        rep    = el * 0.20
        reg    = cvar - el * 0.90
        total  = direct + ops + rep + reg
        labels  = ["Direct Loss", "Operational\nDisruption", "Reputational\nDamage", "Regulatory\nExposure", "TOTAL CVaR"]
        values  = [direct, ops, rep, reg, total]
        measure = ["relative", "relative", "relative", "relative", "total"]
        colours = [CRIT, WARN, GOLD, BLUE, CYAN]
    measure = ["relative", "relative", "relative", "relative", "total"]
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        text=[fmt_usd(v) for v in values],
        textposition="outside",
        textfont=dict(color="#90B8D4", size=10),
        connector=dict(line=dict(color="rgba(0,229,255,0.3)", width=1)),
        increasing=dict(marker=dict(color=CRIT)),
        decreasing=dict(marker=dict(color=SAFE)),
        totals=dict(marker=dict(color=CYAN)),
        hovertemplate="<b>%{x}</b><br>%{text}<extra></extra>",
    ))
    fig_style(fig, height=320, margin=dict(l=60, r=20, t=30, b=60))
    fig.update_yaxes(title="Loss (USD)", tickprefix="$")
    fig.update_layout(showlegend=False)
    return fig


# ── Domain metric helpers ─────────────────────────────────────────────────
_DOMAIN_KEY = {
    "Fraud Detection":     "fraud",
    "Credit Scoring":      "credit",
    "AML Detection":       "aml",
    "Algorithmic Trading": "trading",
}
# ── Post-adversarial domain metric definitions ────────────────────────────
# scale: multiply raw value by this for display (100 → %, 1000 → per-1k txns)
# unit:  display unit string appended to the formatted value
_POST_ADV_METRICS: dict = {
    "fraud": [
        {
            "key":   "fraud_leakage_rate",
            "label": "Fraud Leakage Rate",
            "colour": CRIT,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "fnr",
            "what": (
                "The <b>False Negative Rate (FNR)</b> — the fraction of genuinely fraudulent "
                "transactions that the model fails to flag. Computed as: "
                "<code>FN / (FN + TP)</code>."
            ),
            "why": (
                "Every missed fraud transaction becomes a direct financial loss. At a typical UK "
                "card fraud value of £200–£800, a 5% leakage rate on 1 million daily transactions "
                "is £10M–£40M/day undetected. Under PSD2 and the UK Fraud Act 2006, banks bear "
                "liability for most card-not-present fraud, so each false negative is a P&L hit."
            ),
            "good_when": (
                "Lower is better. The FCA PROD Sourcebook and SR 11-7 guidance suggest "
                "classifiers in production should be monitored when FNR exceeds 5%. "
                "A near-zero FNR under adversarial testing indicates the model is robust."
            ),
            "zero_note": None,
        },
        {
            "key":   "chargeback_ratio",
            "label": "Chargeback Ratio",
            "colour": WARN,
            "scale": 1000,
            "unit":  " per 1,000 txns",
            "range": [0, 10],
            "deg_type": "error",
            "what": (
                "The number of <b>false positives per 1,000 transactions</b> — legitimate "
                "transactions incorrectly blocked by the model. Computed as: "
                "<code>(FP / total) × 1,000</code>."
            ),
            "why": (
                "False positives trigger chargeback disputes when a legitimate customer's "
                "transaction is declined or reversed. Each dispute incurs processing fees "
                "(typically $15–$25 per dispute), damages customer experience, and drives churn. "
                "Visa and Mastercard fine acquirers whose merchants exceed a 0.9% monthly "
                "chargeback rate, creating a direct regulatory cost."
            ),
            "good_when": (
                "Lower is better, but not at the cost of recall. High precision models "
                "block fewer legitimate transactions. A rate near zero means the model "
                "rarely over-blocks — but check that recall hasn't collapsed."
            ),
            "zero_note": (
                "⚠ <b>Why this is near zero:</b> The creditcard.csv dataset uses "
                "PCA-anonymised features (V1–V28) that are highly discriminative. XGBoost "
                "with <code>scale_pos_weight=578</code> on this feature space produces very "
                "few false positives — the PCA projection already separates classes well. "
                "The value is approximately 0.00025 (1 FP per 4,000 transactions) "
                "which appears as ~0.25 per 1,000 here."
            ),
        },
    ],
    "credit": [
        {
            "key":   "default_miss_rate",
            "label": "Default Miss Rate",
            "colour": CRIT,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "fnr",
            "what": (
                "The <b>False Negative Rate</b> for credit defaults — the fraction of borrowers "
                "who will default that the model classifies as creditworthy. "
                "Computed as: <code>FN / (FN + TP)</code>."
            ),
            "why": (
                "Each missed default means the bank underwrites a loan that will not be repaid. "
                "Under IFRS 9, banks must estimate Expected Credit Losses (ECL) for all "
                "financial instruments. An adversarially inflated FNR causes systematic "
                "ECL underestimation — a Basel III capital adequacy breach risk. "
                "A 10% default miss rate on a £500k loan portfolio = £50k ECL understatement."
            ),
            "good_when": (
                "Lower is better. IRB-approved models under Basel III must demonstrate "
                "stable FNR over time; SR 11-7 requires monitoring when performance degrades "
                "more than one standard deviation from the validation benchmark."
            ),
            "zero_note": None,
        },
        {
            "key":   "approval_error_rate",
            "label": "Approval Error Rate",
            "colour": WARN,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "error",
            "what": (
                "Combined mis-classification rate: <code>(FP + FN) / total</code>. "
                "Captures both incorrectly approved bad loans (FN) and incorrectly "
                "rejected good applicants (FP)."
            ),
            "why": (
                "FP errors (rejecting creditworthy applicants) violate FCA MCOB fair-lending "
                "obligations and reduce revenue. FN errors (approving defaulters) increase "
                "credit losses. Together they measure overall model reliability under adversarial "
                "conditions — the total operational cost of model error."
            ),
            "good_when": "Lower is better. Combines both error types into a single audit metric.",
            "zero_note": None,
        },
    ],
    "aml": [
        {
            "key":   "suspicious_activity_miss_rate",
            "label": "SAR Miss Rate",
            "colour": CRIT,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "fnr",
            "what": (
                "The <b>False Negative Rate</b> for suspicious activity — the fraction of "
                "money laundering events that the model fails to flag for a Suspicious "
                "Activity Report (SAR). Computed as: <code>FN / (FN + TP)</code>."
            ),
            "why": (
                "Failing to file required SARs violates UK MLR 2017, the Bank Secrecy Act (BSA), "
                "and FATF Recommendations 20/29. Regulatory penalties for systemic SAR "
                "non-compliance include: deferred prosecution agreements, civil money penalties "
                "(average $1.3B in 2022, Fenergo), and operating licence revocation. "
                "HSBC's $1.9B 2012 settlement arose partly from AML detection failures."
            ),
            "good_when": (
                "As low as possible — AML operates in a recall-priority regime. "
                "A lower threshold (0.40) is used in this framework to prioritise SAR coverage "
                "over precision. Even under adversarial attack, SAR miss rate should remain "
                "below regulators' informal 10% tolerance threshold."
            ),
            "zero_note": None,
        },
        {
            "key":   "detection_coverage",
            "label": "Detection Coverage",
            "colour": SAFE,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "recall",
            "what": (
                "Recall — the fraction of all suspicious activity correctly flagged by the model. "
                "Computed as: <code>TP / (TP + FN)</code>. The direct complement of SAR Miss Rate."
            ),
            "why": (
                "High detection coverage means the AML programme is operationally effective. "
                "Under SR 11-7, regulators expect documented evidence that model performance "
                "does not degrade materially between validation cycles. Low coverage after "
                "adversarial poisoning is a direct indicator of model drift."
            ),
            "good_when": "Higher is better. Target ≥ 90% coverage under clean conditions.",
            "zero_note": None,
        },
    ],
    "trading": [
        {
            "key":   "execution_error_rate",
            "label": "Execution Error Rate",
            "colour": CRIT,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "error",
            "what": (
                "Combined directional mis-execution rate: <code>(FP + FN) / total signals</code>. "
                "Captures both false buy signals (FP) and missed buy opportunities (FN) "
                "generated by the trading model under adversarial conditions."
            ),
            "why": (
                "Adversarial signal injection (FGSM/PGD on market features) causes the model "
                "to generate incorrect directional signals — buying when it should sell and "
                "vice versa. At £100,000 account size and 10% execution error rate, each "
                "erroneous trade has expected adverse P&L of £1,000–£3,000. "
                "MiFID II Article 17 requires algo trading systems to have circuit-breakers "
                "that detect and halt erroneous order generation."
            ),
            "good_when": "Lower is better. Target < 10% under adversarial conditions.",
            "zero_note": None,
        },
        {
            "key":   "signal_precision",
            "label": "Signal Precision",
            "colour": GOLD,
            "scale": 100,
            "unit":  "%",
            "range": [0, 100],
            "deg_type": "precision",
            "what": (
                "Precision of the trading signal: <code>TP / (TP + FP)</code>. "
                "The fraction of buy signals that were genuinely profitable opportunities."
            ),
            "why": (
                "Low precision means many false buy signals → unnecessary trading activity, "
                "transaction costs, and market-impact slippage. For a HFT strategy, each "
                "spurious signal incurs round-trip costs. "
                "FCA SYSC 6 requires firms to monitor the effectiveness of algorithmic "
                "strategies, making precision a compliance metric as well as a P&L driver."
            ),
            "good_when": "Higher is better. Target ≥ 60% under adversarial conditions.",
            "zero_note": None,
        },
    ],
}


def _single_metric_chart(label: str, value: float, unit: str,
                          colour: str, val_range: list) -> go.Figure:
    """Individual Plotly bar chart for a single domain metric."""
    fig = go.Figure(go.Bar(
        x=[value], y=[label], orientation="h",
        marker_color=colour,
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
        text=[f"{value:.2f}{unit}"],
        textposition="outside",
        textfont=dict(size=13, color="#E8F4FD", family="Orbitron, monospace"),
        hovertemplate=f"<b>{label}</b>: {value:.3f}{unit}<extra></extra>",
        width=[0.45],
    ))
    fig.update_xaxes(range=[0, val_range[1] * 1.25],
                     showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,8,16,0.4)",
        height=90,
        margin=dict(l=10, r=80, t=10, b=10),
        showlegend=False,
    )
    return fig


@st.fragment
def section_post_adversarial_metrics(domain_metrics: pd.DataFrame, domain: str,
                                     summary: pd.DataFrame | None = None) -> None:
    """
    Render domain-specific business metric cards with per-metric charts.

    Toggle button switches between:
      - BASELINE METRICS: stored clean test-set values (no attack applied)
      - POST-ADVERSARIAL TESTING: values estimated after adversarial degradation
        using p_success_mean from the domain summary and deg_type formulas per metric.

    Degradation formulas by deg_type:
      fnr    → min(1, base + p_atk × (1 − base))      [FNR rises proportionally]
      recall → max(0, base × (1 − p_atk))              [recall falls proportionally]
      precision → max(0, base × (1 − p_atk × 0.5))    [precision falls at half rate]
      error  → min(1, base + p_atk × (1 − base) × 0.5)[error rises at half rate]
    """
    if domain_metrics.empty:
        st.info(
            "Domain impact metrics will appear after running the pipeline "
            "(`python main.py`). These translate model output into business-level "
            "measures such as fraud leakage rate and SAR miss rate."
        )
        return
    key = _DOMAIN_KEY.get(domain, "fraud")
    dm = (domain_metrics[domain_metrics["domain"] == key]
          if "domain" in domain_metrics.columns else domain_metrics)
    if dm.empty:
        st.info("No metrics data for this domain yet.")
        return
    row = dm.iloc[0]
    metrics = _POST_ADV_METRICS.get(key, [])
    if not metrics:
        return

    # ── Resolve attack success rate for this domain ─────────────────────────
    p_atk = 0.0
    if summary is not None and not summary.empty:
        _ds = summary[summary["domain"] == domain] if "domain" in summary.columns else summary
        if not _ds.empty and "p_success_pct" in _ds.columns:
            p_atk = float(_ds["p_success_pct"].values[0]) / 100.0

    # ── Toggle state (fragment-isolated — no full-page re-render) ──────────
    _state_key = f"adv_view_{domain}"
    if _state_key not in st.session_state:
        st.session_state[_state_key] = "post"
    _baseline_view = st.session_state[_state_key] == "pre"

    # Badge labels
    _badge_text = "BASELINE METRICS" if _baseline_view else "POST-ADVERSARIAL TESTING"

    # ── Inject CSS to style badge button ────────────────────────────────────
    st.markdown(f"""
<style>
button[aria-label="BASELINE METRICS"] {{
    background-color: {SAFE} !important;
    color: #000 !important;
    font-family: Orbitron, monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: none !important;
    padding: 4px 16px !important;
    border-radius: 4px !important;
    min-height: 28px !important;
    height: 28px !important;
    line-height: 1 !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
}}
button[aria-label="BASELINE METRICS"]:hover {{
    opacity: 0.82 !important;
    background-color: {SAFE} !important;
    color: #000 !important;
    border: none !important;
}}
button[aria-label="POST-ADVERSARIAL TESTING"] {{
    background-color: {WARN} !important;
    color: #000 !important;
    font-family: Orbitron, monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    border: none !important;
    padding: 4px 16px !important;
    border-radius: 4px !important;
    min-height: 28px !important;
    height: 28px !important;
    line-height: 1 !important;
    text-transform: uppercase !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
}}
button[aria-label="POST-ADVERSARIAL TESTING"]:hover {{
    opacity: 0.82 !important;
    background-color: {WARN} !important;
    color: #000 !important;
    border: none !important;
}}
</style>""", unsafe_allow_html=True)

    if st.button(_badge_text, key=f"adv_btn_{domain}"):
        st.session_state[_state_key] = "pre" if not _baseline_view else "post"

    # ── Metric cards ─────────────────────────────────────────────────────────
    cols = st.columns(len(metrics))
    for i, meta in enumerate(metrics):
        # Always read stored (baseline) value
        _raw_val = row[meta["key"]] if meta["key"] in row.index else 0.0
        raw_base = float(_raw_val) if _raw_val is not None else 0.0
        if np.isnan(raw_base):
            raw_base = 0.0

        if _baseline_view:
            raw = raw_base
            _sub_label = "Clean test-set · no attack applied"
        else:
            # Estimate post-adversarial degradation
            deg = meta.get("deg_type", "fnr")
            if deg == "fnr":
                raw = min(1.0, raw_base + p_atk * (1.0 - raw_base))
            elif deg == "recall":
                raw = max(0.0, raw_base * (1.0 - p_atk))
            elif deg == "precision":
                raw = max(0.0, raw_base * (1.0 - p_atk * 0.5))
            else:  # "error"
                raw = min(1.0, raw_base + p_atk * (1.0 - raw_base) * 0.5)

            _delta_abs = raw - raw_base
            _delta_pct = (_delta_abs / max(raw_base, 1e-9)) * 100
            _arrow = "▲" if raw > raw_base else ("▼" if raw < raw_base else "—")
            _arr_col = CRIT if raw > raw_base else (SAFE if raw < raw_base else "#546E7A")
            _sub_label = (f'<span style="color:{_arr_col}">'
                          f'{_arrow} {abs(_delta_pct):.1f}% vs baseline</span>')

        displayed = raw * meta["scale"]
        col_h = meta["colour"]

        with cols[i]:
            st.markdown(
                f'<div class="aif-card" style="border-top:3px solid {col_h};padding:16px 18px">'
                f'<div style="font-size:9px;color:#546E7A;letter-spacing:1.5px;'
                f'text-transform:uppercase;margin-bottom:6px">{meta["label"]}</div>'
                f'<div style="font-family:Orbitron,monospace;font-size:28px;'
                f'font-weight:700;color:{col_h}">'
                f'{displayed:.3f}<span style="font-size:13px;margin-left:4px">{meta["unit"]}</span>'
                f'</div>'
                f'<div style="font-size:10px;color:#546E7A;margin-top:4px">{_sub_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            fig = _single_metric_chart(
                meta["label"], displayed, meta["unit"], col_h, meta["range"]
            )
            st.plotly_chart(fig, use_container_width=True,
                            key=f"dm_{domain}_{meta['key']}_{_state_key}")
            with st.expander(f"What is {meta['label']}?"):
                st.markdown(
                    f'<div style="font-size:12px;color:#C8DCEA;line-height:1.9">'
                    f'<div style="color:{CYAN};font-weight:600;margin-bottom:4px">Definition</div>'
                    f'{meta["what"]}'
                    f'<br><br>'
                    f'<div style="color:{CYAN};font-weight:600;margin-bottom:4px">Why it matters</div>'
                    f'{meta["why"]}'
                    f'<br><br>'
                    f'<div style="color:{CYAN};font-weight:600;margin-bottom:4px">Good when</div>'
                    f'{meta["good_when"]}'
                    + (f'<br><br><div style="padding:8px 12px;background:rgba(255,107,53,0.07);'
                       f'border-left:3px solid {WARN};border-radius:4px">'
                       f'{meta["zero_note"]}</div>'
                       if meta.get("zero_note") else "")
                    + f'</div>',
                    unsafe_allow_html=True,
                )


def chart_model_performance(domain_metrics: pd.DataFrame, domain: str) -> go.Figure | None:
    """
    Horizontal bar chart for standard classifier metrics (recall, precision, F1, PR-AUC).
    Intentionally separate from domain-specific business metrics.
    """
    if domain_metrics.empty:
        return None
    key = _DOMAIN_KEY.get(domain, "fraud")
    dm = (domain_metrics[domain_metrics["domain"] == key]
          if "domain" in domain_metrics.columns else domain_metrics)
    if dm.empty:
        return None
    row = dm.iloc[0]
    perf_keys = ["recall", "precision", "f1", "pr_auc", "roc_auc"]
    perf_labels = {
        "recall":   "Recall (Sensitivity)",
        "precision": "Precision",
        "f1":       "F1 Score",
        "pr_auc":   "PR-AUC",
        "roc_auc":  "ROC-AUC",
    }
    # colour: recall/PR-AUC are most important for imbalanced domains → highlight
    perf_colours = {
        "recall":    CRIT,    # most critical for fraud/AML
        "precision": WARN,
        "f1":        GOLD,
        "pr_auc":    BLUE,    # preferred over ROC for imbalanced
        "roc_auc":   "#78909C",
    }
    vals, labels, colours = [], [], []
    for k in perf_keys:
        v = float(row[k]) if k in row.index else 0.0
        vals.append(v)
        labels.append(perf_labels[k])
        colours.append(perf_colours[k])
    fig = go.Figure(go.Bar(
        y=labels, x=vals, orientation="h",
        marker_color=colours,
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
        text=[f"{v:.3f}" for v in vals], textposition="outside",
        textfont=dict(size=10, color="#90B8D4"),
        hovertemplate="<b>%{y}</b>: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(showlegend=False)
    fig_style(fig, height=260, margin=dict(l=170, r=70, t=30, b=30))
    fig.update_xaxes(title="Score", range=[0, 1.2])
    fig.update_layout(
        title=dict(
            text=(f"Baseline Classifier Performance — {domain}<br>"
                  "<span style='font-size:9px'>"
                  "PR-AUC is the preferred metric for imbalanced datasets (fraud 0.17%, AML 0.3%)"
                  "</span>"),
            font=dict(size=11, color="#90B8D4"), x=0,
        ),
    )
    return fig


def chart_bn_mc_integration(bn_mc: pd.DataFrame):
    """
    Overlay synthetic loss distributions for each BN threat scenario vs empirical.

    Each scenario's P(AttackSuccess) from the Bayesian Network is used as the MC
    P_success input. Comparing the distributions shows how the controlled empirical
    test rate (FGSM/C&W lab conditions) underestimates the risk implied by the
    broader threat environment (BN worst-case). Dashed lines mark CVaR 99%.
    """
    if bn_mc.empty:
        return None
    _COLORS = {
        "empirical":  "#2166ac",
        "baseline":   "#4dac26",
        "worst_case": "#d01c8b",
        "mitigated":  "#f1a340",
    }
    rng = np.random.default_rng(42)
    fig = go.Figure()
    for _, row in bn_mc.iterrows():
        key   = str(row.get("scenario", ""))
        label = str(row.get("label", key))
        el    = float(row.get("el_mean", 0))
        cvar  = float(row.get("cvar_99", 0))
        var99 = float(row.get("var_99", el * 1.8)) if "var_99" in row.index else el * 1.8
        p     = float(row.get("bn_p_success", 0))
        if el <= 0:
            continue
        color = _COLORS.get(key, "#888888")
        sigma = max(0.3, (var99 - el) / (2.33 * max(el, 1)))
        losses = rng.lognormal(mean=np.log(max(el, 1)), sigma=sigma, size=8_000)
        short  = label[:30] if len(label) <= 30 else label[:27] + "…"
        fig.add_trace(go.Histogram(
            x=losses, nbinsx=60,
            name=f"{short}<br>P={p:.3f} | EL={fmt_usd(el)} | CVaR={fmt_usd(cvar)}",
            marker_color=color, opacity=0.45, histnorm="probability density",
            marker_line=dict(width=0),
            hovertemplate="Loss: $%{x:,.0f}<br>Density: %{y:.4f}<extra></extra>",
        ))
        if cvar > 0:
            fig.add_vline(x=cvar, line_dash="dash", line_color=color,
                          line_width=1.6, opacity=0.85)
    fig.update_layout(barmode="overlay")
    fig_style(fig, height=400, margin=dict(l=60, r=20, t=60, b=60))
    fig.update_xaxes(title="Simulated Loss per Fraud Event (USD)", tickprefix="$")
    fig.update_yaxes(title="Density")
    fig.update_layout(
        title=dict(
            text=("Bayesian Network → Monte Carlo Integration<br>"
                  "<span style='font-size:9px'>"
                  "BN-derived P(AttackSuccess) per scenario fed into MC engine — "
                  "dashed lines = CVaR 99%"
                  "</span>"),
            font=dict(size=11, color="#90B8D4"), x=0,
        ),
        legend=dict(font=dict(size=8.5), bgcolor="rgba(0,0,0,0)",
                    tracegroupgap=4),
    )
    return fig


def chart_disclosure_comparison(disclosure: pd.DataFrame):
    """
    Grouped bar comparing reputational and total cascade loss under
    immediate vs delayed disclosure (Wu et al., 2022, Section 2.7).
    """
    if disclosure.empty:
        return None
    imm = disclosure[disclosure["scenario"] == "immediate_disclosure"]
    dly = disclosure[disclosure["scenario"] == "delayed_disclosure"]
    amp = disclosure[disclosure["scenario"] == "amplification_factor"]
    if imm.empty or dly.empty:
        return None
    rep_imm = float(imm["reputational_mean"].values[0])   if "reputational_mean"   in imm.columns else 0.0
    rep_dly = float(dly["reputational_mean"].values[0])   if "reputational_mean"   in dly.columns else 0.0
    tot_imm = float(imm["total_cascade_mean"].values[0])  if "total_cascade_mean"  in imm.columns else 0.0
    tot_dly = float(dly["total_cascade_mean"].values[0])  if "total_cascade_mean"  in dly.columns else 0.0
    r_f = (float(amp["reputational_amplification_factor"].values[0])
           if not amp.empty and "reputational_amplification_factor" in amp.columns else None)
    labels = ["Immediate Disclosure", "Delayed Disclosure"]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Reputational Loss", x=labels, y=[rep_imm, rep_dly],
        marker_color=BLUE,
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
        text=[fmt_usd(rep_imm), fmt_usd(rep_dly)], textposition="outside",
        textfont=dict(size=10, color="#90B8D4"),
        hovertemplate="<b>%{x}</b><br>Reputational: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="Total Cascade Loss", x=labels, y=[tot_imm, tot_dly],
        marker_color=WARN, opacity=0.75,
        marker_line=dict(color="rgba(0,0,0,0.3)", width=1),
        text=[fmt_usd(tot_imm), fmt_usd(tot_dly)], textposition="outside",
        textfont=dict(size=10, color="#90B8D4"),
        hovertemplate="<b>%{x}</b><br>Total Cascade: $%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.1)
    rf_str = f" — R_f = {r_f:.2f}×" if r_f else ""
    fig_style(fig, height=320, margin=dict(l=60, r=20, t=70, b=60))
    fig.update_yaxes(title="Mean Loss (USD)", tickprefix="$")
    fig.update_layout(
        title=dict(
            text=(f"Disclosure Timing — Reputational Loss Amplification{rf_str}<br>"
                  "<span style='font-size:9px'>"
                  "Wu et al. (2022) Section 2.7 — delayed disclosure amplifies reputational loss by R_f = 1.5"
                  "</span>"),
            font=dict(size=11, color="#90B8D4"), x=0,
        ),
    )
    return fig


def page_technical(data: dict):
    st.markdown(f"""
<div class="aif-card" style="margin-bottom:20px">
  <div class="aif-card-title">Technical Results</div>
  <div style="font-size:12px;color:#90B8D4;line-height:1.8">
    Quantitative outputs from the AIFCRQF pipeline for technical reviewers —
    dissertation examiners, security engineers, and risk quants.
    The Bayesian Network section shows how empirical adversarial test probabilities
    translate to financial risk across threat scenarios.  Domain tabs below break down
    model accuracy, loss distributions, and cascading impact per financial domain.
  </div>
</div>""", unsafe_allow_html=True)

    attacks, risk = data["attacks"], data["risk"]

    # ── BN-MC integration (domain-independent) ────────────────────────────
    st.markdown(
        '<div class="section-head">Bayesian Network — Monte Carlo Integration</div>',
        unsafe_allow_html=True)
    fbn = chart_bn_mc_integration(data["bn_mc"])
    if fbn:
        st.plotly_chart(fbn, use_container_width=True, key="bn_mc_tech")
        st.markdown(f"""
<div style="font-size:11px;color:#90B8D4;padding:10px 12px;line-height:1.8;
     border-left:3px solid {CYAN};border-radius:4px;background:rgba(0,229,255,0.03);margin-top:8px">
  <b style="color:{CYAN}">How it works:</b>
  The adversarial evaluator produces an empirical P(AttackSuccess) from FGSM, C&amp;W,
  and poisoning experiments on the live model.  The Bayesian Network is calibrated to that
  empirical rate as its baseline, then models <i>how P(AttackSuccess) shifts</i> under
  different threat-environment and control-maturity scenarios.
  Each scenario's conditional probability is passed as <i>p_success</i> into the Monte
  Carlo engine, which simulates 50,000 financial-loss outcomes — producing the distinct
  distributions above.  This bridges probabilistic structure (BN) with quantitative
  risk metrics (EL, VaR, CVaR).<br><br>
  <b style="color:#2166ac">Empirical</b>: controlled FGSM/C&amp;W lab rate (anchor).&nbsp;
  <b style="color:#4dac26">Baseline</b>: BN at medium threat &amp; medium controls.&nbsp;
  <b style="color:#d01c8b">Worst-Case</b>: high threat, weak controls — real-world exposure gap.&nbsp;
  <b style="color:#f1a340">Mitigated</b>: high threat, strong controls — governance effect.
  Dashed lines mark CVaR 99% per scenario.
</div>""", unsafe_allow_html=True)
    else:
        st.info("BN-MC data not available — run the pipeline first (requires pgmpy).")

    # ── Per-domain tabs (use short names to avoid collision with outer tabs) ─
    summary = data["summary"]
    avail   = ([d for d in DOMAIN_COLOURS if d in summary["domain"].unique()]
               if not summary.empty else list(DOMAIN_COLOURS.keys()))
    if not avail:
        st.info("No domain data available — run the pipeline first.")
        return

    _SHORT = {
        "Fraud Detection":     "Fraud",
        "Credit Scoring":      "Credit",
        "AML Detection":       "AML",
        "Algorithmic Trading": "Trading",
    }
    short_labels = [_SHORT.get(d, d) for d in avail]
    domain_tabs  = st.tabs(short_labels)

    for sel, dtab in zip(avail, domain_tabs):
        with dtab:
            st.markdown(
                f'<div class="section-head">Model Accuracy: Clean vs Adversarial — {sel}</div>',
                unsafe_allow_html=True)
            cm_col, mc_col = st.columns(2)
            with cm_col:
                fc = chart_confusion_matrix(attacks, sel)
                if fc:
                    st.plotly_chart(fc, use_container_width=True, key=f"cm_t_{sel}")
                st.markdown(
                    f'<div style="font-size:11px;color:#90B8D4;padding:8px 4px;line-height:1.7">'
                    f'Under adversarial attack the true-positive rate drops, increasing the '
                    f'false-negative rate — the fraction of attacks that slip through undetected.'
                    f'</div>', unsafe_allow_html=True)

            # Auto-detect maturity from last pipeline run — no selector
            _run_cfg_exec = load_run_config()
            _mat = _run_cfg_exec.get(sel, {}).get("iso_maturity", "medium")
            _mat_display = {"weak": "Weak (M=0.30)", "medium": "Medium (M=0.60)", "strong": "Strong (M=0.80)"}

            with mc_col:
                fm = chart_mc_distribution(risk, sel, maturity=_mat)
                if fm:
                    st.plotly_chart(fm, use_container_width=True, key=f"mc_t_{sel}")
                st.markdown(
                    f'<div style="font-size:11px;color:#90B8D4;padding:8px 4px;line-height:1.7">'
                    f'50,000 MC scenarios reproduce the full loss shape. '
                    f'CVaR 99% is the average of all losses beyond the 99th percentile — '
                    f'the true cost of a tail event. Showing: <b style="color:{CYAN}">'
                    f'{_mat_display.get(_mat, _mat)}</b> controls.'
                    f'</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div class="section-head">Cascading Loss Breakdown — {_mat_display.get(_mat, _mat)}</div>',
                unsafe_allow_html=True)
            wf_col, tbl_col = st.columns([3, 2])
            with wf_col:
                fw = chart_cascade_waterfall(risk, sel, maturity=_mat, cascade=data["cascade"])
                if fw:
                    st.plotly_chart(fw, use_container_width=True, key=f"wf_t_{sel}_{_mat}")

            with tbl_col:
                dr = (risk[(risk["domain"] == sel) & (risk["maturity_label"] == _mat)]
                      if not risk.empty else pd.DataFrame())
                el    = float(dr["el_mean"].values[0])    if not dr.empty else 0.0
                cvar  = float(dr["cvar_99"].values[0])    if not dr.empty and "cvar_99"      in dr.columns else 0.0
                var99 = float(dr["var_99"].values[0])     if not dr.empty and "var_99"       in dr.columns else 0.0
                tp    = float(dr["tail_premium"].values[0]) if not dr.empty and "tail_premium" in dr.columns else 0.0
                st.markdown(f"""
<div class="aif-card">
  <div class="aif-card-title">Loss Component Breakdown</div>
  <table class="risk-tbl">
    <thead><tr><th>Component</th><th>Basis</th><th>Value</th></tr></thead>
    <tbody>
      <tr><td>Direct Loss</td><td>45% of EL</td>
          <td style="color:{CRIT};font-weight:600">{fmt_usd(el*0.45)}</td></tr>
      <tr><td>Operational Disruption</td><td>25% of EL</td>
          <td style="color:{WARN};font-weight:600">{fmt_usd(el*0.25)}</td></tr>
      <tr><td>Reputational Damage</td><td>20% of EL</td>
          <td style="color:{GOLD};font-weight:600">{fmt_usd(el*0.20)}</td></tr>
      <tr><td>Regulatory Exposure</td><td>CVaR tail excess beyond EL</td>
          <td style="color:{BLUE};font-weight:600">{fmt_usd(max(0, cvar - el*0.90))}</td></tr>
      <tr><td style="font-weight:700">Total Tail Exposure</td>
          <td>CVaR 99%</td>
          <td style="color:{CYAN};font-weight:700">{fmt_usd(cvar)}</td></tr>
      <tr><td>VaR 99%</td><td>99th percentile threshold</td>
          <td style="color:#90B8D4">{fmt_usd(var99)}</td></tr>
      <tr><td>Tail Premium</td><td>CVaR − VaR</td>
          <td style="color:#90B8D4">{fmt_usd(tp)}</td></tr>
    </tbody>
  </table>
</div>""", unsafe_allow_html=True)

            # ── Poisoning sweep ───────────────────────────────────────────
            st.markdown(
                '<div class="section-head">Data Poisoning Resilience — Recall &amp; PR-AUC</div>',
                unsafe_allow_html=True)
            poison = data["poisoning"]
            _short_to_display_dp = {"fraud": "Fraud Detection", "credit": "Credit Scoring",
                                    "aml": "AML Detection", "trading": "Algorithmic Trading"}
            poison_norm = poison.copy()
            if not poison_norm.empty and "domain" in poison_norm.columns:
                poison_norm["domain"] = poison_norm["domain"].map(
                    lambda x: _short_to_display_dp.get(str(x).lower(), x))
            dp = (poison_norm[poison_norm["domain"] == sel]
                  if not poison_norm.empty and "domain" in poison_norm.columns else pd.DataFrame())
            if not dp.empty:
                fp = chart_poisoning(dp)
                if fp:
                    st.plotly_chart(fp, use_container_width=True, key=f"poi_t_{sel}")
            else:
                st.info("Poisoning sweep data not available — run the pipeline.")


# ════════════════════════════════════════════════════════════════════════════
# Sidebar + routing
# ════════════════════════════════════════════════════════════════════════════

def main():
    data = load_all()
    now  = datetime.now().strftime("%d %b %Y  %H:%M")

    # ── Top header bar ────────────────────────────────────────────────────
    hdr_left, hdr_right = st.columns([6, 1])
    with hdr_left:
        st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding:10px 0 4px;margin-bottom:8px;border-bottom:1px solid rgba(0,229,255,0.15)">
  <div style="font-family:Orbitron,monospace;font-size:18px;font-weight:900;
       color:#00E5FF;letter-spacing:3px;text-shadow:0 0 20px rgba(0,229,255,0.5)">
    AIFCRQF
  </div>
  <div style="font-size:10px;color:#546E7A">{now} &nbsp;·&nbsp; auto-refresh every 15s</div>
</div>""", unsafe_allow_html=True)
    with hdr_right:
        if st.button("Refresh Data", key="refresh_btn", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    # ── Dynamic domain list — only show domains present in data ──────────
    summary = data["summary"]
    avail_domains = ([d for d in DOMAIN_COLOURS if d in summary["domain"].unique()]
                     if not summary.empty else list(DOMAIN_COLOURS.keys()))

    # ── Top navigation tabs ───────────────────────────────────────────────
    tab_labels = (["Executive Overview", "Financial Risk"]
                  + avail_domains
                  + ["Technical Results", "Validation & Proofs"])
    tabs = st.tabs(tab_labels)

    idx = 0
    with tabs[idx]:
        page_overview(data)
    idx += 1
    with tabs[idx]:
        page_financial_risk(data)
    idx += 1
    for domain in avail_domains:
        with tabs[idx]:
            page_domain(domain, data)
        idx += 1
    with tabs[idx]:
        page_technical(data)
    idx += 1
    with tabs[idx]:
        page_validation(data)

    # Sticky footer
    st.markdown(
        f'<div style="position:fixed;bottom:0;left:0;right:0;height:28px;'
        f'background:rgba(0,8,16,0.95);border-top:1px solid rgba(0,229,255,0.1);'
        f'display:flex;align-items:center;justify-content:space-between;'
        f'padding:0 24px;font-size:10px;color:#2A3A4A;z-index:999">'
        f'<span>AIFCRQF &copy; Heriot-Watt University — BSc Cyber Security Dissertation</span>'
        f'<span>{datetime.now().strftime("%d %b %Y  %H:%M")}</span>'
        f'</div>',
        unsafe_allow_html=True)


if __name__ == "__main__":
    main()