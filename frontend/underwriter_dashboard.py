"""
Gigit — Underwriter Dashboard
====================================
A polished Streamlit web app that lets a bank underwriter:

  1. Select or enter a Gig Worker ID.
  2. View the worker's behavioural metrics.
  3. Run the ONNX credit-risk model locally (no cloud dependency).
  4. See the dynamic risk score on an interactive gauge dial.
  5. Explore charts comparing income volatility vs. expense ratio.

Launch:
    python -m streamlit run frontend/underwriter_dashboard.py

Requires:
    - data/mock_gig_worker_data.csv  (run generate_mock_data.py)
    - ml_engine/risk_model.onnx      (run amd_npu_optimizer.py)
    - ml_engine/scaler_params.json   (created during training)
"""

import json
import os
import sys

import numpy as np
import onnxruntime as ort
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Add project root to path so we can import our utilities
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from utils.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    get_all_worker_ids,
    get_worker_features,
    get_worker_summary,
    load_data,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ONNX_MODEL_PATH = os.path.join(PROJECT_ROOT, "ml_engine", "risk_model.onnx")
SCALER_PATH = os.path.join(PROJECT_ROOT, "ml_engine", "scaler_params.json")

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Gigit  |  Credit Scoring Engine",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — clean dark theme with Google Fonts (Inter + JetBrains Mono)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Google Fonts ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ---- Global ---- */
    .stApp {
        background: linear-gradient(160deg, #0b0d17 0%, #111827 40%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
    }
    /* Keep main content stable when sidebar is toggled */
    .stMainBlockContainer {
        max-width: 1200px;
        margin: 0 auto;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    /* Apply font to text elements only — avoid overriding icon fonts */
    .stMarkdown, .stText, p, span:not([data-testid]), li, label, .stCaption {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827 0%, #0b0d17 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    section[data-testid="stSidebar"] .stMarkdown, 
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span:not([data-testid]),
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] label {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 18px 22px;
        backdrop-filter: blur(12px);
        transition: border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(99, 179, 237, 0.25);
    }
    div[data-testid="stMetric"] label {
        color: #94a3b8 !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #f1f5f9 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 500 !important;
        font-size: 1.35rem !important;
    }

    /* ---- Headers ---- */
    h1 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }
    h2 {
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }
    h3 {
        color: #cbd5e1 !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 1.05rem !important;
    }

    /* ---- Body text ---- */
    p, span, li {
        color: #94a3b8;
    }

    /* ---- Divider ---- */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
        margin: 1.5rem 0 !important;
    }

    /* ---- AMD badge ---- */
    .amd-badge {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 5px 14px;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        display: inline-block;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    /* ---- Section label ---- */
    .section-label {
        color: #64748b;
        font-family: 'Inter', sans-serif;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* ---- Recommendation cards ---- */
    .recommendation {
        font-family: 'Inter', sans-serif;
        font-size: 0.88rem;
        font-weight: 500;
        padding: 14px 18px;
        border-radius: 10px;
        margin-top: 8px;
        line-height: 1.5;
    }
    .rec-approve {
        background: rgba(34, 197, 94, 0.08);
        border: 1px solid rgba(34, 197, 94, 0.2);
        color: #86efac;
    }
    .rec-caution {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.2);
        color: #fde047;
    }
    .rec-decline {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        color: #fca5a5;
    }

    /* ---- Footer ---- */
    .footer-text {
        text-align: center;
        color: #475569;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 400;
        letter-spacing: 0.04em;
        padding: 12px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helper: load ONNX session (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_onnx_session():
    """
    Load the ONNX model into an InferenceSession.

    For this MVP we use the default CPU Execution Provider.
    On AMD Ryzen AI hardware, replace with:
        providers=["VitisAIExecutionProvider"]
    to route inference to the on-chip NPU for sub-ms latency.
    """
    if not os.path.exists(ONNX_MODEL_PATH):
        st.error(
            "ONNX model not found.  Please run:\n"
            "```\npython ml_engine/train_model.py\n"
            "python ml_engine/amd_npu_optimizer.py\n```"
        )
        st.stop()
    return ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])


@st.cache_data
def load_scaler_params():
    """Load the StandardScaler mean/scale saved during training."""
    if not os.path.exists(SCALER_PATH):
        st.error("Scaler parameters not found. Please run training first.")
        st.stop()
    with open(SCALER_PATH) as f:
        return json.load(f)


def scale_features(raw: np.ndarray, params: dict) -> np.ndarray:
    """Apply StandardScaler transform using saved mean & scale."""
    mean = np.array(params["mean"], dtype=np.float32)
    scale = np.array(params["scale"], dtype=np.float32)
    return ((raw - mean) / scale).astype(np.float32)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="amd-badge">Powered by AMD Ryzen AI</div>',
        unsafe_allow_html=True,
    )
    st.title("Gigit")
    st.caption("Dynamic Cash-Flow Credit Scoring for Gig Workers")
    st.markdown("---")

    # Load data
    df = load_data()
    worker_ids = get_all_worker_ids(df)

    st.markdown(
        '<div class="section-label">Select Worker</div>',
        unsafe_allow_html=True,
    )
    selected_id = st.selectbox(
        "Worker ID",
        options=worker_ids,
        index=41,  # default to GIG_042
        help="Choose a gig worker to analyse.",
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-label">How It Works</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "1. Extracts behavioural metrics from bank data\n"
        "2. Runs a neural network **locally** via ONNX Runtime\n"
        "3. Outputs a dynamic risk score (0 -- 100)\n"
    )
    st.markdown("---")
    st.caption("MVP Prototype  /  AMD Slingshot 2026")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("## Underwriter Risk Assessment")
st.markdown(f"Analysing worker **{selected_id}** — all inference runs **locally on-device**.")

# Load model & scaler
session = load_onnx_session()
scaler_params = load_scaler_params()

# Get worker data
try:
    raw_features = get_worker_features(selected_id, df)
    summary = get_worker_summary(selected_id, df)
except ValueError as e:
    st.error(str(e))
    st.stop()

# Scale & run inference
scaled_features = scale_features(raw_features, scaler_params)
input_name = session.get_inputs()[0].name
onnx_result = session.run(None, {input_name: scaled_features})
risk_probability = float(onnx_result[0][0][0])       # 0.0 - 1.0
risk_score = round(risk_probability * 100, 1)         # 0 - 100

# Determine risk band
if risk_score < 30:
    risk_band, band_color = "LOW RISK", "#22c55e"
elif risk_score < 60:
    risk_band, band_color = "MEDIUM RISK", "#eab308"
else:
    risk_band, band_color = "HIGH RISK", "#ef4444"

# ---- Row 1: Key metrics ---------------------------------------------------
st.markdown(
    '<div class="section-label">Worker Profile</div>',
    unsafe_allow_html=True,
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Monthly Income", f"Rs. {summary['average_monthly_income']:,.0f}")
c2.metric("Income Volatility", f"{summary['income_volatility_percentage']:.1f}%")
c3.metric("Expense / Income", f"{summary['expense_to_income_ratio']:.2f}")
c4.metric("Tenure", f"{int(summary['work_tenure_months'])} months")

st.markdown("---")

# ---- Row 2: Gauge + charts ------------------------------------------------
col_gauge, col_charts = st.columns([1, 1.3])

# -- Gauge dial (Plotly) -----------------------------------------------------
with col_gauge:
    st.markdown(
        '<div class="section-label">Dynamic Risk Score</div>',
        unsafe_allow_html=True,
    )
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={
                "suffix": " / 100",
                "font": {"size": 42, "color": "#f1f5f9", "family": "Inter"},
            },
            title={
                "text": risk_band,
                "font": {"size": 18, "color": band_color, "family": "Inter"},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 2,
                    "tickcolor": "#334155",
                    "tickfont": {"color": "#64748b", "family": "Inter"},
                },
                "bar": {"color": band_color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.10)"},
                    {"range": [30, 60], "color": "rgba(234,179,8,0.10)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.10)"},
                ],
                "threshold": {
                    "line": {"color": "#f1f5f9", "width": 3},
                    "thickness": 0.8,
                    "value": risk_score,
                },
            },
        )
    )
    gauge.update_layout(
        font=dict(family="Inter"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        margin=dict(t=60, b=20, l=30, r=30),
    )
    st.plotly_chart(gauge, width="stretch")

    # Loan recommendation
    if risk_score < 30:
        st.markdown(
            '<div class="recommendation rec-approve">'
            "<strong>Recommendation:</strong> Approve micro-loan with standard terms."
            "</div>",
            unsafe_allow_html=True,
        )
    elif risk_score < 60:
        st.markdown(
            '<div class="recommendation rec-caution">'
            "<strong>Recommendation:</strong> Approve with enhanced monitoring and lower limit."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="recommendation rec-decline">'
            "<strong>Recommendation:</strong> Decline or require additional collateral."
            "</div>",
            unsafe_allow_html=True,
        )

# -- Bar / radar charts ------------------------------------------------------
with col_charts:
    st.markdown(
        '<div class="section-label">Behavioural Metrics Breakdown</div>',
        unsafe_allow_html=True,
    )

    # Bar chart — income volatility vs expense ratio (normalised 0-100 scale)
    vol_pct = summary["income_volatility_percentage"]
    exp_pct = summary["expense_to_income_ratio"] * 100  # scale to %
    tenure_pct = min(summary["work_tenure_months"] / 120 * 100, 100)
    income_pct = min(summary["average_monthly_income"] / 80_000 * 100, 100)

    bar_fig = go.Figure()
    categories = ["Monthly Income", "Income Volatility", "Expense Ratio", "Work Tenure"]
    values = [income_pct, vol_pct, exp_pct, tenure_pct]
    colors = ["#38bdf8", "#fb923c", "#f87171", "#4ade80"]

    bar_fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=colors,
                line=dict(width=0),
            ),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(color="#cbd5e1", size=12, family="JetBrains Mono"),
        )
    )
    bar_fig.update_layout(
        font=dict(family="Inter"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            title="Percentile / Scale",
            gridcolor="rgba(255,255,255,0.04)",
            color="#64748b",
            range=[0, max(values) * 1.25],
        ),
        xaxis=dict(color="#94a3b8"),
        height=340,
        margin=dict(t=20, b=40, l=50, r=20),
        showlegend=False,
    )
    st.plotly_chart(bar_fig, width="stretch")

st.markdown("---")

# ---- Row 3 — Radar chart (spider) for holistic view -----------------------
st.markdown(
    '<div class="section-label">Holistic Risk Profile</div>',
    unsafe_allow_html=True,
)
radar_cats = [
    "Income Stability",
    "Spending Discipline",
    "Tenure Strength",
    "Earning Power",
    "Overall Safety",
]
# Invert volatility & expense so higher = better
radar_vals = [
    100 - vol_pct,                       # Income stability (inverse of volatility)
    100 - exp_pct,                       # Spending discipline
    tenure_pct,                          # Tenure
    income_pct,                          # Earning power
    100 - risk_score,                    # Overall safety
]
radar_vals += [radar_vals[0]]            # close the polygon
radar_cats += [radar_cats[0]]

radar_fig = go.Figure()
radar_fig.add_trace(
    go.Scatterpolar(
        r=radar_vals,
        theta=radar_cats,
        fill="toself",
        fillcolor="rgba(56,189,248,0.08)",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=5, color="#38bdf8"),
    )
)
radar_fig.update_layout(
    font=dict(family="Inter"),
    polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.05)",
            color="#64748b",
        ),
        angularaxis=dict(color="#94a3b8"),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    height=420,
    margin=dict(t=40, b=40, l=80, r=80),
)
st.plotly_chart(radar_fig, width="stretch")

# ---- Footer ----------------------------------------------------------------
st.markdown("---")
st.markdown(
    '<div class="footer-text">'
    "Gigit  &middot;  Powered by AMD Ryzen AI  &middot;  "
    "Local &amp; Secure Inference  &middot;  No data leaves this device"
    "</div>",
    unsafe_allow_html=True,
)
