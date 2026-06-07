"""Machine Failure Predictor — Streamlit app."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from model import (
    FEATURE_COLS,
    TARGET,
    TYPE_LABELS,
    classification_metrics,
    feature_ranges,
    predict,
    roc_data,
    sample_presets,
    train_models,
)

DATA_PATH = Path(__file__).parent / "ai.csv"

# ── Design tokens ─────────────────────────────────────────────────────────────
C = {
    "bg": "#0b0f19",
    "surface": "#131a2b",
    "card": "#1a2236",
    "border": "#2a3550",
    "text": "#e8edf7",
    "muted": "#8b9bb8",
    "accent": "#22d3ee",
    "accent2": "#818cf8",
    "green": "#34d399",
    "red": "#f87171",
    "amber": "#fbbf24",
    "lr": "#22d3ee",
    "knn": "#a78bfa",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color=C["text"], size=13),
    title_font=dict(size=16, color=C["text"]),
    margin=dict(l=24, r=24, t=56, b=24),
    colorway=[C["accent"], C["accent2"], C["green"], C["red"], C["amber"]],
)

NAV_ITEMS = [
    ("Overview", "🏠", "Dashboard & insights"),
    ("Predict", "🔮", "Live failure scoring"),
    ("Model Analytics", "📊", "Performance metrics"),
    ("Data Explorer", "🗂️", "Telemetry browser"),
    ("Batch scoring", "📤", "Bulk CSV upload"),
]

SENSOR_META = {
    "Type": ("🏭", "Product tier", "L / M / H quality class"),
    "Air temperature [K]": ("🌡️", "Air temp", "Ambient operating temperature"),
    "Process temperature [K]": ("🔥", "Process temp", "Core process heat level"),
    "Rotational speed [rpm]": ("⚡", "RPM", "Spindle rotational speed"),
    "Torque [Nm]": ("🔧", "Torque", "Applied rotational force"),
    "Tool wear [min]": ("⏱️", "Tool wear", "Cumulative tool usage time"),
}

INPUT_KEYS = {
    "Type": "inp_type",
    "Air temperature [K]": "inp_air",
    "Process temperature [K]": "inp_process",
    "Rotational speed [rpm]": "inp_rpm",
    "Torque [Nm]": "inp_torque",
    "Tool wear [min]": "inp_wear",
}

st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg: {C['bg']};
        --surface: {C['surface']};
        --card: {C['card']};
        --border: {C['border']};
        --text: {C['text']};
        --muted: {C['muted']};
        --accent: {C['accent']};
        --accent2: {C['accent2']};
        --green: {C['green']};
        --red: {C['red']};
    }}

    .stApp {{
        background: radial-gradient(ellipse 80% 60% at 10% -10%, rgba(34,211,238,0.08), transparent),
                    radial-gradient(ellipse 60% 50% at 90% 0%, rgba(129,140,248,0.07), transparent),
                    var(--bg);
        font-family: 'DM Sans', sans-serif;
    }}

    .block-container {{
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }}

    h1, h2, h3, h4, h5, h6, p, label, span {{
        font-family: 'DM Sans', sans-serif !important;
    }}

    /* Hide Streamlit chrome that breaks the layout */
    section[data-testid="stSidebar"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="collapsedControl"],
    header[data-testid="stHeader"],
    footer {{
        display: none !important;
    }}
    #MainMenu {{
        visibility: hidden;
    }}
    [data-testid="stAppViewContainer"] {{
        margin-left: 0 !important;
        padding-left: 0 !important;
    }}
    section.main {{
        width: 100% !important;
    }}

    /* ── Top navigation bar ── */
    .top-nav-bar {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 1.5rem;
    }}
    .nav-status {{
        text-align: right;
        font-size: 0.78rem;
        color: var(--muted);
        line-height: 1.5;
    }}
    .nav-status strong {{
        color: var(--green);
    }}
    div[data-testid="stHorizontalBlock"] [data-testid="stRadio"] > div {{
        gap: 0.4rem;
        flex-wrap: wrap;
    }}
    div[data-testid="stHorizontalBlock"] [data-testid="stRadio"] label {{
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 0.45rem 0.85rem !important;
        margin: 0 !important;
        color: var(--muted) !important;
        font-weight: 500 !important;
        transition: all 0.15s ease;
    }}
    div[data-testid="stHorizontalBlock"] [data-testid="stRadio"] label:hover {{
        border-color: rgba(34,211,238,0.35) !important;
        color: var(--text) !important;
    }}
    div[data-testid="stHorizontalBlock"] [data-testid="stRadio"] label[data-checked="true"],
    div[data-testid="stHorizontalBlock"] [data-testid="stRadio"] div[aria-checked="true"] label {{
        background: linear-gradient(135deg, rgba(34,211,238,0.18), rgba(129,140,248,0.12)) !important;
        border-color: rgba(34,211,238,0.45) !important;
        color: var(--accent) !important;
    }}

    /* ── Metrics override ── */
    [data-testid="stMetric"] {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    }}
    [data-testid="stMetric"] label {{
        color: var(--muted) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}
    [data-testid="stMetric"] [data-testid="stMetricValue"] {{
        color: var(--text) !important;
        font-weight: 700 !important;
    }}

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background: var(--surface);
        border-radius: 14px;
        padding: 6px;
        border: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        color: var(--muted);
        font-weight: 500;
        padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, rgba(34,211,238,0.2), rgba(129,140,248,0.15)) !important;
        color: var(--accent) !important;
    }}

    /* ── Forms & inputs ── */
    [data-testid="stForm"] {{
        background: var(--card);
        border: 1px solid var(--border) !important;
        border-radius: 16px;
        padding: 0.5rem;
    }}
    .stSlider > div > div > div {{
        background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    }}
    .stSelectbox label, .stSlider label, .stMultiSelect label,
    .stRadio label, .stFileUploader label {{
        color: var(--muted) !important;
        font-size: 0.85rem !important;
    }}
    [data-testid="stWidgetLabel"] p {{
        color: var(--text) !important;
    }}
    [data-baseweb="select"] > div {{
        background: var(--surface) !important;
        border-color: var(--border) !important;
        color: var(--text) !important;
    }}
    /* ── Buttons ── */
    .stButton > button {{
        border-radius: 12px;
        font-weight: 600;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(34,211,238,0.2);
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #0891b2, #6366f1) !important;
        border: none !important;
    }}

    /* ── Dataframes ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }}

    /* ── Custom components ── */
    .brand-logo {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 0;
    }}
    .brand-icon {{
        width: 44px; height: 44px;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 4px 16px rgba(34,211,238,0.3);
    }}
    .brand-name {{
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.2;
    }}
    .brand-tag {{
        font-size: 0.72rem;
        color: var(--muted);
        letter-spacing: 0.04em;
    }}

    .hero {{
        position: relative;
        background: linear-gradient(135deg, #131a2b 0%, #1a2540 50%, #162032 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.8rem;
        overflow: hidden;
    }}
    .hero::before {{
        content: '';
        position: absolute;
        top: -50%; right: -10%;
        width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(34,211,238,0.12), transparent 70%);
        pointer-events: none;
    }}
    .hero::after {{
        content: '';
        position: absolute;
        bottom: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
    }}
    .hero-badge {{
        display: inline-block;
        background: rgba(34,211,238,0.12);
        border: 1px solid rgba(34,211,238,0.3);
        color: var(--accent);
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.3rem 0.75rem;
        border-radius: 100px;
        margin-bottom: 0.75rem;
    }}
    .hero h1 {{
        margin: 0 0 0.5rem 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
    }}
    .hero p {{
        margin: 0;
        color: var(--muted);
        font-size: 1.05rem;
        max-width: 620px;
        line-height: 1.6;
    }}

    .stat-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 1.8rem;
    }}
    @media (max-width: 900px) {{
        .stat-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    .stat-card {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .stat-card:hover {{
        transform: translateY(-2px);
        border-color: rgba(34,211,238,0.35);
    }}
    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--accent-color, var(--accent));
    }}
    .stat-icon {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    .stat-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 0.25rem;
    }}
    .stat-value {{
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
    }}
    .stat-sub {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.3rem; }}

    .panel {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1rem;
    }}
    .panel-title {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--muted);
        margin-bottom: 1rem;
        font-weight: 600;
    }}

    .sensor-card {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
    }}
    .sensor-card:hover {{ border-color: rgba(34,211,238,0.3); }}
    .sensor-header {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 0.6rem;
    }}
    .sensor-icon {{
        width: 36px; height: 36px;
        background: rgba(34,211,238,0.1);
        border-radius: 10px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }}
    .sensor-name {{ font-weight: 600; color: var(--text); font-size: 0.92rem; }}
    .sensor-hint {{ font-size: 0.75rem; color: var(--muted); }}

    .preset-chip {{
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
    }}
    .preset-chip:hover {{
        border-color: var(--accent);
        background: rgba(34,211,238,0.06);
    }}
    .preset-chip.active {{
        border-color: var(--accent);
        background: rgba(34,211,238,0.12);
        box-shadow: 0 0 0 1px rgba(34,211,238,0.2);
    }}
    .preset-emoji {{ font-size: 1.6rem; margin-bottom: 0.3rem; }}
    .preset-label {{ font-weight: 600; color: var(--text); font-size: 0.88rem; }}
    .preset-desc {{ font-size: 0.72rem; color: var(--muted); margin-top: 0.2rem; }}

    .result-card {{
        border-radius: 18px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid;
        position: relative;
        overflow: hidden;
    }}
    .result-card::before {{
        content: '';
        position: absolute;
        inset: 0;
        opacity: 0.08;
        background: radial-gradient(circle at 50% 0%, white, transparent 70%);
    }}
    .result-safe {{
        background: linear-gradient(160deg, #064e3b 0%, #065f46 100%);
        border-color: rgba(52,211,153,0.4);
        color: #d1fae5;
    }}
    .result-danger {{
        background: linear-gradient(160deg, #7f1d1d 0%, #991b1b 100%);
        border-color: rgba(248,113,113,0.4);
        color: #fee2e2;
    }}
    .result-model {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.75;
        margin-bottom: 0.4rem;
    }}
    .result-status {{
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }}
    .result-prob {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 500;
    }}

    .agreement-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 0.55rem 1.2rem;
        border-radius: 100px;
        font-weight: 600;
        font-size: 0.88rem;
        margin: 1rem 0 1.5rem 0;
    }}
    .agree-pill {{
        background: rgba(52,211,153,0.12);
        border: 1px solid rgba(52,211,153,0.35);
        color: var(--green);
    }}
    .disagree-pill {{
        background: rgba(251,191,36,0.12);
        border: 1px solid rgba(251,191,36,0.35);
        color: var(--amber);
    }}

    .model-pill {{
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }}
    .pill-lr {{ background: rgba(34,211,238,0.15); color: var(--accent); border: 1px solid rgba(34,211,238,0.3); }}
    .pill-knn {{ background: rgba(167,139,250,0.15); color: var(--accent2); border: 1px solid rgba(167,139,250,0.3); }}

    .flow-step {{
        display: flex;
        gap: 14px;
        align-items: flex-start;
        margin-bottom: 1rem;
    }}
    .flow-num {{
        width: 28px; height: 28px; min-width: 28px;
        background: linear-gradient(135deg, var(--accent), var(--accent2));
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.78rem;
        font-weight: 700;
        color: #0b0f19;
    }}
    .flow-text {{ color: var(--muted); font-size: 0.92rem; line-height: 1.5; }}
    .flow-text strong {{ color: var(--text); }}

    .sidebar-status {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1rem;
        margin-top: 1rem;
    }}
    .status-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        background: var(--green);
        border-radius: 50%;
        margin-right: 6px;
        box-shadow: 0 0 8px var(--green);
        animation: pulse 2s infinite;
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}

    .info-banner {{
        background: rgba(34,211,238,0.06);
        border: 1px solid rgba(34,211,238,0.2);
        border-radius: 14px;
        padding: 1rem 1.3rem;
        color: var(--muted);
        font-size: 0.92rem;
        line-height: 1.6;
    }}
    .info-banner strong {{ color: var(--accent); }}

    div[data-testid="stExpander"] {{
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_bundle():
    return train_models(DATA_PATH)


def apply_plotly_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(42,53,80,0.5)", zerolinecolor="rgba(42,53,80,0.5)")
    fig.update_yaxes(gridcolor="rgba(42,53,80,0.5)", zerolinecolor="rgba(42,53,80,0.5)")
    return fig


def hero(title: str, subtitle: str, badge: str = "Predictive Maintenance"):
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-badge">{badge}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stat_card(icon: str, label: str, value: str, sub: str = "", color: str = C["accent"]):
    st.markdown(
        f"""
        <div class="stat-card" style="--accent-color: {color}">
            <div class="stat-icon">{icon}</div>
            <div class="stat-label">{label}</div>
            <div class="stat-value">{value}</div>
            {"<div class='stat-sub'>" + sub + "</div>" if sub else ""}
        </div>
        """,
        unsafe_allow_html=True,
    )


def prob_gauge(prob: float, title: str, accent: str) -> go.Figure:
    color = C["red"] if prob >= 0.5 else C["green"]
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 32, "color": C["text"], "family": "JetBrains Mono"}},
            title={"text": title, "font": {"size": 13, "color": C["muted"]}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": C["muted"], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": C["surface"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(52,211,153,0.15)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.12)"},
                    {"range": [60, 100], "color": "rgba(248,113,113,0.15)"},
                ],
                "threshold": {"line": {"color": accent, "width": 2}, "thickness": 0.8, "value": 50},
            },
        )
    )
    fig.update_layout(height=240, margin=dict(l=24, r=24, t=48, b=8), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def risk_bar(prob: float) -> go.Figure:
    color = C["red"] if prob >= 0.5 else C["green"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 42, "color": C["text"], "family": "JetBrains Mono"}},
        title={"text": "Combined Risk Score", "font": {"size": 14, "color": C["muted"]}},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": color, "thickness": 0.35},
            "bgcolor": C["surface"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 33], "color": "rgba(52,211,153,0.2)"},
                {"range": [33, 66], "color": "rgba(251,191,36,0.15)"},
                {"range": [66, 100], "color": "rgba(248,113,113,0.2)"},
            ],
        },
    ))
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=20, r=20, t=50, b=10))
    return fig


def confusion_fig(matrix: np.ndarray, title: str, accent: str = C["accent"]) -> go.Figure:
    labels = ["No Failure", "Failure"]
    colorscale = [[0, C["surface"]], [0.5, accent], [1, C["accent2"]]]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels,
        text=matrix, texttemplate="%{text}",
        textfont=dict(size=18, color="white"),
        colorscale=colorscale, showscale=False,
    ))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", height=380, yaxis=dict(autorange="reversed"))
    return apply_plotly_style(fig)


def report_df(report: dict) -> pd.DataFrame:
    rows = []
    for label, key in [("No Failure", "0"), ("Failure", "1")]:
        rows.append({
            "Class": label,
            "Precision": f"{report[key]['precision']:.1%}",
            "Recall": f"{report[key]['recall']:.1%}",
            "F1": f"{report[key]['f1-score']:.1%}",
            "Support": int(report[key]["support"]),
        })
    return pd.DataFrame(rows)


def init_widget_state(defaults: dict):
    for col, key in INPUT_KEYS.items():
        if key not in st.session_state:
            st.session_state[key] = defaults[col]


def apply_preset(preset: dict):
    for col, key in INPUT_KEYS.items():
        st.session_state[key] = preset[col]


def render_result_card(model_name: str, label: str, prob: float, css_class: str, pill_class: str):
    st.markdown(
        f"""
        <div class="result-card {css_class}">
            <div class="result-model"><span class="model-pill {pill_class}">{model_name}</span></div>
            <div class="result-status">{"⚠️ FAILURE" if label == "Failure" else "✅ SAFE"}</div>
            <div class="result-prob">{prob:.1%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_overview(bundle):
    hero(
        "MachineGuard AI",
        "Real-time predictive maintenance powered by dual ML classifiers — stop breakdowns before they cost you.",
        badge="Industrial Intelligence",
    )

    total = len(bundle.raw_df)
    failures = int(bundle.raw_df[TARGET].sum())
    failure_rate = failures / total * 100
    lr_acc = (bundle.y_pred_lr == bundle.y_test).mean()

    cols = st.columns(4)
    stats = [
        ("📊", "Total Records", f"{total:,}", "Sensor readings analyzed", C["accent"]),
        ("🔴", "Failure Events", f"{failures:,}", f"{failure_rate:.2f}% of dataset", C["red"]),
        ("🎯", "LR Accuracy", f"{lr_acc:.1%}", "On held-out test set", C["green"]),
        ("🔢", "Optimal K", str(bundle.best_k), "KNN neighbors", C["accent2"]),
    ]
    for col, (icon, label, val, sub, color) in zip(cols, stats):
        with col:
            stat_card(icon, label, val, sub, color)

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Pipeline overview</div>', unsafe_allow_html=True)
            steps = [
                ("Collect", "6 sensor features from industrial machines — temperature, RPM, torque & tool wear."),
                ("Clean", "Drop ID columns & leakage flags (TWF, HDF, PWF, OSF, RNF) that would cheat the model."),
                ("Train", "Logistic Regression + KNN classifiers on 80% of data with standardized features."),
                ("Deploy", "Score live readings instantly — both models vote on failure probability."),
            ]
            for i, (title, desc) in enumerate(steps, 1):
                st.markdown(
                    f'<div class="flow-step"><div class="flow-num">{i}</div>'
                    f'<div class="flow-text"><strong>{title}</strong><br>{desc}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"""
            <div class="panel" style="margin-top:1rem">
                <div class="panel-title">Model comparison</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
                    <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem;">
                        <span class="model-pill pill-lr">Logistic Regression</span>
                        <p style="color:var(--muted);font-size:0.85rem;margin:0.6rem 0 0 0;">
                            Higher <strong style="color:var(--green)">failure recall</strong> — catches more real breakdowns.
                        </p>
                    </div>
                    <div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1rem;">
                        <span class="model-pill pill-knn">KNN k={bundle.best_k}</span>
                        <p style="color:var(--muted);font-size:0.85rem;margin:0.6rem 0 0 0;">
                            Higher <strong style="color:var(--accent)">overall accuracy</strong> on normal ops.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        dist = bundle.raw_df[TARGET].value_counts().sort_index()
        fig = px.pie(
            names=["No Failure", "Failure"],
            values=dist.values,
            color=["No Failure", "Failure"],
            color_discrete_map={"No Failure": C["green"], "Failure": C["red"]},
            hole=0.55,
        )
        fig.update_traces(
            textposition="outside",
            textinfo="percent+label",
            textfont_size=12,
            pull=[0, 0.06],
        )
        fig = apply_plotly_style(fig)
        fig.update_layout(
            title="Class Distribution",
            height=420,
            showlegend=False,
            margin=dict(t=60, b=110, l=70, r=70),
        )
        st.plotly_chart(fig, width="stretch")

        type_counts = bundle.raw_df["Type"].value_counts()
        fig2 = px.bar(
            x=[TYPE_LABELS.get(t, t) for t in type_counts.index],
            y=type_counts.values,
            color=[TYPE_LABELS.get(t, t) for t in type_counts.index],
            color_discrete_map={"Low": C["accent"], "Medium": C["accent2"], "High": C["amber"]},
        )
        fig2.update_layout(title="Records by Product Type", xaxis_title="", yaxis_title="Count", height=280, showlegend=False)
        st.plotly_chart(apply_plotly_style(fig2), width="stretch")

    st.markdown(
        '<div class="info-banner"><strong>Tip:</strong> Head to <strong>Predict</strong> for live scoring — '
        "sliders update risk in real time. Use <strong>Model Analytics</strong> to deep-dive performance.</div>",
        unsafe_allow_html=True,
    )


def page_predict(bundle):
    hero(
        "Live Failure Prediction",
        "Adjust sensor readings below — risk scores update instantly as you tune each parameter.",
        badge="Real-time Scoring",
    )

    presets = sample_presets(bundle.raw_df)
    ranges = feature_ranges(bundle.raw_df)
    init_widget_state(presets["Normal operation"])

    st.markdown('<div class="panel-title" style="margin-bottom:0.75rem">Quick scenarios</div>', unsafe_allow_html=True)
    pcols = st.columns(3)
    preset_meta = {
        "Normal operation": ("✅", "Healthy machine"),
        "Known failure": ("🚨", "Breakdown case"),
        "High tool wear": ("⚠️", "Worn tooling"),
    }
    for col, name in zip(pcols, presets):
        emoji, label = preset_meta[name]
        with col:
            if st.button(f"{emoji} {label}", width="stretch", key=f"preset_{name}"):
                apply_preset(presets[name])
                st.rerun()

    left, right = st.columns([1, 1], gap="large")

    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Sensor inputs</div>', unsafe_allow_html=True)
            machine_type = st.selectbox(
                "Product type",
                options=["L", "M", "H"],
                format_func=lambda x: f"{SENSOR_META['Type'][0]}  {x} — {TYPE_LABELS[x]}",
                key=INPUT_KEYS["Type"],
            )
            air_temp = st.slider(
                "Air temperature [K]",
                float(ranges["Air temperature [K]"][0]),
                float(ranges["Air temperature [K]"][1]),
                step=0.1,
                key=INPUT_KEYS["Air temperature [K]"],
            )
            process_temp = st.slider(
                "Process temperature [K]",
                float(ranges["Process temperature [K]"][0]),
                float(ranges["Process temperature [K]"][1]),
                step=0.1,
                key=INPUT_KEYS["Process temperature [K]"],
            )
            rpm = st.slider(
                "Rotational speed [rpm]",
                int(ranges["Rotational speed [rpm]"][0]),
                int(ranges["Rotational speed [rpm]"][1]),
                key=INPUT_KEYS["Rotational speed [rpm]"],
            )
            torque = st.slider(
                "Torque [Nm]",
                float(ranges["Torque [Nm]"][0]),
                float(ranges["Torque [Nm]"][1]),
                step=0.1,
                key=INPUT_KEYS["Torque [Nm]"],
            )
            tool_wear = st.slider(
                "Tool wear [min]",
                int(ranges["Tool wear [min]"][0]),
                int(ranges["Tool wear [min]"][1]),
                key=INPUT_KEYS["Tool wear [min]"],
            )

    inputs = {
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear,
    }
    result = predict(bundle, inputs)
    combined_prob = (result["lr_prob"] + result["knn_prob"]) / 2

    with right:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Risk assessment</div>', unsafe_allow_html=True)
            st.plotly_chart(risk_bar(combined_prob), width="stretch")

            pill_class = "agree-pill" if result["agreement"] else "disagree-pill"
            pill_icon = "✓" if result["agreement"] else "⚡"
            pill_text = "Both models agree" if result["agreement"] else "Models disagree — investigate"
            st.markdown(
                f'<div class="agreement-pill {pill_class}">{pill_icon}&nbsp; {pill_text}</div>',
                unsafe_allow_html=True,
            )

            r1, r2 = st.columns(2)
            with r1:
                css = "result-danger" if result["lr_label"] == "Failure" else "result-safe"
                render_result_card("Logistic Regression", result["lr_label"], result["lr_prob"], css, "pill-lr")
            with r2:
                css = "result-danger" if result["knn_label"] == "Failure" else "result-safe"
                render_result_card(f"KNN (k={bundle.best_k})", result["knn_label"], result["knn_prob"], css, "pill-knn")

    with st.container(border=True):
        st.markdown('<div class="panel-title">Input snapshot</div>', unsafe_allow_html=True)
        st.json(inputs)


def page_analytics(bundle):
    hero(
        "Model Analytics",
        "Deep performance analysis on the 20% held-out test set — 2,000 unseen machine records.",
        badge="Evaluation Suite",
    )

    lr_m = classification_metrics(bundle.y_test, bundle.y_pred_lr)
    knn_m = classification_metrics(bundle.y_test, bundle.y_pred_knn)

    cols = st.columns(4)
    metrics = [
        ("LR Accuracy", f"{lr_m['accuracy']:.1%}", C["accent"]),
        ("LR Failure Recall", f"{lr_m['recall_failure']:.1%}", C["green"]),
        ("KNN Accuracy", f"{knn_m['accuracy']:.1%}", C["accent2"]),
        ("KNN Failure Recall", f"{knn_m['recall_failure']:.1%}", C["amber"]),
    ]
    for col, (label, val, color) in zip(cols, metrics):
        with col:
            stat_card("📈", label, val, color=color)

    st.markdown("<br>", unsafe_allow_html=True)

    tab_cm, tab_roc, tab_k, tab_feat, tab_report = st.tabs([
        "🎯 Confusion Matrices", "📈 ROC Curve", "🔢 K Tuning", "⚖️ Feature Importance", "📋 Reports",
    ])

    with tab_cm:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                confusion_fig(lr_m["confusion_matrix"], "Logistic Regression", C["accent"]),
                width="stretch",
            )
        with c2:
            st.plotly_chart(
                confusion_fig(knn_m["confusion_matrix"], f"KNN (k={bundle.best_k})", C["accent2"]),
                width="stretch",
            )

    with tab_roc:
        fpr, tpr, roc_auc, _ = roc_data(bundle.y_test, bundle.y_prob_lr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", fill="tozeroy",
            fillcolor="rgba(34,211,238,0.12)",
            name=f"LR (AUC = {roc_auc:.3f})",
            line=dict(color=C["accent"], width=3),
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
            line=dict(dash="dot", color=C["muted"], width=1),
        ))
        fig.update_layout(title=f"ROC Curve — AUC {roc_auc:.3f}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=440)
        st.plotly_chart(apply_plotly_style(fig), width="stretch")

        col1, col2, col3 = st.columns(3)
        col1.metric("AUC Score", f"{roc_auc:.3f}")
        col2.metric("FPR at 50% threshold", f"{fpr[len(fpr)//2]:.1%}")
        col3.metric("TPR at 50% threshold", f"{tpr[len(tpr)//2]:.1%}")

    with tab_k:
        ks = list(bundle.knn_k_accuracies.keys())
        accs = [bundle.knn_k_accuracies[k] for k in ks]
        colors = [C["accent2"] if k != bundle.best_k else C["accent"] for k in ks]
        fig = go.Figure(go.Bar(x=ks, y=accs, marker_color=colors, text=[f"{a:.1%}" for a in accs], textposition="outside"))
        fig.update_layout(title="K Value vs Accuracy", xaxis_title="K (neighbors)", yaxis_title="Accuracy", yaxis_tickformat=".1%", height=400)
        st.plotly_chart(apply_plotly_style(fig), width="stretch")

    with tab_feat:
        coefs = bundle.lr.coef_[0]
        feat_df = pd.DataFrame({"Feature": bundle.feature_cols, "Coefficient": coefs})
        feat_df["Feature"] = feat_df["Feature"].replace({
            "Type": "Product Type",
            "Air temperature [K]": "Air Temp",
            "Process temperature [K]": "Process Temp",
            "Rotational speed [rpm]": "RPM",
            "Torque [Nm]": "Torque",
            "Tool wear [min]": "Tool Wear",
        })
        feat_df["Direction"] = feat_df["Coefficient"].apply(lambda x: "→ Failure" if x > 0 else "→ Safe")
        fig = px.bar(
            feat_df.sort_values("Coefficient"),
            x="Coefficient", y="Feature", orientation="h",
            color="Coefficient",
            color_continuous_scale=[C["green"], C["surface"], C["red"]],
            text="Direction",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(title="Logistic Regression Coefficients", height=400, coloraxis_showscale=False)
        st.plotly_chart(apply_plotly_style(fig), width="stretch")

    with tab_report:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<span class="model-pill pill-lr">Logistic Regression</span>', unsafe_allow_html=True)
            st.dataframe(report_df(lr_m["report"]), width="stretch", hide_index=True)
        with c2:
            st.markdown(f'<span class="model-pill pill-knn">KNN k={bundle.best_k}</span>', unsafe_allow_html=True)
            st.dataframe(report_df(knn_m["report"]), width="stretch", hide_index=True)


def page_explorer(bundle):
    hero(
        "Data Explorer",
        "Interactive telemetry browser — filter, visualize, and inspect all 10,000 machine records.",
        badge="Dataset Insights",
    )

    df = bundle.raw_df.copy()
    df["Type label"] = df["Type"].map(TYPE_LABELS)

    with st.container(border=True):
        st.markdown('<div class="panel-title">Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        type_filter = f1.multiselect("Product type", ["L", "M", "H"], default=["L", "M", "H"])
        fail_filter = f2.selectbox("Failure status", ["All", "Failures only", "No failures only"])
        wear_max = f3.slider(
            "Max tool wear [min]",
            int(df["Tool wear [min]"].min()),
            int(df["Tool wear [min]"].max()),
            int(df["Tool wear [min]"].max()),
        )

    filtered = df[df["Type"].isin(type_filter) & (df["Tool wear [min]"] <= wear_max)]
    if fail_filter == "Failures only":
        filtered = filtered[filtered[TARGET] == 1]
    elif fail_filter == "No failures only":
        filtered = filtered[filtered[TARGET] == 0]

    cols = st.columns(4)
    explorer_stats = [
        ("🗃️", "Rows", f"{len(filtered):,}", C["accent"]),
        ("🔴", "Failures", f"{filtered[TARGET].sum():,}", C["red"]),
        ("⏱️", "Avg Wear", f"{filtered['Tool wear [min]'].mean():.0f} min", C["amber"]),
        ("⚡", "Avg RPM", f"{filtered['Rotational speed [rpm]'].mean():.0f}", C["accent2"]),
    ]
    for col, (icon, label, val, color) in zip(cols, explorer_stats):
        with col:
            stat_card(icon, label, val, color=color)

    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2, t3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📋 Raw Data"])

    numeric_cols = [
        "Air temperature [K]", "Process temperature [K]",
        "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]",
    ]

    with t1:
        c1, c2 = st.columns([1, 2])
        with c1:
            feature = st.selectbox("Select feature", numeric_cols)
        with c2:
            fig = px.histogram(
                filtered, x=feature, color=TARGET,
                color_discrete_map={0: C["green"], 1: C["red"]},
                labels={TARGET: "Status", 0: "Safe", 1: "Failure"},
                barmode="overlay", opacity=0.8, nbins=45,
            )
            fig.update_layout(title=f"Distribution — {feature}", height=420)
            st.plotly_chart(apply_plotly_style(fig), width="stretch")

        fig2 = px.scatter(
            filtered.sample(min(800, len(filtered)), random_state=42),
            x="Rotational speed [rpm]", y="Torque [Nm]",
            color=TARGET, size="Tool wear [min]",
            color_discrete_map={0: C["green"], 1: C["red"]},
            labels={TARGET: "Status", 0: "Safe", 1: "Failure"},
            opacity=0.7,
        )
        fig2.update_layout(title="RPM vs Torque (size = tool wear)", height=420)
        st.plotly_chart(apply_plotly_style(fig2), width="stretch")

    with t2:
        corr_cols = numeric_cols + [TARGET]
        corr = filtered[corr_cols].corr()
        short_names = ["Air Temp", "Process Temp", "RPM", "Torque", "Tool Wear", "Failure"]
        corr.index = short_names
        corr.columns = short_names
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
        fig.update_layout(title="Feature Correlation Matrix", height=500)
        st.plotly_chart(apply_plotly_style(fig), width="stretch")

    with t3:
        display_cols = ["Type", "Type label"] + numeric_cols + [TARGET]
        st.dataframe(filtered[display_cols].head(500), width="stretch", hide_index=True)
        if len(filtered) > 500:
            st.caption(f"Showing 500 of {len(filtered):,} rows.")


def page_batch(bundle):
    hero(
        "Batch Scoring",
        "Upload a CSV to score hundreds of machines at once — download enriched results instantly.",
        badge="Bulk Processing",
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">File requirements</div>
            <p style="color:var(--muted);margin:0;font-size:0.9rem;">
                Required columns: <code style="color:var(--accent);font-family:'JetBrains Mono',monospace;">
                {', '.join(FEATURE_COLS)}</code><br>
                Optional ID & failure-flag columns are ignored automatically.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            """
            <style>
            div[data-testid="stFileUploader"] button span,
            div[data-testid="stFileUploader"] button p {
                display: none !important;
            }
            div[data-testid="stFileUploader"] button {
                font-size: 0.875rem !important;
                color: #e8edf7 !important;
            }
            div[data-testid="stFileUploader"] button::before {
                content: "Browse files";
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown('<div class="panel-title">Upload CSV file</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader("csv", type=["csv"], label_visibility="collapsed", key="batch_csv")
    with c2:
        st.download_button(
            "⬇️ Download template CSV",
            data=pd.DataFrame([sample_presets(bundle.raw_df)["Normal operation"]]).to_csv(index=False),
            file_name="prediction_template.csv",
            mime="text/csv",
            width="stretch",
        )

    if uploaded is None:
        st.markdown(
            '<div class="info-banner">Upload a CSV above or download the template to get started.</div>',
            unsafe_allow_html=True,
        )
        return

    batch_df = pd.read_csv(uploaded)
    missing = [c for c in FEATURE_COLS if c not in batch_df.columns]
    if missing:
        st.error(f"Missing columns: {', '.join(missing)}")
        return

    st.success(f"✅ Loaded **{len(batch_df):,}** rows — ready to score.")

    if st.button("Score all rows", type="primary", width="stretch"):
        with st.spinner("Scoring machines…"):
            rows = []
            for _, row in batch_df.iterrows():
                inputs = {col: row[col] for col in FEATURE_COLS}
                if inputs["Type"] not in ["L", "M", "H"]:
                    st.warning(f"Invalid Type '{inputs['Type']}' — must be L, M, or H.")
                    return
                r = predict(bundle, inputs)
                rows.append({
                    **inputs,
                    "LR prediction": r["lr_label"],
                    "LR probability": round(r["lr_prob"], 4),
                    "KNN prediction": r["knn_label"],
                    "KNN probability": round(r["knn_prob"], 4),
                    "Models agree": r["agreement"],
                })
            out = pd.DataFrame(rows)

        cols = st.columns(4)
        fail_lr = (out["LR prediction"] == "Failure").sum()
        fail_knn = (out["KNN prediction"] == "Failure").sum()
        agree_pct = out["Models agree"].mean()
        for col, (icon, label, val, color) in zip(cols, [
            ("📊", "Rows Scored", f"{len(out):,}", C["accent"]),
            ("🔴", "LR Failures", str(fail_lr), C["red"]),
            ("🟣", "KNN Failures", str(fail_knn), C["accent2"]),
            ("🤝", "Agreement", f"{agree_pct:.1%}", C["green"]),
        ]):
            with col:
                stat_card(icon, label, val, color=color)

        fig = px.histogram(
            out, x="LR probability", nbins=30,
            color_discrete_sequence=[C["accent"]],
            title="Distribution of LR Failure Probabilities",
        )
        fig.update_layout(height=320)
        st.plotly_chart(apply_plotly_style(fig), width="stretch")

        st.dataframe(out, width="stretch", hide_index=True)
        st.download_button(
            "Download results CSV",
            data=out.to_csv(index=False),
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
        )


def render_top_nav(bundle) -> str:
    st.markdown('<div class="top-nav-bar">', unsafe_allow_html=True)
    brand, nav, status = st.columns([1.1, 3.4, 1.1])

    with brand:
        st.markdown(
            """
            <div class="brand-logo">
                <div class="brand-icon">⚙️</div>
                <div>
                    <div class="brand-name">MachineGuard</div>
                    <div class="brand-tag">Predictive Maintenance</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav:
        labels = [f"{icon} {name}" for name, icon, _ in NAV_ITEMS]
        name_map = {f"{icon} {name}": name for name, icon, _ in NAV_ITEMS}
        choice = st.radio(
            "Navigation",
            labels,
            horizontal=True,
            label_visibility="collapsed",
            key="main_nav",
        )
        page = name_map[choice]

    with status:
        st.markdown(
            f"""
            <div class="nav-status">
                <span class="status-dot"></span> <strong>Online</strong><br>
                LR + KNN (k={bundle.best_k})<br>
                {len(bundle.y_test):,} test records
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    return page


def main():
    with st.spinner("Initializing models…"):
        bundle = load_bundle()

    page = render_top_nav(bundle)

    pages = {
        "Overview": page_overview,
        "Predict": page_predict,
        "Model Analytics": page_analytics,
        "Data Explorer": page_explorer,
        "Batch scoring": page_batch,
    }
    pages[page](bundle)


if __name__ == "__main__":
    main()
