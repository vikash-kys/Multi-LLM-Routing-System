import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal dark styling ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3147;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .headline-number {
        font-size: 3rem;
        font-weight: 700;
        color: #00d4aa;
        line-height: 1;
    }
    .headline-label {
        font-size: 0.9rem;
        color: #8892b0;
        margin-top: 4px;
    }
    .savings-banner {
        background: linear-gradient(135deg, #00d4aa20, #0066ff20);
        border: 1px solid #00d4aa40;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=30)
def fetch_stats(days):
    try:
        r = requests.get(f"{API_URL}/v1/stats?days={days}", timeout=5)
        return r.json()
    except Exception as e:
        return None


@st.cache_data(ttl=30)
def fetch_requests(limit=200):
    try:
        r = requests.get(f"{API_URL}/v1/requests?limit={limit}", timeout=5)
        return r.json().get("requests", [])
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_models():
    try:
        r = requests.get(f"{API_URL}/v1/models", timeout=5)
        return r.json()
    except Exception:
        return {}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## LLM Cost Autopilot")
    st.markdown("---")
    days = st.selectbox("Time period", [1, 7, 14, 30], index=1, format_func=lambda d: f"Last {d} day{'s' if d > 1 else ''}")
    st.markdown("---")
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("**API Status**")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("Online")
    except Exception:
        st.error("Offline — start the API first")


# ── Main content ──────────────────────────────────────────────────────────────
st.markdown("# LLM Cost Autopilot")
st.markdown("Intelligent routing layer — cheapest model that meets quality thresholds.")
st.markdown("---")

stats = fetch_stats(days)

if not stats:
    st.warning("Cannot reach API. Make sure the FastAPI server is running on port 8000.")
    st.code("uvicorn app.main:app --reload --port 8000")
    st.stop()

# ── Headline savings banner ───────────────────────────────────────────────────
savings_pct = stats.get("savings_percent", 0)
savings_usd = stats.get("savings_usd", 0)
actual_cost = stats.get("actual_cost_usd", 0)
baseline_cost = stats.get("baseline_cost_usd", 0)

st.markdown(f"""
<div class="savings-banner">
    <div style="font-size:4rem; font-weight:800; color:#00d4aa;">{savings_pct:.1f}%</div>
    <div style="font-size:1.2rem; color:#ccd6f6; margin-top:8px;">cost reduction vs GPT-4o for everything</div>
    <div style="font-size:0.95rem; color:#8892b0; margin-top:8px;">
        Saved ${savings_usd:.4f} — Actual: ${actual_cost:.4f} vs Baseline: ${baseline_cost:.4f}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Key metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(col, value, label, color="#ccd6f6"):
    col.markdown(f"""
    <div class="metric-card">
        <div class="headline-number" style="color:{color}">{value}</div>
        <div class="headline-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

metric_card(c1, f"{stats.get('total_requests', 0):,}", "Total Requests")
metric_card(c2, f"${actual_cost:.4f}", "Actual Cost", "#00d4aa")
metric_card(c3, f"{stats.get('avg_latency_ms', 0):.0f}ms", "Avg Latency")
avg_q = stats.get("avg_quality_score", 0) or 0
metric_card(c4, f"{avg_q:.2f}/5", "Avg Quality Score", "#ffd700" if avg_q >= 4 else "#ff6b6b")
escalation_rate = stats.get("escalation_rate_percent", 0)
metric_card(c5, f"{escalation_rate:.1f}%", "Escalation Rate", "#ff6b6b" if escalation_rate > 10 else "#00d4aa")

st.markdown("<br>", unsafe_allow_html=True)

# ── Charts row 1 ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### Daily Cost")
    daily_data = stats.get("daily", [])
    if daily_data:
        df_daily = pd.DataFrame(daily_data)
        df_daily["date"] = pd.to_datetime(df_daily["date"])
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_daily["date"],
            y=df_daily["cost"],
            name="Actual Cost",
            marker_color="#00d4aa",
        ))
        fig.update_layout(
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=20, b=0),
            height=280,
            xaxis=dict(gridcolor="#2d3147"),
            yaxis=dict(gridcolor="#2d3147", tickprefix="$"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Send some requests first.")

with col_right:
    st.markdown("### Routing Distribution")
    model_data = stats.get("by_model", [])
    if model_data:
        df_model = pd.DataFrame(model_data)
        colors = ["#00d4aa", "#0066ff", "#ffd700", "#ff6b6b", "#a78bfa"]
        fig = px.pie(
            df_model,
            values="count",
            names="routed_model",
            color_discrete_sequence=colors,
            hole=0.45,
        )
        fig.update_layout(
            paper_bgcolor="#1a1d2e",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=20, b=0),
            height=280,
            legend=dict(bgcolor="#1a1d2e"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No routing data yet.")


# ── Charts row 2 ─────────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Tier Distribution")
    tier_data = stats.get("by_tier", [])
    if tier_data:
        df_tier = pd.DataFrame(tier_data)
        tier_names = {1: "Tier 1 — Simple", 2: "Tier 2 — Moderate", 3: "Tier 3 — Complex"}
        df_tier["tier_name"] = df_tier["complexity_tier"].map(tier_names)
        fig = px.bar(
            df_tier,
            x="tier_name",
            y="count",
            color="tier_name",
            color_discrete_sequence=["#00d4aa", "#ffd700", "#ff6b6b"],
        )
        fig.update_layout(
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=20, b=0),
            height=260,
            showlegend=False,
            xaxis=dict(gridcolor="#2d3147"),
            yaxis=dict(gridcolor="#2d3147"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No tier data yet.")

with col_b:
    st.markdown("### Cost: Actual vs GPT-4o Baseline")
    if actual_cost > 0 or baseline_cost > 0:
        fig = go.Figure(go.Bar(
            x=["GPT-4o Baseline", "Actual (Routed)"],
            y=[baseline_cost, actual_cost],
            marker_color=["#ff6b6b", "#00d4aa"],
            text=[f"${baseline_cost:.4f}", f"${actual_cost:.4f}"],
            textposition="auto",
        ))
        fig.update_layout(
            paper_bgcolor="#1a1d2e",
            plot_bgcolor="#1a1d2e",
            font_color="#ccd6f6",
            margin=dict(l=0, r=0, t=20, b=0),
            height=260,
            xaxis=dict(gridcolor="#2d3147"),
            yaxis=dict(gridcolor="#2d3147", tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No cost data yet.")


# ── Recent requests table ─────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Recent Requests")

requests_data = fetch_requests()
if requests_data:
    df = pd.DataFrame(requests_data)

    # Clean up for display
    display_cols = ["timestamp", "prompt_preview", "complexity_tier", "routed_model",
                    "cost_usd", "latency_ms", "quality_score", "was_escalated"]
    df = df[[c for c in display_cols if c in df.columns]]

    tier_map = {1: "Simple", 2: "Moderate", 3: "Complex"}
    if "complexity_tier" in df.columns:
        df["complexity_tier"] = df["complexity_tier"].map(tier_map)

    if "was_escalated" in df.columns:
        df["was_escalated"] = df["was_escalated"].apply(lambda x: "Yes" if x else "No")

    if "cost_usd" in df.columns:
        df["cost_usd"] = df["cost_usd"].apply(lambda x: f"${x:.6f}")

    if "prompt_preview" in df.columns:
        df["prompt_preview"] = df["prompt_preview"].str[:80] + "..."

    st.dataframe(df, use_container_width=True, height=300)
else:
    st.info("No requests logged yet.")


# ── Model pricing reference ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Model Registry")

models_data = fetch_models()
if models_data and "models" in models_data:
    df_models = pd.DataFrame(models_data["models"])
    display = ["display_name", "provider", "quality_tier", "cost_per_1k_input_tokens",
               "cost_per_1k_output_tokens", "avg_latency_ms", "currently_assigned_tiers"]
    df_models = df_models[[c for c in display if c in df_models.columns]]
    df_models["cost_per_1k_input_tokens"] = df_models["cost_per_1k_input_tokens"].apply(lambda x: f"${x:.4f}")
    df_models["cost_per_1k_output_tokens"] = df_models["cost_per_1k_output_tokens"].apply(lambda x: f"${x:.4f}")
    st.dataframe(df_models, use_container_width=True)
