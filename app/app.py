"""The Server Whisperer: live demo dashboard.
Run from the project folder with:  streamlit run app/app.py
"""
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.append(str(ROOT / "src"))

from explain import ACTIONS, KIND_TEXT, METRIC_TEXT, contributions, load_model  # noqa: E402
from features import add_features  # noqa: E402
from simulator import simulate_server  # noqa: E402
from train import baseline_alert, to_alerts  # noqa: E402

st.set_page_config(page_title="The Server Whisperer", page_icon="🌫️", layout="wide")
VIEW_MINUTES = 900
FAULT_LABELS = {
    "memory_leak": "Memory leak",
    "disk_fill": "Disk filling up",
    "cpu_spike": "CPU spike",
}


@st.cache_resource
def load_all():
    model, config = load_model()
    iso = joblib.load("models/isolation_forest.joblib")
    results = json.loads(Path("reports/model_results.json").read_text())["results"]
    return model, iso, config, results


model, iso, config, results = load_all()
FEATURES = config["features"]


def run_scenario(fault, seed):
    raw = simulate_server("demo-server", days=3, fault=fault, seed=seed)
    df = add_features(raw)
    X = df[FEATURES]
    df["prob"] = model.predict_proba(X)[:, 1]
    df["iso"] = -iso.score_samples(X)
    df["model_alert"] = to_alerts(df, df["prob"].values, config["xgb_threshold"]).values
    df["thr_alert"] = baseline_alert(df).values
    return {"df": df, "contribs": contributions(model, X), "fault": fault}


def whisper_level(df, i):
    lo = float(np.median(df["iso"].iloc[:1000]))
    thr = config["if_threshold"]
    return float(np.clip((df["iso"].iloc[i] - lo) / max(thr - lo, 1e-6), 0, 1))


def diagnose(contrib_row):
    """Root cause = the resource (cpu, memory, disk) with the most evidence."""
    pos = np.clip(contrib_row, 0, None)
    by_metric = {}
    for f, v in zip(FEATURES, pos):
        metric = f.split("_")[0]
        by_metric[metric] = by_metric.get(metric, 0) + v
    root = max(("cpu", "memory", "disk"), key=lambda m: by_metric.get(m, 0))
    reasons = []
    for i in np.argsort(pos)[::-1][:3]:
        if pos[i] > 0:
            metric, kind = FEATURES[i].split("_")
            reasons.append(f"{METRIC_TEXT[metric]} {KIND_TEXT[kind]}")
    return root, reasons, ACTIONS[root]


def draw(sc, i, chart_ph, status_ph):
    df = sc["df"]
    start = len(df) - VIEW_MINUTES
    d = df.iloc[start:i + 1]

    fig = go.Figure()
    for m, color in [("cpu", "#e07a5f"), ("memory", "#3d5a80"), ("disk", "#81b29a")]:
        fig.add_trace(go.Scatter(x=d["timestamp"], y=d[m], name=m, mode="lines",
                                 line=dict(color=color)))
    for col, name, symbol, color in [
        ("thr_alert", "Threshold rule alert", "x", "gray"),
        ("model_alert", "Whisperer alert", "triangle-down", "red"),
    ]:
        a = d[col].astype(bool)
        edge = d[a & ~a.shift(1, fill_value=False)]
        fig.add_trace(go.Scatter(x=edge["timestamp"], y=[104] * len(edge), mode="markers",
                                 name=name, marker=dict(symbol=symbol, size=12, color=color)))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(range=[0, 108], title="% used"),
                      legend=dict(orientation="h", y=-0.15))
    chart_ph.plotly_chart(fig)

    row = df.iloc[i]
    level = whisper_level(df, i)
    with status_ph.container():
        c1, c2 = st.columns([1, 2])
        with c1:
            st.progress(level, text=f"Whisper level: {level:.0%}")
            st.caption("Early drift signal (Isolation Forest). Full bar = whisper alert.")
        with c2:
            if row["model_alert"]:
                root, reasons, action = diagnose(sc["contribs"][i])
                st.error(f"⚠️ Failure predicted. Probability {row['prob']:.0%}")
                st.write("**Why:** " + "; ".join(reasons))
                st.write("**Suggested action:** " + action + "  _(suggestion only, nothing is changed automatically)_")
                st.caption(f"Simulator ground truth: crash in {int(row['minutes_to_failure'])} min")
            elif level >= 0.6:
                st.warning("Whispering: something is drifting. Watching closely.")
            else:
                st.success("Healthy: behaving normally.")


def show_summary(sc):
    df = sc["df"]
    w = df[df["is_degrading"]]
    m = w[w["model_alert"]]
    t = w[w["thr_alert"]]
    model_lead = int(m["minutes_to_failure"].iloc[0]) if len(m) else None
    thr_lead = int(t["minutes_to_failure"].iloc[0]) if len(t) else None
    st.subheader("💥 The server crashed. Who saw it coming?")
    c1, c2, c3 = st.columns(3)
    c1.metric("Threshold rule warned", f"{thr_lead} min before" if thr_lead is not None else "No warning")
    c2.metric("Whisperer warned", f"{model_lead} min before" if model_lead is not None else "Missed it")
    if model_lead is not None:
        c3.metric("Extra warning gained", f"{model_lead - (thr_lead or 0)} min")


st.title("🌫️ The Server Whisperer")
st.caption("Every failure whispers first.")

st.sidebar.header("Chaos controls")
fault = st.sidebar.selectbox("Break a server with", list(FAULT_LABELS), format_func=FAULT_LABELS.get)
speed = st.sidebar.select_slider("Playback speed", ["Slow", "Normal", "Fast"], value="Normal")
delay = {"Slow": 0.15, "Normal": 0.07, "Fast": 0.02}[speed]
clicked = st.sidebar.button("💥 Break a server", type="primary")

chart_ph = st.empty()
status_ph = st.empty()

if clicked:
    sc = run_scenario(fault, seed=int(time.time()) % 100000)
    st.session_state["scenario"] = sc
    n = len(sc["df"])
    for i in range(n - VIEW_MINUTES, n, 5):
        draw(sc, i, chart_ph, status_ph)
        time.sleep(delay)
    draw(sc, n - 1, chart_ph, status_ph)
    st.rerun()
elif "scenario" in st.session_state:
    sc = st.session_state["scenario"]
    n = len(sc["df"])
    pos = st.slider("Scrub through time (left = 15 hours before the crash, right = the crash)",
                    n - VIEW_MINUTES, n - 1, n - 1)
    draw(sc, pos, chart_ph, status_ph)
    if pos == n - 1:
        show_summary(sc)
else:
    chart_ph.info("Pick a fault in the sidebar and press **Break a server**.")

st.divider()
base = results["Threshold baseline"]["false_alarms_per_server_day"]
ours = results["XGBoost"]["false_alarms_per_server_day"]
c1, c2 = st.columns([1, 2])
c1.metric("False alarms removed (alert fatigue meter)", f"{1 - ours / base:.0%}",
          f"{base} → {ours} per server-day", delta_color="off")
c2.caption("Measured on held-out simulated servers that no model was trained on. "
           "The old rule alerts when CPU, memory or disk passes 90%.")
