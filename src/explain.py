"""Explain XGBoost alerts in plain language for The Server Whisperer."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

METRIC_TEXT = {
    "cpu": "CPU load", "memory": "memory use", "disk": "disk usage",
    "network": "network traffic", "errors": "error rate",
}
KIND_TEXT = {
    "slope": "is climbing fast",
    "drift": "is well above this server's daily normal",
    "mean15": "is high right now",
    "mean60": "has stayed high for the past hour",
    "max60": "hit a high peak in the past hour",
    "std60": "is swinging unusually",
}
ACTIONS = {
    "memory": "Restart the leaking service and check it for memory leaks",
    "disk": "Clear old logs or add disk space",
    "cpu": "Scale out or throttle incoming requests",
    "network": "Check traffic sources and rate limits",
    "errors": "Review recent application errors and deployments",
}


def load_model():
    model = xgb.XGBClassifier()
    model.load_model("models/xgb.json")
    config = json.loads(Path("models/config.json").read_text())
    return model, config


def contributions(model, X):
    """Per-feature SHAP values from XGBoost itself (last column is the bias)."""
    contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return contribs[:, :-1]


def explain_row(contrib_row, features, top=3):
    order = np.argsort(contrib_row)[::-1][:top]
    reasons = []
    for i in order:
        if contrib_row[i] <= 0:
            continue
        metric, kind = features[i].split("_")
        reasons.append(f"{METRIC_TEXT[metric]} {KIND_TEXT[kind]}")
    top_metric = features[order[0]].split("_")[0]
    return reasons, ACTIONS[top_metric]


if __name__ == "__main__":
    model, config = load_model()
    features = config["features"]
    data = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])
    test_ids = json.loads(Path("reports/model_results.json").read_text())["test_servers"]
    test = data[data.server_id.isin(test_ids)].reset_index(drop=True)
    X = test[features]
    contribs = contributions(model, X)

    # Global picture: which features matter most
    shap.summary_plot(contribs, X, plot_type="bar", max_display=12, show=False)
    plt.tight_layout()
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/shap_summary.png", dpi=150)
    print("Saved reports/shap_summary.png")

    # Local picture: explain each failing test server 45 minutes before it crashes
    print()
    for srv, d in test[test.fault_type != "none"].groupby("server_id"):
        row = d[d.minutes_to_failure == 45]
        if row.empty:
            continue
        i = row.index[0]
        prob = model.predict_proba(X.iloc[[i]])[0, 1]
        reasons, action = explain_row(contribs[i], features)
        print(f"{srv}  (true fault: {d.fault_type.iloc[0]})")
        print(f"  Failure probability: {prob:.0%}")
        for r in reasons:
            print(f"  Why: {r}")
        print(f"  Suggested action: {action}")
        print()
