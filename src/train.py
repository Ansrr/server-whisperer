"""Train and compare models for The Server Whisperer."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

SUFFIXES = ("_mean15", "_mean60", "_std60", "_max60", "_drift", "_slope")
SMOOTH = 3  # a model must stay above its threshold for 3 minutes in a row


def split_servers(df, seed=42):
    rng = np.random.default_rng(seed)
    info = df.groupby("server_id").fault_type.first()
    train, val, test = [], [], []
    for _, grp in info.groupby(info):
        ids = list(grp.index)
        rng.shuffle(ids)
        k = len(ids) // 4
        test += ids[:k]
        val += ids[k:2 * k]
        train += ids[2 * k:]
    return train, val, test


def to_alerts(df, score, thr):
    hit = pd.Series((score > thr).astype(int), index=df.index)
    smooth = hit.groupby(df["server_id"]).transform(
        lambda x: x.rolling(SMOOTH, min_periods=SMOOTH).min())
    return smooth.fillna(0).astype(bool)


def evaluate(df, alert):
    df = df.assign(alert=alert.values)
    prev = df.groupby("server_id")["alert"].shift(1, fill_value=False).astype(bool)
    event = df["alert"] & ~prev
    false_alarms = int((event & ~df.is_degrading).sum())
    rows = []
    for srv, d in df[df.fault_type != "none"].groupby("server_id"):
        hits = d[d.is_degrading & d.alert]
        rows.append({
            "fault_type": d.fault_type.iloc[0],
            "caught": len(hits) > 0,
            "lead": float(hits.minutes_to_failure.iloc[0]) if len(hits) else np.nan,
        })
    det = pd.DataFrame(rows)
    res = {
        "caught": f"{int(det.caught.sum())}/{len(det)}",
        "median_lead_min": float(det.lead.median()),
        "false_alarms_per_server_day": round(false_alarms / (len(df) / 1440), 3),
    }
    for ft, g in det.groupby("fault_type"):
        res[f"lead_{ft}"] = float(g.lead.median())
    return res


def pick_threshold(val, score, budget):
    """Most sensitive threshold whose false alarms stay within the budget."""
    cands = np.quantile(score, [0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995])
    for thr in cands:
        res = evaluate(val, to_alerts(val, score, thr))
        if res["false_alarms_per_server_day"] <= budget:
            return float(thr)
    return float(cands[-1])


def baseline_alert(df):
    return (df.cpu > 90) | (df.memory > 90) | (df.disk > 90)


if __name__ == "__main__":
    data = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])
    FEATURES = [c for c in data.columns if c.endswith(SUFFIXES)]
    train_ids, val_ids, test_ids = split_servers(data)
    train = data[data.server_id.isin(train_ids)]
    val = data[data.server_id.isin(val_ids)]
    test = data[data.server_id.isin(test_ids)]
    print(f"Servers  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")
    print(f"Features: {len(FEATURES)}")

    # Baseline: also gives the false-alarm budget for the models
    budget = evaluate(val, baseline_alert(val))["false_alarms_per_server_day"]
    print(f"False-alarm budget (baseline on validation): {budget} per server-day")

    # Model 1: Isolation Forest, trained on normal behaviour only
    normal = train[~train.is_degrading].sample(150000, random_state=42)
    iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
    iso.fit(normal[FEATURES])
    if_val = -iso.score_samples(val[FEATURES])
    if_thr = pick_threshold(val, if_val, budget)

    # Model 2: XGBoost, predicts "failure within 60 minutes"
    spw = (train.target == 0).sum() / max((train.target == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=30, n_jobs=-1, random_state=42)
    xgb.fit(train[FEATURES], train.target,
            eval_set=[(val[FEATURES], val.target)], verbose=False)
    xgb_val = xgb.predict_proba(val[FEATURES])[:, 1]
    xgb_thr = pick_threshold(val, xgb_val, budget)

    # Final comparison on servers nobody trained or tuned on
    if_test = -iso.score_samples(test[FEATURES])
    xgb_test = xgb.predict_proba(test[FEATURES])[:, 1]
    results = {
        "Threshold baseline": evaluate(test, baseline_alert(test)),
        "Isolation Forest": evaluate(test, to_alerts(test, if_test, if_thr)),
        "XGBoost": evaluate(test, to_alerts(test, xgb_test, xgb_thr)),
    }
    table = pd.DataFrame(results).T
    print()
    print(table.to_string())
    print()
    print("PR-AUC  Isolation Forest:", round(average_precision_score(test.target, if_test), 3))
    print("PR-AUC  XGBoost:         ", round(average_precision_score(test.target, xgb_test), 3))

    # Save everything the dashboard will need later
    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    xgb.save_model("models/xgb.json")
    joblib.dump(iso, "models/isolation_forest.joblib")
    Path("models/config.json").write_text(json.dumps({
        "features": FEATURES, "xgb_threshold": xgb_thr,
        "if_threshold": if_thr, "smooth_minutes": SMOOTH}, indent=2))
    Path("reports/model_results.json").write_text(json.dumps({
        "results": results, "test_servers": test_ids}, indent=2))
