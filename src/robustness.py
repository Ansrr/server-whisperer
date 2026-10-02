"""Repeat the model comparison over several random server splits."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier

from train import SUFFIXES, baseline_alert, evaluate, pick_threshold, split_servers, to_alerts

SEEDS = [1, 2, 3, 4, 5]


def run_once(data, features, seed):
    train_ids, val_ids, test_ids = split_servers(data, seed)
    train = data[data.server_id.isin(train_ids)]
    val = data[data.server_id.isin(val_ids)]
    test = data[data.server_id.isin(test_ids)]
    budget = evaluate(val, baseline_alert(val))["false_alarms_per_server_day"]

    normal = train[~train.is_degrading].sample(150000, random_state=seed)
    iso = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
    iso.fit(normal[features])
    if_thr = pick_threshold(val, -iso.score_samples(val[features]), budget)

    spw = (train.target == 0).sum() / max((train.target == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="aucpr",
        early_stopping_rounds=30, n_jobs=-1, random_state=seed)
    xgb.fit(train[features], train.target,
            eval_set=[(val[features], val.target)], verbose=False)
    xgb_thr = pick_threshold(val, xgb.predict_proba(val[features])[:, 1], budget)

    return {
        "Threshold baseline": evaluate(test, baseline_alert(test)),
        "Isolation Forest": evaluate(
            test, to_alerts(test, -iso.score_samples(test[features]), if_thr)),
        "XGBoost": evaluate(
            test, to_alerts(test, xgb.predict_proba(test[features])[:, 1], xgb_thr)),
    }


def summarize(runs):
    rows = {}
    for method in runs[0]:
        res = [r[method] for r in runs]
        caught = sum(int(x["caught"].split("/")[0]) for x in res)
        total = sum(int(x["caught"].split("/")[1]) for x in res)
        row = {"caught": f"{caught}/{total}"}
        for key in res[0]:
            if key == "caught":
                continue
            vals = np.array([x[key] for x in res], dtype=float)
            row[key] = f"{np.nanmean(vals):.2f} +/- {np.nanstd(vals):.2f}"
        rows[method] = row
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    data = pd.read_csv("data/processed/features.csv", parse_dates=["timestamp"])
    features = [c for c in data.columns if c.endswith(SUFFIXES)]
    runs = []
    for seed in SEEDS:
        print(f"Running split {seed} of {len(SEEDS)} ...", flush=True)
        runs.append(run_once(data, features, seed))
    table = summarize(runs)
    print()
    print(table.to_string())
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/robustness_results.csv")
    Path("reports/robustness_runs.json").write_text(json.dumps(runs, indent=2))
