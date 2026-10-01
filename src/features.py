"""Feature engineering for The Server Whisperer."""
from pathlib import Path

import pandas as pd

METRICS = ["cpu", "memory", "disk", "network", "errors"]
HORIZON = 60  # target = 1 if the server fails within the next 60 minutes


def add_features(df):
    df = df.sort_values(["server_id", "timestamp"]).reset_index(drop=True)
    sid = df["server_id"]
    feats = {}
    for m in METRICS:
        g = df.groupby("server_id")[m]
        mean15 = g.transform(lambda s: s.rolling(15, min_periods=5).mean())
        mean60 = g.transform(lambda s: s.rolling(60, min_periods=20).mean())
        mean1d = g.transform(lambda s: s.rolling(1440, min_periods=240).mean())
        feats[f"{m}_mean15"] = mean15
        feats[f"{m}_mean60"] = mean60
        feats[f"{m}_std60"] = g.transform(lambda s: s.rolling(60, min_periods=20).std())
        feats[f"{m}_max60"] = g.transform(lambda s: s.rolling(60, min_periods=20).max())
        feats[f"{m}_drift"] = mean60 - mean1d
        feats[f"{m}_slope"] = (mean15 - mean15.groupby(sid).shift(60)) / 60
    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    out["target"] = (
        (out.minutes_to_failure >= 0) & (out.minutes_to_failure <= HORIZON)
    ).astype(int)
    return out.dropna(subset=list(feats)).reset_index(drop=True)


if __name__ == "__main__":
    raw = pd.read_csv("data/simulated/fleet_metrics.csv", parse_dates=["timestamp"])
    data = add_features(raw)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    data.to_csv("data/processed/features.csv", index=False)
    print("Rows:", f"{len(data):,}", "| Features:", len(METRICS) * 6)
    print("Failing-soon rows:", int(data.target.sum()),
          f"({data.target.mean():.2%} of all rows)")
