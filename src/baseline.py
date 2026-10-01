"""Threshold baseline: the 'before' result for The Server Whisperer."""
import json
from pathlib import Path

import pandas as pd

CPU_LIMIT = 90
MEM_LIMIT = 90
DISK_LIMIT = 90


def run_baseline(path="data/simulated/fleet_metrics.csv"):
    fleet = pd.read_csv(path, parse_dates=["timestamp"])
    fleet["alert"] = (
        (fleet.cpu > CPU_LIMIT)
        | (fleet.memory > MEM_LIMIT)
        | (fleet.disk > DISK_LIMIT)
    )
    # An alert "event" starts when the rule switches from off to on
    prev = fleet.groupby("server_id")["alert"].shift(1, fill_value=False).astype(bool)
    fleet["alert_event"] = fleet["alert"] & ~prev

    rows = []
    for srv, d in fleet[fleet.fault_type != "none"].groupby("server_id"):
        hits = d[d.is_degrading & d.alert]
        rows.append({
            "server_id": srv,
            "fault_type": d.fault_type.iloc[0],
            "caught": len(hits) > 0,
            "lead_minutes": float(hits.minutes_to_failure.iloc[0]) if len(hits) else None,
        })
    det = pd.DataFrame(rows)

    false_alarms = int((fleet.alert_event & ~fleet.is_degrading).sum())
    server_days = len(fleet) / 1440
    result = {
        "failures_total": len(det),
        "failures_caught": int(det.caught.sum()),
        "median_lead_minutes": float(det.lead_minutes.median()),
        "false_alarms": false_alarms,
        "false_alarms_per_server_day": round(false_alarms / server_days, 2),
    }
    return det, result


if __name__ == "__main__":
    det, result = run_baseline()
    print(det.to_string(index=False))
    print()
    for key, value in result.items():
        print(f"{key}: {value}")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_results.json").write_text(json.dumps(result, indent=2))
