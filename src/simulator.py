"""Server metrics simulator for The Server Whisperer.

Generates per-minute metrics for a fleet of servers. Healthy servers follow a
daily load pattern plus harmless bursts. Faulty servers slowly degrade
(the "whisper") and then fail; their series ends at the failure.
"""
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PER_DAY = 1440
WHISPER_MINUTES = {"memory_leak": 360, "disk_fill": 720, "cpu_spike": 90}


def daily_load(t):
    """0 at midnight, 1 at noon."""
    return 0.5 - 0.5 * np.cos(2 * np.pi * (t % MIN_PER_DAY) / MIN_PER_DAY)


def simulate_server(server_id, days=10, fault=None, seed=0):
    rng = np.random.default_rng(seed)
    n = days * MIN_PER_DAY
    t = np.arange(n)
    load = daily_load(t)

    cpu = 25 + 30 * load + rng.normal(0, 3, n)
    mem = 45 + 8 * load + rng.normal(0, 1.5, n)
    disk = 50 + 0.0002 * t + rng.normal(0, 0.1, n)
    net = 100 + 300 * load + rng.normal(0, 20, n)
    errors = rng.poisson(2, n).astype(float)

    # Harmless bursts: look scary to a threshold rule, but are not failures
    for _ in range(rng.integers(4, 9)):
        pos = rng.integers(0, n - 30)
        cpu[pos:pos + 10] += rng.uniform(25, 40)

    fail_idx = None
    if fault:
        fail_idx = int(rng.integers(int(n * 0.6), n - 60))
        w = WHISPER_MINUTES[fault]
        s = fail_idx - w
        x = np.linspace(0, 1, w)
        if fault == "memory_leak":
            mem[s:fail_idx] += x * (95 - mem[s])
            errors[s:fail_idx] += rng.poisson(10 * x ** 2)
        elif fault == "disk_fill":
            disk[s:fail_idx] += x * (98 - disk[s])
            errors[s:fail_idx] += rng.poisson(6 * x ** 2)
        elif fault == "cpu_spike":
            ramp = (np.exp(3 * x) - 1) / (np.exp(3) - 1)
            cpu[s:fail_idx] += ramp * (99 - cpu[s:fail_idx])
            net[s:fail_idx] += ramp * 400
            errors[s:fail_idx] += rng.poisson(12 * ramp)

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="min"),
        "server_id": server_id,
        "cpu": np.clip(cpu, 0, 100),
        "memory": np.clip(mem, 0, 100),
        "disk": np.clip(disk, 0, 100),
        "network": np.clip(net, 0, None),
        "errors": errors,
        "fault_type": fault or "none",
        "is_degrading": False,
        "minutes_to_failure": np.nan,
    })

    if fail_idx is not None:
        df.loc[fail_idx - w:fail_idx, "is_degrading"] = True
        df["minutes_to_failure"] = fail_idx - t
        df = df.iloc[:fail_idx + 1]  # series ends when the server fails
    return df


def build_fleet(days=10):
    plan = ["none"] * 20 + ["memory_leak"] * 8 + ["disk_fill"] * 8 + ["cpu_spike"] * 8
    frames = [
        simulate_server(f"srv-{i + 1:02d}", days, None if p == "none" else p, seed=i)
        for i, p in enumerate(plan)
    ]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    out = Path("data/simulated")
    out.mkdir(parents=True, exist_ok=True)
    fleet = build_fleet()
    fleet.to_csv(out / "fleet_metrics.csv", index=False)
    print(f"Saved {len(fleet):,} rows for {fleet.server_id.nunique()} servers")
    print(fleet.groupby("fault_type").server_id.nunique())

