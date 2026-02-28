"""Synthetic network telemetry data generator.

Generates realistic perfSONAR-style measurements with configurable anomaly
injection for training and evaluating anomaly detection models.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from netai_anomaly.data.schema import init_db

# Normal metric distributions (mean, std)
NORMAL_PROFILES: dict[str, dict[str, tuple[float, float]]] = {
    "high_bw": {
        "throughput_mbps": (9500.0, 500.0),
        "latency_ms": (5.0, 1.0),
        "packet_loss_pct": (0.01, 0.005),
        "retransmits": (2.0, 1.5),
        "jitter_ms": (0.5, 0.2),
    },
    "medium_bw": {
        "throughput_mbps": (5000.0, 800.0),
        "latency_ms": (15.0, 3.0),
        "packet_loss_pct": (0.05, 0.02),
        "retransmits": (5.0, 3.0),
        "jitter_ms": (1.0, 0.4),
    },
    "low_bw": {
        "throughput_mbps": (1000.0, 200.0),
        "latency_ms": (50.0, 10.0),
        "packet_loss_pct": (0.1, 0.05),
        "retransmits": (10.0, 5.0),
        "jitter_ms": (3.0, 1.0),
    },
}

# Anomaly patterns: each defines how to perturb normal values.
# Functions take (value, rng) to ensure reproducibility.
ANOMALY_TYPES = {
    "slow_link": {
        "throughput_mbps": lambda v, rng: v * rng.uniform(0.05, 0.3),
        "latency_ms": lambda v, rng: v * rng.uniform(3.0, 8.0),
    },
    "high_packet_loss": {
        "packet_loss_pct": lambda _, rng: rng.uniform(5.0, 30.0),
    },
    "excessive_retransmits": {
        "retransmits": lambda _, rng: int(rng.uniform(50, 500)),
    },
    "failed_test": {
        "throughput_mbps": lambda _, rng: 0.0,
        "latency_ms": lambda _, rng: 0.0,
        "packet_loss_pct": lambda _, rng: 100.0,
        "retransmits": lambda _, rng: 0,
    },
    "high_jitter": {
        "jitter_ms": lambda _, rng: rng.uniform(20.0, 100.0),
        "latency_ms": lambda v, rng: v * rng.uniform(1.5, 3.0),
    },
}

SOURCES = [
    "nrp-node-ucsd.edu",
    "nrp-node-chicago.edu",
    "nrp-node-nebraska.edu",
    "nrp-node-michigan.edu",
]
DESTINATIONS = [
    "nrp-node-stanford.edu",
    "nrp-node-mit.edu",
    "nrp-node-georgia.edu",
    "nrp-node-washington.edu",
]


def _generate_normal_sample(
    rng: np.random.Generator,
    profile: dict[str, tuple[float, float]],
) -> dict[str, float]:
    """Generate a single normal measurement."""
    sample: dict[str, float] = {}
    for metric, (mean, std) in profile.items():
        val = rng.normal(mean, std)
        if metric == "retransmits":
            val = max(0, int(round(val)))
        else:
            val = max(0.0, val)
        sample[metric] = val
    return sample


def _apply_anomaly(
    sample: dict[str, float],
    anomaly_type: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Apply an anomaly pattern to a normal sample."""
    perturbed = sample.copy()
    for metric, transform in ANOMALY_TYPES[anomaly_type].items():
        perturbed[metric] = transform(sample[metric], rng)
    return perturbed


def generate_telemetry(
    num_samples: int = 50000,
    anomaly_ratio: float = 0.05,
    seed: int = 42,
    start_time: str | None = None,
    interval_seconds: int = 60,
) -> pd.DataFrame:
    """Generate synthetic network telemetry data.

    Returns a DataFrame with columns matching the ``telemetry`` table schema.
    """
    rng = np.random.default_rng(seed)

    if start_time is None:
        start_time = "2025-01-01T00:00:00"
    base_ts = datetime.fromisoformat(start_time)

    profile_names = list(NORMAL_PROFILES.keys())
    anomaly_names = list(ANOMALY_TYPES.keys())

    records: list[dict[str, Any]] = []
    for i in range(num_samples):
        ts = base_ts + timedelta(seconds=i * interval_seconds)
        src = rng.choice(SOURCES)
        dst = rng.choice(DESTINATIONS)
        profile = NORMAL_PROFILES[rng.choice(profile_names)]

        sample = _generate_normal_sample(rng, profile)
        is_anomaly = rng.random() < anomaly_ratio
        anomaly_type = None

        if is_anomaly:
            anomaly_type = rng.choice(anomaly_names)
            sample = _apply_anomaly(sample, anomaly_type, rng)

        records.append(
            {
                "timestamp": ts.isoformat(),
                "source": src,
                "destination": dst,
                "throughput_mbps": round(sample["throughput_mbps"], 2),
                "latency_ms": round(sample["latency_ms"], 3),
                "packet_loss_pct": round(sample["packet_loss_pct"], 4),
                "retransmits": int(sample["retransmits"]),
                "jitter_ms": round(sample["jitter_ms"], 3),
                "is_anomaly": int(is_anomaly),
                "anomaly_type": anomaly_type,
            }
        )

    return pd.DataFrame(records)


def save_to_sqlite(df: pd.DataFrame, db_path: str) -> None:
    """Persist a telemetry DataFrame to the SQLite database."""
    conn = init_db(db_path)
    df.to_sql("telemetry", conn, if_exists="replace", index=False)
    conn.close()


def load_from_sqlite(db_path: str) -> pd.DataFrame:
    """Load telemetry data from the SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM telemetry ORDER BY timestamp", conn)
    conn.close()
    return df
