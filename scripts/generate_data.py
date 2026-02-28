#!/usr/bin/env python3
"""Generate synthetic network telemetry data and store it in SQLite."""

from __future__ import annotations

import argparse
import logging

from netai_anomaly.data.generator import generate_telemetry, save_to_sqlite

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic network telemetry")
    parser.add_argument("--db-path", default="data/network_telemetry.db", help="SQLite DB path")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples")
    parser.add_argument("--anomaly-ratio", type=float, default=0.05, help="Anomaly ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    logger.info(
        "Generating %d samples (anomaly_ratio=%.2f, seed=%d)",
        args.num_samples,
        args.anomaly_ratio,
        args.seed,
    )

    df = generate_telemetry(
        num_samples=args.num_samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed,
    )

    logger.info("Anomalies: %d / %d (%.2f%%)", df["is_anomaly"].sum(), len(df),
                100 * df["is_anomaly"].mean())
    logger.info("Anomaly types:\n%s", df[df["is_anomaly"] == 1]["anomaly_type"].value_counts().to_string())

    save_to_sqlite(df, args.db_path)
    logger.info("Saved to %s", args.db_path)


if __name__ == "__main__":
    main()
