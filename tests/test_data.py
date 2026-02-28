"""Tests for data generation and SQLite schema."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from netai_anomaly.data.generator import (
    generate_telemetry,
    load_from_sqlite,
    save_to_sqlite,
)
from netai_anomaly.data.schema import init_db


class TestSchema:
    def test_init_db_creates_tables(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()
            assert "telemetry" in tables
            assert "network_tests" in tables
            assert "throughput_results" in tables
            assert "latency_results" in tables
            assert "traceroute_hops" in tables

    def test_init_db_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn1 = init_db(db_path)
            conn1.close()
            conn2 = init_db(db_path)
            conn2.close()


class TestGenerator:
    def test_generate_default(self):
        df = generate_telemetry(num_samples=100, seed=42)
        assert len(df) == 100
        assert "throughput_mbps" in df.columns
        assert "latency_ms" in df.columns
        assert "packet_loss_pct" in df.columns
        assert "retransmits" in df.columns
        assert "jitter_ms" in df.columns
        assert "is_anomaly" in df.columns
        assert "anomaly_type" in df.columns

    def test_anomaly_ratio(self):
        df = generate_telemetry(num_samples=10000, anomaly_ratio=0.1, seed=42)
        actual_ratio = df["is_anomaly"].mean()
        assert 0.07 < actual_ratio < 0.13  # within reasonable range

    def test_no_negative_values(self):
        df = generate_telemetry(num_samples=1000, seed=42)
        for col in ["throughput_mbps", "latency_ms", "packet_loss_pct", "jitter_ms"]:
            assert (df[col] >= 0).all(), f"{col} has negative values"

    def test_retransmits_are_integers(self):
        df = generate_telemetry(num_samples=100, seed=42)
        assert df["retransmits"].dtype in ("int64", "int32", "int")

    def test_anomaly_types_present(self):
        df = generate_telemetry(num_samples=5000, anomaly_ratio=0.2, seed=42)
        anomaly_types = df[df["is_anomaly"] == 1]["anomaly_type"].unique()
        assert len(anomaly_types) >= 3  # at least 3 of the 5 types

    def test_timestamps_ordered(self):
        df = generate_telemetry(num_samples=100, seed=42)
        assert df["timestamp"].is_monotonic_increasing

    def test_reproducibility(self):
        df1 = generate_telemetry(num_samples=100, seed=42)
        df2 = generate_telemetry(num_samples=100, seed=42)
        pd.testing.assert_frame_equal(df1, df2)


class TestSQLiteIO:
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            df_orig = generate_telemetry(num_samples=50, seed=42)
            save_to_sqlite(df_orig, db_path)
            df_loaded = load_from_sqlite(db_path)
            assert len(df_loaded) == 50
            assert set(df_orig.columns).issubset(set(df_loaded.columns))

    def test_load_preserves_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            df = generate_telemetry(num_samples=20, seed=42)
            save_to_sqlite(df, db_path)
            df_loaded = load_from_sqlite(db_path)
            assert df_loaded["throughput_mbps"].dtype == "float64"
