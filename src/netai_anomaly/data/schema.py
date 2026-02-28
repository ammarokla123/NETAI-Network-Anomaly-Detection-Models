"""SQLite schema for network telemetry storage.

Mirrors the data layout used by perfSONAR and traceroute collectors on NRP.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS network_tests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    destination TEXT    NOT NULL,
    test_type   TEXT    NOT NULL  -- throughput | latency | trace
);

CREATE TABLE IF NOT EXISTS throughput_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id         INTEGER NOT NULL REFERENCES network_tests(id),
    throughput_mbps REAL    NOT NULL,
    retransmits     INTEGER NOT NULL DEFAULT 0,
    duration_sec    REAL    NOT NULL DEFAULT 10.0
);

CREATE TABLE IF NOT EXISTS latency_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id         INTEGER NOT NULL REFERENCES network_tests(id),
    latency_ms      REAL    NOT NULL,
    jitter_ms       REAL    NOT NULL DEFAULT 0.0,
    packet_loss_pct REAL    NOT NULL DEFAULT 0.0,
    packets_sent    INTEGER NOT NULL DEFAULT 100
);

CREATE TABLE IF NOT EXISTS traceroute_hops (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id     INTEGER NOT NULL REFERENCES network_tests(id),
    hop_number  INTEGER NOT NULL,
    hop_ip      TEXT    NOT NULL,
    rtt_ms      REAL,
    is_timeout  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS telemetry (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    destination     TEXT    NOT NULL,
    throughput_mbps REAL    NOT NULL,
    latency_ms      REAL    NOT NULL,
    packet_loss_pct REAL    NOT NULL,
    retransmits     INTEGER NOT NULL,
    jitter_ms       REAL    NOT NULL,
    is_anomaly      INTEGER NOT NULL DEFAULT 0,
    anomaly_type    TEXT    DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_telemetry_ts      ON telemetry(timestamp);
CREATE INDEX IF NOT EXISTS idx_telemetry_src_dst ON telemetry(source, destination);
CREATE INDEX IF NOT EXISTS idx_telemetry_anomaly ON telemetry(is_anomaly);
"""


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Create the database and tables if they don't exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn
