"""SQLite persistence for spans and LLM metrics, so traces survive restarts and can be queried."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS spans (
  trace_id TEXT, span_id TEXT PRIMARY KEY, parent_id TEXT, name TEXT,
  start_ns INTEGER, end_ns INTEGER, status TEXT, attributes TEXT
);
CREATE INDEX IF NOT EXISTS ix_spans_trace ON spans(trace_id);
CREATE TABLE IF NOT EXISTS requests (
  trace_id TEXT PRIMARY KEY, session_id TEXT, user_id TEXT, question TEXT, started_at REAL,
  latency_ms REAL, llm_calls INTEGER, tool_calls INTEGER, tool_errors INTEGER,
  input_tokens INTEGER, output_tokens INTEGER, thoughts_tokens INTEGER, cached_tokens INTEGER,
  cost_usd REAL, reflections INTEGER, tokens_saved_by_trimming INTEGER, per_agent TEXT
);
CREATE TABLE IF NOT EXISTS model_calls (
  trace_id TEXT, agent TEXT, model TEXT, started_at REAL, latency_ms REAL, ttft_ms REAL,
  input_tokens INTEGER, output_tokens INTEGER, thoughts_tokens INTEGER, cached_tokens INTEGER,
  tokens_per_sec REAL, cost_usd REAL, finish_reason TEXT
);
CREATE INDEX IF NOT EXISTS ix_model_calls_trace ON model_calls(trace_id);
CREATE TABLE IF NOT EXISTS tool_calls (
  trace_id TEXT, agent TEXT, tool TEXT, started_at REAL, latency_ms REAL, error INTEGER, result_chars INTEGER
);
"""


class TelemetryStore:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def write(self, table: str, rows: list[dict]) -> None:
        if not rows:
            return
        cols = list(rows[0].keys())
        sql = f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        with self._lock, self._conn() as c:
            c.executemany(sql, [tuple(r[k] for k in cols) for r in rows])

    def query(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, params)]

    def trace_tree(self, trace_id: str) -> list[dict]:
        spans = self.query("SELECT * FROM spans WHERE trace_id = ? ORDER BY start_ns", (trace_id,))
        if not spans:
            return []
        t0 = spans[0]["start_ns"]
        by_id = {s["span_id"]: s for s in spans}
        depth: dict[str, int] = {}

        def d(s):
            if s["span_id"] not in depth:
                p = by_id.get(s["parent_id"])
                depth[s["span_id"]] = 0 if p is None else d(p) + 1
            return depth[s["span_id"]]

        return [
            {
                "span_id": s["span_id"], "parent_id": s["parent_id"], "name": s["name"], "depth": d(s),
                "offset_ms": round((s["start_ns"] - t0) / 1e6, 1),
                "duration_ms": round((s["end_ns"] - s["start_ns"]) / 1e6, 1),
                "status": s["status"], "attributes": json.loads(s["attributes"]),
            }
            for s in spans
        ]
