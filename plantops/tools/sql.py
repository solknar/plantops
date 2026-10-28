"""Guarded, read-only SQL access to the customer warehouse.

The guard lives in SQLite itself rather than in a regex over the query text:
  * the connection is opened with mode=ro, so writes fail at the storage layer;
  * an authorizer callback allows only SELECT-shaped operations and replaces PII
    columns with NULL (SQLITE_IGNORE), so `SELECT *` still works but leaks nothing;
  * a progress handler aborts runaway queries, and results are row-capped.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from opentelemetry import trace

from plantops.ingest.structured import PII_COLUMNS

tracer = trace.get_tracer("plantops.sql")

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


class SqlGuardError(ValueError):
    pass


@dataclass
class SqlResult:
    columns: list[str]
    rows: list[list]
    row_count: int
    truncated: bool
    elapsed_ms: float
    cached: bool

    def as_dict(self) -> dict:
        return {
            "columns": self.columns,
            "rows": self.rows,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "cached": self.cached,
        }


def _authorizer(action, arg1, arg2, dbname, source):
    if action == sqlite3.SQLITE_READ and arg2 in PII_COLUMNS.get(arg1, ()):
        return sqlite3.SQLITE_IGNORE  # column reads as NULL
    if action == sqlite3.SQLITE_READ and arg1 and arg1.startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _normalize(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"\s+", " ", sql).strip().rstrip(";").strip()


class Warehouse:
    def __init__(self, db_path: Path, max_rows: int = 50, timeout_ms: int = 2000, cache_size: int = 128):
        self.db_path = db_path
        self.max_rows = max_rows
        self.timeout_ms = timeout_ms
        self._cache: OrderedDict[tuple, SqlResult] = OrderedDict()
        self._cache_size = cache_size

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.set_authorizer(_authorizer)
        return conn

    def query(self, sql: str) -> SqlResult:
        norm = _normalize(sql)
        if not norm:
            raise SqlGuardError("empty query")
        if ";" in norm:
            raise SqlGuardError("only a single statement is allowed")
        if not re.match(r"(?i)^(select|with)\b", norm):
            raise SqlGuardError("only SELECT queries are allowed")

        # Key on the warehouse file version so a re-ingest invalidates the cache.
        key = (norm, self.db_path.stat().st_mtime_ns)
        with tracer.start_as_current_span("sql.execute") as span:
            span.set_attribute("db.system", "sqlite")
            span.set_attribute("db.statement", norm[:2000])
            if key in self._cache:
                self._cache.move_to_end(key)
                hit = self._cache[key]
                span.set_attribute("sql.cached", True)
                return SqlResult(hit.columns, hit.rows, hit.row_count, hit.truncated, 0.0, True)

            conn = self._connect()
            deadline = time.perf_counter() + self.timeout_ms / 1000
            conn.set_progress_handler(lambda: 1 if time.perf_counter() > deadline else 0, 10_000)
            t0 = time.perf_counter()
            try:
                cur = conn.execute(norm)
                rows = cur.fetchmany(self.max_rows + 1)
                columns = [d[0] for d in cur.description or []]
            except sqlite3.DatabaseError as e:
                msg = str(e)
                if "interrupted" in msg:
                    msg = f"query exceeded {self.timeout_ms} ms timeout"
                elif "not authorized" in msg:
                    msg = "query touches objects or operations that are not permitted (read-only SELECT only)"
                span.set_attribute("sql.error", msg)
                raise SqlGuardError(msg) from e
            finally:
                conn.close()
            elapsed = (time.perf_counter() - t0) * 1000
            truncated = len(rows) > self.max_rows
            rows = [list(r) for r in rows[: self.max_rows]]
            result = SqlResult(columns, rows, len(rows), truncated, elapsed, False)
            span.set_attribute("sql.rows", len(rows))
            span.set_attribute("sql.truncated", truncated)
            span.set_attribute("sql.cached", False)
            self._cache[key] = result
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return result
