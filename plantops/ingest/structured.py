"""Structured pipeline: CSV exports -> SQLite warehouse, plus retrievable schema cards.

Two outputs feed the agents:
  * the warehouse itself, queried read-only by the data analyst agent;
  * "schema cards" (one per table, with business descriptions and sample values) and
    free-text ticket notes, embedded into the vector store so retrieval can find the
    right tables and the right historical repairs.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from plantops.ingest.chunking import Chunk

# Business semantics an FDE captures in discovery; the raw CSV headers don't say any of this.
TABLE_DESCRIPTIONS: dict[str, str] = {
    "plants": "One row per manufacturing plant (DET Detroit, HOU Houston, AUS Austin).",
    "production_lines": "Production lines within each plant and the product family they build.",
    "machines": "Every machine on every line. model is HX-500 (hydraulic press), CNC-7 (machining center), RW-3 (robotic welder) or CV-200 (conveyor).",
    "downtime_events": "Each period a machine was stopped. duration_min is minutes stopped. Unplanned downtime means exactly reason_code IN ('MECH','ELEC'); PM, CHG, MAT and QUAL are planned or operational stops and are never unplanned. error_code is the alarm raised, if any.",
    "maintenance_tickets": "CMMS work orders. status open/closed, priority high/medium/low, part_used is the part consumed, notes is the technician's free-text description.",
    "parts": "Spare part catalog with unit cost and preferred supplier.",
    "inventory": "Current on-hand stock per plant and part, with the reorder point. Stock is low when on_hand < reorder_point.",
    "suppliers": "Supplier master data.",
    "purchase_orders": "Spare-part purchase orders. A PO is on time when received_date <= promised_date; received_date is empty when not yet received.",
    "quality_inspections": "Weekly quality inspection results per line: units inspected, defects found, most common defect type.",
    "technicians": "Maintenance technicians. Contains PII (email, phone) that is redacted from agent access.",
}

# Columns the agents must never see (enforced in tools/sql.py as well).
PII_COLUMNS: dict[str, set[str]] = {"technicians": {"email", "phone"}}


def _infer_type(values: list[str]) -> str:
    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "TEXT"
    try:
        for v in non_empty:
            int(v)
        return "INTEGER"
    except ValueError:
        pass
    try:
        for v in non_empty:
            float(v)
        return "REAL"
    except ValueError:
        return "TEXT"


def load_warehouse(csv_dir: Path, db_path: Path) -> dict[str, int]:
    """(Re)build the SQLite warehouse from CSVs. Returns row counts per table."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    counts = {}
    for path in sorted(csv_dir.glob("*.csv")):
        table = path.stem
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
        types = {c: _infer_type([r[c] for r in rows]) for c in cols}
        conn.execute(f'CREATE TABLE "{table}" ({", ".join(f"{c} {types[c]}" for c in cols)})')

        def cast(v: str, t: str):
            if v == "":
                return None
            return int(v) if t == "INTEGER" else float(v) if t == "REAL" else v

        conn.executemany(
            f'INSERT INTO "{table}" VALUES ({", ".join("?" * len(cols))})',
            [tuple(cast(r[c], types[c]) for c in cols) for r in rows],
        )
        for c in cols:
            if c.endswith("_id"):
                conn.execute(f'CREATE INDEX "ix_{table}_{c}" ON "{table}"({c})')
        counts[table] = len(rows)
    conn.commit()
    conn.close()
    tmp.replace(db_path)  # atomic swap so readers never see a half-built warehouse
    return counts


MAX_CATEGORIES = 12


def categorical_values(conn: sqlite3.Connection, table: str, column: str, typ: str) -> list[str] | None:
    """All distinct values of a low-cardinality text column, or None.

    Text-to-SQL breaks on guessed literals ('High' vs 'high'), so the full value set goes into
    the schema the agent reads.
    """
    if typ.upper() != "TEXT" or column.endswith("_id"):
        return None
    vals = [r[0] for r in conn.execute(f'SELECT DISTINCT "{column}" FROM "{table}" WHERE "{column}" IS NOT NULL LIMIT {MAX_CATEGORIES + 1}')]
    if len(vals) > MAX_CATEGORIES or any(len(str(v)) > 40 for v in vals):
        return None
    return sorted(str(v) for v in vals)


def schema_cards(db_path: Path) -> list[Chunk]:
    conn = sqlite3.connect(db_path)
    cards = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
        hidden = PII_COLUMNS.get(table, set())
        cols = [(r[1], r[2]) for r in conn.execute(f'PRAGMA table_info("{table}")') if r[1] not in hidden]
        n = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
        lines = [f"Table {table} ({n} rows). {TABLE_DESCRIPTIONS.get(table, '')}", "Columns:"]
        for name, typ in cols:
            values = categorical_values(conn, table, name, typ)
            if values:
                lines.append(f"- {name} {typ} values (exact, case-sensitive): {' | '.join(values)}")
                continue
            samples = [
                str(r[0])
                for r in conn.execute(f'SELECT DISTINCT {name} FROM "{table}" WHERE {name} IS NOT NULL LIMIT 4')
            ]
            lines.append(f"- {name} {typ} e.g. {', '.join(s[:40] for s in samples)}")
        cards.append(Chunk(f"schema_{table}", "warehouse.db", f"Schema: {table}", table, "\n".join(lines), "schema", {"table": table}))
    conn.close()
    return cards


def ticket_note_chunks(db_path: Path) -> list[Chunk]:
    """Turn ticket free-text into retrievable chunks with filterable metadata."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    chunks = []
    q = """SELECT t.ticket_id, t.machine_id, t.plant_id, t.opened_at, t.status, t.priority,
                  t.error_code, t.part_used, t.notes, m.model
           FROM maintenance_tickets t JOIN machines m USING(machine_id)"""
    for r in conn.execute(q):
        text = (
            f"Ticket {r['ticket_id']} ({r['status']}, {r['priority']} priority) opened {r['opened_at']} "
            f"on {r['model']} {r['machine_id']} at plant {r['plant_id']}. Error {r['error_code']}. "
            f"Part used: {r['part_used'] or 'none'}. Notes: {r['notes']}"
        )
        chunks.append(Chunk(
            f"ticket_{r['ticket_id']}", "maintenance_tickets", f"Maintenance ticket {r['ticket_id']}",
            r["error_code"] or "", text, "maintenance_ticket",
            {"plant_id": r["plant_id"], "machine_id": r["machine_id"], "equipment_model": r["model"], "error_code": r["error_code"] or ""},
        ))
    conn.close()
    return chunks
