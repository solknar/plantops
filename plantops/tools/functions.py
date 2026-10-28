"""Tool functions exposed to agents (in-process) and to the MCP server.

Each returns a JSON-serializable dict and never raises: errors come back as
{"error": ...} so a ReAct agent can read the observation and correct itself.
Results are kept compact on purpose; tool output is the largest driver of
prompt tokens in a multi-step agent.
"""

from __future__ import annotations

import sqlite3

from plantops.ingest.structured import PII_COLUMNS, TABLE_DESCRIPTIONS, categorical_values
from plantops.retrieval.store import FILTER_FIELDS
from plantops.tools import runtime
from plantops.tools.sql import SqlGuardError


def list_tables() -> dict:
    """List the warehouse tables with a one-line business description and row count."""
    wh = runtime.warehouse()
    out = []
    for name, desc in TABLE_DESCRIPTIONS.items():
        try:
            n = wh.query(f'SELECT COUNT(*) FROM "{name}"').rows[0][0]
        except SqlGuardError:
            continue
        out.append({"table": name, "rows": n, "description": desc})
    return {"tables": out}


def describe_table(table: str) -> dict:
    """Return the columns, types, allowed category values and a few sample rows for one table.

    Args:
        table: Table name exactly as returned by list_tables.
    """
    if table not in TABLE_DESCRIPTIONS:
        return {"error": f"unknown table {table!r}", "known_tables": sorted(TABLE_DESCRIPTIONS)}
    wh = runtime.warehouse()
    conn = sqlite3.connect(f"file:{wh.db_path}?mode=ro", uri=True)
    hidden = PII_COLUMNS.get(table, set())
    cols = [{"name": r[1], "type": r[2]} for r in conn.execute(f'PRAGMA table_info("{table}")') if r[1] not in hidden]
    for c in cols:
        values = categorical_values(conn, table, c["name"], c["type"])
        if values:
            c["values"] = values
    conn.close()
    sample = wh.query(f'SELECT {", ".join(c["name"] for c in cols)} FROM "{table}" LIMIT 3')
    return {
        "table": table,
        "description": TABLE_DESCRIPTIONS[table],
        "columns": cols,
        "redacted_columns": sorted(hidden),
        "sample_rows": sample.rows,
    }


def run_sql(query: str) -> dict:
    """Run one read-only SQLite SELECT against the warehouse and return up to 50 rows.

    Use SQLite syntax (e.g. date('2026-09-30', '-90 day'), strftime). Aggregate in SQL
    rather than pulling raw rows. PII columns come back as NULL.

    Args:
        query: A single SELECT or WITH ... SELECT statement.
    """
    try:
        return runtime.warehouse().query(query).as_dict() | {"sql": query}
    except SqlGuardError as e:
        return {"error": str(e), "sql": query}


def search_documents(
    query: str,
    doc_type: str = "",
    equipment_model: str = "",
    plant_id: str = "",
    error_code: str = "",
    k: int = 5,
) -> dict:
    """Hybrid (semantic + keyword) search over manuals, SOPs, policies, incident reports and maintenance ticket notes.

    Args:
        query: Natural-language search query. Include exact codes (E-42, P-1001) when known.
        doc_type: Optional filter: equipment_manual, sop, policy, incident_report, maintenance_ticket, schema.
        equipment_model: Optional filter: HX-500, CNC-7, RW-3 or CV-200.
        plant_id: Optional filter: DET, HOU or AUS. Only tickets and incident reports carry a plant.
        error_code: Optional filter on the alarm code, e.g. E-42. Only tickets carry an error code.
        k: Number of results (1-10).
    """
    filters = {"doc_type": doc_type, "equipment_model": equipment_model, "plant_id": plant_id, "error_code": error_code}
    try:
        hits = runtime.store().search(query, k=max(1, min(int(k), 10)), filters=filters)
    except ValueError as e:
        return {"error": str(e), "allowed_filters": list(FILTER_FIELDS)}
    return {"results": [h.as_tool_result() for h in hits], "count": len(hits)}


def find_relevant_tables(question: str) -> dict:
    """Retrieve the schema cards of the warehouse tables most relevant to a question (schema RAG for text-to-SQL).

    Args:
        question: The analytical question you need to answer with SQL.
    """
    hits = runtime.store().search(question, k=4, filters={"doc_type": "schema"})
    return {"tables": [{"table": h.metadata.get("table") or h.section, "card": h.text} for h in hits]}


def fetch_chunk(chunk_id: str) -> dict:
    """Fetch the full text of a document chunk by its chunk_id, to verify a citation.

    Args:
        chunk_id: The chunk_id returned by search_documents, e.g. 'hx500_hydraulic_press_manual#cf3483d0ee'.
    """
    hit = runtime.store().get(chunk_id)
    if hit is None:
        return {"error": f"chunk {chunk_id!r} not found"}
    return hit.as_tool_result(max_chars=4000)


DATA_TOOLS = [find_relevant_tables, list_tables, describe_table, run_sql]
DOC_TOOLS = [search_documents, fetch_chunk]
VERIFY_TOOLS = [fetch_chunk, run_sql]
ALL_TOOLS = [find_relevant_tables, list_tables, describe_table, run_sql, search_documents, fetch_chunk]
