import pytest

from plantops.retrieval.store import build_where
from plantops.tools import functions as fn
from plantops.tools import runtime


def test_hybrid_search_finds_error_code_doc(settings):
    res = fn.search_documents("What does E-42 mean on the HX-500?", doc_type="equipment_manual")
    assert res["results"][0]["chunk_id"].startswith("hx500_hydraulic_press_manual")


def test_metadata_filters_narrow_results(settings):
    res = fn.search_documents("wire feed jam", doc_type="maintenance_ticket", plant_id="HOU", k=10)
    assert res["count"] > 0
    assert all(r["plant_id"] == "HOU" and r["doc_type"] == "maintenance_ticket" for r in res["results"])


def test_filter_whitelist_blocks_injection():
    with pytest.raises(ValueError):
        build_where({"text": "x"})
    with pytest.raises(ValueError):
        build_where({"plant_id": "DET' OR 1=1 --"})
    assert build_where({"plant_id": "DET", "doc_type": ""}) == "`plant_id` = 'DET'"
    assert "error" in fn.search_documents("x", plant_id="DET' OR '1'='1")


def test_fetch_chunk_round_trip(settings):
    hit = fn.search_documents("lockout tagout steps")["results"][0]
    full = fn.fetch_chunk(hit["chunk_id"])
    assert full["chunk_id"] == hit["chunk_id"] and len(full["text"]) >= len(hit["text"].rstrip(" …"))
    assert "error" in fn.fetch_chunk("nope#123")


def test_schema_rag_picks_relevant_tables(settings):
    tables = [t["table"] for t in fn.find_relevant_tables("which suppliers deliver purchase orders late?")["tables"]]
    assert {"suppliers", "purchase_orders"} & set(tables[:2])


def test_sql_redacts_pii(settings):
    res = fn.run_sql("SELECT name, email, phone FROM technicians LIMIT 3")
    assert res["rows"] and all(r[1] is None and r[2] is None for r in res["rows"])
    star = fn.run_sql("SELECT * FROM technicians LIMIT 1")
    assert "@" not in str(star["rows"])
    desc = fn.describe_table("technicians")
    assert desc["redacted_columns"] == ["email", "phone"]
    assert all(c["name"] not in ("email", "phone") for c in desc["columns"])


@pytest.mark.parametrize("sql,fragment", [
    ("DELETE FROM plants", "only SELECT"),
    ("SELECT 1; DROP TABLE plants", "single statement"),
    ("SELECT name FROM sqlite_master", "prohibited"),
    ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c", "timeout"),
    ("SELECT nope FROM plants", "no such column"),
])
def test_sql_guard_rejects(settings, sql, fragment):
    res = fn.run_sql(sql)
    assert fragment in res["error"]


def test_sql_row_cap_and_cache(settings):
    first = fn.run_sql("SELECT * FROM downtime_events")
    assert first["row_count"] == 50 and first["truncated"] and not first["cached"]
    assert fn.run_sql("SELECT *   FROM downtime_events;")["cached"]


def test_warehouse_is_read_only_at_storage_layer(settings):
    conn = runtime.warehouse()._connect()
    conn.set_authorizer(None)
    with pytest.raises(Exception, match="readonly"):
        conn.execute("DELETE FROM plants")
