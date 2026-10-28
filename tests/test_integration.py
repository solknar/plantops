import json

from fastapi.testclient import TestClient

from plantops import api
from plantops.evals import runner
from plantops.mcp_server import build_server
from plantops.testing import Call
from plantops.tools import runtime


async def test_mcp_server_exposes_guarded_tools(settings):
    server = build_server()
    names = {t.name for t in await server.list_tools()}
    assert names == {"find_relevant_tables", "list_tables", "describe_table", "run_sql", "search_documents", "fetch_chunk"}
    tool = next(t for t in await server.list_tools() if t.name == "run_sql")
    assert tool.annotations.read_only_hint is True and tool.annotations.destructive_hint is False

    res = await server.call_tool("run_sql", {"query": "SELECT email FROM technicians LIMIT 1"})
    payload = res.structured_content or json.loads(res.content[0].text)
    payload = payload.get("result", payload)
    assert payload["rows"] == [[None]]


def test_eval_scoring_handles_numbers_and_formatting():
    s = runner.score_answer("DET had 2,401 minutes; see P-1001.", ["DET", "2401", "P-1001"], [])
    assert s["fact_recall"] == 1.0
    # 240 must not match inside 2401
    assert runner.score_answer("2401 minutes", ["240"], [])["missing"] == ["240"]
    s = runner.score_answer("There are seven steps", [], [["7", "seven"], ["verify"]])
    assert s["fact_recall"] == 0.5 and s["missing"] == ["verify"]


def test_expected_facts_come_from_the_warehouse(settings):
    item = next(i for i in runner.load_golden() if i["id"] == "hyb-suppliers")
    facts = runner.expected_facts(item, settings.warehouse_path)
    assert facts == ["95%", "Lone Star Industrial", "Midwest Hydraulics", "Precision Spindle GmbH", "WeldTech Supply"]


def test_retrieval_eval_meets_threshold_offline(settings):
    report = {"retrieval": runner.retrieval_eval(runtime.store(), runner.load_golden())}
    assert report["retrieval"]["retrieval_recall_at_k"] >= runner.THRESHOLDS["retrieval_recall_at_k"]
    assert runner.gate(report) == []


def test_citation_check(settings):
    hit = runtime.store().search("supplier on-time delivery", k=1)[0]
    good = runner.check_citations(f"95% [doc:{hit.id}]", runtime.store(), ["supplier_quality_agreement"])
    assert good["citation_validity"] == 1.0 and good["cites_relevant_doc"]
    bad = runner.check_citations("95% [doc:made_up#123]", runtime.store(), ["supplier_quality_agreement"])
    assert bad["citation_validity"] == 0.0


def test_api_chat_metrics_and_trace(service, monkeypatch):
    monkeypatch.setattr(api, "_service", service)
    service.llm.script.update({"coordinator": ["**Answer**: ok"], "critic": [Call("exit_loop", {})]})
    client = TestClient(api.app)

    res = client.post("/api/chat", json={"question": "How many open high-priority tickets?"}).json()
    assert res["answer"] == "**Answer**: ok" and res["session_id"]
    follow_up = client.post("/api/chat", json={"question": "And at Detroit?", "session_id": res["session_id"]}).json()
    assert follow_up["session_id"] == res["session_id"]

    m = client.get("/api/metrics").json()
    assert m["requests"] >= 2 and m["by_model"] and "p95" in m["latency_ms"]
    spans = client.get(f"/api/traces/{res['trace_id']}").json()
    names = [s["name"] for s in spans]
    assert names[0] == "POST /api/chat" and "plantops.request" in names  # HTTP span wraps the agent run
    assert client.get("/api/traces/doesnotexist").status_code == 404
    assert "PlantOps" in client.get("/").text


def test_numeric_fact_matching():
    from plantops.evals.runner import _contains

    assert _contains("PM share is 8.46%", "8.5")
    assert not _contains("PM share is 8.44%", "8.5")
    assert not _contains("about 2.4 tickets", "2")
    assert _contains("There are 2.", "2")
    assert _contains("2,401 minutes", "2401")


def test_schema_cards_list_exact_category_values(settings):
    from plantops.ingest.structured import schema_cards

    card = next(c for c in schema_cards(settings.warehouse_path) if c.section == "maintenance_tickets")
    assert "status TEXT values (exact, case-sensitive): closed | open" in card.text
