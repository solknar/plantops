from google.genai import types

from plantops.agents.context_budget import ContextBudgetPlugin
from plantops.telemetry.metrics import cost_usd
from plantops.testing import Call

SQL = "SELECT supplier_id FROM purchase_orders WHERE received_date IS NOT NULL GROUP BY 1 HAVING AVG(received_date <= promised_date) < 0.95"


def _script_with_one_revision():
    return {
        "coordinator": [
            Call("knowledge_researcher", {"request": "What on-time rate does QA-SUP-002 require?"}),
            Call("data_analyst", {"request": "Suppliers with on-time rate below 95%"}),
            "**Answer**: draft",
        ],
        "knowledge_researcher": [Call("search_documents", {"query": "on-time delivery rate", "doc_type": "policy"}), "95% [doc:x]"],
        "data_analyst": [Call("find_relevant_tables", {"question": "supplier on-time"}), Call("run_sql", {"query": SQL}), "S-01, S-02"],
        "critic": [Call("run_sql", {"query": SQL}), "1. Missing supplier names.", Call("exit_loop", {})],
        "refiner": ["**Answer**: revised with names"],
    }


async def test_full_graph_delegation_and_reflection(service):
    service.llm.script.update(_script_with_one_revision())
    res = await service.ask("Which suppliers are below the on-time requirement?", session_id="t-reflect")

    assert res["answer"] == "**Answer**: revised with names"
    assert res["approved_by_critic"] is True
    calls = [(s["agent"], s["name"]) for s in res["steps"] if s["type"] == "tool_call"]
    assert ("coordinator", "knowledge_researcher") in calls and ("coordinator", "data_analyst") in calls
    assert calls[-1] == ("critic", "exit_loop")

    m = res["metrics"]
    assert m["reflections"] == 2 and m["agent_runs"]["refiner"] == 1
    assert set(m["per_agent"]) == {"coordinator", "knowledge_researcher", "data_analyst", "critic", "refiner"}
    assert m["llm_calls"] == 12 and m["input_tokens"] > 0 and m["cost_usd"] > 0

    # The specialists ran inside child runners but share the request trace.
    names = [s["name"] for s in service.telemetry.trace_tree(res["trace_id"])]
    for expected in ("plantops.request", "invoke_agent data_analyst", "execute_tool run_sql", "sql.execute", "retrieval.hybrid_search"):
        assert expected in names
    stored = service.telemetry.query("SELECT llm_calls, reflections FROM requests WHERE trace_id = ?", (res["trace_id"],))
    assert stored == [{"llm_calls": 12, "reflections": 2}]


async def test_critic_approval_short_circuits_refiner(service):
    service.llm.script.update({
        "coordinator": ["**Answer**: plant manager approves POs over $5,000 [doc:spare_parts_policy#x]"],
        "critic": [Call("exit_loop", {})],
        "refiner": ["SHOULD NOT RUN"],
    })
    res = await service.ask("Who approves a $12,000 PO?", session_id="t-approve")
    assert res["approved_by_critic"] and "plant manager" in res["answer"]
    assert "refiner" not in res["metrics"]["agent_runs"]
    service.llm.script["refiner"].clear()


async def test_critic_and_refiner_get_question_via_state_not_history(service):
    service.llm.script.update({"coordinator": ["**Answer**: d1"], "critic": [Call("exit_loop", {})]})
    service.llm.seen.clear()
    await service.ask("What is the LOTO procedure?", session_id="t-state")
    critic_req = service.llm.seen["critic"][0]
    si = critic_req.config.system_instruction
    text = si if isinstance(si, str) else "".join(p.text for p in si.parts)
    assert "What is the LOTO procedure?" in text and "**Answer**: d1" in text
    assert not critic_req.contents or all(c.role != "model" for c in critic_req.contents)


async def test_runaway_react_loop_is_stopped_by_max_llm_calls(settings):
    from dataclasses import replace

    from google.adk.sessions import InMemorySessionService

    from plantops.service import PlantOpsService
    from plantops.testing import ScriptedLlm

    # The analyst never stops calling run_sql; only the code-level ceiling can end the loop.
    llm = ScriptedLlm({
        "coordinator": [Call("data_analyst", {"request": "count open tickets"}), "**Answer**: draft"],
        "data_analyst": [Call("run_sql", {"query": "SELECT 1"})] * 50,
    })
    svc = PlantOpsService(replace(settings, max_llm_calls=5), worker=llm, strong=llm,
                          session_service=InMemorySessionService(), synchronous_tracing=True)
    res = await svc.ask("How many open tickets?", session_id="t-runaway")

    assert res["limit_reached"] is True and res["approved_by_critic"] is False
    assert res["answer"].startswith("Stopped after reaching the limit of 5 model calls")
    assert res["steps"][-1]["type"] == "stopped"
    assert len(llm.seen["data_analyst"]) == 5  # the 6th call is refused before reaching the model
    assert "critic" not in llm.seen


def test_max_llm_calls_default_leaves_headroom(settings):
    # Real runs peaked at 23 model calls in the top-level run; the default must stay well above that.
    assert settings.max_llm_calls == 40


def _fr(name, payload):
    return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(name=name, response=payload))])


def _fc(name):
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args={}))])


def test_context_budget_elides_old_tool_results_and_keeps_pairing():
    big = {"rows": [["x" * 50] for _ in range(100)]}
    contents = [types.Content(role="user", parts=[types.Part(text="q")])]
    for i in range(6):
        contents += [_fc(f"t{i}"), _fr(f"t{i}", big)]
    plugin = ContextBudgetPlugin(keep_recent_tool_results=2, max_tool_result_chars=100_000)
    trimmed, saved = plugin.trim(contents)

    responses = [p.function_response for c in trimmed for p in c.parts if p.function_response]
    assert [r.name for r in responses] == [f"t{i}" for i in range(6)]  # every call still has its response
    assert all("elided" in r.response for r in responses[:4])
    assert all("rows" in r.response for r in responses[4:])
    assert saved > 4 * 5000
    assert "rows" in contents[1 + 2 * 0 + 1].parts[0].function_response.response  # input not mutated


def test_context_budget_truncates_oversized_recent_result():
    contents = [_fc("t"), _fr("t", {"blob": "y" * 20_000})]
    trimmed, saved = ContextBudgetPlugin(max_tool_result_chars=1000).trim(contents)
    r = trimmed[1].parts[0].function_response.response
    assert "truncated" in r and len(r["truncated"]) == 1000 and saved > 18_000


def test_cost_model():
    # 1M uncached input + 1M output on gemini-3.8-flash at list price
    assert round(cost_usd("gemini-3.8-flash", 1_000_000, 1_000_000), 4) == 4.5
    # Cached tokens are billed at the cached rate
    assert round(cost_usd("gemini-3.8-flash", 1_000_000, 0, cached_tokens=1_000_000), 4) == 0.075
    assert cost_usd("models/gemini-2.5-flash", 1000, 0) == 0.0003
    assert cost_usd("unknown-model", 1000, 1000) == 0.0
