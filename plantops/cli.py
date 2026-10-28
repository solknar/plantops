"""Command line entry point: `plantops <command>`."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys


def _require_key() -> None:
    from plantops.config import get_settings

    get_settings()  # loads .env
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").upper() == "TRUE"):
        sys.exit("GOOGLE_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey and put it in .env")


def cmd_generate(args) -> None:
    from plantops.config import get_settings
    from plantops.data_gen import generate

    print(json.dumps(generate(get_settings().structured_dir), indent=1))


def cmd_ingest(args) -> None:
    from dataclasses import asdict

    from plantops.config import get_settings
    from plantops.ingest.embeddings import make_embedder
    from plantops.ingest.pipeline import run_ingest
    from plantops.telemetry.store import TelemetryStore
    from plantops.telemetry.tracing import flush, setup_tracing

    s = get_settings()
    if s.embedder == "gemini":
        _require_key()
    if not any(s.structured_dir.glob("*.csv")):
        cmd_generate(args)
    setup_tracing(TelemetryStore(s.telemetry_path))
    stats = run_ingest(s, make_embedder(s), full=args.full)
    flush()
    print(json.dumps(asdict(stats), indent=1))


def cmd_ask(args) -> None:
    _require_key()
    from plantops.config import get_settings
    from plantops.service import PlantOpsService

    svc = PlantOpsService(get_settings())
    res = asyncio.run(svc.ask(args.question, user_id=args.user, session_id=args.session))
    if args.json:
        print(json.dumps(res, indent=1, default=str))
        return
    print(res["answer"])
    m = res["metrics"]
    print(
        f"\n--- session {res['session_id']} · trace {res['trace_id']} · critic {'approved' if res['approved_by_critic'] else 'not approved'}\n"
        f"--- {m.get('latency_ms')} ms · {m.get('llm_calls')} LLM calls · {m.get('tool_calls')} tool calls · "
        f"{m.get('input_tokens')} in / {m.get('output_tokens')} out tokens · {m.get('output_tokens_per_sec')} tok/s · "
        f"${m.get('cost_usd')} list price · {m.get('reflections')} reflection pass(es)"
    )


def cmd_eval(args) -> None:
    from plantops.config import get_settings
    from plantops.evals import runner
    from plantops.tools import runtime

    s = get_settings()
    items = runner.load_golden(args.ids.split(",") if args.ids else None)
    if args.limit:
        items = items[: args.limit]
    report: dict = {"settings": {"worker": s.worker_model, "strong": s.strong_model, "embedder": s.embedder, "max_reflections": s.max_reflections}}
    report["retrieval"] = runner.retrieval_eval(runtime.store(), items, k=args.k)
    if not args.retrieval_only:
        _require_key()
        from plantops.service import PlantOpsService

        svc = PlantOpsService(s)
        judge = runner.Judge(s.strong_model) if args.judge else None
        report["e2e"] = asyncio.run(runner.e2e_eval(svc, runtime.store(), s, items, judge=judge, pause_s=args.pause))
    report["gate_failures"] = runner.gate(report)
    path = runner.write_report(report, s.var_dir / "evals")
    print(path.with_suffix(".md").read_text())
    print(f"Full report: {path}")
    sys.exit(1 if report["gate_failures"] else 0)


def cmd_serve(args) -> None:
    import uvicorn

    _require_key()
    uvicorn.run("plantops.api:app", host=args.host, port=args.port, reload=False)


def cmd_metrics(args) -> None:
    from plantops.config import get_settings
    from plantops.telemetry.metrics import aggregate
    from plantops.telemetry.store import TelemetryStore

    print(json.dumps(aggregate(TelemetryStore(get_settings().telemetry_path)), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="plantops", description="Multi-agent GenAI copilot for plant operations")
    sub = ap.add_subparsers(required=True)

    p = sub.add_parser("generate-data", help="write the synthetic CSV exports")
    p.set_defaults(fn=cmd_generate)

    p = sub.add_parser("ingest", help="load warehouse + embed documents (incremental)")
    p.add_argument("--full", action="store_true", help="drop and rebuild the vector index")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="ask the multi-agent system a question")
    p.add_argument("question")
    p.add_argument("--session", help="continue an existing session id")
    p.add_argument("--user", default="cli")
    p.add_argument("--json", action="store_true", help="print the full response incl. steps and metrics")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval", help="run the evaluation pipeline")
    p.add_argument("--retrieval-only", action="store_true", help="offline retrieval metrics only (no LLM calls)")
    p.add_argument("--judge", action="store_true", help="add an LLM-as-judge score per answer")
    p.add_argument("--ids", help="comma-separated golden item ids")
    p.add_argument("--limit", type=int)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--pause", type=float, default=5.0, help="seconds between questions (free-tier quotas)")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("serve", help="run the API + dashboard")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("metrics", help="print aggregate LLM metrics across recent requests")
    p.set_defaults(fn=cmd_metrics)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
