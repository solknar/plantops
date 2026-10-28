"""Evaluation pipeline: retrieval quality, answer correctness, grounding, latency and cost.

Expected answers for data questions are computed from the warehouse at eval time
(expected_sql), so the golden set never drifts from the data. Two tiers:

  * retrieval (offline, no LLM): recall@k and MRR of hybrid search against labeled docs;
  * end-to-end (Gemini): fact recall in the final answer, citation validity, critic approval,
    optional LLM-as-judge score, plus latency/tokens/cost per question.

The run fails (non-zero exit) when aggregate scores fall below the thresholds, so it can gate CI.
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plantops.config import Settings
from plantops.retrieval.store import VectorStore

GOLDEN = Path(__file__).with_name("golden.json")
THRESHOLDS = {"retrieval_recall_at_k": 0.85, "fact_recall": 0.80, "citation_validity": 0.90}
# Models drift on formatting ("[doc: `id`]"), so allow spaces and backticks around the id.
_CITE_RE = re.compile(r"\[doc:\s*`?([^\]\s`]+)`?\s*\]")


def load_golden(ids: list[str] | None = None) -> list[dict]:
    items = json.loads(GOLDEN.read_text())
    return [i for i in items if not ids or i["id"] in ids]


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.1f}".rstrip("0").rstrip(".") if v != int(v) else str(int(v))
    return str(v)


def expected_facts(item: dict, warehouse: Path) -> list[str]:
    facts = list(item.get("must_include", []))
    if "expected_sql" in item:
        conn = sqlite3.connect(f"file:{warehouse}?mode=ro", uri=True)
        rows = conn.execute(item["expected_sql"]).fetchall()
        conn.close()
        for r in rows:
            facts.extend(_fmt(r[c]) for c in item.get("fact_columns", [0]) if r[c] is not None)
    return list(dict.fromkeys(facts))


def _contains(answer: str, fact: str) -> bool:
    norm = re.sub(r"(?<=\d),(?=\d{3})", "", answer.lower())
    f = fact.lower()
    if re.fullmatch(r"\d+(\.\d+)?", f):
        if re.search(rf"(?<![\d.]){re.escape(f)}(?!\d|\.\d)", norm):
            return True
        # A decimal fact also matches a more precise number that rounds to it (8.46 -> 8.5).
        places = len(f.partition(".")[2])
        if places:
            return any(
                len(n.partition(".")[2]) > places and round(float(n), places) == float(f)
                for n in re.findall(r"(?<![\d.])\d+\.\d+", norm)
            )
        return False
    return f in norm


def score_answer(answer: str, facts: list[str], any_groups: list[list[str]]) -> dict:
    found = [f for f in facts if _contains(answer, f)]
    groups_ok = [g for g in any_groups if any(_contains(answer, f) for f in g)]
    total = len(facts) + len(any_groups)
    return {
        "fact_recall": (len(found) + len(groups_ok)) / total if total else 1.0,
        "missing": [f for f in facts if f not in found] + [" | ".join(g) for g in any_groups if g not in groups_ok],
    }


def check_citations(answer: str, store: VectorStore, relevant_docs: list[str]) -> dict:
    ids = list(dict.fromkeys(_CITE_RE.findall(answer)))
    valid = [i for i in ids if store.get(i) is not None]
    on_target = [i for i in valid if i.split("#")[0] in relevant_docs] if relevant_docs else valid
    return {
        "doc_citations": len(ids),
        "citation_validity": len(valid) / len(ids) if ids else (1.0 if not relevant_docs else 0.0),
        "cites_relevant_doc": bool(on_target) if relevant_docs else None,
        "has_sql_evidence": "[sql]" in answer.lower(),
    }


def retrieval_eval(store: VectorStore, items: list[dict], k: int = 5) -> dict:
    rows = []
    for item in items:
        rel = item.get("relevant_docs")
        if not rel:
            continue
        hits = store.search(item["question"], k=k)
        docs = [h.doc_id for h in hits]
        rank = next((i + 1 for i, d in enumerate(docs) if d in rel), None)
        rows.append({
            "id": item["id"],
            "recall_at_k": len(set(rel) & set(docs)) / len(rel),
            "mrr": 1 / rank if rank else 0.0,
            "retrieved": docs,
        })
    return {
        "k": k,
        "retrieval_recall_at_k": round(statistics.mean(r["recall_at_k"] for r in rows), 3) if rows else None,
        "mrr": round(statistics.mean(r["mrr"] for r in rows), 3) if rows else None,
        "items": rows,
    }


@dataclass
class Judge:
    model: str

    def score(self, question: str, answer: str, facts: list[str]) -> dict:
        from google import genai
        from google.genai import types

        prompt = (
            "You grade answers from an operations copilot. Score 1-5 for correctness and completeness "
            "against the reference facts (5 = all facts correct and the question fully answered; 1 = wrong "
            "or unsupported). Penalize invented numbers.\n\n"
            f"Question: {question}\nReference facts: {facts}\nAnswer:\n{answer}"
        )
        schema = types.Schema(
            type="OBJECT",
            properties={"score": types.Schema(type="INTEGER"), "reason": types.Schema(type="STRING")},
            required=["score", "reason"],
        )
        resp = genai.Client().models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json", response_schema=schema),
        )
        return json.loads(resp.text)


async def e2e_eval(service, store: VectorStore, settings: Settings, items: list[dict], judge: Judge | None = None, pause_s: float = 0.0) -> dict:
    rows = []
    run_id = time.strftime("%Y%m%d%H%M%S")
    for item in items:
        facts = expected_facts(item, settings.warehouse_path)
        t0 = time.perf_counter()
        try:
            res = await service.ask(item["question"], user_id="eval", session_id=f"eval-{run_id}-{item['id']}")
            error = None
        except Exception as e:  # keep going; one failing question shouldn't hide the rest
            res, error = {"answer": "", "metrics": {}, "approved_by_critic": False, "trace_id": None}, repr(e)
        answer = res["answer"] or ""
        row = {
            "id": item["id"],
            "category": item["category"],
            "error": error,
            **score_answer(answer, facts, item.get("must_include_any", [])),
            **check_citations(answer, store, item.get("relevant_docs", [])),
            "approved_by_critic": res["approved_by_critic"],
            "latency_ms": res["metrics"].get("latency_ms", (time.perf_counter() - t0) * 1000),
            "cost_usd": res["metrics"].get("cost_usd", 0.0),
            "tokens": res["metrics"].get("input_tokens", 0) + res["metrics"].get("output_tokens", 0) + res["metrics"].get("thoughts_tokens", 0),
            "output_tokens_per_sec": res["metrics"].get("output_tokens_per_sec"),
            "reflections": res["metrics"].get("reflections"),
            "trace_id": res["trace_id"],
            "answer": answer,
        }
        if judge and answer:
            try:
                row["judge"] = judge.score(item["question"], answer, facts)
            except Exception as e:
                row["judge"] = {"score": None, "reason": f"judge failed: {e!r}"}
        rows.append(row)
        if pause_s:
            time.sleep(pause_s)  # stay under free-tier requests-per-minute quotas

    lat = sorted(r["latency_ms"] for r in rows)
    judged = [r["judge"]["score"] for r in rows if r.get("judge", {}).get("score")]
    return {
        "fact_recall": round(statistics.mean(r["fact_recall"] for r in rows), 3),
        "full_pass_rate": round(sum(not r["missing"] for r in rows) / len(rows), 3),
        "citation_validity": round(statistics.mean(r["citation_validity"] for r in rows), 3),
        "critic_approval_rate": round(sum(r["approved_by_critic"] for r in rows) / len(rows), 3),
        "judge_mean": round(statistics.mean(judged), 2) if judged else None,
        "latency_ms_p50": round(lat[len(lat) // 2], 1),
        "latency_ms_p95": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
        "avg_cost_usd": round(statistics.mean(r["cost_usd"] for r in rows), 6),
        "avg_tokens": round(statistics.mean(r["tokens"] for r in rows)),
        "errors": sum(1 for r in rows if r["error"]),
        "items": rows,
    }


def gate(report: dict) -> list[str]:
    failures = []
    for key, minimum in THRESHOLDS.items():
        section = report.get("retrieval", {}) if key.startswith("retrieval") else report.get("e2e", {})
        value = section.get(key)
        if value is not None and value < minimum:
            failures.append(f"{key}={value} < {minimum}")
    return failures


def write_report(report: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"eval-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str))

    md = [f"# PlantOps eval {stamp}", ""]
    r = report.get("retrieval")
    if r:
        md += [f"**Retrieval** recall@{r['k']}: {r['retrieval_recall_at_k']} · MRR: {r['mrr']}", ""]
        md += ["| id | recall | MRR | top docs |", "|---|---|---|---|"]
        md += [f"| {i['id']} | {i['recall_at_k']:.2f} | {i['mrr']:.2f} | {', '.join(i['retrieved'][:3])} |" for i in r["items"]]
        md.append("")
    e = report.get("e2e")
    if e:
        md += [
            f"**End-to-end** fact recall: {e['fact_recall']} · full pass: {e['full_pass_rate']} · citation validity: {e['citation_validity']} · "
            f"critic approval: {e['critic_approval_rate']} · judge: {e['judge_mean']}",
            f"**Latency** p50 {e['latency_ms_p50']} ms · p95 {e['latency_ms_p95']} ms · **avg cost** ${e['avg_cost_usd']} · avg tokens {e['avg_tokens']}",
            "",
            "| id | cat | facts | missing | cites | approved | latency ms | tokens | cost $ |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        md += [
            f"| {i['id']} | {i['category']} | {i['fact_recall']:.2f} | {', '.join(i['missing']) or '-'} | {i['citation_validity']:.2f} | "
            f"{'yes' if i['approved_by_critic'] else 'no'} | {i['latency_ms']:.0f} | {i['tokens']} | {i['cost_usd']:.5f} |"
            for i in e["items"]
        ]
    if report.get("gate_failures") is not None:
        md += ["", f"**Gate**: {'PASS' if not report['gate_failures'] else 'FAIL — ' + '; '.join(report['gate_failures'])}"]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    return path
