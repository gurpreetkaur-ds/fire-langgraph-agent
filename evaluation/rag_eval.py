"""RAG evaluation: index a small corpus, ask questions with ground truth, and measure retrieval and answer quality.

    python -m evaluation.rag_eval                 # live: Fire embeddings + the real routed LangGraph workflow + LLM judge
    python -m evaluation.rag_eval --retrieval-only --offline   # deterministic, no network: retrieval metrics only

Runs in an isolated temporary SQLite database (it never touches the app database) and writes
evaluations/rag/results/rag_<timestamp>.json and a Markdown report next to it.

Metrics (per case, then averaged):
- retrieval precision  = share of sources given to the generator that are relevant (contain a relevant phrase)
- retrieval recall     = share of relevant phrases found in at least one retrieved source (ground truth)
- citation correctness = share of the answer's citations that point to a relevant source
- fact recall          = share of expected facts present in the answer
- abstention accuracy  = unanswerable questions answered with "not in your documents" (and no citations)
- faithfulness / answer relevance = the existing LLM-judge evaluators (live mode)
- latency              = end-to-end workflow time
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from .config import BASE_DIR

RAG_DIR = BASE_DIR / "evaluations" / "rag"


def _relevant(text: str, phrases: List[str]) -> bool:
    low = text.lower()
    return any(p.lower() in low for p in phrases)


async def _prepare(tmp: Path, offline: bool) -> str:
    os.environ["DATABASE_URL"] = f"sqlite:///{tmp / 'rag_eval.db'}"
    os.environ["UPLOAD_DIR"] = str(tmp / "uploads")
    os.environ["EVAL_ENABLED"] = "false"  # this script scores answers itself; don't queue background evaluations
    if offline:
        # the toy hash embedder's cosine scores are far lower than real embeddings'; 0.30 is tuned for the latter
        os.environ.setdefault("RAG_SIMILARITY_THRESHOLD", "0.05")
    from app.settings import reload_app_settings

    reload_app_settings()
    from evaluation.db import init_engine, run_migrations

    run_migrations()
    init_engine()
    if offline:
        sys.path.insert(0, str(BASE_DIR))
        from app.rag.embeddings import set_embedder
        from tests.conftest import HashEmbedder

        set_embedder(HashEmbedder())

    import hashlib

    from app.models import User, new_id
    from app.rag import store
    from app.rag.extract import detect_type
    from app.rag.ingest import IngestionWorker
    from app.settings import get_app_settings
    from evaluation.db import session_scope

    with session_scope() as db:
        user = User(username="rag-eval", display_name="RAG eval", password_hash="!", preferences={})
        db.add(user)
        db.flush()
        user_id = user.id
    for path in sorted((RAG_DIR / "docs").iterdir()):
        data = path.read_bytes()
        ftype = detect_type(path.name, data)
        doc_id = new_id("doc")
        rel = f"{user_id}/{doc_id}.{ftype}"
        target = get_app_settings().upload_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        store.create_document(user_id, doc_id=doc_id, filename=path.name, file_type=ftype, size_bytes=len(data),
                              sha256=hashlib.sha256(data).hexdigest(), storage_path=rel)
    await IngestionWorker().drain()
    failed = [d["filename"] for d in store.list_documents(user_id) if d["status"] != "ready"]
    if failed:
        raise SystemExit(f"Indexing failed for: {failed}")
    return user_id


async def _retrieve_only(user_id: str, question: str) -> List[dict]:
    from app.rag import retrieval, store
    from app.rag.embeddings import get_embedder
    from app.settings import get_app_settings

    s = get_app_settings()
    vecs = (await get_embedder().embed_many([question])).vectors
    hits = store.search(user_id, vecs, s.candidate_k)
    ranked, _ = retrieval.rerank(question, hits, top_k=s.top_k, threshold=s.similarity_threshold, relative_cutoff=s.relative_cutoff)
    sources, _ = retrieval.build_context(ranked, s.context_char_budget)
    return sources


async def _run_case(user_id: str, case: dict, retrieval_only: bool, judge_engine) -> dict:
    from app.rag.retrieval import says_not_found

    t0 = time.perf_counter()
    out: dict = {"id": case["id"], "question": case["question"], "answerable": case["answerable"]}
    if retrieval_only:
        sources = await _retrieve_only(user_id, case["question"])
        answer, citations, route = None, [], "rag"
    else:
        from app.runner import WorkflowRun

        run = WorkflowRun(case["question"], user_id=user_id, force_route=None, kb_mode="always")
        async for _ in run.stream():
            pass
        sources = run.state.get("sources") or []
        answer = run.state.get("final_answer") or ""
        citations = run.state.get("citations") or []
        route = run.state.get("route")
        out["error"] = run.error
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    out["route"] = route
    phrases = case["relevant_phrases"]
    by_id = {s["id"]: s for s in sources}
    out["sources"] = [{"id": s["id"], "file": s["filename"], "page": s.get("page"), "score": s["score"],
                       "relevant": _relevant(s["text"], phrases)} for s in sources]
    if case["answerable"]:
        out["retrieval_precision"] = (sum(_relevant(s["text"], phrases) for s in sources) / len(sources)) if sources else 0.0
        out["retrieval_recall"] = sum(any(p.lower() in s["text"].lower() for s in sources) for p in phrases) / len(phrases)
    if answer is not None:
        out["answer"] = answer
        if case["answerable"]:
            facts = case["expected_facts"]
            # an expected fact may list accepted spellings separated by "|" (e.g. "two|2")
            out["fact_recall"] = (sum(any(alt.lower() in answer.lower() for alt in f.split("|")) for f in facts) / len(facts)
                                  if facts else None)
            out["citation_correctness"] = (sum(_relevant(by_id[c["source_id"]]["text"], phrases) for c in citations if c["source_id"] in by_id)
                                           / len(citations)) if citations else 0.0
        else:
            out["abstained"] = says_not_found(answer) and not citations
        if judge_engine is not None and answer:
            from evaluation.schemas import Citation, ContextItem, EvalInput

            data = EvalInput(trace_id="rag_eval", question=case["question"], answer=answer, success=True,
                             context=[ContextItem(id=s["id"], source=s["filename"], text=s["text"]) for s in sources],
                             citations=[Citation(**c) for c in citations])
            for r in await judge_engine.evaluate(data):
                out[r.metric] = r.score
                out[r.metric + "_reason"] = r.reason
    return out


def _mean(rows: List[dict], key: str) -> Optional[float]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _report(summary: dict, rows: List[dict]) -> str:
    pct = lambda v: "n/a" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    lines = [f"# RAG evaluation — {summary['run_id']}", "",
             f"Mode: **{summary['mode']}** · cases: {summary['cases']} · embedding: {summary['embedding_model']} · "
             f"top-k {summary['top_k']}, threshold {summary['threshold']}, relative cutoff {summary['relative_cutoff']}, chunk {summary['chunk_size']}/{summary['chunk_overlap']}", "",
             "| Metric | Score |", "|---|---|"]
    for k, label in (("retrieval_precision", "Retrieval precision"), ("retrieval_recall", "Retrieval recall"),
                     ("citation_correctness", "Citation correctness"), ("fact_recall", "Answer fact recall"),
                     ("abstention_accuracy", "Abstention on unanswerable"), ("faithfulness", "Faithfulness (LLM judge)"),
                     ("relevance", "Answer relevance (LLM judge)"), ("citation_accuracy", "Citation support (LLM judge)")):
        if summary.get(k) is not None:
            lines.append(f"| {label} | {pct(summary[k])} |")
    lines += [f"| Mean latency | {summary['latency_ms_mean']} ms |", f"| p95 latency | {summary['latency_ms_p95']} ms |", "",
              "## Cases", "", "| Case | Route | Precision | Recall | Citations | Facts | Abstained | Latency |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r.get('route')} | {pct(r.get('retrieval_precision'))} | {pct(r.get('retrieval_recall'))} | "
                     f"{pct(r.get('citation_correctness'))} | {pct(r.get('fact_recall'))} | {'' if r['answerable'] else r.get('abstained')} | {r['latency_ms']} ms |")
    return "\n".join(lines) + "\n"


async def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrieval-only", action="store_true", help="skip answer generation (no chat LLM calls)")
    ap.add_argument("--offline", action="store_true", help="deterministic local embedder instead of Fire (implies --retrieval-only)")
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM-judge metrics")
    ap.add_argument("--case", action="append", help="run only these case ids")
    args = ap.parse_args(argv)
    retrieval_only = args.retrieval_only or args.offline

    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    cases = json.loads((RAG_DIR / "cases.json").read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] in set(args.case)]
    with tempfile.TemporaryDirectory(prefix="rag_eval_") as tmp:
        user_id = await _prepare(Path(tmp), args.offline)
        judge_engine = None
        if not retrieval_only and not args.no_judge:
            from evaluation.engine import EvaluationEngine
            from evaluation.evaluators.citation import CitationEvaluator
            from evaluation.evaluators.faithfulness import FaithfulnessEvaluator
            from evaluation.evaluators.relevance import RelevanceEvaluator

            judge_engine = EvaluationEngine(evaluators=[FaithfulnessEvaluator, RelevanceEvaluator, CitationEvaluator])
        rows = []
        for case in cases:
            row = await _run_case(user_id, case, retrieval_only, judge_engine)
            rows.append(row)
            print(f"  {row['id']:<14} precision={row.get('retrieval_precision')} recall={row.get('retrieval_recall')} "
                  f"citations={row.get('citation_correctness')} facts={row.get('fact_recall')} abstained={row.get('abstained')} "
                  f"{row['latency_ms']} ms", flush=True)
    from app.settings import get_app_settings

    s = get_app_settings()
    lat = sorted(r["latency_ms"] for r in rows)
    unans = [r for r in rows if not r["answerable"] and "abstained" in r]
    summary = {
        "run_id": "rag_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        "mode": "offline retrieval-only" if args.offline else ("retrieval-only" if retrieval_only else "live end-to-end"),
        "cases": len(rows), "embedding_model": "test-hash-embedding" if args.offline else s.embedding_model,
        "top_k": s.top_k, "threshold": s.similarity_threshold, "relative_cutoff": s.relative_cutoff, "chunk_size": s.chunk_size, "chunk_overlap": s.chunk_overlap,
        "retrieval_precision": _mean(rows, "retrieval_precision"), "retrieval_recall": _mean(rows, "retrieval_recall"),
        "citation_correctness": _mean(rows, "citation_correctness"), "fact_recall": _mean(rows, "fact_recall"),
        "abstention_accuracy": round(sum(r["abstained"] for r in unans) / len(unans), 4) if unans else None,
        "faithfulness": _mean(rows, "faithfulness"), "relevance": _mean(rows, "relevance"), "citation_accuracy": _mean(rows, "citation_accuracy"),
        "latency_ms_mean": round(statistics.mean(lat)) if lat else None, "latency_ms_p95": lat[min(len(lat) - 1, int(0.95 * len(lat)))] if lat else None,
    }
    out_dir = RAG_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{summary['run_id']}.json").write_text(json.dumps({"summary": summary, "cases": rows}, indent=2))
    (out_dir / f"{summary['run_id']}.md").write_text(_report(summary, rows))
    print(json.dumps(summary, indent=2))
    print(f"Report: {out_dir / (summary['run_id'] + '.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
