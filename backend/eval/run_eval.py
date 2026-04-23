"""Offline RAG evaluation for the voice-rag-agent pipeline.

Usage (from backend/):
    ./.venv/bin/python -m eval.run_eval

Fixture: a fictional handbook (backend/eval/fixture.py) paired with
hand-written golden Q&A (backend/eval/golden.json). First run ingests the
fixture as PDF into a fixed session_id; subsequent runs reuse it.

Metrics (see eval/metrics.py): faithfulness, answer_relevancy,
context_precision, context_recall. Answer LLM: Cerebras qwen-3-235b.
Judge LLM: same (separate instance). Embeddings: HuggingFace all-MiniLM-L6-v2 (local).

Ragas itself is NOT used — its 0.2.x async plumbing (asyncio.timeout,
anyio, nest_asyncio) is broken on Python 3.14. The metrics follow the
same definitions but are implemented as plain sync LLM calls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fitz
import pandas as pd
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from agents.ingestion_agent import IngestPDFAgent  # noqa: E402
from agents.retrieval_agent import RetrievalAgent  # noqa: E402
from agents.answer_agent import AnswerAgent  # noqa: E402
from eval.fixture import HANDBOOK_TEXT  # noqa: E402
from eval.metrics import Judge, Sample, build_embedder, score_sample  # noqa: E402

load_dotenv(BACKEND_DIR / ".env")

EVAL_DIR = Path(__file__).parent
GOLDEN_PATH = EVAL_DIR / "golden.json"
RESULTS_PATH = EVAL_DIR / "results.json"
SESSION_ID = "eval-nimbuscraft-v1"


def _build_pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(50, 50, page.rect.width - 50, page.rect.height - 50)
    page.insert_textbox(rect, text, fontsize=9, fontname="helv")
    return doc.tobytes()


def ensure_fixture_ingested() -> None:
    chunks_dir = Path(os.getenv("CHUNKS_DIR", str(BACKEND_DIR / "chunks_store")))
    chunks_file = chunks_dir / f"{SESSION_ID}.json"
    if chunks_file.exists():
        print(f"[setup] Fixture already ingested at session_id={SESSION_ID}")
        return

    print(f"[setup] Ingesting fixture as session_id={SESSION_ID}")
    pdf_bytes = _build_pdf_bytes(HANDBOOK_TEXT)
    IngestPDFAgent().run(pdf_bytes=pdf_bytes, session_id=SESSION_ID)
    print("[setup] Done.")


def run_pipeline(qa_pairs: list[dict]) -> list[Sample]:
    retriever = RetrievalAgent()
    answerer = AnswerAgent()

    samples = []
    for i, pair in enumerate(qa_pairs, 1):
        q = pair["question"]
        gt = pair["ground_truth"]
        print(f"[pipeline] ({i}/{len(qa_pairs)}) {q}")
        chunks = retriever.run(query=q, session_id=SESSION_ID, k=5)
        ans = answerer.run(query=q, chunks=chunks)
        samples.append(Sample(question=q, answer=ans, contexts=chunks, ground_truth=gt))
    return samples


METRIC_COLS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def _save_partial(samples: list[Sample], rows: list[dict]) -> None:
    """Write whatever scored rows we have so far, so we never lose work on crash."""
    full = [
        {
            "question": s.question,
            "answer": s.answer,
            "contexts": s.contexts,
            "ground_truth": s.ground_truth,
            **{c: row.get(c) for c in METRIC_COLS},
        }
        for s, row in zip(samples, rows)
    ]
    RESULTS_PATH.write_text(json.dumps(full, indent=2, default=str), encoding="utf-8")


def score(samples: list[Sample]) -> pd.DataFrame:
    judge = Judge(api_key=os.getenv("CEREBRAS_API_KEY"))
    embedder = build_embedder()
    rows = []
    for i, s in enumerate(samples, 1):
        print(f"[metrics] Scoring ({i}/{len(samples)}) {s.question[:70]}")
        rows.append(score_sample(s, judge, embedder))
        _save_partial(samples[: len(rows)], rows)
    return pd.DataFrame(rows)


def main() -> None:
    if not os.getenv("CEREBRAS_API_KEY"):
        sys.exit("CEREBRAS_API_KEY is required for the answer LLM and judge (set it in backend/.env).")

    ensure_fixture_ingested()

    qa_pairs = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    samples = run_pipeline(qa_pairs)
    df = score(samples)

    print("\n=== Per-question scores ===")
    with pd.option_context("display.max_colwidth", 55, "display.width", 200):
        print(df[["question", *METRIC_COLS]].to_string(index=False))

    print("\n=== Mean scores ===")
    print(df[METRIC_COLS].mean(numeric_only=True).round(3).to_string())
    print(f"\n[metrics] Full results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
