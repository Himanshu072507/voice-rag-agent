"""Self-contained RAG evaluation metrics.

We roll our own instead of using Ragas because Ragas 0.2.x is incompatible
with Python 3.14 (asyncio.timeout / anyio / nest_asyncio breakage). These
implementations follow the same definitions as the Ragas metrics but are
plain sync calls over a Gemini judge — no async glue.

Metrics:
  faithfulness        — fraction of answer claims grounded in contexts
  answer_relevancy    — cosine similarity between question and LLM-generated
                        alt-questions derived from the answer
  context_precision   — mean of (relevance indicator × precision@k) over ranks
  context_recall      — fraction of ground-truth claims entailed by contexts
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np
from langchain_cerebras import ChatCerebras
from langchain_community.embeddings import HuggingFaceEmbeddings


@dataclass
class Sample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def _extract_json_array(text: str) -> list:
    """Pull the first JSON array out of an LLM response, tolerant of stray prose."""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return []


class Judge:
    """Thin wrapper around Cerebras for deterministic short JSON-returning prompts."""

    def __init__(self, api_key: str, model: str = "llama3.1-8b"):
        self.llm = ChatCerebras(model=model, temperature=0, api_key=api_key)

    def call(self, system: str, user: str) -> str:
        last_err = None
        for attempt in range(8):
            try:
                out = self.llm.invoke([
                    ("system", system),
                    ("human", user),
                ])
                return out.content if hasattr(out, "content") else str(out)
            except Exception as e:
                msg = str(e)
                if "token_quota_exceeded" in msg:
                    raise
                if "429" in msg or "queue_exceeded" in msg or "503" in msg:
                    last_err = e
                    time.sleep(min(60, 5 * (attempt + 1)))
                    continue
                raise
        raise last_err


# ----- faithfulness -----

_CLAIMS_SYSTEM = (
    "You extract atomic factual claims from text. Return a JSON array of "
    "short standalone statements. No commentary."
)

_VERIFY_SYSTEM = (
    "You verify whether a claim is entailed by a context passage. "
    "Return ONLY a JSON array of objects: "
    '[{"claim": "...", "verdict": 1 or 0}]. '
    "Verdict 1 = the claim is supported by the context. 0 = not supported."
)


def faithfulness(sample: Sample, judge: Judge) -> float:
    claims_raw = judge.call(
        _CLAIMS_SYSTEM,
        f"Text:\n{sample.answer}\n\nReturn a JSON array of claims.",
    )
    claims = [c for c in _extract_json_array(claims_raw) if isinstance(c, str)]
    if not claims and sample.answer.strip():
        # Short answer where claim extraction returned empty — treat the whole
        # answer as a single claim so the metric doesn't degenerate to NaN.
        claims = [sample.answer.strip()]
    if not claims:
        return np.nan

    context = "\n\n".join(sample.contexts)
    verify_raw = judge.call(
        _VERIFY_SYSTEM,
        f"Context:\n{context}\n\nClaims:\n{json.dumps(claims)}\n\nReturn the JSON array.",
    )
    verdicts = _extract_json_array(verify_raw)
    supported = sum(1 for v in verdicts if isinstance(v, dict) and v.get("verdict") in (1, "1", True))
    return supported / max(len(claims), 1)


# ----- answer relevancy -----

_QGEN_SYSTEM = (
    "Given an answer, generate 3 different questions that this answer fully "
    "addresses. Return a JSON array of 3 strings, no prose."
)


def answer_relevancy(sample: Sample, judge: Judge, embedder) -> float:
    gen_raw = judge.call(
        _QGEN_SYSTEM,
        f"Answer:\n{sample.answer}\n\nReturn the JSON array.",
    )
    gen = [q for q in _extract_json_array(gen_raw) if isinstance(q, str)]
    if not gen:
        return np.nan

    vecs = embedder.embed_documents([sample.question, *gen])
    q_vec = np.array(vecs[0])
    sims = []
    for v in vecs[1:]:
        v = np.array(v)
        denom = np.linalg.norm(q_vec) * np.linalg.norm(v)
        if denom == 0:
            continue
        sims.append(float(np.dot(q_vec, v) / denom))
    return float(np.mean(sims)) if sims else np.nan


# ----- context precision (reference-based) -----

_RELEVANT_SYSTEM = (
    "Decide whether the given context passage is useful for answering the "
    "question AND is consistent with the reference answer. "
    'Respond with ONLY JSON: {"relevant": 1 or 0}.'
)


def context_precision(sample: Sample, judge: Judge) -> float:
    verdicts = []
    for ctx in sample.contexts:
        raw = judge.call(
            _RELEVANT_SYSTEM,
            (
                f"Question: {sample.question}\n"
                f"Reference answer: {sample.ground_truth}\n"
                f"Context: {ctx}\n\n"
                "Return the JSON."
            ),
        )
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        v = 0
        if m:
            try:
                v = int(json.loads(m.group(0)).get("relevant", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                v = 0
        verdicts.append(v)

    if not verdicts:
        return np.nan
    # Ranked precision@k averaged over hits, as per Ragas definition.
    hits = 0
    weighted = 0.0
    for i, v in enumerate(verdicts, 1):
        if v:
            hits += 1
            weighted += hits / i
    return weighted / max(sum(verdicts), 1) if any(verdicts) else 0.0


# ----- context recall (reference-based) -----

_GT_CLAIMS_SYSTEM = (
    "Break the reference answer into atomic factual claims. "
    "Return a JSON array of short standalone statements."
)

_ATTRIB_SYSTEM = (
    "Decide whether each claim can be attributed (fully inferred) to the context. "
    'Return ONLY: [{"claim": "...", "attributed": 1 or 0}].'
)


def context_recall(sample: Sample, judge: Judge) -> float:
    claims_raw = judge.call(
        _GT_CLAIMS_SYSTEM,
        f"Reference answer:\n{sample.ground_truth}\n\nReturn the JSON array.",
    )
    claims = [c for c in _extract_json_array(claims_raw) if isinstance(c, str)]
    if not claims:
        return np.nan

    context = "\n\n".join(sample.contexts)
    raw = judge.call(
        _ATTRIB_SYSTEM,
        f"Context:\n{context}\n\nClaims:\n{json.dumps(claims)}\n\nReturn the JSON array.",
    )
    results = _extract_json_array(raw)
    attributed = sum(
        1 for r in results if isinstance(r, dict) and r.get("attributed") in (1, "1", True)
    )
    return attributed / max(len(claims), 1)


def build_embedder():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def score_sample(sample: Sample, judge: Judge, embedder) -> dict:
    scores: dict = {"question": sample.question}
    for name, fn in [
        ("faithfulness", lambda: faithfulness(sample, judge)),
        ("answer_relevancy", lambda: answer_relevancy(sample, judge, embedder)),
        ("context_precision", lambda: context_precision(sample, judge)),
        ("context_recall", lambda: context_recall(sample, judge)),
    ]:
        try:
            scores[name] = fn()
        except Exception as e:
            print(f"  [warn] {name}: {type(e).__name__}: {e}")
            scores[name] = None
    return scores
