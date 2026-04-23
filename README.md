Goal: Build a multi-agent voice RAG system where users upload a PDF, chat with it via a GPT-style UI, and receive answers as both text and spoken audio.

Architecture: Four specialized LangChain LCEL agents in a sequential pipeline — PDF Ingestion → Retrieval → Answer Generation → TTS. FastAPI backend with a hybrid BM25 + dense vector store (ChromaDB) reranked by Cohere; Next.js 15 frontend. Runs locally.

## Tech Stack

| Layer         | Technology                                                            |
| ------------- | --------------------------------------------------------------------- |
| Frontend      | Next.js 15, shadcn/ui, wavesurfer.js                                  |
| Backend       | FastAPI, Python 3.11+                                                 |
| RAG Framework | LangChain LCEL                                                        |
| Retrieval     | BM25 + ChromaDB (dense) fused via EnsembleRetriever + Cohere rerank   |
| Embeddings    | HuggingFace all-MiniLM-L6-v2 (local, free)                            |
| LLM           | Cerebras llama3.1-8b (free tier, via langchain-cerebras)              |
| TTS           | Browser Web Speech API (free, built-in)                               |
| PDF Parsing   | PyMuPDF                                                               |
| Hosting       | Localhost only                                                        |

## Agent Pipeline

| Agent             | Input                  | Output                       |
| ----------------- | ---------------------- | ---------------------------- |
| PDF Ingestion     | PDF bytes + session_id | ChromaDB collection + chunks |
| Retrieval         | Query + session_id     | Top-5 reranked chunks        |
| Answer Generation | Query + chunks         | Answer text                  |
| TTS               | Answer text            | Spoken audio (browser-side)  |

## Retrieval Pipeline

1. **BM25** over per-session chunks persisted to `backend/chunks_store/{session_id}.json`.
2. **Dense** ChromaDB search using HuggingFace `all-MiniLM-L6-v2` embeddings.
3. **Fusion** via LangChain `EnsembleRetriever` (RRF, weights 0.5/0.5), each over-fetching k=20.
4. **Rerank** the fused top-20 down to k with `CohereRerank(model="rerank-v3.5")`. If `COHERE_API_KEY` is unset, ensemble output is returned raw.

Legacy sessions ingested before BM25 was added silently fall back to dense-only retrieval.

## Offline Evaluation

`backend/eval/` runs the full ingest → retrieve → answer pipeline against a fictional handbook fixture and 10 hand-written Q&A pairs, scoring four Ragas-equivalent metrics (faithfulness, answer_relevancy, context_precision, context_recall) via an LLM judge.

Ragas itself is not used — its 0.2.x async plumbing is broken on Python 3.14. The metrics are re-implemented as plain sync calls over a Cerebras `llama3.1-8b` judge, with HF `all-MiniLM-L6-v2` embeddings for the answer_relevancy cosine. Results are saved incrementally so a mid-run crash never loses work.

Run it (from `backend/`):

```bash
./.venv/bin/python -m eval.run_eval
```

Latest scores (2026-04-23, hybrid retrieval + soft prompt):

| Metric              | Score  |
| ------------------- | ------ |
| faithfulness        | 1.000  |
| answer_relevancy    | 0.755  |
| context_precision   | 1.000  |
| context_recall      | 0.875* |

\* recall is averaged over 4 of 10 questions — the smaller judge returns empty claim-arrays on short ground-truths, which falls through to NaN. Fix is queued.

## Environment Variables

Place in `backend/.env` (gitignored):

| Variable            | Required | Purpose                                       |
| ------------------- | -------- | --------------------------------------------- |
| `CEREBRAS_API_KEY`  | yes      | Answer LLM + eval judge                       |
| `COHERE_API_KEY`    | no       | Reranker (retrieval works without it)         |
| `AUDIO_DIR`         | no       | Defaults to `./audio`                         |
| `CHROMA_DIR`        | no       | Defaults to `./chroma_db`                     |
| `CHUNKS_DIR`        | no       | Defaults to `./chunks_store`                  |

## Quickstart

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (separate shell)
cd frontend
npm install
npm run dev
```

<img width="1470" height="956" alt="image" src="https://github.com/user-attachments/assets/a373e2a7-a282-4b00-80aa-96ea42226b93" />
