import os
import json
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_cohere import CohereRerank


_retriever_cache: dict[str, object] = {}

# Over-fetch candidates from the hybrid ensemble before Cohere reranks down to k.
OVER_FETCH = 20
RERANK_MODEL = "rerank-v3.5"


class RetrievalAgent:
    def __init__(self, chroma_dir: str = None, chunks_dir: str = None):
        self.chroma_dir = chroma_dir or os.getenv("CHROMA_DIR", "./chroma_db")
        self.chunks_dir = chunks_dir or os.getenv("CHUNKS_DIR", "./chunks_store")

    def _build_embeddings(self):
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if hf_token:
            return HuggingFaceInferenceAPIEmbeddings(
                api_key=hf_token, model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def _load_chunks(self, session_id: str) -> list[str] | None:
        path = os.path.join(self.chunks_dir, f"{session_id}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_retriever(self, session_id: str, k: int):
        cache_key = f"{session_id}:{k}"
        if cache_key in _retriever_cache:
            return _retriever_cache[cache_key]

        vectorstore = Chroma(
            collection_name=f"session_{session_id}",
            embedding_function=self._build_embeddings(),
            persist_directory=self.chroma_dir,
        )
        fetch_n = max(OVER_FETCH, k)
        dense = vectorstore.as_retriever(search_kwargs={"k": fetch_n})

        chunks = self._load_chunks(session_id)
        if chunks:
            bm25 = BM25Retriever.from_texts(chunks)
            bm25.k = fetch_n
            base = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])
        else:
            # Legacy session ingested before BM25 rollout — dense-only.
            base = dense

        cohere_key = os.getenv("COHERE_API_KEY")
        if cohere_key:
            reranker = CohereRerank(
                cohere_api_key=cohere_key, model=RERANK_MODEL, top_n=k
            )
            retriever = ContextualCompressionRetriever(
                base_compressor=reranker, base_retriever=base
            )
        else:
            retriever = base

        _retriever_cache[cache_key] = retriever
        return retriever

    def run(self, query: str, session_id: str, k: int = 5) -> list[str]:
        retriever = self._get_retriever(session_id, k)
        docs = retriever.invoke(query)
        return [doc.page_content for doc in docs[:k]]
