import json
from unittest.mock import MagicMock, patch
import agents.retrieval_agent as retrieval_module
from agents.retrieval_agent import RetrievalAgent


def _clear_cache():
    retrieval_module._retriever_cache.clear()


def test_retrieval_hybrid_returns_top_chunks(monkeypatch, tmp_path):
    _clear_cache()
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    session_id = "hybrid-session"
    (chunks_dir / f"{session_id}.json").write_text(
        json.dumps([
            "Paris is the capital of France.",
            "France is in Western Europe.",
            "Berlin is the capital of Germany.",
        ]),
        encoding="utf-8",
    )

    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Paris is the capital of France."
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "France is in Western Europe."

    mock_ensemble = MagicMock()
    mock_ensemble.invoke.return_value = [mock_doc1, mock_doc2]

    with patch("agents.retrieval_agent.Chroma"), \
         patch("agents.retrieval_agent.HuggingFaceEmbeddings"), \
         patch("agents.retrieval_agent.BM25Retriever") as mock_bm25_cls, \
         patch("agents.retrieval_agent.EnsembleRetriever", return_value=mock_ensemble) as mock_ensemble_cls:
        mock_bm25_cls.from_texts.return_value = MagicMock()
        agent = RetrievalAgent(chroma_dir=str(tmp_path), chunks_dir=str(chunks_dir))
        chunks = agent.run(query="capital of France?", session_id=session_id)

    assert chunks == ["Paris is the capital of France.", "France is in Western Europe."]
    mock_bm25_cls.from_texts.assert_called_once()
    mock_ensemble_cls.assert_called_once()
    ensemble_kwargs = mock_ensemble_cls.call_args[1]
    assert ensemble_kwargs["weights"] == [0.5, 0.5]


def test_retrieval_uses_cohere_rerank_when_key_set(monkeypatch, tmp_path):
    _clear_cache()
    monkeypatch.setenv("COHERE_API_KEY", "fake-cohere-key")

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    session_id = "rerank-session"
    (chunks_dir / f"{session_id}.json").write_text(
        json.dumps(["chunk a", "chunk b", "chunk c"]), encoding="utf-8"
    )

    mock_doc = MagicMock()
    mock_doc.page_content = "reranked chunk"
    mock_compression_retriever = MagicMock()
    mock_compression_retriever.invoke.return_value = [mock_doc]

    with patch("agents.retrieval_agent.Chroma"), \
         patch("agents.retrieval_agent.HuggingFaceEmbeddings"), \
         patch("agents.retrieval_agent.BM25Retriever") as mock_bm25_cls, \
         patch("agents.retrieval_agent.EnsembleRetriever"), \
         patch("agents.retrieval_agent.CohereRerank") as mock_rerank_cls, \
         patch(
             "agents.retrieval_agent.ContextualCompressionRetriever",
             return_value=mock_compression_retriever,
         ) as mock_compression_cls:
        mock_bm25_cls.from_texts.return_value = MagicMock()
        agent = RetrievalAgent(chroma_dir=str(tmp_path), chunks_dir=str(chunks_dir))
        chunks = agent.run(query="anything", session_id=session_id, k=5)

    assert chunks == ["reranked chunk"]
    mock_rerank_cls.assert_called_once()
    rerank_kwargs = mock_rerank_cls.call_args[1]
    assert rerank_kwargs["cohere_api_key"] == "fake-cohere-key"
    assert rerank_kwargs["top_n"] == 5
    mock_compression_cls.assert_called_once()


def test_retrieval_passes_correct_collection(monkeypatch, tmp_path):
    _clear_cache()
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "abc-123.json").write_text(json.dumps(["some chunk"]), encoding="utf-8")

    mock_ensemble = MagicMock()
    mock_ensemble.invoke.return_value = []

    with patch("agents.retrieval_agent.Chroma") as mock_chroma_cls, \
         patch("agents.retrieval_agent.HuggingFaceEmbeddings"), \
         patch("agents.retrieval_agent.BM25Retriever") as mock_bm25_cls, \
         patch("agents.retrieval_agent.EnsembleRetriever", return_value=mock_ensemble):
        mock_bm25_cls.from_texts.return_value = MagicMock()
        agent = RetrievalAgent(chroma_dir=str(tmp_path), chunks_dir=str(chunks_dir))
        agent.run(query="test", session_id="abc-123")
        call_kwargs = mock_chroma_cls.call_args[1]
        assert call_kwargs["collection_name"] == "session_abc-123"


def test_retrieval_falls_back_to_dense_when_chunks_missing(monkeypatch, tmp_path):
    _clear_cache()
    monkeypatch.delenv("COHERE_API_KEY", raising=False)

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()  # empty — simulates legacy session

    mock_doc = MagicMock()
    mock_doc.page_content = "dense only result"
    mock_dense = MagicMock()
    mock_dense.invoke.return_value = [mock_doc]
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = mock_dense

    with patch("agents.retrieval_agent.Chroma", return_value=mock_vectorstore), \
         patch("agents.retrieval_agent.HuggingFaceEmbeddings"), \
         patch("agents.retrieval_agent.BM25Retriever") as mock_bm25_cls, \
         patch("agents.retrieval_agent.EnsembleRetriever") as mock_ensemble_cls:
        agent = RetrievalAgent(chroma_dir=str(tmp_path), chunks_dir=str(chunks_dir))
        chunks = agent.run(query="anything", session_id="legacy")

    assert chunks == ["dense only result"]
    mock_bm25_cls.from_texts.assert_not_called()
    mock_ensemble_cls.assert_not_called()
