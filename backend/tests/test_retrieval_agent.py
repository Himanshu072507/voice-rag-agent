import pytest
from unittest.mock import MagicMock, patch
from agents.retrieval_agent import RetrievalAgent


def test_retrieval_returns_top_chunks(tmp_path):
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Paris is the capital of France."
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "France is in Western Europe."

    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = [mock_doc1, mock_doc2]

    with patch("agents.retrieval_agent.Chroma", return_value=mock_vectorstore), \
         patch("agents.retrieval_agent.OpenAIEmbeddings"):
        agent = RetrievalAgent(chroma_dir=str(tmp_path))
        chunks = agent.run(query="What is the capital of France?", session_id="test-session-123")

    assert len(chunks) == 2
    assert chunks[0] == "Paris is the capital of France."


def test_retrieval_passes_correct_collection(tmp_path):
    mock_vectorstore = MagicMock()
    mock_vectorstore.similarity_search.return_value = []

    with patch("agents.retrieval_agent.Chroma") as mock_chroma_cls, \
         patch("agents.retrieval_agent.OpenAIEmbeddings"):
        mock_chroma_cls.return_value = mock_vectorstore
        agent = RetrievalAgent(chroma_dir=str(tmp_path))
        agent.run(query="test", session_id="abc-123")
        call_kwargs = mock_chroma_cls.call_args[1]
        assert call_kwargs["collection_name"] == "session_abc-123"
