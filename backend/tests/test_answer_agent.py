# backend/tests/test_answer_agent.py
import pytest
from unittest.mock import MagicMock, patch
from agents.answer_agent import AnswerAgent

def test_answer_agent_returns_string():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Paris is the capital of France."

    agent = AnswerAgent()
    agent._chain = mock_chain
    result = agent.run(
        query="What is the capital of France?",
        chunks=["The capital of France is Paris.", "France is in Europe."]
    )

    assert result == "Paris is the capital of France."
    mock_chain.invoke.assert_called_once()

def test_answer_agent_includes_chunks_in_prompt():
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "Some answer."

    agent = AnswerAgent()
    agent._chain = mock_chain
    agent.run(
        query="test query",
        chunks=["chunk one", "chunk two"]
    )

    call_args = mock_chain.invoke.call_args[0][0]
    assert "chunk one" in call_args["context"]
    assert "chunk two" in call_args["context"]
    assert call_args["question"] == "test query"
