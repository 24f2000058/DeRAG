"""Tests for generation/llm_client.py.

The fast tests below (build_prompt, and generate_streaming/generate_blocking
against mocked HTTP responses) need no external services.

The one test marked @pytest.mark.slow requires a real `ollama serve` running
and reachable at OLLAMA_HOST (see .env.example: http://localhost:11434),
with the gemma4:12b model pulled. Start it, then run:

    uv run pytest -m slow tests/test_generation_llm_client.py
"""

from unittest.mock import Mock, patch

import pytest

from generation.llm_client import build_prompt, generate_blocking, generate_streaming


PLAIN_CONTEXT_DOCS = [
    {
        "chroma_id": "hash1",
        "text": "The Eiffel Tower is located in Paris, France.",
        "similarity": 0.91,
        "metadata": {"source_file": "landmarks.txt", "chunk_id": 0, "ipfs_cid": "Qm1", "doc_hash": "hash1"},
    },
    {
        "chroma_id": "hash2",
        "text": "The Statue of Liberty is located in New York Harbor.",
        "similarity": 0.77,
        "metadata": {"source_file": "landmarks.txt", "chunk_id": 1, "ipfs_cid": "Qm2", "doc_hash": "hash2"},
    },
]

ENRICHED_CONTEXT_DOCS = [
    {
        **PLAIN_CONTEXT_DOCS[0],
        "exists": True,
        "reputation": 5,
        "normalized_reputation": 0.62,
        "score": 0.85,
    },
    {
        **PLAIN_CONTEXT_DOCS[1],
        "exists": False,
        "reputation": 0,
        "normalized_reputation": 0.12,
        "score": 0.60,
    },
]


def test_build_prompt_includes_the_query():
    prompt = build_prompt("Where is the Eiffel Tower?", PLAIN_CONTEXT_DOCS)

    assert "Where is the Eiffel Tower?" in prompt


def test_build_prompt_includes_each_source_text():
    prompt = build_prompt("Where is the Eiffel Tower?", PLAIN_CONTEXT_DOCS)

    assert "The Eiffel Tower is located in Paris, France." in prompt
    assert "The Statue of Liberty is located in New York Harbor." in prompt


def test_build_prompt_includes_source_file_attribution():
    prompt = build_prompt("Where is the Eiffel Tower?", PLAIN_CONTEXT_DOCS)

    assert "landmarks.txt" in prompt


def test_build_prompt_instructs_context_only_and_insufficient_context_disclosure():
    prompt = build_prompt("Where is the Eiffel Tower?", PLAIN_CONTEXT_DOCS)

    assert "only" in prompt.lower()
    assert "insufficient" in prompt.lower() or "does not contain enough" in prompt.lower()


def test_build_prompt_instructs_citation_of_source_numbers():
    prompt = build_prompt("Where is the Eiffel Tower?", PLAIN_CONTEXT_DOCS)

    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "cite" in prompt.lower() or "source number" in prompt.lower()


def test_build_prompt_works_identically_with_trust_weighted_rerank_enriched_shape():
    plain_prompt = build_prompt("Where is the Eiffel Tower?", PLAIN_CONTEXT_DOCS)
    enriched_prompt = build_prompt("Where is the Eiffel Tower?", ENRICHED_CONTEXT_DOCS)

    assert plain_prompt == enriched_prompt


def test_build_prompt_falls_back_to_chroma_id_when_source_file_missing():
    docs = [
        {
            "chroma_id": "hash3",
            "text": "Some unattributed fact.",
            "similarity": 0.5,
            "metadata": {"chunk_id": 0, "ipfs_cid": "", "doc_hash": "hash3"},
        }
    ]

    prompt = build_prompt("What is the fact?", docs)

    assert "hash3" in prompt


def test_generate_blocking_returns_message_content_from_mocked_response():
    mock_response = Mock()
    mock_response.json.return_value = {
        "model": "gemma4:12b",
        "created_at": "2026-08-25T06:33:38.932837598Z",
        "message": {"role": "assistant", "content": "2 + 2 is 4."},
        "done": True,
        "done_reason": "stop",
    }
    with patch("generation.llm_client.requests.post", return_value=mock_response) as mock_post:
        result = generate_blocking("What is 2+2?")

    assert result == "2 + 2 is 4."
    called_url, called_kwargs = mock_post.call_args
    assert called_url[0] == "http://localhost:11434/api/chat"
    body = called_kwargs["json"]
    assert body["model"] == "gemma4:12b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["messages"] == [{"role": "user", "content": "What is 2+2?"}]
    assert body["options"]["temperature"] == 0.1
    assert body["options"]["num_predict"] == 512


def test_generate_streaming_yields_text_chunks_from_mocked_ndjson_response():
    ndjson_lines = [
        b'{"model":"gemma4:12b","message":{"role":"assistant","content":"The"},"done":false}',
        b'{"model":"gemma4:12b","message":{"role":"assistant","content":" Eiffel"},"done":false}',
        b'{"model":"gemma4:12b","message":{"role":"assistant","content":" Tower"},"done":false}',
        b'{"model":"gemma4:12b","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop"}',
    ]
    mock_response = Mock()
    mock_response.iter_lines.return_value = ndjson_lines
    with patch("generation.llm_client.requests.post", return_value=mock_response) as mock_post:
        chunks = list(generate_streaming("Where is the Eiffel Tower?"))

    assert chunks == ["The", " Eiffel", " Tower"]
    called_url, called_kwargs = mock_post.call_args
    assert called_url[0] == "http://localhost:11434/api/chat"
    assert called_kwargs["json"]["stream"] is True
    assert called_kwargs["stream"] is True


@pytest.mark.slow
def test_generate_blocking_against_real_ollama_service():
    result = generate_blocking("Answer in one short sentence: what is 2+2?")

    assert isinstance(result, str)
    assert len(result) > 0
