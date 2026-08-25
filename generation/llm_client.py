import json
import os
from typing import Iterator

import requests

MODEL = "gemma4:12b"
TEMPERATURE = 0.1
MAX_NEW_TOKENS = 512


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _source_label(doc: dict) -> str:
    return doc["metadata"].get("source_file") or doc["chroma_id"]


def build_prompt(query: str, context_docs: list[dict]) -> str:
    sources = "\n".join(
        f"[{i}] ({_source_label(doc)}): {doc['text']}" for i, doc in enumerate(context_docs, start=1)
    )
    return (
        "You are answering a question using ONLY the provided context sources. "
        "Do not use outside knowledge.\n\n"
        "Context sources:\n"
        f"{sources}\n\n"
        f"Question: {query}\n\n"
        "Instructions:\n"
        "- Answer only using the context above.\n"
        "- If the context does not contain enough information to answer, say so "
        "plainly instead of guessing.\n"
        "- Cite which source number(s) (e.g. [1], [2]) you drew the answer from."
    )


def _request_body(prompt: str, stream: bool) -> dict:
    return {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "think": False,
        "options": {"temperature": TEMPERATURE, "num_predict": MAX_NEW_TOKENS},
    }


def generate_blocking(prompt: str) -> str:
    url = f"{_ollama_host()}/api/chat"
    response = requests.post(url, json=_request_body(prompt, stream=False))
    return response.json()["message"]["content"]


def generate_streaming(prompt: str) -> Iterator[str]:
    url = f"{_ollama_host()}/api/chat"
    response = requests.post(url, json=_request_body(prompt, stream=True), stream=True)
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        content = chunk["message"]["content"]
        if content:
            yield content
