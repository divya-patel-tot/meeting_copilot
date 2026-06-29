from __future__ import annotations

from collections.abc import Iterator

from app.core.groq_client import get_groq_client
from app.utils.config import settings


def ask_llm(prompt: str) -> str:
    """Send a single user message to Groq LLM and return the response text."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def generate_suggestion_streaming(messages: list[dict]) -> Iterator[str]:
    """Stream chat completion deltas from Groq."""
    client = get_groq_client()
    stream = client.chat.completions.create(
        model=settings.GROQ_LLM_MODEL,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
