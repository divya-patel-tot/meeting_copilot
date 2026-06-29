from __future__ import annotations

import json
import re
from typing import Any

from app.core.groq_client import get_groq_client
from app.utils.config import settings


def chat_complete(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """Single non-streaming Groq chat completion."""
    client = get_groq_client()
    kwargs: dict[str, Any] = {
        "model": settings.GROQ_LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating fenced code blocks."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)
