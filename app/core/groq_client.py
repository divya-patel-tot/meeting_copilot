from groq import Groq

from app.utils.config import settings

_client: Groq | None = None


def reset_groq_client() -> None:
    """Drop cached client so the next call picks up a new API key."""
    global _client
    _client = None


def get_groq_client() -> Groq:
    """Return a module-level Groq client singleton."""
    global _client
    if _client is None:
        _client = Groq(api_key=settings.GROQ_API_KEY)
    return _client
