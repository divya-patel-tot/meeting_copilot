from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.utils.paths import BASE_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
    )

    GROQ_API_KEY: str = ""
    GROQ_STT_MODEL: str = "whisper-large-v3-turbo"
    GROQ_LLM_MODEL: str = "llama-3.3-70b-versatile"

    # LangGraph suggestion pipeline: retrieve → filter → draft → verify → revise
    USE_LANGGRAPH_SUGGESTIONS: bool = True
    SUGGESTION_MAX_REVISIONS: int = 1


settings = Settings()


def reload_settings() -> None:
    """Reload settings from .env after the file is updated."""
    global settings
    settings = Settings()
