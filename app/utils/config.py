from pathlib import Path
from typing import Any

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from app.utils.paths import BASE_DIR


class Settings(BaseSettings):
    """Application settings loaded from the install-dir ``.env`` file only.

    OS-level environment variables (e.g. a system ``GROQ_API_KEY``) are
    intentionally ignored so the in-app Settings / setup wizard always wins.
    """

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
        **kwargs: Any,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Omit env_settings — do not read GROQ_* (or any field) from the OS env.
        return (init_settings, dotenv_settings, file_secret_settings)


settings = Settings()


def reload_settings() -> None:
    """Reload settings from .env after the file is updated.

    Mutates the existing ``settings`` object in place so modules that
    imported ``from app.utils.config import settings`` see updated values
    without requiring an app restart.
    """
    global settings
    fresh = Settings()
    for field_name in type(fresh).model_fields:
        object.__setattr__(settings, field_name, getattr(fresh, field_name))

    try:
        from app.core.groq_client import reset_groq_client

        reset_groq_client()
    except ImportError:
        pass
