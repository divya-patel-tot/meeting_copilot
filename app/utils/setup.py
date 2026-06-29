from __future__ import annotations

from pathlib import Path
from typing import Callable

import requests
from dotenv import set_key
from groq import Groq

from app.utils.config import reload_settings, settings
from app.utils.paths import (
    ENV_PATH,
    MODELS_DIR,
    SETUP_MARKER_PATH,
    SILERO_VAD_FILENAME,
    SILERO_VAD_URL,
)


def silero_vad_path() -> Path:
    return MODELS_DIR / SILERO_VAD_FILENAME


def is_setup_complete() -> bool:
    """True when marker, API key, and VAD model are all present."""
    return (
        SETUP_MARKER_PATH.exists()
        and bool(settings.GROQ_API_KEY.strip())
        and silero_vad_path().exists()
    )


def write_setup_marker() -> None:
    SETUP_MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETUP_MARKER_PATH.write_text("ok\n", encoding="utf-8")


def save_groq_api_key(api_key: str) -> None:
    key = api_key.strip()
    if not ENV_PATH.exists():
        ENV_PATH.write_text(
            "GROQ_API_KEY=\n"
            "GROQ_STT_MODEL=whisper-large-v3-turbo\n"
            "GROQ_LLM_MODEL=llama-3.3-70b-versatile\n",
            encoding="utf-8",
        )
    set_key(str(ENV_PATH), "GROQ_API_KEY", key)
    reload_settings()


def validate_groq_api_key(api_key: str) -> None:
    """Raise on invalid key after a minimal Groq API call."""
    client = Groq(api_key=api_key.strip())
    client.models.list()


def download_silero_vad(
    on_progress: Callable[[int, int | None], None] | None = None,
) -> Path:
    """Download Silero VAD ONNX to MODELS_DIR. Returns destination path."""
    dest = silero_vad_path()
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(SILERO_VAD_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        total = int(response.headers.get("Content-Length", 0)) or None
        downloaded = 0
        tmp = dest.with_suffix(".onnx.part")
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                if on_progress is not None:
                    on_progress(downloaded, total)

    tmp.replace(dest)
    if on_progress is not None and total is None:
        on_progress(downloaded, downloaded)
    return dest


def warm_up_embedder() -> None:
    """Force fastembed's one-time model download and a trivial embed call."""
    from app.utils.std_streams import ensure_std_streams, silence_download_progress_bars

    ensure_std_streams()
    silence_download_progress_bars()

    from app.core.rag.embedder import Embedder

    embedder = Embedder()
    embedder.embed(["setup warmup"])
