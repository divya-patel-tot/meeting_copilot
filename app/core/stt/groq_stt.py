from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from app.core.groq_client import get_groq_client
from app.utils.config import settings


@dataclass(frozen=True)
class TranscriptionTimings:
    text: str
    encode_seconds: float
    api_seconds: float

    @property
    def total_seconds(self) -> float:
        return self.encode_seconds + self.api_seconds


def warm_up_groq_connection() -> None:
    """Force DNS + TLS setup before the first real STT request."""
    get_groq_client().models.list()


def _encode_wav_bytes(audio_array: np.ndarray, samplerate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, samplerate, format="WAV")
    buffer.seek(0)
    return buffer.read()


def transcribe_audio_bytes(audio_array: np.ndarray, samplerate: int) -> str:
    """Encode audio to in-memory WAV and send to Groq STT."""
    return transcribe_audio_bytes_timed(audio_array, samplerate).text


def transcribe_audio_bytes_timed(
    audio_array: np.ndarray,
    samplerate: int,
) -> TranscriptionTimings:
    """Transcribe in-memory audio and return separate encode vs API timings."""
    encode_start = time.time()
    wav_bytes = _encode_wav_bytes(audio_array, samplerate)
    encode_seconds = time.time() - encode_start

    api_start = time.time()
    client = get_groq_client()
    transcription = client.audio.transcriptions.create(
        file=("audio.wav", wav_bytes),
        model=settings.GROQ_STT_MODEL,
    )
    api_seconds = time.time() - api_start

    return TranscriptionTimings(
        text=transcription.text,
        encode_seconds=encode_seconds,
        api_seconds=api_seconds,
    )


def transcribe_audio_file(file_path: str) -> str:
    """Send an audio file to Groq STT and return the transcript."""
    client = get_groq_client()
    with open(file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=(file_path, audio_file.read()),
            model=settings.GROQ_STT_MODEL,
        )
    return transcription.text
