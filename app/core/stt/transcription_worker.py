from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from app.core.audio.vad import SileroVAD
from app.core.stt.groq_stt import transcribe_audio_bytes_timed
from app.core.stt.transcript_buffer import (
    SPEAKER_THEM,
    Speaker,
    TranscriptBuffer,
    TranscriptEntry,
)
from app.core.stt.transcript_quality import is_meaningful_transcript

OnResultCallback = Callable[[TranscriptEntry], None]


def _save_debug_wav(path: Path, audio: np.ndarray, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, samplerate)


class TranscriptionWorker:
    """Submit segment audio to Groq STT on a background thread pool."""

    def __init__(
        self,
        buffer: TranscriptBuffer,
        on_result: OnResultCallback,
        *,
        max_workers: int = 4,
    ) -> None:
        self._buffer = buffer
        self._on_result = on_result
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_segment(
        self,
        segment_index: int,
        audio: np.ndarray,
        wav_path: str | Path,
        segment_closed_time: float,
        audio_duration_seconds: float,
        *,
        speaker: Speaker = SPEAKER_THEM,
        samplerate: int = SileroVAD.SAMPLE_RATE,
    ) -> None:
        self._executor.submit(
            self._run_job,
            segment_index,
            np.asarray(audio, dtype=np.float32).copy(),
            Path(wav_path),
            segment_closed_time,
            audio_duration_seconds,
            speaker,
            samplerate,
        )

    def _run_job(
        self,
        segment_index: int,
        audio: np.ndarray,
        wav_path: Path,
        segment_closed_time: float,
        audio_duration_seconds: float,
        speaker: Speaker,
        samplerate: int,
    ) -> None:
        closed_at = datetime.fromtimestamp(segment_closed_time)

        threading.Thread(
            target=_save_debug_wav,
            args=(wav_path, audio, samplerate),
            daemon=True,
        ).start()

        try:
            result = transcribe_audio_bytes_timed(audio, samplerate)
            latency = time.time() - segment_closed_time
            is_valid = is_meaningful_transcript(result.text)
            entry = TranscriptEntry(
                segment_index=segment_index,
                timestamp=closed_at,
                audio_duration_seconds=audio_duration_seconds,
                text=result.text.strip(),
                latency_seconds=latency,
                encode_seconds=result.encode_seconds,
                api_seconds=result.api_seconds,
                is_valid=is_valid,
                speaker=speaker,
            )
        except Exception as exc:
            entry = TranscriptEntry(
                segment_index=segment_index,
                timestamp=closed_at,
                audio_duration_seconds=audio_duration_seconds,
                text="",
                latency_seconds=time.time() - segment_closed_time,
                is_valid=False,
                speaker=speaker,
                error=str(exc),
            )

        self._buffer.add(entry)
        self._on_result(entry)

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
