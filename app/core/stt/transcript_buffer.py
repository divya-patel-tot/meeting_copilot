from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Speaker = Literal["you", "them"]
SPEAKER_YOU: Speaker = "you"
SPEAKER_THEM: Speaker = "them"


@dataclass
class TranscriptEntry:
    segment_index: int
    timestamp: datetime
    audio_duration_seconds: float
    text: str
    latency_seconds: float
    is_valid: bool
    speaker: Speaker = SPEAKER_THEM
    encode_seconds: float = 0.0
    api_seconds: float = 0.0
    error: str | None = None


class TranscriptBuffer:
    """Thread-safe in-memory store for finalized segment transcripts."""

    def __init__(self) -> None:
        self._entries: list[TranscriptEntry] = []
        self._lock = threading.Lock()

    def add(self, entry: TranscriptEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def get_recent(self, n: int = 6) -> list[TranscriptEntry]:
        """Return the *n* most recent entries in spoken order (by segment timestamp)."""
        with self._lock:
            ordered = sorted(self._entries, key=lambda entry: entry.timestamp)
            return ordered[-n:]

    def get_all(self) -> list[TranscriptEntry]:
        with self._lock:
            return list(self._entries)
