from __future__ import annotations

import threading

from app.core.llm.llm_utils import chat_complete
from app.core.stt.transcript_buffer import SPEAKER_THEM, SPEAKER_YOU, TranscriptEntry

_MEMORY_SYSTEM = """You maintain a running summary of a live meeting for a copilot assistant.

Rules:
- Only include facts explicitly stated in the transcript or approved document context.
- Never infer, guess, or add details not present in the input.
- Keep the summary under 120 words.
- Track: topics discussed, concrete facts/numbers/policies mentioned, and any open questions.
- Use neutral third-person phrasing (e.g. "They asked about…", "The host said…")."""


class ConversationMemory:
    """Thread-safe session memory: rolling summary + recent transcript window."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._summary = ""

    def get_summary(self) -> str:
        with self._lock:
            return self._summary

    def reset(self) -> None:
        with self._lock:
            self._summary = ""

    def update_after_suggestion(
        self,
        recent_transcript: list[TranscriptEntry],
        latest_them_text: str,
        final_suggestion: str,
        retrieved_chunks: list[dict],
    ) -> None:
        """Merge the latest exchange into the rolling summary."""
        transcript_lines = _format_lines(recent_transcript[-6:])
        doc_lines = "\n".join(
            f'- [{chunk.get("source", "unknown")}] {chunk.get("text", "").strip()}'
            for chunk in retrieved_chunks
        ) or "(none)"

        user_content = (
            f"Prior summary:\n{self.get_summary() or '(empty)'}\n\n"
            f"Recent transcript:\n{transcript_lines}\n\n"
            f"Latest [Them] message:\n{latest_them_text}\n\n"
            f"Approved reply suggestion for the host:\n{final_suggestion}\n\n"
            f"Document context used:\n{doc_lines}\n\n"
            "Write the updated summary."
        )

        try:
            updated = chat_complete(
                [
                    {"role": "system", "content": _MEMORY_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
            )
        except Exception:
            return

        with self._lock:
            self._summary = updated.strip()


def _format_lines(entries: list[TranscriptEntry]) -> str:
    lines: list[str] = []
    for entry in entries:
        if not entry.is_valid or not entry.text.strip():
            continue
        tag = "[You]" if entry.speaker == SPEAKER_YOU else "[Them]"
        lines.append(f'{tag}: "{entry.text.strip()}"')
    return "\n".join(lines) or "(none)"
