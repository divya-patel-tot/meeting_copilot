from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.core.llm.groq_llm import generate_suggestion_streaming
from app.core.llm.prompt_builder import build_messages
from app.core.rag.knowledge_base import KnowledgeBase
from app.core.stt.transcript_buffer import TranscriptBuffer, TranscriptEntry

OnTokenCallback = Callable[[int, str], None]
OnCompleteCallback = Callable[[int, str, float, float, int], None]
OnErrorCallback = Callable[[int, str], None]
OnRetrievalCallback = Callable[[int, list[dict], list[dict]], None]


class SuggestionWorker:
    """Generate LLM suggestions sequentially on a background thread."""

    def __init__(
        self,
        on_token: OnTokenCallback,
        on_complete: OnCompleteCallback,
        on_error: OnErrorCallback,
        *,
        on_retrieval: OnRetrievalCallback | None = None,
        relevance_threshold: float = 0.55,
        top_k: int = 4,
        max_workers: int = 1,
    ) -> None:
        self._on_token = on_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_retrieval = on_retrieval
        self._relevance_threshold = relevance_threshold
        self._top_k = top_k
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._completed_count = 0
        self._total_times: list[float] = []

    @property
    def completed_count(self) -> int:
        return self._completed_count

    @property
    def average_total_time(self) -> float | None:
        if not self._total_times:
            return None
        return sum(self._total_times) / len(self._total_times)

    def submit_transcript_entry(
        self,
        entry: TranscriptEntry,
        transcript_buffer: TranscriptBuffer,
        knowledge_base: KnowledgeBase,
    ) -> None:
        self._executor.submit(
            self._run_job,
            entry,
            transcript_buffer,
            knowledge_base,
        )

    def _run_job(
        self,
        entry: TranscriptEntry,
        transcript_buffer: TranscriptBuffer,
        knowledge_base: KnowledgeBase,
    ) -> None:
        segment_index = entry.segment_index
        request_time = time.time()
        time_to_first_token: float | None = None
        parts: list[str] = []

        try:
            candidates = knowledge_base.query_candidates(
                entry.text,
                top_k=self._top_k,
            )
            retrieved = [
                match
                for match in candidates
                if match["score"] >= self._relevance_threshold
            ]
            if self._on_retrieval is not None:
                self._on_retrieval(segment_index, candidates, retrieved)
            recent = transcript_buffer.get_recent(6)
            messages = build_messages(recent, retrieved)

            for delta in generate_suggestion_streaming(messages):
                if time_to_first_token is None:
                    time_to_first_token = time.time() - request_time
                parts.append(delta)
                self._on_token(segment_index, delta)

            full_text = "".join(parts).strip()
            total_time = time.time() - request_time
            ttft = time_to_first_token if time_to_first_token is not None else total_time
            self._completed_count += 1
            self._total_times.append(total_time)
            self._on_complete(segment_index, full_text, ttft, total_time, len(retrieved))
        except Exception as exc:
            self._on_error(segment_index, str(exc))

    def shutdown(self, *, wait: bool = True) -> None:
        self._executor.shutdown(wait=wait)
