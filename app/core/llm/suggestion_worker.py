from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from app.core.llm.conversation_memory import ConversationMemory
from app.core.llm.groq_llm import generate_suggestion_streaming
from app.core.llm.prompt_builder import build_messages
from app.core.llm.suggestion_graph import run_suggestion_graph
from app.core.rag.knowledge_base import KnowledgeBase
from app.core.stt.transcript_buffer import TranscriptBuffer, TranscriptEntry
from app.utils.config import settings

OnTokenCallback = Callable[[int, str], None]
OnCompleteCallback = Callable[[int, str, float, float, int], None]
OnErrorCallback = Callable[[int, str], None]
OnRetrievalCallback = Callable[[int, list[dict], list[dict]], None]


def _emit_pseudo_stream(text: str, segment_index: int, on_token: OnTokenCallback) -> None:
    """Emit verified text in small chunks so the UI keeps a streaming feel."""
    chunk_size = 12
    for offset in range(0, len(text), chunk_size):
        on_token(segment_index, text[offset : offset + chunk_size])


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
        conversation_memory: ConversationMemory | None = None,
    ) -> None:
        self._on_token = on_token
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_retrieval = on_retrieval
        self._relevance_threshold = relevance_threshold
        self._top_k = top_k
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._memory = conversation_memory or ConversationMemory()
        self._completed_count = 0
        self._total_times: list[float] = []

    @property
    def conversation_memory(self) -> ConversationMemory:
        return self._memory

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
            score_filtered = [
                match
                for match in candidates
                if match["score"] >= self._relevance_threshold
            ]
            recent = transcript_buffer.get_recent(8)
            summary = self._memory.get_summary()
            retrieved: list[dict] = []

            if settings.USE_LANGGRAPH_SUGGESTIONS:
                graph_result = run_suggestion_graph(
                    segment_index=segment_index,
                    recent_transcript=recent,
                    conversation_summary=summary,
                    candidates=score_filtered or candidates,
                    max_revisions=settings.SUGGESTION_MAX_REVISIONS,
                )
                retrieved = graph_result["retrieved_chunks"]
                if self._on_retrieval is not None:
                    self._on_retrieval(segment_index, candidates, retrieved)

                final_text = graph_result["final_suggestion"].strip()
                if final_text:
                    time_to_first_token = time.time() - request_time
                    _emit_pseudo_stream(final_text, segment_index, self._on_token)
                    parts.append(final_text)

                self._memory.update_after_suggestion(
                    recent,
                    graph_result["latest_them_text"],
                    final_text,
                    retrieved,
                )
            else:
                retrieved = score_filtered
                if self._on_retrieval is not None:
                    self._on_retrieval(segment_index, candidates, retrieved)
                messages = build_messages(recent, retrieved, conversation_summary=summary)

                for delta in generate_suggestion_streaming(messages):
                    if time_to_first_token is None:
                        time_to_first_token = time.time() - request_time
                    parts.append(delta)
                    self._on_token(segment_index, delta)

                final_text = "".join(parts).strip()
                self._memory.update_after_suggestion(
                    recent,
                    entry.text.strip(),
                    final_text,
                    retrieved,
                )

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
