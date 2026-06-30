from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

from PyQt6.QtCore import QObject, pyqtSignal

from app.core.audio.listener import (
    ContinuousListener,
    FinalizedSegment,
    SOURCE_LOOPBACK,
    SOURCE_MICROPHONE,
    SpeechSegmenter,
)
from app.core.audio.vad import SileroVAD
from app.core.llm.conversation_memory import ConversationMemory
from app.core.llm.suggestion_worker import SuggestionWorker
from app.core.rag.knowledge_base import KnowledgeBase
from app.core.stt.groq_stt import warm_up_groq_connection
from app.core.stt.transcript_buffer import (
    SPEAKER_THEM,
    SPEAKER_YOU,
    Speaker,
    TranscriptBuffer,
    TranscriptEntry,
)
from app.core.stt.transcription_worker import TranscriptionWorker
from app.utils.config import settings
from app.utils.paths import MODELS_DIR


@dataclass
class _CaptureStream:
    name: str
    listener: ContinuousListener
    segmenter: SpeechSegmenter
    speaker: Speaker


class PipelineController(QObject):
    """Owns the dev_assistant pipeline and bridges worker callbacks to Qt signals."""

    status_changed = pyqtSignal(str)
    transcript_ready = pyqtSignal(object)  # TranscriptEntry
    suggestion_started = pyqtSignal(int, str)  # segment_index, transcript_text
    suggestion_retrieval = pyqtSignal(int, object)  # segment_index, list[dict]
    retrieval_debug = pyqtSignal(int, list)  # segment_index, [(score, source, passed)]
    suggestion_token = pyqtSignal(int, str)  # segment_index, delta
    suggestion_complete = pyqtSignal(int, str, dict)  # segment_index, full_text, stats
    suggestion_error = pyqtSignal(int, str)  # segment_index, error message
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._session_thread: threading.Thread | None = None
        self._stream_threads: list[threading.Thread] = []
        self._listening = False
        self._show_retrieval_debug = False
        self._segment_counter = 0
        self._segment_counter_lock = threading.Lock()

        self._knowledge_base = KnowledgeBase()
        self._transcript_buffer = TranscriptBuffer()
        self._conversation_memory = ConversationMemory()

        self._streams: list[_CaptureStream] = []
        self._transcription_worker: TranscriptionWorker | None = None
        self._suggestion_worker: SuggestionWorker | None = None

    @property
    def is_listening(self) -> bool:
        with self._lock:
            return self._listening

    @property
    def knowledge_base_stats(self) -> dict[str, Any]:
        return self._knowledge_base.get_stats()

    @property
    def knowledge_base(self) -> KnowledgeBase:
        return self._knowledge_base

    def set_retrieval_debug(self, enabled: bool) -> None:
        self._show_retrieval_debug = enabled

    def start_listening(
        self,
        mic_device_index: int,
        loopback_device_index: int | None = None,
        *,
        mic_only_testing: bool = False,
    ) -> None:
        with self._lock:
            if self._listening:
                return
            self._listening = True

        self._stop_event.clear()
        self._session_thread = threading.Thread(
            target=self._run_capture_session,
            args=(mic_device_index, loopback_device_index, mic_only_testing),
            daemon=True,
            name="pipeline-session",
        )
        self._session_thread.start()

    def stop_listening(self, *, wait: bool = True, timeout: float = 2.0) -> None:
        self._stop_event.set()
        thread = self._session_thread
        if wait and thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        if thread is not None and thread.is_alive():
            self._force_shutdown()

    def reload_audio_devices(
        self,
        mic_device_index: int,
        loopback_device_index: int | None,
        *,
        mic_only_testing: bool = False,
    ) -> None:
        """Hot-swap capture endpoints by restarting the active session."""
        with self._lock:
            if not self._listening:
                return
        self.stop_listening(wait=True, timeout=3.0)
        with self._lock:
            if self._listening:
                return
        self.start_listening(
            mic_device_index,
            loopback_device_index,
            mic_only_testing=mic_only_testing,
        )

    def _run_capture_session(
        self,
        mic_device_index: int,
        loopback_device_index: int | None,
        mic_only_testing: bool,
    ) -> None:
        try:
            if not settings.GROQ_API_KEY.strip():
                raise RuntimeError("GROQ_API_KEY is not set in .env")

            model_path = MODELS_DIR / "silero_vad.onnx"
            if not model_path.exists():
                raise FileNotFoundError(f"Silero VAD model not found: {model_path}")

            self.status_changed.emit("Warming up...")
            warm_up_groq_connection()

            with self._segment_counter_lock:
                self._segment_counter = 0
            self._conversation_memory.reset()

            mic_vad = SileroVAD(str(model_path))
            mic_vad.reset_state()
            mic_listener = ContinuousListener(
                mic_device_index, source_mode=SOURCE_MICROPHONE
            )
            mic_segmenter = SpeechSegmenter(mic_vad)
            mic_speaker: Speaker = SPEAKER_THEM if mic_only_testing else SPEAKER_YOU

            self._streams = [
                _CaptureStream(
                    name="mic",
                    listener=mic_listener,
                    segmenter=mic_segmenter,
                    speaker=mic_speaker,
                )
            ]

            if not mic_only_testing:
                if loopback_device_index is None:
                    raise RuntimeError("No loopback capture device configured.")
                call_vad = SileroVAD(str(model_path))
                call_vad.reset_state()
                call_listener = ContinuousListener(
                    loopback_device_index, source_mode=SOURCE_LOOPBACK
                )
                call_segmenter = SpeechSegmenter(call_vad)
                self._streams.append(
                    _CaptureStream(
                        name="call",
                        listener=call_listener,
                        segmenter=call_segmenter,
                        speaker=SPEAKER_THEM,
                    )
                )

            self._transcription_worker = TranscriptionWorker(
                self._transcript_buffer,
                on_result=self._on_transcript,
            )
            self._suggestion_worker = SuggestionWorker(
                on_token=self._on_suggestion_token,
                on_complete=self._on_suggestion_complete,
                on_error=self._on_suggestion_error,
                on_retrieval=self._on_retrieval,
                conversation_memory=self._conversation_memory,
            )

            for stream in self._streams:
                stream.listener.start()

            self._stream_threads = []
            for stream in self._streams:
                thread = threading.Thread(
                    target=self._run_stream_loop,
                    args=(stream,),
                    daemon=True,
                    name=f"pipeline-{stream.name}",
                )
                thread.start()
                self._stream_threads.append(thread)

            self.status_changed.emit("Listening")

            while not self._stop_event.is_set():
                dead = [s for s in self._streams if not s.listener.is_running]
                if dead:
                    names = ", ".join(s.name for s in dead)
                    self.error_occurred.emit(
                        f"Audio capture stopped unexpectedly ({names}). "
                        "Check your microphone or playback device."
                    )
                    break
                if not any(t.is_alive() for t in self._stream_threads):
                    break
                self._stop_event.wait(timeout=0.25)
        except Exception as exc:
            self.error_occurred.emit(str(exc))
        finally:
            self._shutdown_pipeline()
            with self._lock:
                self._listening = False
            self.status_changed.emit("Idle")

    def _run_stream_loop(self, stream: _CaptureStream) -> None:
        while not self._stop_event.is_set() and stream.listener.is_running:
            try:
                chunk = stream.listener.chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            finalized = stream.segmenter.process_chunk(chunk, verbose=False)
            if finalized is not None:
                self._submit_segment(finalized, stream.speaker)

    def _shutdown_pipeline(self) -> None:
        for stream in self._streams:
            try:
                stream.listener.stop()
            except Exception:
                pass

        for thread in self._stream_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        if self._transcription_worker is not None:
            for stream in self._streams:
                flushed = stream.segmenter.flush()
                if flushed is not None:
                    self._submit_segment(flushed, stream.speaker)

        if self._transcription_worker is not None:
            self._transcription_worker.shutdown(wait=True)
        if self._suggestion_worker is not None:
            self._suggestion_worker.shutdown(wait=True)

        self._streams = []
        self._stream_threads = []
        self._transcription_worker = None
        self._suggestion_worker = None

    def _force_shutdown(self) -> None:
        for stream in self._streams:
            try:
                stream.listener.stop()
            except Exception:
                pass
        if self._transcription_worker is not None:
            self._transcription_worker.shutdown(wait=False)
        if self._suggestion_worker is not None:
            self._suggestion_worker.shutdown(wait=False)
        with self._lock:
            self._listening = False
        self.status_changed.emit("Idle")

    def _submit_segment(self, segment: FinalizedSegment, speaker: Speaker) -> None:
        if self._transcription_worker is None:
            return
        with self._segment_counter_lock:
            self._segment_counter += 1
            segment_index = self._segment_counter
        self._transcription_worker.submit_segment(
            segment_index=segment_index,
            audio=segment.audio,
            wav_path=segment.wav_path,
            segment_closed_time=segment.closed_at,
            audio_duration_seconds=segment.audio_duration_seconds,
            speaker=speaker,
        )

    def _on_transcript(self, entry: TranscriptEntry) -> None:
        self.transcript_ready.emit(entry)
        if (
            entry.is_valid
            and not entry.error
            and entry.speaker == SPEAKER_THEM
        ):
            self.suggestion_started.emit(entry.segment_index, entry.text)
            if self._suggestion_worker is not None:
                self._suggestion_worker.submit_transcript_entry(
                    entry,
                    self._transcript_buffer,
                    self._knowledge_base,
                )

    def _on_retrieval(
        self,
        segment_index: int,
        candidates: list[dict],
        filtered: list[dict],
    ) -> None:
        passed_sources = {match["source"] for match in filtered}
        context_rows = [
            {
                "source": str(match["source"]),
                "score": float(match["score"]),
                "text": str(match.get("text", ""))[:240],
                "used": match["source"] in passed_sources,
            }
            for match in candidates
        ]
        self.suggestion_retrieval.emit(segment_index, context_rows)

        if not self._show_retrieval_debug:
            return
        rows = [
            (float(match["score"]), str(match["source"]), match["source"] in passed_sources)
            for match in candidates
        ]
        self.retrieval_debug.emit(segment_index, rows)

    def _on_suggestion_token(self, segment_index: int, delta: str) -> None:
        self.suggestion_token.emit(segment_index, delta)

    def _on_suggestion_complete(
        self,
        segment_index: int,
        full_text: str,
        time_to_first_token: float,
        total_time: float,
        num_chunks_used: int,
    ) -> None:
        stats = {
            "first_token_seconds": time_to_first_token,
            "total_seconds": total_time,
            "num_chunks_used": num_chunks_used,
        }
        self.suggestion_complete.emit(segment_index, full_text, stats)

    def _on_suggestion_error(self, segment_index: int, reason: str) -> None:
        self.suggestion_error.emit(segment_index, reason)
