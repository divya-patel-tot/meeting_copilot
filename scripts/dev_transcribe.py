#!/usr/bin/env python3
"""Dev runner: live capture + VAD segmentation + async Groq STT transcripts."""

from __future__ import annotations

import argparse
import queue
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.audio.device_manager import (
    get_input_device_name,
    print_input_devices,
    resolve_input_device,
)
from app.core.audio.listener import ContinuousListener, FinalizedSegment, SpeechSegmenter
from app.core.audio.vad import SileroVAD
from app.core.stt.groq_stt import warm_up_groq_connection
from app.core.stt.transcript_buffer import TranscriptBuffer, TranscriptEntry
from app.core.stt.transcription_worker import TranscriptionWorker
from app.utils.config import settings
from app.utils.paths import MODELS_DIR

_print_lock = threading.Lock()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen, segment speech, and transcribe via Groq STT (dev tool).",
    )
    parser.add_argument(
        "device",
        nargs="?",
        type=int,
        help="PortAudio input device index (skip interactive menu)",
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device_flag",
        type=int,
        help="PortAudio input device index (same as positional arg)",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List input sources and exit",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Hide live VAD level/status lines (transcript prints remain)",
    )
    return parser.parse_args()


def _print_transcript_result(entry: TranscriptEntry) -> None:
    with _print_lock:
        timing = (
            f"encode: {entry.encode_seconds:.2f}s, "
            f"api: {entry.api_seconds:.2f}s, "
            f"total: {entry.latency_seconds:.2f}s"
        )
        if entry.error:
            print(f"[Segment #{entry.segment_index}] STT FAILED — {entry.error}")
        elif entry.is_valid:
            print(
                f'[Segment #{entry.segment_index}] '
                f'("{entry.audio_duration_seconds:.1f}s audio, {timing}") '
                f'→ "{entry.text}"'
            )
        else:
            detail = entry.text or "(empty)"
            print(
                f"[Segment #{entry.segment_index}] discarded — "
                f"no meaningful speech detected ({detail!r}) [{timing}]"
            )


def _submit_segment(worker: TranscriptionWorker, segment: FinalizedSegment) -> None:
    worker.submit_segment(
        segment_index=segment.segment_index,
        audio=segment.audio,
        wav_path=segment.wav_path,
        segment_closed_time=segment.closed_at,
        audio_duration_seconds=segment.audio_duration_seconds,
    )


def _print_summary(buffer: TranscriptBuffer, segment_count: int) -> None:
    entries = buffer.get_all()
    valid = [e for e in entries if e.is_valid and not e.error]
    discarded = [e for e in entries if not e.is_valid and not e.error]
    failed = [e for e in entries if e.error]

    print(f"\nSegments captured: {segment_count}")
    print(f"Transcripts: {len(valid)} valid, {len(discarded)} discarded, {len(failed)} failed")
    if valid:
        avg_latency = sum(e.latency_seconds for e in valid) / len(valid)
        avg_api = sum(e.api_seconds for e in valid) / len(valid)
        print(f"Average end-to-end latency (valid): {avg_latency:.2f}s")
        print(f"Average Groq API time (valid): {avg_api:.2f}s")


def main() -> int:
    args = _parse_args()

    if args.list:
        print_input_devices()
        return 0

    if not settings.GROQ_API_KEY.strip():
        print("GROQ_API_KEY is not set in .env", file=sys.stderr)
        return 1

    print("Warming up connection...")
    try:
        warm_up_groq_connection()
    except Exception as exc:
        print(f"Warm-up failed: {exc}", file=sys.stderr)
        return 1
    print("Ready.")

    device_index = args.device_flag if args.device_flag is not None else args.device
    device_index = resolve_input_device(device_index)

    model_path = MODELS_DIR / "silero_vad.onnx"
    if not model_path.exists():
        print(f"Silero VAD model not found: {model_path}", file=sys.stderr)
        return 1

    device_name = get_input_device_name(device_index)
    vad = SileroVAD(str(model_path))
    vad.reset_state()

    listener = ContinuousListener(device_index)
    segmenter = SpeechSegmenter(vad)
    buffer = TranscriptBuffer()
    worker = TranscriptionWorker(buffer, on_result=_print_transcript_result)
    verbose = not args.quiet

    print(f"\nListening + transcribing on [{device_index}] {device_name}")
    print("Ctrl+C to stop")
    if verbose:
        print(
            f"\n  Pause ~{SpeechSegmenter.DEFAULT_MIN_SILENCE_MS / 1000:.1f}s between "
            "phrases to split segments.\n"
            "  Transcripts appear asynchronously (capture keeps running).\n"
        )

    listener.start()
    try:
        while listener.is_running:
            try:
                chunk = listener.chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            finalized = segmenter.process_chunk(chunk, verbose=verbose)
            if finalized is not None:
                _submit_segment(worker, finalized)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        flushed = segmenter.flush()
        if flushed is not None:
            _submit_segment(worker, flushed)
        listener.stop()
        worker.shutdown(wait=True)

    _print_summary(buffer, segmenter.segment_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
