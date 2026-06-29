#!/usr/bin/env python3
"""Dev runner: continuous mic capture + Silero VAD speech segmentation."""

from __future__ import annotations

import argparse
import queue
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.core.audio.device_manager import (
    get_input_device_name,
    print_input_devices,
    resolve_input_device,
)
from app.core.audio.listener import ContinuousListener, SpeechSegmenter
from app.core.audio.vad import SileroVAD
from app.utils.paths import MODELS_DIR


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for speech segments via Silero VAD (dev tool).",
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
        help="Hide live VAD level/status lines (segments still print)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if args.list:
        print_input_devices()
        return 0

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
    verbose = not args.quiet

    print(f"\nListening on [{device_index}] {device_name}")
    print("Ctrl+C to stop")
    if verbose:
        print(
            "  Pause ~1.5s between phrases to split into separate segments.\n"
            "  If level stays 0.000, wrong mic or Windows mic permission blocked.\n"
        )

    listener.start()
    try:
        while listener.is_running:
            try:
                chunk = listener.chunk_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            segmenter.process_chunk(chunk, verbose=verbose)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        segmenter.flush()
        listener.stop()

    print(
        f"\n{segmenter.segment_count} segment(s) detected, "
        f"total speech time: {segmenter.total_speech_seconds:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
