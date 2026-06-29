from __future__ import annotations

import queue
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from app.core.audio.vad import SileroVAD
from app.utils.paths import DEBUG_SEGMENTS_DIR


@dataclass(frozen=True)
class FinalizedSegment:
    """Metadata returned when SpeechSegmenter finalizes a segment."""

    segment_index: int
    wav_path: Path
    audio: np.ndarray
    audio_duration_seconds: float
    closed_at: float


def _resample_to_16k(samples: np.ndarray, native_rate: int) -> np.ndarray:
    """Resample one native-rate block to exactly one 512-sample 16 kHz VAD chunk."""
    target_len = SileroVAD.CHUNK_SIZE
    if len(samples) == 0:
        return np.zeros(target_len, dtype=np.float32)

    if native_rate == SileroVAD.SAMPLE_RATE:
        out = samples[:target_len]
        if len(out) < target_len:
            out = np.pad(out, (0, target_len - len(out)))
        return out.astype(np.float32)

    # Exact 3:1 ratio (e.g. 48000 -> 16000): average each triplet.
    if native_rate % SileroVAD.SAMPLE_RATE == 0:
        ratio = native_rate // SileroVAD.SAMPLE_RATE
        needed = target_len * ratio
        block = samples[:needed]
        if len(block) < needed:
            block = np.pad(block, (0, needed - len(block)))
        return block.reshape(target_len, ratio).mean(axis=1).astype(np.float32)

    x_in = np.arange(len(samples), dtype=np.float64)
    x_out = np.linspace(0, len(samples) - 1, target_len)
    return np.interp(x_out, x_in, samples).astype(np.float32)


class _SegmentState(Enum):
    IDLE = auto()
    SPEAKING = auto()


class ContinuousListener:
    """Callback-based capture; resamples device-native audio to 16 kHz VAD chunks."""

    VAD_CHUNK_SIZE = SileroVAD.CHUNK_SIZE
    VAD_SAMPLE_RATE = SileroVAD.SAMPLE_RATE

    def __init__(self, device_index: int) -> None:
        self.device_index = device_index
        self.chunk_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._native_buffer = np.array([], dtype=np.float32)

        device_info = sd.query_devices(device_index)
        self._channels = 1
        self.native_rate = self._pick_native_rate(device_index, device_info)
        self._native_blocksize = int(
            round(self.VAD_CHUNK_SIZE * self.native_rate / self.VAD_SAMPLE_RATE)
        )

        sd.check_input_settings(
            device=device_index,
            channels=self._channels,
            samplerate=self.native_rate,
        )

    @staticmethod
    def _pick_native_rate(device_index: int, device_info: dict) -> int:
        """Prefer 48000 Hz when supported (clean 3:1 resample to 16 kHz)."""
        default_rate = int(device_info["default_samplerate"])
        candidates = []
        for rate in (48000, default_rate, 44100):
            if rate not in candidates:
                candidates.append(rate)
        for rate in candidates:
            try:
                sd.check_input_settings(
                    device=device_index, channels=1, samplerate=rate
                )
                return rate
            except sd.PortAudioError:
                continue
        return default_rate

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            print(f"Audio stream status: {status}", file=sys.stderr)

        mono = indata[:, 0].astype(np.float32)
        self._native_buffer = np.concatenate([self._native_buffer, mono])

        while len(self._native_buffer) >= self._native_blocksize:
            native_chunk = self._native_buffer[: self._native_blocksize]
            self._native_buffer = self._native_buffer[self._native_blocksize :]
            vad_chunk = _resample_to_16k(native_chunk, self.native_rate)
            self.chunk_queue.put(vad_chunk)

    def start(self) -> None:
        if self._stream is not None:
            return
        try:
            self._stream = sd.InputStream(
                device=self.device_index,
                channels=self._channels,
                samplerate=self.native_rate,
                blocksize=self._native_blocksize,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except sd.PortAudioError as exc:
            raise sd.PortAudioError(
                f"Could not open device {self.device_index} at "
                f"{self.native_rate} Hz: {exc}"
            ) from exc

        print(
            f"Capture: {self.native_rate} Hz -> VAD: {self.VAD_SAMPLE_RATE} Hz "
            f"({self.VAD_CHUNK_SIZE}-sample chunks)"
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self._native_buffer = np.array([], dtype=np.float32)

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active


class SpeechSegmenter:
    """State machine that turns VAD probabilities into saved speech segments."""

    CHUNK_MS = (SileroVAD.CHUNK_SIZE / SileroVAD.SAMPLE_RATE) * 1000
    # Tunable: lower min_silence_ms = faster transcript trigger; higher = fewer
    # mid-sentence splits on natural breath pauses.
    DEFAULT_MIN_SILENCE_MS = 900

    def __init__(
        self,
        vad: SileroVAD,
        output_dir=DEBUG_SEGMENTS_DIR,
        speech_start_threshold: float = 0.5,
        speech_continue_threshold: float = 0.35,
        min_audio_level: float = 0.04,
        min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
        min_speech_ms: int = 250,
    ) -> None:
        self.vad = vad
        self.output_dir = output_dir
        self.speech_start_threshold = speech_start_threshold
        self.speech_continue_threshold = speech_continue_threshold
        self.min_audio_level = min_audio_level
        self.min_silence_chunks = max(1, int(round(min_silence_ms / self.CHUNK_MS)))
        self.min_speech_samples = int(
            SileroVAD.SAMPLE_RATE * min_speech_ms / 1000
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._state = _SegmentState.IDLE
        self._buffer: list[np.ndarray] = []
        self._silence_chunks = 0
        self._segment_count = 0
        self._total_speech_samples = 0
        self.last_probability = 0.0
        self.last_level = 0.0
        self._chunks_seen = 0

    @property
    def is_speaking(self) -> bool:
        return self._state == _SegmentState.SPEAKING

    @property
    def segment_count(self) -> int:
        return self._segment_count

    @property
    def total_speech_seconds(self) -> float:
        return self._total_speech_samples / SileroVAD.SAMPLE_RATE

    def _is_speech(self, probability: float, level: float) -> bool:
        """True when this chunk counts as speech (level gate + VAD hysteresis)."""
        if level < self.min_audio_level:
            return False
        if self._state == _SegmentState.SPEAKING:
            return probability >= self.speech_continue_threshold
        return probability >= self.speech_start_threshold

    def process_chunk(
        self, chunk: np.ndarray, *, verbose: bool = False
    ) -> FinalizedSegment | None:
        self._chunks_seen += 1
        self.last_level = float(np.abs(chunk).max())
        probability = self.vad.get_speech_probability(chunk)
        self.last_probability = probability
        is_speech = self._is_speech(probability, self.last_level)

        if self._state == _SegmentState.IDLE:
            if is_speech:
                self._state = _SegmentState.SPEAKING
                self._buffer = [chunk.copy()]
                self._silence_chunks = 0
                if verbose:
                    print(
                        f">> Speech started (vad={probability:.3f}, level={self.last_level:.3f})"
                    )
            elif verbose and self._chunks_seen % 31 == 0:
                self._print_live_status()
            return None

        # SPEAKING
        self._buffer.append(chunk.copy())

        if is_speech:
            self._silence_chunks = 0
            if verbose and self._chunks_seen % 15 == 0:
                self._print_live_status()
            return None

        self._silence_chunks += 1
        if verbose and self._silence_chunks == 1:
            print(
                f">> Pause detected (vad={probability:.3f}, level={self.last_level:.3f})"
            )
        elif verbose and self._silence_chunks == self.min_silence_chunks:
            print(f">> Closing segment after {self._silence_chunks} silent chunks...")
        if self._silence_chunks < self.min_silence_chunks:
            return None

        return self._finalize_segment()

    def _print_live_status(self) -> None:
        state = "SPEAKING" if self.is_speaking else "idle"
        print(
            f"   [{state}] vad={self.last_probability:.3f} "
            f"level={self.last_level:.3f} "
            f"(speak, then pause ~{self.min_silence_chunks * self.CHUNK_MS / 1000:.1f}s to split segments)"
        )

    def flush(self) -> FinalizedSegment | None:
        """Finalize an in-progress segment when stopping the listener."""
        if self._state == _SegmentState.SPEAKING and self._buffer:
            return self._finalize_segment(force=True)
        return None

    def _finalize_segment(self, force: bool = False) -> FinalizedSegment | None:
        audio = np.concatenate(self._buffer)

        if not force and self._silence_chunks >= self.min_silence_chunks:
            trim = self.min_silence_chunks * SileroVAD.CHUNK_SIZE
            if len(audio) > trim:
                audio = audio[:-trim]

        duration = len(audio) / SileroVAD.SAMPLE_RATE

        self._buffer = []
        self._silence_chunks = 0
        self._state = _SegmentState.IDLE

        if not force and len(audio) < self.min_speech_samples:
            return None

        self._segment_count += 1
        self._total_speech_samples += len(audio)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.output_dir / f"segment_{self._segment_count}_{timestamp}.wav"
        closed_at = time.time()

        print(
            f"[Segment #{self._segment_count}] duration: {duration:.1f}s - queued {path.name}"
        )
        return FinalizedSegment(
            segment_index=self._segment_count,
            wav_path=path,
            audio=audio,
            audio_duration_seconds=duration,
            closed_at=closed_at,
        )
