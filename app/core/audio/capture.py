import numpy as np
import sounddevice as sd


def record_test_clip(
    device_index: int,
    duration_seconds: float = 2.0,
    samplerate: int = 16000,
) -> np.ndarray:
    """Record a mono audio clip from the given input device."""
    frames = int(duration_seconds * samplerate)
    recording = sd.rec(
        frames,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    return recording.flatten()
