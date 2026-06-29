import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.core.stt.groq_stt import transcribe_audio_file
        from app.utils.config import settings

        if not settings.GROQ_API_KEY.strip():
            return False, "SKIPPED - no API key in .env"

        samplerate = 16000
        duration = 2.0
        t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
        tone = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            sf.write(wav_path, tone, samplerate)
            transcript = transcribe_audio_file(wav_path)
            if not isinstance(transcript, str):
                return False, f"Expected str transcript, got {type(transcript).__name__}"
            return True, f"API call succeeded, transcript: {transcript!r}"
        finally:
            Path(wav_path).unlink(missing_ok=True)
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    if message.startswith("SKIPPED"):
        print(f"SKIPPED: {message}")
    else:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {message}")
