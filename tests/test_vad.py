import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.core.audio.vad import SileroVAD
        from app.utils.paths import MODELS_DIR

        model_path = MODELS_DIR / "silero_vad.onnx"
        if not model_path.exists():
            return False, f"Model not found at {model_path}"

        vad = SileroVAD(str(model_path))
        silence = np.zeros(512, dtype=np.float32)
        probability = vad.get_speech_probability(silence)

        if not isinstance(probability, float):
            return False, f"Expected float, got {type(probability).__name__}"
        if not 0.0 <= probability <= 1.0:
            return False, f"Probability out of range: {probability}"
        return True, f"Speech probability on silence: {probability:.4f}"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    status = "PASS" if success else "FAIL"
    print(f"{status}: {message}")
