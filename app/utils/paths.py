import os
import sys
from pathlib import Path


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS")


if _is_frozen():
    _BUNDLE_DIR = Path(sys._MEIPASS)
    BASE_DIR = Path(sys.executable).resolve().parent
    MODELS_DIR = _BUNDLE_DIR / "app" / "assets" / "models"
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    MODELS_DIR = BASE_DIR / "app" / "assets" / "models"

DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
DEBUG_SEGMENTS_DIR = DATA_DIR / "debug_segments"
FASTEMBED_CACHE_DIR = DATA_DIR / "fastembed_cache"
SETUP_MARKER_PATH = DATA_DIR / ".setup_complete"
ENV_PATH = BASE_DIR / ".env"
SMOKE_TEST_LOG_PATH = BASE_DIR / "smoke_test.log"


def resource_path(*parts: str) -> Path:
    """Resolve a project resource path in dev and PyInstaller bundle layouts."""
    if _is_frozen():
        return _BUNDLE_DIR.joinpath(*parts)
    return BASE_DIR.joinpath(*parts)


SILERO_VAD_FILENAME = "silero_vad.onnx"
SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/"
    f"{SILERO_VAD_FILENAME}"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)
if not _is_frozen():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("FASTEMBED_CACHE_PATH", str(FASTEMBED_CACHE_DIR))
FASTEMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
