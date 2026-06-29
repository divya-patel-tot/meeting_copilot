from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "app" / "assets" / "models"
DATA_DIR = BASE_DIR / "data"
KB_DIR = DATA_DIR / "knowledge_base"
DEBUG_SEGMENTS_DIR = DATA_DIR / "debug_segments"

DATA_DIR.mkdir(parents=True, exist_ok=True)
KB_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_SEGMENTS_DIR.mkdir(parents=True, exist_ok=True)
