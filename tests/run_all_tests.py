import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests import test_audio_devices
from tests import test_config
from tests import test_doc_parsing
from tests import test_embeddings
from tests import test_groq_llm
from tests import test_groq_stt
from tests import test_pyqt
from tests import test_vad
from tests import test_vectorstore

TESTS = [
    ("pyqt", test_pyqt),
    ("config", test_config),
    ("audio_devices", test_audio_devices),
    ("doc_parsing", test_doc_parsing),
    ("embeddings", test_embeddings),
    ("vectorstore", test_vectorstore),
    ("vad", test_vad),
    ("groq_llm", test_groq_llm),
    ("groq_stt", test_groq_stt),
]


def _status(success: bool, message: str) -> str:
    if message.startswith("SKIPPED"):
        return "SKIPPED"
    return "PASS" if success else "FAIL"


def main() -> int:
    results: list[tuple[str, str, str]] = []
    hard_failures = 0

    print("Running dependency verification tests...\n")

    for name, module in TESTS:
        success, message = module.run()
        status = _status(success, message)
        if status == "FAIL":
            hard_failures += 1
        results.append((name, status, message.replace("\n", " | ")))

    name_width = max(len(r[0]) for r in results)
    print(f"{'Test':<{name_width}}  | Status   | Message")
    print("-" * (name_width + 60))

    for name, status, message in results:
        print(f"{name:<{name_width}}  | {status:<8} | {message}")

    print()
    passed = sum(1 for _, s, _ in results if s == "PASS")
    skipped = sum(1 for _, s, _ in results if s == "SKIPPED")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    print(f"Summary: {passed} passed, {skipped} skipped, {failed} failed")

    return 1 if hard_failures else 0


if __name__ == "__main__":
    sys.exit(main())
