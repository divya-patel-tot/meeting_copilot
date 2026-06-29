import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.core.llm.groq_llm import ask_llm
        from app.utils.config import settings

        if not settings.GROQ_API_KEY.strip():
            return False, "SKIPPED - no API key in .env"

        response = ask_llm("Reply with exactly: OK")
        if "OK" in response.upper():
            return True, f"Response: {response.strip()[:100]}"
        return False, f"Response did not contain OK: {response!r}"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    if message.startswith("SKIPPED"):
        print(f"SKIPPED: {message}")
    else:
        status = "PASS" if success else "FAIL"
        print(f"{status}: {message}")
