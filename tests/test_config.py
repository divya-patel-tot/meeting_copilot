"""Settings load from .env only — not from OS environment variables."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.utils.config import Settings

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            env_file = tmp_path / ".env"
            env_file.write_text(
                "GROQ_API_KEY=from_env_file\n"
                "GROQ_STT_MODEL=whisper-large-v3-turbo\n"
                "GROQ_LLM_MODEL=llama-3.3-70b-versatile\n",
                encoding="utf-8",
            )

            previous = os.environ.get("GROQ_API_KEY")
            os.environ["GROQ_API_KEY"] = "from_system_environment"
            try:
                loaded = Settings(_env_file=env_file)
                if loaded.GROQ_API_KEY != "from_env_file":
                    return (
                        False,
                        f"Expected file key, got {loaded.GROQ_API_KEY!r} "
                        "(system env may be overriding .env)",
                    )

                missing = tmp_path / "missing.env"
                empty = Settings(_env_file=missing)
                if empty.GROQ_API_KEY != "":
                    return (
                        False,
                        f"Expected empty key without .env, got {empty.GROQ_API_KEY!r}",
                    )
            finally:
                if previous is None:
                    os.environ.pop("GROQ_API_KEY", None)
                else:
                    os.environ["GROQ_API_KEY"] = previous

        return True, "Settings ignore OS GROQ_API_KEY; .env file is authoritative"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    print(("PASS" if success else "FAIL") + ": " + message)
