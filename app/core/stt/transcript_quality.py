from __future__ import annotations

import re
import unicodedata


def is_meaningful_transcript(text: str) -> bool:
    """Return False for empty, whitespace-only, or punctuation/noise-only STT output."""
    stripped = text.strip()
    if not stripped:
        return False

    # Keep letters and numbers from any script; drop punctuation/symbols.
    chars = [
        ch
        for ch in stripped
        if unicodedata.category(ch)[0] in {"L", "N"}
    ]
    if len(chars) < 2:
        return False

    collapsed = re.sub(r"\s+", " ", stripped)
    if collapsed in {".", "..", "...", "-", "--", "_"}:
        return False

    return True
