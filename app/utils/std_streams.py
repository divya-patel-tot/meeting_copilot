"""Restore usable stdout/stderr in PyInstaller windowed (console=False) builds."""

from __future__ import annotations

import io
import os
import sys


def ensure_std_streams() -> None:
    """Replace null stdout/stderr so tqdm/HuggingFace can write safely."""
    if sys.stdout is None:
        sys.stdout = io.StringIO()
    if sys.stderr is None:
        sys.stderr = io.StringIO()


def silence_download_progress_bars() -> None:
    """Disable tqdm-style progress output for first-run model downloads."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
