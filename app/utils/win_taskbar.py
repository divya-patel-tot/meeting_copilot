"""Windows taskbar / shell branding helpers."""

from __future__ import annotations

import sys


def apply_windows_app_user_model_id(app_id: str) -> None:
    """Assign a unique AppUserModelID so Windows uses our icon, not python.exe.

    Must run before QApplication is created. Without this, launching via
    ``python.exe -m app.main`` shows the Python icon in the taskbar.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass
