"""Frameless-window helpers (optional acrylic blur)."""

from __future__ import annotations

import sys
from ctypes import Structure, byref, c_int, c_size_t, c_void_p, cast, pointer, sizeof, windll

from PyQt6.QtWidgets import QWidget

RESIZE_MARGIN = 10


class _ACCENT_POLICY(Structure):
    _fields_ = [
        ("AccentState", c_int),
        ("AccentFlags", c_int),
        ("GradientColor", c_int),
        ("AnimationId", c_int),
    ]


class _WINDOWCOMPOSITIONATTRIBDATA(Structure):
    _fields_ = [
        ("Attribute", c_int),
        ("Data", c_void_p),
        ("SizeOfData", c_size_t),
    ]


WCA_ACCENT_POLICY = 19
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4


def try_enable_acrylic(window: QWidget, *, color_abgr: int = 0x9912141C) -> bool:
    """Best-effort Windows acrylic blur; returns False if unavailable."""
    if sys.platform != "win32":
        return False

    try:
        hwnd = int(window.winId())
        accent = _ACCENT_POLICY()
        accent.AccentState = ACCENT_ENABLE_ACRYLICBLURBEHIND
        accent.AccentFlags = 2
        accent.GradientColor = color_abgr
        accent.AnimationId = 0

        data = _WINDOWCOMPOSITIONATTRIBDATA()
        data.Attribute = WCA_ACCENT_POLICY
        data.Data = cast(pointer(accent), c_void_p)
        data.SizeOfData = sizeof(accent)

        set_attr = windll.user32.SetWindowCompositionAttribute
        set_attr.argtypes = [c_void_p, c_void_p]
        set_attr.restype = c_int
        return bool(set_attr(hwnd, byref(data)))
    except Exception:
        return False
