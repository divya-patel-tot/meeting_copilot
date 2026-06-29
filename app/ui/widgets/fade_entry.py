from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QLabel, QWidget


def fade_in_widget(widget: QWidget, *, duration_ms: int = 180) -> None:
    """Fade a widget in once; safe to call when reusing an existing effect."""
    effect = widget.graphicsEffect()
    if not isinstance(effect, QGraphicsOpacityEffect):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)

    effect.setOpacity(0.0)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration_ms)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    widget._fade_anim = anim  # type: ignore[attr-defined]


class FadeEntryLabel(QLabel):
    """Single transcript line that fades in when added."""

    def __init__(self, text: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setObjectName("transcriptEntry")
        fade_in_widget(self)
