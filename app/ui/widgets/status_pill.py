from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PyQt6.QtWidgets import QGraphicsOpacityEffect, QHBoxLayout, QLabel, QWidget


class StatusPill(QWidget):
    """Rounded status badge with optional pulsing dot while listening."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusPill")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 12, 4)
        layout.setSpacing(8)

        self._dot = QWidget()
        self._dot.setObjectName("statusDot")
        self._dot.setFixedSize(8, 8)
        self._dot_effect = QGraphicsOpacityEffect(self._dot)
        self._dot.setGraphicsEffect(self._dot_effect)

        self._label = QLabel("Idle")
        self._label.setObjectName("statusPillLabel")

        layout.addWidget(self._dot, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._label, alignment=Qt.AlignmentFlag.AlignVCenter)

        self._pulse_anim: QPropertyAnimation | None = None
        self._listening = False

    def set_status(self, text: str, *, listening: bool) -> None:
        self._label.setText(text)
        self.setProperty("listening", listening)
        self.style().unpolish(self)
        self.style().polish(self)

        if listening and not self._listening:
            self._start_pulse()
        elif not listening and self._listening:
            self._stop_pulse()
        self._listening = listening

    def _start_pulse(self) -> None:
        self._stop_pulse()
        self._dot_effect.setOpacity(1.0)
        anim = QPropertyAnimation(self._dot_effect, b"opacity", self)
        anim.setDuration(900)
        anim.setStartValue(1.0)
        anim.setEndValue(0.35)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim.setLoopCount(-1)
        anim.start()
        self._pulse_anim = anim

    def _stop_pulse(self) -> None:
        if self._pulse_anim is not None:
            self._pulse_anim.stop()
            self._pulse_anim = None
        self._dot_effect.setOpacity(1.0)
