from __future__ import annotations

from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QMainWindow, QPushButton, QWidget

from app.ui.branding import APP_NAME
from app.ui.icon_loader import app_logo_pixmap


class CustomTitleBar(QWidget):
    """Frameless window title bar with logo, drag-to-move, and window controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("customTitleBar")
        self.setFixedHeight(40)
        self._drag_origin: QPoint | None = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 8, 0)
        layout.setSpacing(8)

        logo = QLabel()
        logo.setObjectName("titleBarLogo")
        logo.setFixedSize(22, 22)
        pixmap = app_logo_pixmap(22)
        if not pixmap.isNull():
            logo.setPixmap(pixmap)
        layout.addWidget(logo)

        title = QLabel(APP_NAME)
        title.setObjectName("titleBarLabel")
        layout.addWidget(title)
        layout.addStretch()

        self._minimize_btn = QPushButton("−")
        self._minimize_btn.setObjectName("titleBarButton")
        self._minimize_btn.setFixedSize(32, 28)
        self._minimize_btn.clicked.connect(self._minimize_window)

        self._maximize_btn = QPushButton("□")
        self._maximize_btn.setObjectName("titleBarButton")
        self._maximize_btn.setFixedSize(32, 28)
        self._maximize_btn.clicked.connect(self._toggle_maximize)

        self._close_btn = QPushButton("×")
        self._close_btn.setObjectName("titleBarCloseButton")
        self._close_btn.setFixedSize(32, 28)
        self._close_btn.clicked.connect(self._close_window)

        layout.addWidget(self._minimize_btn)
        layout.addWidget(self._maximize_btn)
        layout.addWidget(self._close_btn)

    def _window(self) -> QMainWindow:
        return self.window()  # type: ignore[return-value]

    def _minimize_window(self) -> None:
        self._window().showMinimized()

    def _toggle_maximize(self) -> None:
        window = self._window()
        if window.isMaximized():
            window.showNormal()
            self._maximize_btn.setText("□")
        else:
            window.showMaximized()
            self._maximize_btn.setText("❐")

    def _close_window(self) -> None:
        self._window().close()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            window = self._window()
            if window.isMaximized():
                super().mousePressEvent(event)
                return
            self._drag_origin = (
                event.globalPosition().toPoint() - window.frameGeometry().topLeft()
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        window = self._window()
        if (
            event.buttons() & Qt.MouseButton.LeftButton
            and self._drag_origin is not None
            and not window.isMaximized()
        ):
            window.move(event.globalPosition().toPoint() - self._drag_origin)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_origin = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximize()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)
