from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class GlassPanel(QWidget):
    """Rounded sub-panel with a section heading and content slot."""

    def __init__(
        self,
        title: str,
        *,
        variant: str = "transcript",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName(
            "suggestionGlassPanel" if variant == "suggestion" else "transcriptGlassPanel"
        )
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(10)

        heading = QLabel(title)
        heading.setObjectName("panelHeading")
        layout.addWidget(heading)

        self.content_area = QWidget()
        self.content_area.setObjectName("panelContent")
        self.content_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(0)
        layout.addWidget(self.content_area, stretch=1)
