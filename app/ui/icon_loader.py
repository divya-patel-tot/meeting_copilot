from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QPainter, QPainterPath, QPixmap

from app.utils.paths import resource_path

_ICON_CACHE: QIcon | None = None
_IOS_RADIUS_RATIO = 0.22


def _rounded_pixmap(source: QPixmap, size: int) -> QPixmap:
    """Apply iOS-style rounded corners to a square pixmap."""
    if source.isNull() or size <= 0:
        return QPixmap()

    scaled = source.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    result = QPixmap(size, size)
    result.fill(Qt.GlobalColor.transparent)

    painter = QPainter(result)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

    radius = max(2.0, size * _IOS_RADIUS_RATIO)
    path = QPainterPath()
    path.addRoundedRect(0.0, 0.0, float(size), float(size), radius, radius)
    painter.setClipPath(path)

    x = (size - scaled.width()) // 2
    y = (size - scaled.height()) // 2
    painter.drawPixmap(x, y, scaled)
    painter.end()
    return result


def _load_source_pixmap() -> QPixmap:
    rounded = resource_path("app", "assets", "logo_rounded.png")
    if rounded.exists():
        pixmap = QPixmap(str(rounded.resolve()))
        if not pixmap.isNull():
            return pixmap

    ico = resource_path("app", "assets", "logo.ico")
    if ico.exists():
        icon = QIcon(str(ico.resolve()))
        sizes = icon.availableSizes()
        if sizes:
            return icon.pixmap(sizes[-1])

    png = resource_path("app", "assets", "logo.png")
    if png.exists():
        return QPixmap(str(png.resolve()))
    return QPixmap()


def app_icon() -> QIcon:
    """Load the CuePilot AI application icon from bundled assets."""
    global _ICON_CACHE
    if _ICON_CACHE is not None and not _ICON_CACHE.isNull():
        return _ICON_CACHE

    source = _load_source_pixmap()
    if source.isNull():
        return QIcon()

    icon = QIcon()
    for size in (16, 24, 32, 48, 64, 128, 256):
        icon.addPixmap(_rounded_pixmap(source, size))

    _ICON_CACHE = icon
    return icon


def app_logo_pixmap(size: int = 22) -> QPixmap:
    """Scaled logo with iOS-style rounded corners for title bar branding."""
    return _rounded_pixmap(_load_source_pixmap(), size)
