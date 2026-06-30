from __future__ import annotations

from PyQt6.QtCore import QObject, QTimer, pyqtSignal

from app.core.audio.device_manager import ActiveAudioDevices, get_active_audio_devices


class AudioDeviceMonitor(QObject):
    """Polls for default audio endpoint changes and emits when they differ."""

    devices_changed = pyqtSignal(object)  # ActiveAudioDevices

    def __init__(self, parent: QObject | None = None, *, poll_interval_ms: int = 1500) -> None:
        super().__init__(parent)
        self._poll_interval_ms = poll_interval_ms
        self._last_fingerprint: str | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(self._poll_interval_ms)
        self._timer.timeout.connect(self._poll)

    def start(self) -> None:
        snapshot = get_active_audio_devices()
        self._last_fingerprint = snapshot.fingerprint
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()

    def refresh_now(self) -> ActiveAudioDevices:
        """Force a snapshot read; emit only if changed unless force_emit."""
        snapshot = get_active_audio_devices()
        if snapshot.fingerprint != self._last_fingerprint:
            self._last_fingerprint = snapshot.fingerprint
            self.devices_changed.emit(snapshot)
        return snapshot

    def _poll(self) -> None:
        snapshot = get_active_audio_devices()
        if snapshot.fingerprint == self._last_fingerprint:
            return
        self._last_fingerprint = snapshot.fingerprint
        self.devices_changed.emit(snapshot)
