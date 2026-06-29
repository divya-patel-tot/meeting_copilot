from PyQt6.QtCore import QSettings

ORGANIZATION = "MeetingResponder"
APPLICATION = "MeetingCopilot"

SOURCE_LOOPBACK = "loopback"
SOURCE_MICROPHONE = "microphone"
SOURCE_ADVANCED = "advanced"


def _settings() -> QSettings:
    return QSettings(ORGANIZATION, APPLICATION)


def get_audio_device_index() -> int | None:
    value = _settings().value("audio/device_index")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def set_audio_device_index(device_index: int) -> None:
    _settings().setValue("audio/device_index", int(device_index))


def get_audio_source_mode() -> str:
    value = _settings().value("audio/source_mode", SOURCE_LOOPBACK)
    if value in {SOURCE_LOOPBACK, SOURCE_MICROPHONE, SOURCE_ADVANCED}:
        return str(value)
    return SOURCE_LOOPBACK


def set_audio_source_mode(source_mode: str) -> None:
    _settings().setValue("audio/source_mode", source_mode)


def get_advanced_capture_backend() -> str:
    return str(_settings().value("audio/advanced_backend", "sounddevice"))


def set_advanced_capture_backend(backend: str) -> None:
    _settings().setValue("audio/advanced_backend", backend)


def get_mic_only_testing() -> bool:
    return _settings().value("audio/mic_only_testing", False, type=bool)


def set_mic_only_testing(enabled: bool) -> None:
    _settings().setValue("audio/mic_only_testing", bool(enabled))
