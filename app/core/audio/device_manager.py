from __future__ import annotations

import sys
from dataclasses import dataclass

import sounddevice as sd

from app.core.audio.output_detector import (
    format_output_display,
    get_default_input_device_info,
    get_default_output_device_info,
)

_HOST_API_PREFERENCE = (
    "Windows WASAPI",
    "Windows WDM-KS",
    "Windows DirectSound",
    "MME",
)


@dataclass(frozen=True)
class ActiveAudioDevices:
    """Resolved active capture endpoints for the current system defaults."""

    mic_index: int | None
    output_index: int | None
    loopback_index: int | None
    mic_name: str
    output_label: str
    has_mic: bool
    has_output: bool
    has_loopback: bool

    @property
    def fingerprint(self) -> str:
        return (
            f"mic={self.mic_index}|out={self.output_index}|"
            f"loop={self.loopback_index}|m={self.mic_name}|o={self.output_label}"
        )


def _valid_device_index(index: int | None) -> int | None:
    """PortAudio uses -1 when no default device exists; treat that as missing."""
    if index is None:
        return None
    try:
        resolved = int(index)
    except (TypeError, ValueError):
        return None
    return resolved if resolved >= 0 else None


def _safe_query_device(index: int) -> dict | None:
    try:
        return sd.query_devices(index)
    except sd.PortAudioError:
        return None


def _host_api_name(hostapi_index: int) -> str:
    try:
        hostapis = sd.query_hostapis()
        if 0 <= hostapi_index < len(hostapis):
            return str(hostapis[hostapi_index]["name"])
    except sd.PortAudioError:
        pass
    return ""


def _is_sound_mapper(name: str) -> bool:
    lowered = name.casefold()
    return "sound mapper" in lowered or lowered.startswith("mapper -")


def _normalize_device_name(name: str) -> str:
    text = _friendly_portaudio_name(name)
    for token in (" [Loopback]", " (loopback)", " - Input", " - Output"):
        text = text.replace(token, "")
    return text.strip().casefold()


def _names_match_fuzzy(left: str, right: str) -> bool:
    a = _normalize_device_name(left)
    b = _normalize_device_name(right)
    if not a or not b:
        return False
    return a == b or a in b or b in a


def _device_is_input(index: int) -> bool:
    device = _safe_query_device(index)
    return device is not None and int(device.get("max_input_channels", 0)) > 0


def _device_is_output(index: int) -> bool:
    device = _safe_query_device(index)
    return device is not None and int(device.get("max_output_channels", 0)) > 0


def _find_portaudio_index_by_name(
    friendly_name: str,
    *,
    want_input: bool,
) -> int | None:
    if not friendly_name or friendly_name == "Unknown":
        return None
    try:
        devices = sd.query_devices()
    except sd.PortAudioError:
        return None

    channel_key = "max_input_channels" if want_input else "max_output_channels"
    matches: list[tuple[int, dict]] = []
    for index, device in enumerate(devices):
        if int(device.get(channel_key, 0)) <= 0:
            continue
        if _names_match_fuzzy(str(device["name"]), friendly_name):
            matches.append((index, device))

    if not matches:
        return None
    if len(matches) == 1:
        return matches[0][0]

    ranked = sorted(
        matches,
        key=lambda item: _device_preference_score(item[1], item[0], None, want_input),
        reverse=True,
    )
    return ranked[0][0]


def _device_preference_score(
    device: dict,
    index: int,
    default_index: int | None,
    want_input: bool,
) -> int:
    score = 0
    name = str(device.get("name", ""))
    host = _host_api_name(int(device.get("hostapi", -1)))

    if default_index is not None and index == default_index:
        score += 120
    if host in _HOST_API_PREFERENCE:
        score += 60 - _HOST_API_PREFERENCE.index(host) * 10
    if not _is_sound_mapper(name):
        score += 40

    channel_key = "max_input_channels" if want_input else "max_output_channels"
    score += min(int(device.get(channel_key, 0)), 4) * 2

    lowered = name.casefold()
    if any(token in lowered for token in ("microphone array", "internal", "built-in", "realtek")):
        score += 8
    if _is_sound_mapper(name):
        score -= 50
    return score


def get_default_input_index() -> int | None:
    """Return PortAudio index of the system default input device, if set."""
    try:
        default = sd.default.device
        return _valid_device_index(default[0])
    except sd.PortAudioError:
        return None


def get_default_output_index() -> int | None:
    """Return PortAudio index of the system default output device, if set."""
    try:
        default = sd.default.device
        return _valid_device_index(default[1])
    except sd.PortAudioError:
        return None


def _enumerate_input_devices(*, mark_default_index: int | None = None) -> list[dict]:
    """Return all audio input devices (no resolve_best_* calls — avoids recursion)."""
    try:
        devices = sd.query_devices()
    except sd.PortAudioError:
        return []
    if mark_default_index is None:
        mark_default_index = get_default_input_index()
    result = []
    for index, device in enumerate(devices):
        max_input = device.get("max_input_channels", 0)
        if max_input > 0:
            result.append(
                {
                    "index": index,
                    "name": device["name"],
                    "max_input_channels": max_input,
                    "default_samplerate": device["default_samplerate"],
                    "is_default": index == mark_default_index,
                }
            )
    return result


def resolve_best_input_index() -> int | None:
    """Pick the best available microphone: default → Windows default → ranked fallback."""
    default_index = get_default_input_index()
    if default_index is not None and _device_is_input(default_index):
        return default_index

    windows_default = get_default_input_device_info()
    matched = _find_portaudio_index_by_name(
        str(windows_default["name"]),
        want_input=True,
    )
    if matched is not None:
        return matched

    devices = _enumerate_input_devices()
    if not devices:
        return None

    ranked = sorted(
        devices,
        key=lambda d: _device_preference_score(
            _safe_query_device(d["index"]) or {},
            d["index"],
            default_index,
            True,
        ),
        reverse=True,
    )
    return ranked[0]["index"]


def resolve_best_output_index() -> int | None:
    """Pick the best available playback device for loopback pairing."""
    default_index = get_default_output_index()
    if default_index is not None and _device_is_output(default_index):
        return default_index

    windows_default = get_default_output_device_info()
    matched = _find_portaudio_index_by_name(
        str(windows_default["name"]),
        want_input=False,
    )
    if matched is not None:
        return matched

    devices = list_output_devices()
    if not devices:
        return None

    ranked = sorted(
        devices,
        key=lambda d: _device_preference_score(
            _safe_query_device(d["index"]) or {},
            d["index"],
            default_index,
            False,
        ),
        reverse=True,
    )
    return ranked[0]["index"]


def get_active_audio_devices() -> ActiveAudioDevices:
    """Resolve current mic, output, and loopback indices with fallback."""
    mic_index = resolve_best_input_index()
    output_index = resolve_best_output_index()

    if mic_index is not None:
        device = _safe_query_device(mic_index)
        mic_name = (
            _friendly_portaudio_name(device["name"])
            if device is not None
            else f"device {mic_index}"
        )
    else:
        mic_name = "No microphone detected"

    output_info = get_default_output_device_info()
    output_label = format_output_display(
        str(output_info["name"]),
        str(output_info["form_factor"]),
    )

    loopback_index: int | None = None
    has_loopback = False
    if output_index is not None:
        try:
            loopback_index = resolve_loopback_capture_index(output_index)
            has_loopback = True
        except RuntimeError:
            loopback_index = None

    return ActiveAudioDevices(
        mic_index=mic_index,
        output_index=output_index,
        loopback_index=loopback_index,
        mic_name=mic_name,
        output_label=output_label,
        has_mic=mic_index is not None,
        has_output=output_index is not None,
        has_loopback=has_loopback,
    )


def list_output_devices() -> list[dict]:
    """Return all audio output devices."""
    try:
        devices = sd.query_devices()
    except sd.PortAudioError:
        return []
    default_index = get_default_output_index()
    result = []
    for index, device in enumerate(devices):
        max_output = device.get("max_output_channels", 0)
        if max_output > 0:
            result.append(
                {
                    "index": index,
                    "name": device["name"],
                    "max_output_channels": max_output,
                    "default_samplerate": device["default_samplerate"],
                    "is_default": index == default_index,
                }
            )
    return result


def format_output_device(device: dict) -> str:
    default_tag = " - system default" if device.get("is_default") else ""
    rate = int(device["default_samplerate"])
    return (
        f"[{device['index']}] {device['name']} "
        f"({device['max_output_channels']} ch, {rate} Hz){default_tag}"
    )


def get_default_mic_friendly_name() -> str:
    """Short friendly name for the active microphone."""
    active = get_active_audio_devices()
    return active.mic_name


def get_default_output_friendly_label() -> str:
    """Friendly label for the active playback device."""
    return get_active_audio_devices().output_label


def _friendly_portaudio_name(name: str) -> str:
    """Strip redundant host-API prefixes from PortAudio device names."""
    for prefix in ("Windows WASAPI, ", "Windows DirectSound, ", "MME, "):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _loopback_base_name(name: str) -> str:
    return name.replace(" [Loopback]", "").strip()


def _loopback_names_match(output_name: str, loopback_name: str) -> bool:
    out = _friendly_portaudio_name(output_name).casefold()
    loop = _loopback_base_name(loopback_name).casefold()
    return out == loop or out in loop or loop in out


def _default_loopback_index(pa) -> int | None:  # noqa: ANN001
    """Best-effort default loopback device from pyaudiowpatch."""
    fallback: int | None = None
    for index in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(index)
        if not info.get("isLoopbackDevice"):
            continue
        if fallback is None:
            fallback = index
        if info.get("isDefaultDevice"):
            return index
    return fallback


def resolve_loopback_capture_index(output_index: int | None = None) -> int:
    """Find pyaudiowpatch loopback index for a playback device, with fallbacks."""
    if sys.platform != "win32":
        raise RuntimeError("WASAPI loopback capture is only supported on Windows")

    import pyaudiowpatch as pyaudio

    if output_index is None:
        output_index = resolve_best_output_index()
    if output_index is None:
        raise RuntimeError("No playback device found for loopback capture")

    device = _safe_query_device(output_index)
    if device is None:
        raise RuntimeError(f"Could not query output device index {output_index}")
    output_name = device["name"]

    pa = pyaudio.PyAudio()
    try:
        matches: list[tuple[int, dict]] = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if not info.get("isLoopbackDevice"):
                continue
            if _loopback_names_match(output_name, info["name"]):
                matches.append((index, info))

        if matches:
            if len(matches) > 1:
                for index, info in matches:
                    if info.get("isDefaultDevice"):
                        return index
            return matches[0][0]

        default_loopback = _default_loopback_index(pa)
        if default_loopback is not None:
            return default_loopback

        available = [
            pa.get_device_info_by_index(i)["name"]
            for i in range(pa.get_device_count())
            if pa.get_device_info_by_index(i).get("isLoopbackDevice")
        ]
        raise RuntimeError(
            f"No loopback capture device found for output {output_name!r}. "
            f"Available loopback devices: {available or '(none)'}"
        )
    finally:
        pa.terminate()


def list_advanced_capture_devices() -> list[dict]:
    """All selectable capture endpoints for the Advanced device picker."""
    devices: list[dict] = []
    for device in list_input_devices():
        devices.append(
            {
                "label": f"Mic: {format_input_device(device)}",
                "index": device["index"],
                "source_mode": "microphone",
                "backend": "sounddevice",
            }
        )

    if sys.platform == "win32":
        try:
            import pyaudiowpatch as pyaudio

            pa = pyaudio.PyAudio()
            try:
                for index in range(pa.get_device_count()):
                    info = pa.get_device_info_by_index(index)
                    if not info.get("isLoopbackDevice"):
                        continue
                    devices.append(
                        {
                            "label": f"Loopback: [{index}] {info['name']}",
                            "index": index,
                            "source_mode": "loopback",
                            "backend": "pyaudiowpatch",
                        }
                    )
            finally:
                pa.terminate()
        except Exception:
            pass

    return devices


def is_valid_input_index(device_index: int) -> bool:
    return any(d["index"] == device_index for d in list_input_devices())


def is_valid_loopback_index(device_index: int) -> bool:
    if sys.platform != "win32":
        return False
    try:
        import pyaudiowpatch as pyaudio

        pa = pyaudio.PyAudio()
        try:
            info = pa.get_device_info_by_index(device_index)
            return bool(info.get("isLoopbackDevice"))
        finally:
            pa.terminate()
    except Exception:
        return False


def list_input_devices() -> list[dict]:
    """Return all audio input devices with index, name, channels, and sample rate."""
    best_index = resolve_best_input_index()
    devices = _enumerate_input_devices()
    if best_index is None:
        return devices
    for device in devices:
        device["is_default"] = device["index"] == best_index
    return devices


def format_input_device(device: dict) -> str:
    """Human-readable one-line description of an input device."""
    default_tag = " - system default" if device.get("is_default") else ""
    rate = int(device["default_samplerate"])
    return (
        f"[{device['index']}] {device['name']} "
        f"({device['max_input_channels']} ch, {rate} Hz){default_tag}"
    )


def print_input_devices() -> list[dict]:
    """Print all input devices and return the list."""
    devices = list_input_devices()
    if not devices:
        print("No input devices found.")
        return devices

    print("Available input sources:")
    for menu_num, device in enumerate(devices, start=1):
        print(f"  {menu_num}) {format_input_device(device)}")
    return devices


def resolve_input_device(device_index: int | None = None) -> int:
    """Return a PortAudio input device index (interactive menu if not provided)."""
    devices = list_input_devices()
    if not devices:
        raise SystemExit("No input devices found. Connect a mic or headset and try again.")

    if device_index is not None:
        known = {d["index"] for d in devices}
        if device_index not in known:
            print_input_devices()
            raise SystemExit(
                f"Device index {device_index} is not a valid input. "
                "Pick an index from the list above."
            )
        return device_index

    print_input_devices()
    print()
    default_menu = next(
        (i for i, d in enumerate(devices, start=1) if d.get("is_default")),
        1,
    )
    while True:
        raw = input(
            f"Select input source (1-{len(devices)}, Enter={default_menu} for default): "
        ).strip()
        if raw == "":
            choice = default_menu
        elif not raw.isdigit():
            print("Enter a number from the list, or press Enter for the default.")
            continue
        else:
            choice = int(raw)
        if 1 <= choice <= len(devices):
            selected = devices[choice - 1]
            if "Sound Mapper" in selected["name"]:
                print(
                    "Note: Sound Mapper is a Windows router and often unreliable. "
                    "Prefer your USB/headset mic if audio seems silent."
                )
            print(f"Using: {format_input_device(selected)}")
            return selected["index"]
        print(f"Enter a number between 1 and {len(devices)}.")


def get_input_device_name(device_index: int) -> str:
    """Return the display name for an input device index."""
    for device in list_input_devices():
        if device["index"] == device_index:
            return device["name"]
    return f"device {device_index}"
