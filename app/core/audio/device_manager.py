import sys

import sounddevice as sd

from app.core.audio.output_detector import format_output_display, get_default_output_device_info


def get_default_input_index() -> int | None:
    """Return PortAudio index of the system default input device, if set."""
    default = sd.default.device
    if default[0] is not None:
        return int(default[0])
    return None


def get_default_output_index() -> int | None:
    """Return PortAudio index of the system default output device, if set."""
    default = sd.default.device
    if default[1] is not None:
        return int(default[1])
    return None


def list_output_devices() -> list[dict]:
    """Return all audio output devices."""
    devices = sd.query_devices()
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
    """Short friendly name for the default microphone."""
    default_index = get_default_input_index()
    if default_index is None:
        return "Unknown microphone"
    return _friendly_portaudio_name(sd.query_devices(default_index)["name"])


def get_default_output_friendly_label() -> str:
    """Friendly label for the default playback device (name + form factor)."""
    info = get_default_output_device_info()
    return format_output_display(str(info["name"]), str(info["form_factor"]))


def _friendly_portaudio_name(name: str) -> str:
    """Strip redundant host-API prefixes from PortAudio device names."""
    for prefix in ("Windows WASAPI, ", "Windows DirectSound, ", "MME, "):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _loopback_base_name(name: str) -> str:
    return name.replace(" [Loopback]", "").strip()


def _names_match(output_name: str, loopback_name: str) -> bool:
    out = _friendly_portaudio_name(output_name).casefold()
    loop = _loopback_base_name(loopback_name).casefold()
    return out == loop or out in loop or loop in out


def resolve_loopback_capture_index(output_index: int | None = None) -> int:
    """Find the pyaudiowpatch loopback input index for a playback device.

    sounddevice 0.4.6–0.5.5 does not expose WasapiSettings(loopback=True) and
    the bundled PortAudio binary has no loopback devices, so Windows loopback
    capture uses pyaudiowpatch while microphone capture stays on sounddevice.
    """
    if sys.platform != "win32":
        raise RuntimeError("WASAPI loopback capture is only supported on Windows")

    import pyaudiowpatch as pyaudio

    if output_index is None:
        output_index = get_default_output_index()
    if output_index is None:
        raise RuntimeError("No default output device found")

    output_name = sd.query_devices(output_index)["name"]
    pa = pyaudio.PyAudio()
    try:
        matches: list[tuple[int, dict]] = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if not info.get("isLoopbackDevice"):
                continue
            if _names_match(output_name, info["name"]):
                matches.append((index, info))

        if not matches:
            available = [
                pa.get_device_info_by_index(i)["name"]
                for i in range(pa.get_device_count())
                if pa.get_device_info_by_index(i).get("isLoopbackDevice")
            ]
            raise RuntimeError(
                f"No loopback capture device found for output {output_name!r}. "
                f"Available loopback devices: {available or '(none)'}"
            )

        if len(matches) > 1:
            for index, info in matches:
                if info.get("isDefaultDevice"):
                    return index
        return matches[0][0]
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


def list_input_devices() -> list[dict]:
    """Return all audio input devices with index, name, channels, and sample rate."""
    devices = sd.query_devices()
    default_index = get_default_input_index()
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
                    "is_default": index == default_index,
                }
            )
    return result


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
