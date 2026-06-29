import sounddevice as sd


def get_default_input_index() -> int | None:
    """Return PortAudio index of the system default input device, if set."""
    default = sd.default.device
    if default[0] is not None:
        return int(default[0])
    return None


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
