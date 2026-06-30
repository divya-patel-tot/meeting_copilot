"""Default Windows playback/capture device metadata via pycaw."""

from __future__ import annotations

import sys

# Windows EndpointFormFactor values (mmdeviceapi.h)
_FORM_FACTOR_NAMES: dict[int, str] = {
    0: "RemoteNetworkDevice",
    1: "Speakers",
    2: "LineLevel",
    3: "Headphones",
    4: "Microphone",
    5: "Headset",
    6: "Handset",
    7: "UnknownDigitalPassthrough",
    8: "SPDIF",
    9: "HDMI",
    10: "Unknown",
}

_DEVPKEY_FORM_FACTOR = "{B3F8FA53-0004-438E-9003-51A46E139BFC} 0"


def _unknown_endpoint() -> dict[str, str | bool]:
    return {"name": "Unknown", "form_factor": "Unknown", "is_default": True}


def _form_factor_from_properties(properties: dict) -> str:
    raw = properties.get(_DEVPKEY_FORM_FACTOR)
    if raw is None:
        return "Unknown"
    try:
        return _FORM_FACTOR_NAMES.get(int(raw), "Unknown")
    except (TypeError, ValueError):
        return "Unknown"


def format_output_display(name: str, form_factor: str) -> str:
    """Build a friendly label like 'Headphones (USB Audio)'."""
    if not name or name == "Unknown":
        return form_factor if form_factor != "Unknown" else "Unknown output"
    if form_factor != "Unknown" and form_factor.lower() in name.lower():
        return name
    if "(" in name and name.endswith(")"):
        return f"{form_factor} {name}" if form_factor != "Unknown" else name
    return f"{form_factor} ({name})" if form_factor != "Unknown" else name


def _pycaw_default_endpoint(getter_name: str) -> dict[str, str | bool]:
    if sys.platform != "win32":
        return _unknown_endpoint()

    try:
        from pycaw.pycaw import AudioUtilities

        getter = getattr(AudioUtilities, getter_name, None)
        if getter is None:
            return _unknown_endpoint()
        device = getter()
        if device is None:
            return _unknown_endpoint()

        name = device.FriendlyName or "Unknown"
        form_factor = _form_factor_from_properties(device.properties)
        return {
            "name": name,
            "form_factor": form_factor,
            "is_default": True,
        }
    except Exception:
        return _unknown_endpoint()


def get_default_output_device_info() -> dict[str, str | bool]:
    """Return default playback device metadata from Windows."""
    return _pycaw_default_endpoint("GetSpeakers")


def get_default_input_device_info() -> dict[str, str | bool]:
    """Return default recording device metadata from Windows."""
    return _pycaw_default_endpoint("GetMicrophone")
