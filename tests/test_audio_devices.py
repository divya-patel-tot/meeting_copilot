import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from app.core.audio.device_manager import list_input_devices

        devices = list_input_devices()
        lines = [f"Found {len(devices)} input device(s):"]
        for device in devices:
            lines.append(
                f"  [{device['index']}] {device['name']} "
                f"({device['max_input_channels']} ch, {device['default_samplerate']} Hz)"
            )
        if not devices:
            lines.append(
                "  (No input devices — expected if no mic/virtual cable installed yet.)"
            )
        else:
            lines.append(
                "  Note: CABLE Output / BlackHole may appear here once a virtual audio driver is installed."
            )
        return True, "\n".join(lines)
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    status = "PASS" if success else "FAIL"
    print(f"{status}: {message}")
