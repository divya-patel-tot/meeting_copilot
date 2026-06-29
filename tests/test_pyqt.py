import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run() -> tuple[bool, str]:
    try:
        from PyQt6.QtWidgets import QApplication

        from app.ui.main_window import MainWindow

        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        if window.windowTitle() != "Meeting Responder":
            return False, f"Unexpected window title: {window.windowTitle()!r}"
        app.quit()
        return True, "MainWindow created successfully"
    except Exception as exc:
        return False, str(exc)


if __name__ == "__main__":
    success, message = run()
    status = "PASS" if success else "FAIL"
    print(f"{status}: {message}")
