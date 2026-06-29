import sys

from app.utils.std_streams import ensure_std_streams

ensure_std_streams()

from PyQt6.QtWidgets import QApplication, QDialog, QSystemTrayIcon

from app.core.pipeline_controller import PipelineController
from app.ui.main_window import MainWindow
from app.ui.setup_wizard import SetupWizard
from app.ui.system_tray import SystemTray
from app.utils.config import reload_settings
from app.utils.paths import resource_path
from app.utils.setup import is_setup_complete


def _load_stylesheet() -> str:
    qss_path = resource_path("app", "ui", "styles", "dark_theme.qss")
    if qss_path.exists():
        return qss_path.read_text(encoding="utf-8")
    return ""


def main() -> int:
    if "--smoke-test" in sys.argv:
        from app.utils.frozen_smoke_test import run_smoke_test

        return run_smoke_test()
    app = QApplication(sys.argv)
    app.setApplicationName("Meeting Responder")

    stylesheet = _load_stylesheet()
    if stylesheet:
        app.setStyleSheet(stylesheet)

    if not is_setup_complete():
        wizard = SetupWizard()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            return 0
        reload_settings()

    controller = PipelineController()
    window = MainWindow(controller)

    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = SystemTray(window)
        window.set_system_tray(tray)
        tray.show()
    else:
        tray = None

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
