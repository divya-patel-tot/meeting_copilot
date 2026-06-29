from __future__ import annotations

from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from app.ui.main_window import MainWindow


class SystemTray(QSystemTrayIcon):
    """System tray icon with shortcuts mirroring MainWindow actions."""

    def __init__(self, main_window: MainWindow) -> None:
        super().__init__(main_window)
        self._window = main_window

        icon = QApplication.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        self.setIcon(icon)
        self.setToolTip("Meeting Responder")

        self._menu = QMenu()
        self._show_hide_action = QAction("Hide Window", self)
        self._listen_action = QAction("Start Listening", self)
        self._settings_action = QAction("Settings...", self)
        self._quit_action = QAction("Quit", self)

        self._show_hide_action.triggered.connect(self._window.toggle_window_visibility)
        self._listen_action.triggered.connect(self._window.toggle_listening)
        self._settings_action.triggered.connect(self._window.open_settings)
        self._quit_action.triggered.connect(self._window.quit_application)

        self._menu.addAction(self._show_hide_action)
        self._menu.addAction(self._listen_action)
        self._menu.addSeparator()
        self._menu.addAction(self._settings_action)
        self._menu.addSeparator()
        self._menu.addAction(self._quit_action)
        self.setContextMenu(self._menu)

        self.activated.connect(self._on_activated)
        main_window.listening_state_changed.connect(self._update_listen_action)
        main_window.window_visibility_changed.connect(self._update_show_hide_action)

        self._update_listen_action(main_window.is_listening_active())
        self._update_show_hide_action(main_window.isVisible())

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self._window.toggle_window_visibility()

    def _update_listen_action(self, listening: bool) -> None:
        self._listen_action.setText("Stop Listening" if listening else "Start Listening")

    def _update_show_hide_action(self, visible: bool) -> None:
        self._show_hide_action.setText("Hide Window" if visible else "Show Window")
