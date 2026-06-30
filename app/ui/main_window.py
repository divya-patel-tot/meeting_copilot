from __future__ import annotations

import html
from dataclasses import dataclass, field
from datetime import datetime

from PyQt6.QtCore import QEvent, Qt, QPoint, QUrl, QRect, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QShowEvent, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.audio.device_manager import (
    ActiveAudioDevices,
    get_active_audio_devices,
    get_default_mic_friendly_name,
    get_default_output_friendly_label,
    is_valid_input_index,
    is_valid_loopback_index,
    list_advanced_capture_devices,
)
from app.core.audio.device_monitor import AudioDeviceMonitor
from app.core.pipeline_controller import PipelineController
from app.core.stt.transcript_buffer import SPEAKER_YOU, TranscriptEntry
from app.ui.branding import APP_NAME
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.icon_loader import app_icon
from app.ui.widgets.custom_title_bar import CustomTitleBar
from app.ui.widgets.glass_panel import GlassPanel
from app.ui.widgets.status_pill import StatusPill
from app.ui.window_utils import RESIZE_MARGIN
from app.utils.app_settings import (
    SOURCE_LOOPBACK,
    SOURCE_MICROPHONE,
    get_audio_device_index,
    get_mic_only_testing,
    set_advanced_capture_backend,
    set_audio_device_index,
    set_mic_only_testing,
)


@dataclass
class _SuggestionRecord:
    segment_index: int
    transcript_text: str
    body: str = ""
    stats: dict | None = None
    error: str | None = None
    streaming: bool = True
    started_at: datetime | None = None
    retrieval_matches: list[dict] = field(default_factory=list)


class MainWindow(QMainWindow):
    listening_state_changed = pyqtSignal(bool)
    window_visibility_changed = pyqtSignal(bool)

    def __init__(self, controller: PipelineController) -> None:
        super().__init__()
        self._controller = controller
        self._transcript_entries: list[TranscriptEntry] = []
        self._system_transcript_lines: list[str] = []
        self._suggestions: dict[int, _SuggestionRecord] = {}
        self._selected_segment: int | None = None
        self._follow_live_suggestions = True
        self._tray = None
        self._resize_edge: str | None = None
        self._resize_origin: QPoint | None = None
        self._resize_start_geom: QRect | None = None
        self._auto_resume_listening = False
        self._reloading_audio = False
        self._device_monitor = AudioDeviceMonitor(self)

        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.setMinimumSize(680, 460)
        self.resize(880, 580)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._build_ui()
        self._connect_signals()
        self._populate_devices()
        self._update_start_stop_ui(listening=False)
        self._device_monitor.devices_changed.connect(self._on_audio_devices_changed)
        self._device_monitor.start()

        self.setMouseTracking(True)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def set_system_tray(self, tray) -> None:  # noqa: ANN001
        self._tray = tray

    def is_listening_active(self) -> bool:
        return self._controller.is_listening

    def toggle_window_visibility(self) -> None:
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()

    def toggle_listening(self) -> None:
        self._on_start_stop_clicked()

    def open_settings(self) -> None:
        dialog = SettingsDialog(self._controller, self)
        dialog.exec()
        self._apply_saved_device()

    def quit_application(self) -> None:
        if self._controller.is_listening:
            self._controller.stop_listening(wait=True, timeout=2.0)
        QApplication.instance().quit()

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.window_visibility_changed.emit(True)

    def hideEvent(self, event) -> None:  # noqa: ANN001
        super().hideEvent(event)
        self.window_visibility_changed.emit(False)

    def changeEvent(self, event) -> None:  # noqa: ANN001
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            maximized = self.isMaximized()
            self._title_bar._maximize_btn.setText("❐" if maximized else "□")

    def eventFilter(self, obj, event):  # noqa: ANN001
        if not self.isVisible() or self.isMinimized():
            return super().eventFilter(obj, event)

        event_type = event.type()
        if event_type in {
            QEvent.Type.MouseMove,
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
        }:
            if hasattr(event, "globalPosition"):
                global_pos = event.globalPosition().toPoint()
            else:
                return super().eventFilter(obj, event)

            if not self.frameGeometry().contains(global_pos):
                if event_type == QEvent.Type.MouseMove and self._resize_edge is None:
                    self.unsetCursor()
                return super().eventFilter(obj, event)

            if event_type == QEvent.Type.MouseButtonPress:
                if (
                    event.button() == Qt.MouseButton.LeftButton
                    and not self.isMaximized()
                ):
                    edge = self._edge_at(global_pos)
                    if edge is not None:
                        self._resize_edge = edge
                        self._resize_origin = global_pos
                        self._resize_start_geom = self.geometry()
                        return True
            elif event_type == QEvent.Type.MouseMove:
                if self._resize_edge is not None and (
                    event.buttons() & Qt.MouseButton.LeftButton
                ):
                    self._apply_resize(global_pos)
                    return True
                edge = self._edge_at(global_pos)
                self.setCursor(
                    self._cursor_for_edge(edge)
                    if edge
                    else Qt.CursorShape.ArrowCursor
                )
            elif event_type == QEvent.Type.MouseButtonRelease:
                if self._resize_edge is not None:
                    self._resize_edge = None
                    self._resize_origin = None
                    self._resize_start_geom = None

        return super().eventFilter(obj, event)

    def _edge_at(self, global_pos: QPoint) -> str | None:
        if self.isMaximized():
            return None

        local = self.mapFromGlobal(global_pos)
        rect = self.rect()
        margin = RESIZE_MARGIN

        left = local.x() <= margin
        right = local.x() >= rect.width() - margin
        top = local.y() <= margin
        bottom = local.y() >= rect.height() - margin

        if not any((left, right, top, bottom)):
            return None

        edge = ""
        if top:
            edge += "t"
        if bottom:
            edge += "b"
        if left:
            edge += "l"
        if right:
            edge += "r"
        return edge or None

    @staticmethod
    def _cursor_for_edge(edge: str | None) -> Qt.CursorShape:
        if edge in {"tl", "br"}:
            return Qt.CursorShape.SizeFDiagCursor
        if edge in {"tr", "bl"}:
            return Qt.CursorShape.SizeBDiagCursor
        if edge in {"l", "r"}:
            return Qt.CursorShape.SizeHorCursor
        if edge in {"t", "b"}:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def _apply_resize(self, global_pos: QPoint) -> None:
        if (
            self._resize_edge is None
            or self._resize_origin is None
            or self._resize_start_geom is None
        ):
            return

        delta = global_pos - self._resize_origin
        geom = QRect(self._resize_start_geom)
        edge = self._resize_edge
        min_w = self.minimumWidth()
        min_h = self.minimumHeight()

        if "r" in edge:
            geom.setWidth(max(min_w, geom.width() + delta.x()))
        if "b" in edge:
            geom.setHeight(max(min_h, geom.height() + delta.y()))
        if "l" in edge:
            new_width = max(min_w, geom.width() - delta.x())
            geom.setLeft(geom.left() + (geom.width() - new_width))
            geom.setWidth(new_width)
        if "t" in edge:
            new_height = max(min_h, geom.height() - delta.y())
            geom.setTop(geom.top() + (geom.height() - new_height))
            geom.setHeight(new_height)

        self.setGeometry(geom)

    def _build_ui(self) -> None:
        self._card = QWidget()
        self._card.setObjectName("glassCard")
        self._card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setCentralWidget(self._card)

        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(0, 0, 0, 0)
        card_layout.setSpacing(0)

        self._title_bar = CustomTitleBar(self._card)
        card_layout.addWidget(self._title_bar)

        body = QWidget()
        body.setObjectName("glassCardBody")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(12, 8, 12, 12)
        body_layout.setSpacing(8)

        top_controls = QHBoxLayout()
        top_controls.setSpacing(10)

        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("secondaryButton")
        settings_btn.clicked.connect(self.open_settings)
        top_controls.addWidget(settings_btn)

        top_controls.addStretch()

        self._start_stop_btn = QPushButton("Start")
        self._start_stop_btn.setObjectName("startStopButton")
        self._start_stop_btn.setMinimumWidth(96)
        top_controls.addWidget(self._start_stop_btn)

        self._status_pill = StatusPill()
        top_controls.addWidget(self._status_pill)

        body_layout.addLayout(top_controls)

        source_row = QHBoxLayout()
        source_row.setSpacing(8)

        device_label = QLabel("Capture")
        device_label.setObjectName("controlLabel")
        device_label.setFixedWidth(88)
        source_row.addWidget(device_label, alignment=Qt.AlignmentFlag.AlignTop)

        source_column = QVBoxLayout()
        source_column.setSpacing(6)

        mic_name = get_default_mic_friendly_name()
        output_label = get_default_output_friendly_label()
        self._capture_info = QLabel(
            f"🎤 Your mic ({mic_name})  +  🔊 System audio ({output_label})"
        )
        self._capture_info.setObjectName("captureInfoLabel")
        self._capture_info.setWordWrap(True)
        source_column.addWidget(self._capture_info)

        self._mic_only_checkbox = QCheckBox("Mic only (testing mode)")
        self._mic_only_checkbox.setToolTip(
            "Capture only your microphone and treat all speech as [Them] "
            "for suggestions — same as earlier single-stream testing."
        )
        self._mic_only_checkbox.setChecked(get_mic_only_testing())
        self._mic_only_checkbox.toggled.connect(self._on_mic_only_toggled)
        source_column.addWidget(self._mic_only_checkbox)

        advanced_row = QHBoxLayout()
        self._advanced_toggle = QPushButton("Advanced…")
        self._advanced_toggle.setObjectName("linkButton")
        self._advanced_toggle.setCheckable(True)
        advanced_row.addWidget(self._advanced_toggle)
        advanced_row.addStretch()
        source_column.addLayout(advanced_row)

        self._advanced_device_combo = QComboBox()
        self._advanced_device_combo.hide()
        source_column.addWidget(self._advanced_device_combo)

        source_row.addLayout(source_column, stretch=1)
        body_layout.addLayout(source_row)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.setObjectName("mainSplitter")
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(8)

        self._transcript_panel = GlassPanel("Live Transcript", variant="transcript")
        self._transcript_view = QTextBrowser()
        self._transcript_view.setObjectName("transcriptPanel")
        self._transcript_view.setOpenExternalLinks(False)
        self._transcript_view.setReadOnly(True)
        self._transcript_view.setPlaceholderText(
            "Live transcript will build up here as speech is detected… "
            "Click any line to view its suggestion."
        )
        self._transcript_view.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self._transcript_view.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._transcript_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._transcript_panel.content_layout.addWidget(self._transcript_view)

        self._suggestion_panel = GlassPanel("AI Response", variant="suggestion")
        self._suggestion_view = QTextBrowser()
        self._suggestion_view.setObjectName("suggestionPanel")
        self._suggestion_view.setOpenExternalLinks(False)
        self._suggestion_view.setReadOnly(True)
        self._suggestion_view.setPlaceholderText(
            "AI suggestions with source context will appear here…"
        )
        self._suggestion_view.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self._suggestion_panel.content_layout.addWidget(self._suggestion_view)

        self._splitter.addWidget(self._transcript_panel)
        self._splitter.addWidget(self._suggestion_panel)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([500, 500])

        body_layout.addWidget(self._splitter, stretch=1)

        debug_row = QHBoxLayout()
        self._debug_checkbox = QCheckBox("Show retrieval scores")
        debug_row.addWidget(self._debug_checkbox)
        debug_row.addStretch()
        body_layout.addLayout(debug_row)

        self._debug_view = QTextEdit()
        self._debug_view.setObjectName("debugPanel")
        self._debug_view.setReadOnly(True)
        self._debug_view.setMaximumHeight(110)
        self._debug_view.setVisible(False)
        body_layout.addWidget(self._debug_view)

        card_layout.addWidget(body, stretch=1)

        self._start_stop_btn.clicked.connect(self._on_start_stop_clicked)
        self._debug_checkbox.toggled.connect(self._on_debug_toggled)
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)
        self._advanced_device_combo.currentIndexChanged.connect(
            self._on_advanced_device_changed
        )

    def _connect_signals(self) -> None:
        self._controller.status_changed.connect(self._on_status_changed)
        self._controller.transcript_ready.connect(self._on_transcript_ready)
        self._controller.suggestion_started.connect(self._on_suggestion_started)
        self._controller.suggestion_retrieval.connect(self._on_suggestion_retrieval)
        self._controller.retrieval_debug.connect(self._on_retrieval_debug)
        self._controller.suggestion_token.connect(self._on_suggestion_token)
        self._controller.suggestion_complete.connect(self._on_suggestion_complete)
        self._controller.suggestion_error.connect(self._on_suggestion_error)
        self._controller.error_occurred.connect(self._on_error_occurred)
        self._transcript_view.anchorClicked.connect(self._on_transcript_anchor_clicked)

    def _populate_devices(self) -> None:
        self._advanced_device_combo.clear()

        advanced_devices = list_advanced_capture_devices()
        if not advanced_devices:
            self._advanced_toggle.setEnabled(False)
        else:
            for device in advanced_devices:
                self._advanced_device_combo.addItem(
                    device["label"],
                    device,
                )

            saved_index = get_audio_device_index()
            if saved_index is not None:
                for row in range(self._advanced_device_combo.count()):
                    data = self._advanced_device_combo.itemData(row)
                    if isinstance(data, dict) and data.get("index") == saved_index:
                        self._advanced_toggle.setChecked(True)
                        self._advanced_device_combo.setCurrentIndex(row)
                        self._advanced_device_combo.show()
                        break

        self._mic_only_checkbox.setChecked(get_mic_only_testing())
        self._refresh_capture_info()
        active = get_active_audio_devices()
        self._start_stop_btn.setEnabled(active.has_mic)

    def _validate_advanced_selection(self) -> None:
        """Drop Advanced override if the saved device was unplugged."""
        if not self._advanced_toggle.isChecked():
            return
        data = self._advanced_device_combo.currentData()
        if not isinstance(data, dict):
            self._advanced_toggle.setChecked(False)
            self._advanced_device_combo.hide()
            return
        index = int(data["index"])
        source_mode = str(data.get("source_mode"))
        valid = (
            source_mode == SOURCE_MICROPHONE and is_valid_input_index(index)
        ) or (
            source_mode == SOURCE_LOOPBACK and is_valid_loopback_index(index)
        )
        if not valid:
            self._advanced_toggle.setChecked(False)
            self._advanced_device_combo.hide()

    def _on_audio_devices_changed(self, snapshot: object) -> None:
        if not isinstance(snapshot, ActiveAudioDevices):
            return

        was_listening = self._controller.is_listening
        if was_listening:
            self._auto_resume_listening = True

        self._validate_advanced_selection()
        self._populate_devices()
        self._refresh_capture_info()

        if not snapshot.has_mic:
            if was_listening:
                self._auto_resume_listening = True
                self._controller.stop_listening()
                self._append_transcript_system(
                    "Microphone disconnected — capture paused. "
                    "Connect a mic to resume."
                )
            else:
                self._auto_resume_listening = False
            return

        if self._auto_resume_listening or was_listening:
            try:
                mic_index, loopback_index, mic_only = self._resolve_capture_devices()
            except RuntimeError as exc:
                self._controller.stop_listening()
                self._append_transcript_system(f"Audio device change: {exc}")
                self._auto_resume_listening = False
                return

            if was_listening or self._auto_resume_listening:
                self._append_transcript_system(
                    f"Audio devices updated — mic: {snapshot.mic_name}; "
                    f"output: {snapshot.output_label}"
                )
                self._reloading_audio = True
                try:
                    if self._controller.is_listening:
                        self._controller.reload_audio_devices(
                            mic_index,
                            loopback_index,
                            mic_only_testing=mic_only,
                        )
                    else:
                        self._controller.start_listening(
                            mic_index,
                            loopback_index,
                            mic_only_testing=mic_only,
                        )
                finally:
                    self._reloading_audio = False
            self._auto_resume_listening = False

    def _on_mic_only_toggled(self, checked: bool) -> None:
        set_mic_only_testing(checked)
        self._refresh_capture_info()

    def _refresh_capture_info(self) -> None:
        mic_name = get_default_mic_friendly_name()
        output_label = get_default_output_friendly_label()
        if self._mic_only_checkbox.isChecked():
            self._capture_info.setText(
                f"🎤 Mic only — testing mode ({mic_name}); all speech tagged [Them]"
            )
        else:
            self._capture_info.setText(
                f"🎤 Your mic ({mic_name})  +  🔊 System audio ({output_label})"
            )

    def _on_advanced_toggled(self, checked: bool) -> None:
        self._advanced_device_combo.setVisible(checked)

    def _on_advanced_device_changed(self) -> None:
        if not self._advanced_toggle.isChecked():
            return
        data = self._advanced_device_combo.currentData()
        if isinstance(data, dict):
            set_audio_device_index(int(data["index"]))
            set_advanced_capture_backend(str(data.get("backend", "sounddevice")))

    def _apply_saved_device(self) -> None:
        self._populate_devices()

    def _resolve_capture_devices(self) -> tuple[int, int | None, bool]:
        mic_only = self._mic_only_checkbox.isChecked()
        active = get_active_audio_devices()

        mic_index = active.mic_index
        if mic_index is None:
            raise RuntimeError(
                "No microphone found. Connect a headset or enable the built-in mic "
                "in Windows Sound settings."
            )

        loopback_index: int | None = active.loopback_index if not mic_only else None
        if not mic_only and loopback_index is None:
            raise RuntimeError(
                "No system-audio loopback device found. Connect speakers/headphones "
                "and set a default playback device in Windows Sound settings."
            )

        if self._advanced_toggle.isChecked():
            data = self._advanced_device_combo.currentData()
            if not isinstance(data, dict):
                raise RuntimeError("No advanced capture device selected.")
            mode = str(data.get("source_mode"))
            index = int(data["index"])
            if mode == SOURCE_MICROPHONE:
                if not is_valid_input_index(index):
                    raise RuntimeError("Selected microphone is no longer available.")
                mic_index = index
            elif mode == SOURCE_LOOPBACK:
                if not is_valid_loopback_index(index):
                    raise RuntimeError("Selected loopback device is no longer available.")
                loopback_index = index

        return mic_index, loopback_index, mic_only

    def _on_start_stop_clicked(self) -> None:
        if self._controller.is_listening:
            self._auto_resume_listening = False
            self._controller.stop_listening()
            return

        try:
            mic_index, loopback_index, mic_only = self._resolve_capture_devices()
        except RuntimeError as exc:
            self._append_transcript_system(str(exc))
            return

        set_mic_only_testing(mic_only)
        set_audio_device_index(int(mic_index))
        self._auto_resume_listening = True
        self._controller.start_listening(
            int(mic_index),
            loopback_index,
            mic_only_testing=mic_only,
        )

    def _on_debug_toggled(self, checked: bool) -> None:
        self._controller.set_retrieval_debug(checked)
        self._debug_view.setVisible(checked)
        if not checked:
            self._debug_view.clear()

    def _on_status_changed(self, status: str) -> None:
        if status == "Warming up...":
            self._clear_session_history()
        listening = status in {"Listening", "Warming up..."}
        self._status_pill.set_status(status, listening=listening)
        self._update_start_stop_ui(listening=listening)

    def _clear_session_history(self) -> None:
        self._transcript_entries.clear()
        self._system_transcript_lines.clear()
        self._suggestions.clear()
        self._selected_segment = None
        self._follow_live_suggestions = True
        self._transcript_view.clear()
        self._suggestion_view.clear()

    def _update_start_stop_ui(self, *, listening: bool) -> None:
        if self._reloading_audio:
            listening = True
        if listening:
            self._start_stop_btn.setText("Stop")
            self._start_stop_btn.setProperty("listening", True)
            self._mic_only_checkbox.setEnabled(False)
            self._advanced_toggle.setEnabled(False)
            self._advanced_device_combo.setEnabled(False)
        else:
            self._start_stop_btn.setText("Start")
            self._start_stop_btn.setProperty("listening", False)
            has_mic = get_active_audio_devices().has_mic
            self._mic_only_checkbox.setEnabled(has_mic)
            self._advanced_toggle.setEnabled(self._advanced_device_combo.count() > 0)
            self._advanced_device_combo.setEnabled(True)
        self._start_stop_btn.style().unpolish(self._start_stop_btn)
        self._start_stop_btn.style().polish(self._start_stop_btn)
        self.listening_state_changed.emit(listening)

    def _on_transcript_ready(self, entry: object) -> None:
        if not isinstance(entry, TranscriptEntry):
            return
        self._transcript_entries.append(entry)
        self._append_transcript_entry(entry)

    def _format_transcript_html(
        self,
        entry: TranscriptEntry,
        *,
        selected: bool = False,
    ) -> str:
        timestamp = html.escape(entry.timestamp.strftime("%H:%M:%S"))
        is_you = entry.speaker == SPEAKER_YOU
        speaker = "You" if is_you else "Them"
        align = "right" if is_you else "left"
        margin = "margin:10px 0 10px 56px;" if is_you else "margin:10px 56px 10px 0;"

        if entry.error:
            body = f"STT failed: {html.escape(entry.error)}"
        elif entry.is_valid:
            body = html.escape(entry.text)
        else:
            detail = html.escape(entry.text or "(empty)")
            body = f"<span style='color:#94a3b8;font-style:italic'>Discarded: {detail}</span>"

        if is_you:
            bubble_bg = "#eff6ff"
            bubble_border = "#bfdbfe"
            meta_color = "#64748b"
        else:
            bubble_bg = "#ffffff"
            bubble_border = "#e2e8f0"
            meta_color = "#64748b"

        if selected and not is_you:
            bubble_border = "#2563eb"
            shadow = "box-shadow:0 0 0 2px rgba(37,99,235,0.18);"
        else:
            shadow = "box-shadow:0 1px 2px rgba(15,23,42,0.05);"

        meta = (
            f"<div style='font-size:11px;color:{meta_color};margin-bottom:4px;"
            f"text-align:{align};'>{speaker} · {timestamp}</div>"
        )
        bubble = (
            f"<div style='background:{bubble_bg};border:1px solid {bubble_border};"
            f"border-radius:14px;padding:10px 14px;{shadow}"
            f"display:inline-block;max-width:100%;text-align:left;"
            f"color:#0f172a;font-size:13px;line-height:1.45;'>{body}</div>"
        )
        wrapper_open = (
            f"<div style='{margin}text-align:{align};'>"
            f"<a href='seg:{entry.segment_index}' style='text-decoration:none;color:inherit;'>"
        )
        wrapper_close = "</a></div>"
        return wrapper_open + meta + bubble + wrapper_close

    def _format_system_html(self, line: str) -> str:
        return (
            "<div style='margin:8px 0;text-align:center;'>"
            f"<span style='font-size:11px;color:#94a3b8;background:#f8fafc;"
            f"border:1px solid #eef2f6;border-radius:12px;padding:4px 12px;'>"
            f"{html.escape(line)}</span></div>"
        )

    def _format_suggestion_html(self, record: _SuggestionRecord) -> str:
        status = "Generating…" if record.streaming else "Complete"
        if record.error:
            status = "Failed"
        started = (
            record.started_at.strftime("%H:%M:%S")
            if record.started_at
            else "—"
        )

        parts = [
            "<div style='font-family:Segoe UI,sans-serif;color:#0f172a;'>",
            (
                "<div style='background:#f8fafc;border:1px solid #e8edf3;"
                "border-radius:12px;padding:12px 14px;margin-bottom:12px;'>"
                f"<div style='font-size:11px;font-weight:600;color:#64748b;"
                f"letter-spacing:0.06em;text-transform:uppercase;'>Triggered by</div>"
                f"<div style='font-size:13px;color:#334155;margin-top:6px;line-height:1.45;'>"
                f"\"{html.escape(record.transcript_text)}\"</div>"
                f"<div style='font-size:11px;color:#94a3b8;margin-top:8px;'>"
                f"Segment #{record.segment_index} · {started} · {status}</div>"
                "</div>"
            ),
        ]

        if record.retrieval_matches:
            parts.append(
                "<div style='font-size:11px;font-weight:600;color:#64748b;"
                "letter-spacing:0.06em;text-transform:uppercase;margin-bottom:8px;'>"
                "Knowledge sources</div>"
            )
            for match in record.retrieval_matches[:6]:
                score = float(match.get("score", 0.0))
                source = html.escape(str(match.get("source", "unknown")))
                snippet = html.escape(str(match.get("text", ""))[:160])
                used = match.get("used", False)
                badge_bg = "#ecfdf5" if used else "#f8fafc"
                badge_color = "#047857" if used else "#64748b"
                badge_border = "#a7f3d0" if used else "#e2e8f0"
                label = "Used" if used else "Below threshold"
                parts.append(
                    f"<div style='background:{badge_bg};border:1px solid {badge_border};"
                    f"border-radius:10px;padding:10px 12px;margin-bottom:8px;'>"
                    f"<div style='display:flex;justify-content:space-between;'>"
                    f"<span style='font-size:12px;font-weight:600;color:#334155;'>{source}</span>"
                    f"<span style='font-size:11px;color:{badge_color};font-weight:600;'>"
                    f"{label} · {score:.0%}</span></div>"
                    f"<div style='font-size:12px;color:#64748b;margin-top:6px;line-height:1.4;'>"
                    f"{snippet}</div></div>"
                )
        elif not record.streaming:
            parts.append(
                "<div style='font-size:12px;color:#94a3b8;font-style:italic;"
                "margin-bottom:12px;'>No knowledge base documents matched this message.</div>"
            )

        response_body = html.escape(record.body).replace("\n", "<br>")
        parts.append(
            "<div style='font-size:11px;font-weight:600;color:#64748b;"
            "letter-spacing:0.06em;text-transform:uppercase;margin:4px 0 8px;'>"
            "Suggested response</div>"
            f"<div style='background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;"
            f"padding:14px 16px;font-size:13px;line-height:1.55;color:#0f172a;"
            f"box-shadow:0 1px 3px rgba(15,23,42,0.04);'>{response_body or '…'}</div>"
        )

        if record.error:
            parts.append(
                f"<div style='margin-top:12px;color:#dc2626;font-size:12px;'>"
                f"Error: {html.escape(record.error)}</div>"
            )
        elif record.streaming:
            parts.append(
                "<div style='margin-top:12px;font-size:12px;color:#2563eb;'>"
                "● Generating response…</div>"
            )
        elif record.stats:
            first_token = float(record.stats.get("first_token_seconds", 0.0))
            total = float(record.stats.get("total_seconds", 0.0))
            num_chunks = int(record.stats.get("num_chunks_used", 0))
            parts.append(
                f"<div style='margin-top:12px;font-size:11px;color:#94a3b8;'>"
                f"Retrieved {num_chunks} chunks · first token {first_token:.2f}s · "
                f"total {total:.2f}s</div>"
            )

        parts.append("</div>")
        return "".join(parts)

    def _refresh_transcript_view(self, *, scroll_to_bottom: bool = False) -> None:
        bar = self._transcript_view.verticalScrollBar()
        pinned_to_bottom = scroll_to_bottom or bar.value() >= bar.maximum() - 24
        previous_scroll = bar.value()

        blocks: list[str] = []
        for line in self._system_transcript_lines:
            blocks.append(self._format_system_html(line))
        blocks.extend(
            self._format_transcript_html(
                entry,
                selected=entry.segment_index == self._selected_segment,
            )
            for entry in self._transcript_entries
        )
        self._transcript_view.setHtml("<br>".join(blocks) if blocks else "")

        if pinned_to_bottom:
            self._scroll_to_bottom(self._transcript_view)
        else:
            bar.setValue(min(previous_scroll, bar.maximum()))

    def _append_transcript_entry(self, entry: TranscriptEntry) -> None:
        self._refresh_transcript_view(scroll_to_bottom=True)

    def _on_transcript_anchor_clicked(self, url: QUrl) -> None:
        link = url.toString()
        if not link.startswith("seg:"):
            return
        try:
            segment_index = int(link.removeprefix("seg:"))
        except ValueError:
            return

        self._selected_segment = segment_index
        latest = max(self._suggestions.keys(), default=None)
        self._follow_live_suggestions = segment_index == latest
        self._refresh_transcript_view()
        self._show_suggestion_for_segment(segment_index)

    def _entry_for_segment(self, segment_index: int) -> TranscriptEntry | None:
        for entry in self._transcript_entries:
            if entry.segment_index == segment_index:
                return entry
        return None

    def _is_displaying_segment(self, segment_index: int) -> bool:
        if self._follow_live_suggestions:
            latest = max(self._suggestions.keys(), default=None)
            return segment_index == latest
        return self._selected_segment == segment_index

    def _show_suggestion_for_segment(self, segment_index: int) -> None:
        record = self._suggestions.get(segment_index)
        if record is None:
            entry = self._entry_for_segment(segment_index)
            if entry is not None and entry.speaker == SPEAKER_YOU:
                self._suggestion_view.setHtml(
                    "<div style='color:#64748b;font-size:13px;padding:8px;'>"
                    "AI responses are generated for <b>Them</b> messages only.</div>"
                )
            else:
                self._suggestion_view.setHtml(
                    "<div style='color:#64748b;font-size:13px;padding:8px;'>"
                    "No AI response is available for this message yet.</div>"
                )
            return

        self._suggestion_view.setHtml(self._format_suggestion_html(record))
        self._scroll_to_bottom(self._suggestion_view)

    def _append_transcript_system(self, line: str) -> None:
        self._system_transcript_lines.append(line)
        self._refresh_transcript_view(scroll_to_bottom=True)

    def _on_suggestion_started(self, segment_index: int, transcript_text: str) -> None:
        self._suggestions[segment_index] = _SuggestionRecord(
            segment_index=segment_index,
            transcript_text=transcript_text,
            started_at=datetime.now(),
        )
        if self._follow_live_suggestions:
            self._selected_segment = segment_index
            self._refresh_transcript_view()
            self._show_suggestion_for_segment(segment_index)
        else:
            self._refresh_transcript_view()

    def _on_suggestion_retrieval(self, segment_index: int, rows: object) -> None:
        if not isinstance(rows, list):
            return
        record = self._suggestions.get(segment_index)
        if record is not None:
            record.retrieval_matches = rows
        if self._is_displaying_segment(segment_index):
            self._show_suggestion_for_segment(segment_index)

    def _on_retrieval_debug(self, segment_index: int, rows: list) -> None:
        if not self._debug_checkbox.isChecked():
            return
        lines = [f"Segment #{segment_index} retrieval:"]
        if not rows:
            lines.append("  (no candidates)")
        else:
            for score, source, passed in rows:
                marker = "" if passed else " (below threshold)"
                lines.append(f"  score={score:.2f} source={source}{marker}")
        self._debug_view.append("\n".join(lines))
        self._scroll_to_bottom(self._debug_view)

    def _on_suggestion_token(self, segment_index: int, delta: str) -> None:
        record = self._suggestions.get(segment_index)
        if record is not None:
            record.body += delta
        if self._is_displaying_segment(segment_index):
            self._show_suggestion_for_segment(segment_index)

    def _on_suggestion_complete(self, segment_index: int, full_text: str, stats: dict) -> None:
        record = self._suggestions.get(segment_index)
        if record is not None:
            record.body = full_text
            record.stats = stats
            record.streaming = False
        if self._is_displaying_segment(segment_index):
            self._show_suggestion_for_segment(segment_index)

    def _on_suggestion_error(self, segment_index: int, reason: str) -> None:
        record = self._suggestions.get(segment_index)
        if record is not None:
            record.error = reason
            record.streaming = False
        if self._is_displaying_segment(segment_index):
            self._show_suggestion_for_segment(segment_index)

    def _on_error_occurred(self, message: str) -> None:
        self._append_transcript_system(f"Error: {message}")
        if "Audio capture stopped" not in message:
            return
        if not get_active_audio_devices().has_mic:
            return
        try:
            mic_index, loopback_index, mic_only = self._resolve_capture_devices()
        except RuntimeError as exc:
            self._append_transcript_system(str(exc))
            return
        self._append_transcript_system("Attempting to resume audio capture...")
        self._auto_resume_listening = True
        self._controller.start_listening(
            mic_index,
            loopback_index,
            mic_only_testing=mic_only,
        )

    @staticmethod
    def _scroll_to_bottom(editor: QTextEdit) -> None:
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        editor.setTextCursor(cursor)

    def closeEvent(self, event: QCloseEvent) -> None:
        self._device_monitor.stop()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self._controller.is_listening:
            self._controller.stop_listening(wait=True, timeout=2.0)
        event.accept()
