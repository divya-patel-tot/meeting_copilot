from __future__ import annotations

import html

from PyQt6.QtCore import QEvent, Qt, QPoint, QRect, pyqtSignal
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.audio.device_manager import (
    get_default_input_index,
    get_default_mic_friendly_name,
    get_default_output_friendly_label,
    list_advanced_capture_devices,
    resolve_loopback_capture_index,
)
from app.core.pipeline_controller import PipelineController
from app.core.stt.transcript_buffer import SPEAKER_YOU, TranscriptEntry
from app.ui.dialogs.settings_dialog import SettingsDialog
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


class MainWindow(QMainWindow):
    listening_state_changed = pyqtSignal(bool)
    window_visibility_changed = pyqtSignal(bool)

    def __init__(self, controller: PipelineController) -> None:
        super().__init__()
        self._controller = controller
        self._current_suggestion_segment: int | None = None
        self._tray = None
        self._resize_edge: str | None = None
        self._resize_origin: QPoint | None = None
        self._resize_start_geom: QRect | None = None

        self.setWindowTitle("Meeting Responder")
        self.setMinimumSize(900, 560)
        self.resize(1200, 760)

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        self._build_ui()
        self._connect_signals()
        self._populate_devices()
        self._update_start_stop_ui(listening=False)

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
        body_layout.setContentsMargins(14, 10, 14, 14)
        body_layout.setSpacing(10)

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

        self._transcript_panel = GlassPanel("Transcript", variant="transcript")
        self._transcript_view = QTextEdit()
        self._transcript_view.setObjectName("transcriptPanel")
        self._transcript_view.setReadOnly(True)
        self._transcript_view.setPlaceholderText(
            "Live transcript will build up here as speech is detected…"
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

        self._suggestion_panel = GlassPanel("Suggestion", variant="suggestion")
        self._suggestion_view = QTextEdit()
        self._suggestion_view.setObjectName("suggestionPanel")
        self._suggestion_view.setReadOnly(True)
        self._suggestion_view.setPlaceholderText(
            "Streaming reply suggestions will appear here…"
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
        self._controller.retrieval_debug.connect(self._on_retrieval_debug)
        self._controller.suggestion_token.connect(self._on_suggestion_token)
        self._controller.suggestion_complete.connect(self._on_suggestion_complete)
        self._controller.suggestion_error.connect(self._on_suggestion_error)
        self._controller.error_occurred.connect(self._on_error_occurred)

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
        has_mic = get_default_input_index() is not None
        self._start_stop_btn.setEnabled(has_mic)

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

        mic_index = get_default_input_index()
        if mic_index is None:
            raise RuntimeError("No default microphone found.")

        loopback_index: int | None = None
        if not mic_only:
            loopback_index = resolve_loopback_capture_index()

        if self._advanced_toggle.isChecked():
            data = self._advanced_device_combo.currentData()
            if not isinstance(data, dict):
                raise RuntimeError("No advanced capture device selected.")
            if str(data.get("source_mode")) == SOURCE_MICROPHONE:
                mic_index = int(data["index"])
            elif str(data.get("source_mode")) == SOURCE_LOOPBACK:
                loopback_index = int(data["index"])

        return mic_index, loopback_index, mic_only

    def _on_start_stop_clicked(self) -> None:
        if self._controller.is_listening:
            self._controller.stop_listening()
            return

        try:
            mic_index, loopback_index, mic_only = self._resolve_capture_devices()
        except RuntimeError as exc:
            self._append_transcript_system(str(exc))
            return

        set_mic_only_testing(mic_only)
        set_audio_device_index(int(mic_index))
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
        listening = status in {"Listening", "Warming up..."}
        self._status_pill.set_status(status, listening=listening)
        self._update_start_stop_ui(listening=listening)

    def _update_start_stop_ui(self, *, listening: bool) -> None:
        if listening:
            self._start_stop_btn.setText("Stop")
            self._start_stop_btn.setProperty("listening", True)
            self._mic_only_checkbox.setEnabled(False)
            self._advanced_toggle.setEnabled(False)
            self._advanced_device_combo.setEnabled(False)
        else:
            self._start_stop_btn.setText("Start")
            self._start_stop_btn.setProperty("listening", False)
            has_mic = get_default_input_index() is not None
            self._mic_only_checkbox.setEnabled(has_mic)
            self._advanced_toggle.setEnabled(self._advanced_device_combo.count() > 0)
            self._advanced_device_combo.setEnabled(True)
        self._start_stop_btn.style().unpolish(self._start_stop_btn)
        self._start_stop_btn.style().polish(self._start_stop_btn)
        self.listening_state_changed.emit(listening)

    def _on_transcript_ready(self, entry: object) -> None:
        if not isinstance(entry, TranscriptEntry):
            return
        self._append_transcript_entry(entry)

    @staticmethod
    def _format_transcript_html(entry: TranscriptEntry) -> str:
        timestamp = html.escape(entry.timestamp.strftime("%H:%M:%S"))
        if entry.speaker == SPEAKER_YOU:
            tag_html = '<span style="color:#8ab4ff;font-weight:600">[You]</span>'
            text_color = "#c8d4ea"
        else:
            tag_html = '<span style="color:#7dcea0;font-weight:600">[Them]</span>'
            text_color = "#d8dee9"

        if entry.error:
            body = f"STT failed: {html.escape(entry.error)}"
        elif entry.is_valid:
            body = (
                f'"{html.escape(entry.text)}" '
                f'<span style="color:#8892a8">(STT {entry.latency_seconds:.2f}s)</span>'
            )
        else:
            detail = html.escape(entry.text or "(empty)")
            body = f"discarded: {detail!r}"

        return (
            f'<span style="color:#8892a8">[{timestamp}]</span> '
            f"{tag_html} "
            f'<span style="color:{text_color}">#{entry.segment_index} — {body}</span>'
        )

    def _append_transcript_entry(self, entry: TranscriptEntry) -> None:
        bar = self._transcript_view.verticalScrollBar()
        pinned_to_bottom = bar.value() >= bar.maximum() - 24

        cursor = self._transcript_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._transcript_view.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(self._format_transcript_html(entry))
        self._transcript_view.setTextCursor(cursor)

        if pinned_to_bottom:
            self._scroll_to_bottom(self._transcript_view)

    def _append_transcript_system(self, line: str) -> None:
        bar = self._transcript_view.verticalScrollBar()
        pinned_to_bottom = bar.value() >= bar.maximum() - 24

        cursor = self._transcript_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self._transcript_view.document().isEmpty():
            cursor.insertBlock()
        cursor.insertHtml(
            f'<span style="color:#8892a8">[system]</span> '
            f'<span style="color:#d8dee9">{html.escape(line)}</span>'
        )
        self._transcript_view.setTextCursor(cursor)

        if pinned_to_bottom:
            self._scroll_to_bottom(self._transcript_view)

    def _on_suggestion_started(self, segment_index: int, transcript_text: str) -> None:
        self._current_suggestion_segment = segment_index
        self._suggestion_view.clear()
        header = f"Replying to [Them]: \"{transcript_text}\"\n\n"
        self._suggestion_view.setPlainText(header)
        cursor = self._suggestion_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._suggestion_view.setTextCursor(cursor)

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
        if self._current_suggestion_segment != segment_index:
            return
        cursor = self._suggestion_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(delta)
        self._suggestion_view.setTextCursor(cursor)

    def _on_suggestion_complete(self, segment_index: int, _full_text: str, stats: dict) -> None:
        if self._current_suggestion_segment != segment_index:
            return
        first_token = float(stats.get("first_token_seconds", 0.0))
        total = float(stats.get("total_seconds", 0.0))
        num_chunks = int(stats.get("num_chunks_used", 0))
        footer = (
            f"\n\n— retrieved {num_chunks} chunks · "
            f"first token: {first_token:.2f}s · total: {total:.2f}s —"
        )
        cursor = self._suggestion_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(footer)
        self._suggestion_view.setTextCursor(cursor)

    def _on_suggestion_error(self, segment_index: int, reason: str) -> None:
        self._suggestion_view.append(
            f"\n\nSuggestion failed (segment #{segment_index}): {reason}"
        )
        self._scroll_to_bottom(self._suggestion_view)

    def _on_error_occurred(self, message: str) -> None:
        self._append_transcript_system(f"Error: {message}")

    @staticmethod
    def _scroll_to_bottom(editor: QTextEdit) -> None:
        cursor = editor.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        editor.setTextCursor(cursor)

    def closeEvent(self, event: QCloseEvent) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if self._controller.is_listening:
            self._controller.stop_listening(wait=True, timeout=2.0)
        event.accept()
