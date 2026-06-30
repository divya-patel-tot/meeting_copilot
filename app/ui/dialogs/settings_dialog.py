from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.ui.branding import APP_NAME
from app.ui.icon_loader import app_icon
from app.core.audio.device_manager import format_input_device, list_input_devices
from app.core.pipeline_controller import PipelineController
from app.ui.setup_wizard import _ApiKeyWorker
from app.utils.app_settings import get_audio_device_index, set_audio_device_index
from app.utils.config import settings


class _IngestWorker(QThread):
    finished_ok = pyqtSignal(int)
    failed = pyqtSignal(str)

    def __init__(self, controller: PipelineController, file_path: str) -> None:
        super().__init__()
        self._controller = controller
        self._file_path = file_path

    def run(self) -> None:
        try:
            count = self._controller.knowledge_base.ingest_file(self._file_path)
            self.finished_ok.emit(count)
        except Exception as exc:
            self.failed.emit(str(exc))


class SettingsDialog(QDialog):
    """Manage API key, audio device, and knowledge base documents."""

    def __init__(self, controller: PipelineController, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._api_worker: _ApiKeyWorker | None = None
        self._ingest_worker: _IngestWorker | None = None
        self._device_saved_index: int | None = get_audio_device_index()

        self.setWindowTitle(f"{APP_NAME} — Settings")
        self.setWindowIcon(app_icon())
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)

        api_group = QGroupBox("Groq API key")
        api_layout = QFormLayout(api_group)
        self._api_key_input = QLineEdit()
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        if settings.GROQ_API_KEY.strip():
            self._api_key_input.setPlaceholderText("Saved — paste a new key to replace")
        else:
            self._api_key_input.setPlaceholderText("gsk_...")
        api_layout.addRow("API key:", self._api_key_input)
        self._api_key_status = QLabel()
        self._api_key_status.setStyleSheet("color: #9aa4b2;")
        api_layout.addRow("", self._api_key_status)
        root.addWidget(api_group)

        audio_group = QGroupBox("Audio device")
        audio_layout = QFormLayout(audio_group)
        self._device_combo = QComboBox()
        self._populate_devices()
        audio_layout.addRow("Microphone:", self._device_combo)
        root.addWidget(audio_group)

        kb_group = QGroupBox("Knowledge base")
        kb_layout = QVBoxLayout(kb_group)
        self._sources_list = QListWidget()
        kb_layout.addWidget(self._sources_list)
        kb_buttons = QHBoxLayout()
        self._add_doc_btn = QPushButton("Add Document...")
        self._remove_doc_btn = QPushButton("Remove Selected")
        kb_buttons.addWidget(self._add_doc_btn)
        kb_buttons.addWidget(self._remove_doc_btn)
        kb_buttons.addStretch()
        kb_layout.addLayout(kb_buttons)
        self._kb_status = QLabel()
        self._kb_status.setStyleSheet("color: #9aa4b2;")
        kb_layout.addWidget(self._kb_status)
        root.addWidget(kb_group)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Close
        )
        self._buttons.button(QDialogButtonBox.StandardButton.Save).clicked.connect(
            self._save_settings
        )
        self._buttons.rejected.connect(self.reject)
        root.addWidget(self._buttons)

        self._add_doc_btn.clicked.connect(self._add_document)
        self._remove_doc_btn.clicked.connect(self._remove_selected_document)
        self._refresh_sources_list()

    def _populate_devices(self) -> None:
        self._device_combo.clear()
        devices = list_input_devices()
        if not devices:
            self._device_combo.addItem("No input devices found", -1)
            return

        saved = get_audio_device_index()
        selected_row = 0
        for row, device in enumerate(devices):
            self._device_combo.addItem(format_input_device(device), device["index"])
            if saved is not None and device["index"] == saved:
                selected_row = row
            elif saved is None and device.get("is_default"):
                selected_row = row
        self._device_combo.setCurrentIndex(selected_row)

    def _refresh_sources_list(self) -> None:
        self._sources_list.clear()
        stats = self._controller.knowledge_base.get_stats()
        for source in stats.get("sources", []):
            self._sources_list.addItem(source)
        total = stats.get("total_chunks", 0)
        self._kb_status.setText(f"{total} chunk(s) across {len(stats.get('sources', []))} document(s)")

    def _add_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Add document to knowledge base",
            "",
            "Documents (*.pdf *.docx *.txt)",
        )
        if not path:
            return

        self._add_doc_btn.setEnabled(False)
        self._remove_doc_btn.setEnabled(False)
        self._kb_status.setText(f"Ingesting {Path(path).name}…")

        worker = _IngestWorker(self._controller, path)
        worker.finished_ok.connect(self._on_ingest_ok)
        worker.failed.connect(self._on_ingest_failed)
        self._ingest_worker = worker
        worker.start()

    def _on_ingest_ok(self, chunk_count: int) -> None:
        self._add_doc_btn.setEnabled(True)
        self._remove_doc_btn.setEnabled(True)
        self._refresh_sources_list()
        if chunk_count == 0:
            QMessageBox.warning(self, "Ingest", "No text chunks were created from that file.")
        else:
            self._kb_status.setText(f"Ingested {chunk_count} chunk(s).")

    def _on_ingest_failed(self, message: str) -> None:
        self._add_doc_btn.setEnabled(True)
        self._remove_doc_btn.setEnabled(True)
        self._refresh_sources_list()
        QMessageBox.critical(self, "Ingest failed", message)

    def _remove_selected_document(self) -> None:
        item = self._sources_list.currentItem()
        if item is None:
            QMessageBox.information(self, "Remove document", "Select a document to remove.")
            return

        source = item.text()
        confirm = QMessageBox.question(
            self,
            "Remove document",
            f"Remove all chunks from \"{source}\"?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        removed = self._controller.knowledge_base.remove_source(source)
        self._refresh_sources_list()
        self._kb_status.setText(f"Removed {removed} chunk(s) from {source}.")

    def _save_settings(self) -> None:
        device_index = self._device_combo.currentData()
        if device_index is None or int(device_index) < 0:
            QMessageBox.warning(self, "Settings", "Select a valid microphone.")
            return

        set_audio_device_index(int(device_index))
        self._device_saved_index = int(device_index)

        api_key = self._api_key_input.text().strip()
        if not api_key:
            self._api_key_status.setText("Device saved. API key unchanged.")
            self._api_key_status.setStyleSheet("color: #2ecc71;")
            return

        self._api_key_status.setText("Validating API key…")
        self._api_key_status.setStyleSheet("color: #9aa4b2;")
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(False)

        worker = _ApiKeyWorker(api_key)
        worker.finished_ok.connect(self._on_api_key_saved)
        worker.failed.connect(self._on_api_key_failed)
        self._api_worker = worker
        worker.start()

    def _on_api_key_saved(self) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
        self._api_key_input.clear()
        self._api_key_input.setPlaceholderText("Saved — paste a new key to replace")
        self._api_key_status.setText("API key saved and validated.")
        self._api_key_status.setStyleSheet("color: #2ecc71;")

    def _on_api_key_failed(self, message: str) -> None:
        self._buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(True)
        self._api_key_status.setText(f"Invalid API key: {message}")
        self._api_key_status.setStyleSheet("color: #e74c3c;")

    def selected_device_index(self) -> int | None:
        value = self._device_combo.currentData()
        if value is None or int(value) < 0:
            return None
        return int(value)
