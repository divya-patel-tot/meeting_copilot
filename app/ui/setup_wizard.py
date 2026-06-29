from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.utils.config import settings
from app.utils.setup import (
    download_silero_vad,
    is_setup_complete,
    save_groq_api_key,
    silero_vad_path,
    validate_groq_api_key,
    warm_up_embedder,
    write_setup_marker,
)


class _ApiKeyWorker(QThread):
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, api_key: str) -> None:
        super().__init__()
        self._api_key = api_key

    def run(self) -> None:
        try:
            save_groq_api_key(self._api_key)
            validate_groq_api_key(self._api_key)
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _VadDownloadWorker(QThread):
    progress = pyqtSignal(int, int)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            download_silero_vad(
                on_progress=lambda done, total: self.progress.emit(done, total or 0),
            )
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class _EmbedderWorker(QThread):
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def run(self) -> None:
        try:
            warm_up_embedder()
            self.finished_ok.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class SetupWizard(QDialog):
    """First-launch setup: Groq API key, Silero VAD model, embedder warmup."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Meeting Responder — Setup")
        self.setModal(True)
        self.setMinimumWidth(520)

        self._needs_api_key = not settings.GROQ_API_KEY.strip()
        self._needs_vad = not silero_vad_path().exists()
        self._needs_embedder = True

        self._page_keys: list[str] = []
        self._workers: list[QThread] = []

        self._stack = QStackedWidget()
        self._back_btn = QPushButton("Back")
        self._next_btn = QPushButton("Next")
        self._next_btn.setDefault(True)
        self._back_btn.clicked.connect(self._go_back)
        self._next_btn.clicked.connect(self._go_next)

        root = QVBoxLayout(self)
        self._heading = QLabel()
        self._heading.setObjectName("setupHeading")
        root.addWidget(self._heading)
        root.addWidget(self._stack, stretch=1)

        nav = QHBoxLayout()
        nav.addStretch()
        nav.addWidget(self._back_btn)
        nav.addWidget(self._next_btn)
        root.addLayout(nav)

        self._build_pages()
        self._current_index = 0
        self._show_page(0)

    def _build_pages(self) -> None:
        intro = QWidget()
        intro_layout = QVBoxLayout(intro)
        intro_layout.addWidget(
            QLabel(
                "Welcome. This one-time setup prepares speech detection, "
                "your Groq API connection, and knowledge-base components."
            )
        )
        steps = []
        if self._needs_api_key:
            steps.append("• Connect your Groq API key")
        if self._needs_vad:
            steps.append("• Download the speech detection model (~2 MB)")
        steps.append("• Prepare embedding models for document search")
        intro_layout.addWidget(QLabel("\n".join(steps)))
        intro_layout.addStretch()
        self._add_page("intro", intro)

        if self._needs_api_key:
            api_page = QWidget()
            api_layout = QVBoxLayout(api_page)
            api_layout.addWidget(
                QLabel("Paste your free Groq API key (console.groq.com):")
            )
            self._api_key_input = QLineEdit()
            self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self._api_key_input.setPlaceholderText("gsk_...")
            api_layout.addWidget(self._api_key_input)
            self._api_key_error = QLabel()
            self._api_key_error.setStyleSheet("color: #e74c3c;")
            self._api_key_error.setWordWrap(True)
            self._api_key_error.hide()
            api_layout.addWidget(self._api_key_error)
            self._api_key_progress = QLabel("Validating API key…")
            self._api_key_progress.hide()
            api_layout.addWidget(self._api_key_progress)
            api_layout.addStretch()
            self._add_page("api_key", api_page)

        if self._needs_vad:
            vad_page = QWidget()
            vad_layout = QVBoxLayout(vad_page)
            self._vad_label = QLabel("Downloading speech detection model…")
            vad_layout.addWidget(self._vad_label)
            self._vad_progress = QProgressBar()
            self._vad_progress.setRange(0, 100)
            self._vad_progress.setValue(0)
            vad_layout.addWidget(self._vad_progress)
            self._vad_error = QLabel()
            self._vad_error.setStyleSheet("color: #e74c3c;")
            self._vad_error.setWordWrap(True)
            self._vad_error.hide()
            vad_layout.addWidget(self._vad_error)
            vad_layout.addStretch()
            self._add_page("vad", vad_page)

        embed_page = QWidget()
        embed_layout = QVBoxLayout(embed_page)
        self._embed_label = QLabel("Preparing knowledge base components…")
        embed_layout.addWidget(self._embed_label)
        self._embed_progress = QProgressBar()
        self._embed_progress.setRange(0, 0)
        embed_layout.addWidget(self._embed_progress)
        self._embed_error = QLabel()
        self._embed_error.setStyleSheet("color: #e74c3c;")
        self._embed_error.setWordWrap(True)
        self._embed_error.hide()
        embed_layout.addWidget(self._embed_error)
        embed_layout.addStretch()
        self._add_page("embedder", embed_page)

        done = QWidget()
        done_layout = QVBoxLayout(done)
        done_layout.addWidget(QLabel("Setup complete. You're ready to use Meeting Responder."))
        done_layout.addStretch()
        self._add_page("done", done)

    def _add_page(self, key: str, widget: QWidget) -> None:
        self._page_keys.append(key)
        self._stack.addWidget(widget)

    @property
    def _current_key(self) -> str:
        return self._page_keys[self._current_index]

    def _show_page(self, index: int) -> None:
        self._current_index = index
        self._stack.setCurrentIndex(index)
        key = self._current_key

        titles = {
            "intro": "First-time setup",
            "api_key": "Groq API key",
            "vad": "Speech detection model",
            "embedder": "Knowledge base components",
            "done": "All set",
        }
        self._heading.setText(titles.get(key, "Setup"))

        self._back_btn.setEnabled(index > 0 and key not in {"vad", "embedder"})
        if key == "intro":
            self._next_btn.setText("Next")
            self._next_btn.setEnabled(True)
        elif key == "api_key":
            self._next_btn.setText("Save & validate")
            self._next_btn.setEnabled(True)
        elif key == "vad":
            self._next_btn.setEnabled(False)
            self._start_vad_download()
        elif key == "embedder":
            self._next_btn.setEnabled(False)
            self._start_embedder_warmup()
        elif key == "done":
            self._next_btn.setText("Finish")
            self._next_btn.setEnabled(True)

    def _go_back(self) -> None:
        if self._current_index > 0:
            self._show_page(self._current_index - 1)

    def _go_next(self) -> None:
        key = self._current_key
        if key == "vad" and self._next_btn.text() == "Retry":
            self._next_btn.setText("Next")
            self._start_vad_download()
            return
        if key == "embedder" and self._next_btn.text() == "Retry":
            self._next_btn.setText("Next")
            self._start_embedder_warmup()
            return

        if key == "intro":
            self._show_page(self._current_index + 1)
        elif key == "api_key":
            self._validate_api_key()
        elif key == "done":
            write_setup_marker()
            self.accept()
        elif key in {"vad", "embedder"}:
            return
        else:
            self._show_page(self._current_index + 1)

    def _validate_api_key(self) -> None:
        api_key = self._api_key_input.text().strip()
        if not api_key:
            self._api_key_error.setText("Please enter your Groq API key.")
            self._api_key_error.show()
            return

        self._api_key_error.hide()
        self._api_key_progress.show()
        self._next_btn.setEnabled(False)
        self._back_btn.setEnabled(False)

        worker = _ApiKeyWorker(api_key)
        worker.finished_ok.connect(self._on_api_key_ok)
        worker.failed.connect(self._on_api_key_failed)
        self._workers.append(worker)
        worker.start()

    def _on_api_key_ok(self) -> None:
        self._api_key_progress.hide()
        self._needs_api_key = False
        self._next_btn.setEnabled(True)
        self._back_btn.setEnabled(True)
        self._show_page(self._current_index + 1)

    def _on_api_key_failed(self, message: str) -> None:
        self._api_key_progress.hide()
        self._api_key_error.setText(
            f"Invalid API key or connection failed: {message}\n"
            "Check the key at console.groq.com and try again."
        )
        self._api_key_error.show()
        self._next_btn.setEnabled(True)
        self._back_btn.setEnabled(True)

    def _start_vad_download(self) -> None:
        if not self._needs_vad:
            self._show_page(self._current_index + 1)
            return

        self._vad_error.hide()
        self._vad_progress.setRange(0, 100)
        self._vad_progress.setValue(0)
        worker = _VadDownloadWorker()
        worker.progress.connect(self._on_vad_progress)
        worker.finished_ok.connect(self._on_vad_ok)
        worker.failed.connect(self._on_vad_failed)
        self._workers.append(worker)
        worker.start()

    def _on_vad_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            self._vad_progress.setRange(0, total)
            self._vad_progress.setValue(downloaded)
            pct = int(downloaded * 100 / total)
            self._vad_label.setText(f"Downloading speech detection model… {pct}%")
        else:
            self._vad_progress.setRange(0, 0)
            self._vad_label.setText("Downloading speech detection model…")

    def _on_vad_ok(self) -> None:
        self._needs_vad = False
        self._vad_progress.setRange(0, 100)
        self._vad_progress.setValue(100)
        self._vad_label.setText("Speech detection model ready.")
        self._show_page(self._current_index + 1)

    def _on_vad_failed(self, message: str) -> None:
        self._vad_error.setText(f"Download failed: {message}")
        self._vad_error.show()
        self._next_btn.setText("Retry")
        self._next_btn.setEnabled(True)

    def _start_embedder_warmup(self) -> None:
        self._embed_error.hide()
        worker = _EmbedderWorker()
        worker.finished_ok.connect(self._on_embedder_ok)
        worker.failed.connect(self._on_embedder_failed)
        self._workers.append(worker)
        worker.start()

    def _on_embedder_ok(self) -> None:
        self._embed_label.setText("Knowledge base components ready.")
        self._embed_progress.setRange(0, 100)
        self._embed_progress.setValue(100)
        self._show_page(self._current_index + 1)

    def _on_embedder_failed(self, message: str) -> None:
        self._embed_error.setText(f"Preparation failed: {message}")
        self._embed_error.show()
        self._next_btn.setText("Retry")
        self._next_btn.setEnabled(True)

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if not is_setup_complete():
            reply = QMessageBox.question(
                self,
                "Exit setup?",
                "Setup is not complete. Exit without finishing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()

    def reject(self) -> None:
        if not is_setup_complete():
            reply = QMessageBox.question(
                self,
                "Exit setup?",
                "Setup is not complete. Exit without finishing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        super().reject()
