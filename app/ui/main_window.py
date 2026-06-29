from PyQt6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Meeting Responder")
        self.setMinimumSize(400, 200)

        label = QLabel("Environment OK")
        label.setStyleSheet("font-size: 18px; padding: 20px;")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(label)
        self.setCentralWidget(container)
