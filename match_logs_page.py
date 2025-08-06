# match_logs_page.py
import os
import csv
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QPushButton
from PyQt5.QtGui import QIcon

LOG_FILE = "logs/match_logs.csv"

class MatchLogsPage(QWidget):
    def __init__(self):
        super().__init__()

        ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

        self.setWindowTitle("Match Logs")
        self.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, "app_icon.ico")))        
        self.resize(500, 500)

        layout = QVBoxLayout()

        self.title = QLabel("Match Logs")
        self.title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.title)

        self.log_list = QListWidget()
        layout.addWidget(self.log_list)

        self.clear_button = QPushButton("Clear Logs")
        self.clear_button.clicked.connect(self.clear_logs)
        layout.addWidget(self.clear_button)

        self.setLayout(layout)

        self.load_logs()

    def add_log(self, label, score):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{timestamp} - ✅ Match: {label} (Score: {score:.2f})"
        self.log_list.insertItem(0, entry)
        self.append_log_to_file(timestamp, label, score)

    def append_log_to_file(self, timestamp, label, score):
        os.makedirs("logs", exist_ok=True)
        is_new = not os.path.exists(LOG_FILE)
        with open(LOG_FILE, "a", newline="") as file:
            writer = csv.writer(file)
            if is_new:
                writer.writerow(["Timestamp", "Match Label", "Score"])
            writer.writerow([timestamp, label, f"{score:.2f}"])

    def load_logs(self):
        if not os.path.exists(LOG_FILE):
            return

        with open(LOG_FILE, newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            # Skip header
            for row in reversed(rows[1:]):  # Reverse order to show latest on top
                timestamp, label, score = row
                entry = f"{timestamp} - ✅ Match: {label} (Score: {score})"
                self.log_list.addItem(entry)

    def clear_logs(self):
        self.log_list.clear()
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
