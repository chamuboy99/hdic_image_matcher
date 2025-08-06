from PyQt5.QtWidgets import QApplication,QMessageBox, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
import sys
from pathlib import Path
from PyQt5.QtGui import QIcon

class PreviewUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HDIC Matcher")
        self.setWindowIcon(QIcon("assets/app_icon.ico"))
        self.resize(650, 650)

        self.image_label = QLabel()
        self.result_label = QLabel("Match Results Will Appear Here")

        self.toggle_button = QPushButton("Start Camera")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)

        self.upload_button = QPushButton("Upload Victim")
        self.upload_button.setObjectName("Upload_Victim")
        self.upload_button.clicked.connect(self.upload)

        self.upload_button2 = QPushButton("Upload Multiple Victims")
        self.upload_button2.setObjectName("Upload_Multiple_Victims")

        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("Quit")
        self.quit_button.clicked.connect(self.close)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.upload_button)
        layout2.addWidget(self.upload_button2)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.toggle_button)
        layout.addLayout(layout2)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

    
        self.apply_styles()

    def apply_styles(self):
        qss_file = Path("styles.qss")
        if qss_file.exists():
            with open(qss_file, "r") as f:
                self.setStyleSheet(f.read())

    def upload(self):
        QMessageBox.warning(self, "No Images", f"No image files found.")
            

if __name__ == '__main__':
    app = QApplication(sys.argv)
    preview = PreviewUI()
    preview.show()
    sys.exit(app.exec_())
