import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication,QSplashScreen, QStackedLayout, QSizePolicy, QLabel, QPushButton, QVBoxLayout,QHBoxLayout, QWidget, QFileDialog, QMessageBox, QProgressDialog
from PyQt5.QtGui import QImage, QPixmap, QMovie, QIcon
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtMultimedia import QSoundEffect
from PyQt5.QtCore import QObject, pyqtSignal, QUrl

import shutil
import json
from pathlib import Path
from threading import Thread

def preload_tensorflow():
    _ = tf.constant([0.0]) + tf.constant([0.0])

from facenet_pytorch import MTCNN

from MTCNN_preprocess_gpu import align_and_crop_face_from_array
from embedder_512D import generate_embedding2
from matcher import encode_embedding_to_hv, match_hv_to_class

from match_logs_page import MatchLogsPage

# Constants
PADDING = 10
TARGET_SIZE = (160, 160)
BLUR_THRESHOLD = 40.0
STABILITY_THRESHOLD = 5    # Number of stable frames needed
CAPTURE_FRAME_COUNT = 5   # Number of frames to capture for batch
MATCH_DISPLAY_TIME_MS = 2000

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

        self.setWindowTitle("HDIC Matcher")
        self.setWindowIcon(QIcon(os.path.join(ASSETS_DIR, "app_icon.ico")))
        self.resize(650, 650)

        self.setup_sound()

        self.logs_page = MatchLogsPage()

        self.class_hypervectors = {}
        self.load_all_class_hypervectors()

        self.paused_gif = QLabel()
        self.paused_gif.setAlignment(Qt.AlignCenter)
        self.movie = QMovie("assets/triangle.gif")
        self.paused_gif.setMovie(self.movie)
        self.movie.start()

        self.processing_image = QLabel()
        self.processing_image.setAlignment(Qt.AlignCenter)
        self.processing_pixmap = QPixmap("assets/processing.png")
        self.processing_image.setPixmap(self.processing_pixmap)
        
        self.image_label = QLabel()
        self.result_label = QLabel("Match Results Will Appear Here")

        self.image_label.setFixedSize(640, 480)
        self.paused_gif.setFixedSize(640, 480)
        self.processing_image.setFixedSize(640, 480)

        size_policy = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.image_label.setSizePolicy(size_policy)
        self.paused_gif.setSizePolicy(size_policy)
        self.processing_image.setSizePolicy(size_policy)

        self.toggle_button = QPushButton("Start Camera")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(True)
        self.toggle_button.clicked.connect(self.toggle_camera)

        self.upload_button = QPushButton("Upload Victim")
        self.upload_button.setObjectName("Upload_Victim")
        self.upload_button.clicked.connect(self.upload_victim)

        self.upload_button2 = QPushButton("Upload Multiple Victims")
        self.upload_button2.setObjectName("Upload_Multiple_Victims")
        self.upload_button2.clicked.connect(self.upload_multiple_victims)

        self.view_logs_button = QPushButton("View Detection Logs")
        self.view_logs_button.setObjectName("View_Detection_Logs")
        self.view_logs_button.clicked.connect(self.show_logs_page)

        self.quit_button = QPushButton("Quit")
        self.quit_button.setObjectName("Quit")
        self.quit_button.clicked.connect(self.close)

        layout2 = QHBoxLayout()
        layout2.addWidget(self.upload_button)
        layout2.addWidget(self.upload_button2)

        layout = QVBoxLayout()
        self.stack = QStackedLayout()
        self.stack.addWidget(self.image_label)      # index 0 - live camera 
        self.stack.addWidget(self.paused_gif)       # index 1 - paused
        self.stack.addWidget(self.processing_image)   # index 2 - processing

        layout.addLayout(self.stack)

        layout.addWidget(self.result_label)
        layout.addWidget(self.toggle_button)
        layout.addLayout(layout2)
        layout.addWidget(self.view_logs_button)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        # Setup webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # Setup MTCNN on GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=device)

        # Face stability & capture buffers
        self.camera_running = False
        self.prev_box = None
        self.stable_count = 0
        self.face_buffer = []
        self.processing = False
        self.showing_match = False

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

        self.apply_styles()

    def show_logs_page(self):
        self.logs_page.show()

    def setup_sound(self):
        self.sound = QSoundEffect()
        self.sound.setSource(QUrl.fromLocalFile("assets/beep-warning-6387.wav"))
        self.sound.setVolume(0.9)

    def play_alert_sound(self):
        self.sound.play()
    
    def apply_styles(self):
        qss_file = Path("styles.qss")
        if qss_file.exists():
            with open(qss_file, "r") as f:
                self.setStyleSheet(f.read())

    def load_all_class_hypervectors(self):
        self.class_hypervectors.clear()
        hv_dir = Path("class_hypervectors")
        if not hv_dir.exists():
            return

        for file in hv_dir.glob("*.json"):
            label = file.stem.replace("_class_hv", "")
            with open(file, "r") as f:
                data = json.load(f)
                self.class_hypervectors[label] = np.array(data, dtype=np.uint8)

        # ✅ Build person_ids and packed class_hv_matrix
        self.person_ids = list(self.class_hypervectors.keys())
        hv_list = list(self.class_hypervectors.values())
        self.class_hv_matrix = np.array([np.packbits(hv) for hv in hv_list], dtype=np.uint8)

    def toggle_camera(self):
        if self.toggle_button.isChecked():
            # Stop camera
            self.camera_running = False
            self.toggle_button.setText("Start Camera")
            self.result_label.setText("Camera Paused.")
        else:
            # Start camera
            self.camera_running = True
            self.toggle_button.setText("Stop Camera")
            self.result_label.setText("Match Results Will Appear Here.")
            # Reset any buffer/state on restart
            self.prev_box = None
            self.stable_count = 0
            self.face_buffer.clear()
            self.processing = False
            self.showing_match = False
    
    def upload_victim(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select folder containing victim images", os.getcwd())
        if not folder_path:
            return
        
        person_id = os.path.basename(folder_path)
        raw_person_dir = os.path.join("images", "raw", person_id)
        embedding_person_dir = os.path.join("embeddings", person_id)
        hypervector_person_dir = os.path.join("hypervectors", person_id)

        os.makedirs(raw_person_dir, exist_ok=True)
        os.makedirs(embedding_person_dir, exist_ok=True)
        os.makedirs(hypervector_person_dir, exist_ok=True)

        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        total = len(image_files)

        if total == 0:
            QMessageBox.warning(self, "No Images", f"No image files found in {folder_path}.")
            return
        
        # Setup progress dialog
        progress = QProgressDialog("Processing Images", "Cancel", 0, total, self)
        progress.setWindowTitle("Processing")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        hypervectors = []

        for i, img_name in enumerate(image_files):
            progress.setValue(i)
            if progress.wasCanceled():
                QMessageBox.warning(self, "Cancelled", "Image processing was cancelled.")
                return

            src_path = os.path.join(folder_path, img_name)
            dst_path = os.path.join(raw_person_dir, img_name)
            shutil.copy2(src_path, dst_path)

            img = cv2.imread(dst_path)
            cropped_face = align_and_crop_face_from_array(
                img,
                self.mtcnn,
                padding=PADDING,
                target_size=TARGET_SIZE,
                blur_threshold=BLUR_THRESHOLD
            )
            if cropped_face is None:
                continue

            embedding = generate_embedding2(cropped_face)
            emb_path = os.path.join(embedding_person_dir, img_name.rsplit('.', 1)[0] + '.json')
            with open(emb_path, 'w') as f:
                json.dump({'embedding': embedding.tolist()}, f)

            hv = encode_embedding_to_hv(embedding)
            hv_path = os.path.join(hypervector_person_dir, img_name.rsplit('.', 1)[0] + '_hv.json')
            with open(hv_path, 'w') as f:
                json.dump(hv.tolist(), f)

            hypervectors.append(hv)

        progress.setValue(total)

        if hypervectors:
            summed = np.sum(hypervectors, axis=0)
            threshold = len(hypervectors) / 2
            class_hv = (summed > threshold).astype(np.uint8)

            class_path = os.path.join("class_hypervectors", f"{person_id}_class_hv.json")
            with open(class_path, 'w') as f:
                json.dump(class_hv.tolist(), f)
            
            self.load_all_class_hypervectors()

            QMessageBox.information(self, "Success", f"Victim '{person_id}' added to the database.")
        else:
            QMessageBox.warning(self, "Error", f"No valid faces found in {person_id}.")

    def upload_multiple_victims(self):
        # Select the parent folder
        folder_path = QFileDialog.getExistingDirectory(self, "Select parent folder containing victim folders", os.getcwd())
        if not folder_path:
            return

        # Get all subfolders (each representing a person)
        subfolders = [os.path.join(folder_path, d) for d in os.listdir(folder_path)
                    if os.path.isdir(os.path.join(folder_path, d))]

        if not subfolders:
            QMessageBox.warning(self, "Error", "No victim folders found.")
            return

        # Count total number of images to process
        total_images = sum(
            len([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
            for folder in subfolders
        )
        if total_images == 0:
            QMessageBox.warning(self, "Error", "No images found in any folders.")
            return

        # Setup progress dialog
        progress = QProgressDialog("Processing victim images", "Cancel", 0, total_images, self)
        progress.setWindowTitle("Uploading Victims")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        processed_count = 0

        for person_folder in subfolders:
            person_id = os.path.basename(person_folder)
            raw_person_dir = os.path.join("images", "raw", person_id)
            embedding_person_dir = os.path.join("embeddings", person_id)
            hypervector_person_dir = os.path.join("hypervectors", person_id)

            os.makedirs(raw_person_dir, exist_ok=True)
            os.makedirs(embedding_person_dir, exist_ok=True)
            os.makedirs(hypervector_person_dir, exist_ok=True)

            image_files = [f for f in os.listdir(person_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            hypervectors = []

            for img_name in image_files:
                progress.setValue(processed_count)
                if progress.wasCanceled():
                    QMessageBox.warning(self, "Cancelled", "Upload cancelled.")
                    return

                src_path = os.path.join(person_folder, img_name)
                dst_path = os.path.join(raw_person_dir, img_name)
                shutil.copy2(src_path, dst_path)

                img = cv2.imread(dst_path)
                cropped_face = align_and_crop_face_from_array(
                    img,
                    self.mtcnn,
                    padding=PADDING,
                    target_size=TARGET_SIZE,
                    blur_threshold=BLUR_THRESHOLD
                )
                if cropped_face is None:
                    processed_count += 1
                    continue

                embedding = generate_embedding2(cropped_face)
                emb_path = os.path.join(embedding_person_dir, img_name.rsplit('.', 1)[0] + '.json')
                with open(emb_path, 'w') as f:
                    json.dump({'embedding': embedding.tolist()}, f)

                hv = encode_embedding_to_hv(embedding)
                hv_path = os.path.join(hypervector_person_dir, img_name.rsplit('.', 1)[0] + '_hv.json')
                with open(hv_path, 'w') as f:
                    json.dump(hv.tolist(), f)

                hypervectors.append(hv)
                processed_count += 1

            # Save class hypervector
            if hypervectors:
                summed = np.sum(hypervectors, axis=0)
                threshold = len(hypervectors) / 2
                class_hv = (summed > threshold).astype(np.uint8)

                class_path = os.path.join("class_hypervectors", f"{person_id}_class_hv.json")
                with open(class_path, 'w') as f:
                    json.dump(class_hv.tolist(), f)

        progress.setValue(total_images)
        self.load_all_class_hypervectors()
        QMessageBox.information(self, "Success", "All victims stored to the database.")

    def update_frame(self):
        if not self.camera_running:
            self.stack.setCurrentIndex(1) # show triangle gif
            return
        
        if self.processing:
            self.stack.setCurrentIndex(2)  # show processing image
            return

        if self.showing_match:
            self.stack.setCurrentIndex(0)
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)

        if boxes is not None and len(boxes) > 0:
            # Select largest face
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            idx = int(np.argmax(areas))
            box = boxes[idx].astype(int)

            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check stability by IoU with previous box
            if self.prev_box is not None:
                iou = self.compute_iou(box, self.prev_box)
                if iou > 0.7:
                    self.stable_count += 1
                else:
                    self.stable_count = 0
                    self.face_buffer.clear()
            else:
                self.stable_count = 0
                self.face_buffer.clear()

            self.prev_box = box

            # Capture frames once stable count reached
            if self.stable_count >= STABILITY_THRESHOLD:
                self.face_buffer.append(frame.copy())
                # Limit to CAPTURE_FRAME_COUNT frames
                if len(self.face_buffer) == CAPTURE_FRAME_COUNT:
                    self.processing = True
                    self.result_label.setText("🔄 Processing match... please wait.")
                    QTimer.singleShot(100, self.process_batch)  # Slight delay to update UI
                    return
        else:
            self.prev_box = None
            self.stable_count = 0
            self.face_buffer.clear()
            self.result_label.setText("No face detected.")

        # Show current frame
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)
        self.stack.setCurrentIndex(0)

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def process_batch(self):
        # For each captured frame:
        hypervectors = []
        for frame_bgr in self.face_buffer:
            cropped_face = align_and_crop_face_from_array(
                frame_bgr,
                self.mtcnn,
                padding=PADDING,
                target_size=TARGET_SIZE,
                blur_threshold=BLUR_THRESHOLD
            )
            if cropped_face is None:
                continue

            embedding = generate_embedding2(cropped_face)  # cropped_face is RGB np.ndarray
            hv = encode_embedding_to_hv(embedding)
            hypervectors.append(hv)

        if not hypervectors:
            self.result_label.setText("❌ No valid faces for matching.")
            self.face_buffer.clear()
            self.stable_count = 0
            self.processing = False
            return

        # Average hypervectors by majority vote
        summed = np.sum(hypervectors, axis=0)
        threshold = len(hypervectors) / 2
        class_hv = (summed > threshold).astype(np.uint8)

        label, score = match_hv_to_class(class_hv, self.class_hv_matrix, self.person_ids)

        if label is not None and score > 0.7:
            self.result_label.setStyleSheet("color: lightgreen")
            self.result_label.setText(f"✅ Match found: {label} (Score: {score:.2f})")
            QTimer.singleShot(0, self.play_alert_sound)
            self.logs_page.add_log(label, score)

        else:
            self.result_label.setStyleSheet("color: tomato")
            self.result_label.setText("❌ No match found.")

        # Reset for next capture
        self.face_buffer.clear()
        self.stable_count = 0
        self.processing = False

        # Pause camera feed display while match is shown
        self.showing_match = True
        QTimer.singleShot(MATCH_DISPLAY_TIME_MS, self.resume_feed)

    
    def resume_feed(self):
        self.showing_match = False
        self.result_label.setStyleSheet("color: #aaaaaa")
        self.result_label.setText("Match result will appear here.")


    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

class TensorFlowPreloader(QObject):
    finished = pyqtSignal()

    def start(self):
        Thread(target=self._preload, daemon=True).start()

    def _preload(self):
        _ = tf.constant([0.0]) + tf.constant([0.0])
        self.finished.emit()

if __name__ == '__main__':
    app = QApplication(sys.argv)

    splash_pix = QPixmap("assets/startup.png")
    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setWindowFlag(Qt.FramelessWindowHint)
    splash.show()

    window = CameraApp()

    def start_main_window():
        splash.finish(window)
        window.show()

    preloader = TensorFlowPreloader()
    preloader.finished.connect(start_main_window)
    preloader.start()

    sys.exit(app.exec_())
