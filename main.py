import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import sys
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer

from facenet_pytorch import MTCNN

from MTCNN_preprocess_gpu import align_and_crop_face_from_array
from embedder_512D import generate_embedding2
from matcher import encode_embedding_to_hv
from matcher import match_hv_to_class

# Constants
PADDING = 10
TARGET_SIZE = (160, 160)
BLUR_THRESHOLD = 40.0
STABILITY_THRESHOLD = 5  # Number of stable frames before capture

class CameraApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition with Auto Capture")
        self.resize(800, 650)

        self.image_label = QLabel()
        self.result_label = QLabel("Match result will appear here.")
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.quit_button)
        self.setLayout(layout)

        # Setup webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # Setup MTCNN on GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=device)

        # For face stability check
        self.prev_box = None
        self.stable_count = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~30 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces and landmarks
        boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)

        if boxes is not None and len(boxes) > 0:
            # Select largest face
            areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
            idx = int(np.argmax(areas))
            box = boxes[idx].astype(int)

            # Draw bounding box on RGB frame
            x1, y1, x2, y2 = box
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check face stability (IoU or position difference)
            if self.prev_box is not None:
                iou = self.compute_iou(box, self.prev_box)
                if iou > 0.7:
                    self.stable_count += 1
                else:
                    self.stable_count = 0
            else:
                self.stable_count = 0

            self.prev_box = box

            # If stable for enough frames, run matcher
            if self.stable_count >= STABILITY_THRESHOLD:
                label, score = self.run_matcher(frame)
                if label is not None:
                    self.result_label.setText(f"Match: {label} (Score: {score:.3f})")
                else:
                    self.result_label.setText("No match found.")
                self.stable_count = 0  # reset after capture

        else:
            self.prev_box = None
            self.stable_count = 0
            self.result_label.setText("No face detected.")

        # Convert back to BGR for Qt display
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_bgr.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def compute_iou(self, boxA, boxB):
        # Compute intersection over union for two boxes
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def run_matcher(self, frame_bgr):
        # Align & crop largest face from frame BGR
        cropped_face = align_and_crop_face_from_array(
            frame_bgr,
            self.mtcnn,
            padding=PADDING,
            target_size=TARGET_SIZE,
            blur_threshold=BLUR_THRESHOLD
        )
        if cropped_face is None:
            print("No usable face found for matching.")
            return None, None

        # Generate embedding
        embedding = generate_embedding2(cropped_face)  # cropped_face is RGB np.ndarray

        # Encode to hypervector
        query_hv = encode_embedding_to_hv(embedding)

        # Match against class hypervectors
        label, score = match_hv_to_class(query_hv)
        return label, score

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
