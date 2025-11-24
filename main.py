import sys
import cv2
import numpy as np
import time
import os
import pyttsx3
import threading
import queue
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from google.cloud import vision
from google.cloud.vision_v1 import types
from ultralytics import YOLO  # Use YOLOv8 models

# -----------------------------
# NOTE: Secret credentials removed.
# Set your Google Vision credentials via environment variable before running:
# export GOOGLE_APPLICATION_CREDENTIALS="path/to/your_credentials.json"
# -----------------------------

# Load YOLO models
model = YOLO('product.pt')  # Path to trained YOLO model (product mode)

# Directory to save cropped images
output_dir = 'output_crops'
os.makedirs(output_dir, exist_ok=True)

# Initialize TTS engine and queue
tts_engine = pyttsx3.init()
tts_queue = queue.Queue()

def tts_worker():
    """Background worker that reads text from a queue and speaks it."""
    while True:
        text = tts_queue.get()
        if text is None:
            break
        tts_engine.say(text)
        tts_engine.runAndWait()
        tts_queue.task_done()

tts_thread = threading.Thread(target=tts_worker)
tts_thread.daemon = True
tts_thread.start()

# Google Vision OCR function
def google_vision_ocr(image):
    """Run Google Vision OCR on a numpy image and return extracted text."""
    client = vision.ImageAnnotatorClient()

    # Encode image in memory and send to Google Vision API
    if isinstance(image, np.ndarray):
        success, encoded_image = cv2.imencode('.jpg', image)
        content = encoded_image.tobytes()
        image = types.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        if texts:
            return texts[0].description  # Return full detected text
    return ""

# OCR + keyword extraction (product tag info)
def extract_tag_info(cropped_image):
    """Extract a short product description from the tag region."""
    text = google_vision_ocr(cropped_image)
    words = text.split()
    product_info = " ".join(words[:20])  # Take first 20 words
    return product_info

def extract_nutri_info(cropped_image):
    """Extract full nutrition text from the nutrition region."""
    return google_vision_ocr(cropped_image)

# Compute image sharpness using Laplacian variance
def calculate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# Crop a bounding box region from an image
def crop_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]


def process_detections(frame, detections, alpha=0.6, conf_threshold=0.80, sharpness_threshold=300):
    """
    Select the best crop based on confidence and sharpness.

    alpha: weight between confidence and sharpness (0.0 <= alpha <= 1.0)
    sharpness_threshold: minimum sharpness to accept a crop
    """
    best_crop = None
    best_score = -1

    for detection in detections:
        box = detection[:4].tolist()
        conf = detection[4].item()

        # Filter out low-confidence detections
        if conf < conf_threshold:
            continue

        cropped = crop_box(frame, box)

        # Compute sharpness and filter out blurry crops
        sharpness = calculate_sharpness(cropped)
        if sharpness < sharpness_threshold:
            continue

        # Combined score
        score = alpha * conf + (1 - alpha) * sharpness

        # Update best crop
        if score > best_score:
            best_score = score
            best_crop = cropped

    return best_crop, best_score

class VideoApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

        # Load YOLO models
        self.model = YOLO('corner.pt')    # Corner mode YOLO model
        self.product_model = YOLO('product.pt')  # Product mode YOLO model
        self.current_model = self.model  # Active model

        # Move models to GPU
        self.model.to('cuda')
        self.product_model.to('cuda')

        # Input video path
        self.video_path = 'test4.MOV'
        self.cap = cv2.VideoCapture(self.video_path)

        # Initial variables
        self.prev_frame = None
        self.pixel_diffs = []            # Store frame-to-frame diff values
        self.CALIBRATION_FRAMES = 30     # Number of frames for calibration

        self.no_movement_start = None

        # Frame update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        # Start processing after a short delay
        self.start_timer = QTimer(self)
        self.start_timer.setSingleShot(True)
        self.start_timer.timeout.connect(self.start_processing)
        self.start_timer.start(1000)  # Wait 1 second

        # TTS timer
        self.tts_timer = QTimer(self)
        self.tts_timer.setInterval(500)
        self.tts_timer.setSingleShot(True)
        self.tts_timer.timeout.connect(self.read_tts_queue)

        # Delay timer for switching back to product mode
        self.product_mode_delay_timer = QTimer(self)
        self.product_mode_delay_timer.setInterval(8000)
        self.product_mode_delay_timer.setSingleShot(True)
        self.product_mode_delay_timer.timeout.connect(self.enable_product_mode)

        # Crop storage
        self.tag_crops = []
        self.nutri_crops = []

        # Detection flags
        self.tag_found = False
        self.nutri_found = False
        self.hand_detected = False

        # Frame counter for product mode
        self.frame_count = 0

        # Whether switching to product mode is allowed
        self.can_switch_to_product_mode = True

        # Corner TTS read flag
        self.corner_read = False

        # Thresholds
        self.SHARPNESS_THRESHOLD = 0
        self.NO_MOVEMENT_DURATION = 0.5

        # Original crops (before any drawing)
        self.original_tag_crops = []
        self.original_nutri_crops = []

    def start_processing(self):
        """Start reading and processing video frames."""
        self.timer.start(10)

    def enable_product_mode(self):
        """Re-enable product mode switching after cooldown."""
        self.can_switch_to_product_mode = True
        self.hand_detected = False
        self.corner_read = False
        print("corner_read reset to False")

    def read_tts_queue(self):
        """Read one text from queue and speak it."""
        if not tts_queue.empty():
            text = tts_queue.get()
            print(f"TTS: {text}")  # Debug
            tts_thread = threading.Thread(target=self.tts_speak, args=(text,))
            tts_thread.start()

    def tts_speak(self, text):
        """Speak text immediately and restart the timer."""
        tts_engine.say(text)
        tts_engine.runAndWait()
        self.tts_timer.start()

    def initUI(self):
        """Initialize the UI layout."""
        self.setWindowTitle('ViewMate')
        self.setGeometry(100, 100, 1280, 960)

        self.original_label = QLabel(self)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(640, 480)

        self.detected_label = QLabel(self)
        self.detected_label.setAlignment(Qt.AlignCenter)
        self.detected_label.setFixedSize(640, 480)

        self.tag_crop_label = QLabel(self)
        self.tag_crop_label.setAlignment(Qt.AlignCenter)
        self.tag_crop_label.setFixedSize(320, 240)

        self.nutri_crop_label = QLabel(self)
        self.nutri_crop_label.setAlignment(Qt.AlignCenter)
        self.nutri_crop_label.setFixedSize(320, 240)

        self.ocr_result = QTextEdit(self)
        self.ocr_result.setReadOnly(True)
        self.ocr_result.setFixedSize(320, 240)

        layout = QHBoxLayout()
        layout.addWidget(self.original_label)
        layout.addWidget(self.detected_label)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.tag_crop_label)
        right_layout.addWidget(self.nutri_crop_label)
        right_layout.addWidget(self.ocr_result)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)

    def update_frame(self):
        """Switch processing logic based on current mode."""
        if self.current_model == self.product_model:
            self.process_product_mode()
        else:
            self.process_corner_mode()

    def process_corner_mode(self):
        """Corner recognition mode."""
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        self.display_frame(frame, self.original_label)

        frame_width = frame.shape[1]
        left_boundary = frame_width // 3
        right_boundary = 2 * frame_width // 3

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_frame is not None:
            sharpness = calculate_sharpness(frame)

            if sharpness > self.SHARPNESS_THRESHOLD:
                if self.no_movement_start is None:
                    self.no_movement_start = time.time()
                elif time.time() - self.no_movement_start >= self.NO_MOVEMENT_DURATION:
                    print("Camera sharpness is high for the required duration. Running detection...")

                    results = self.current_model(frame, verbose=False)
                    self.detect_objects(frame, results)

                    self.no_movement_start = None
            else:
                self.no_movement_start = None

        self.prev_frame = gray_frame

        # Switch to product mode if hand detected
        if self.hand_detected and self.can_switch_to_product_mode:
            print("Switching to product mode")
            self.current_model = self.product_model
            self.frame_count = 0
            self.tag_found = False
            self.nutri_found = False
            self.hand_detected = False

    def process_product_mode(self):
        """Product info reading mode."""
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            return

        self.display_frame(frame, self.original_label)

        results = self.current_model(frame, verbose=False)
        self.detect_objects(frame, results)

        if self.frame_count > 0:
            self.frame_count += 1

        # After enough frames, pick best crops and run OCR
        if self.frame_count >= 120 and self.tag_found and self.nutri_found:
            best_tag = self.select_best_crop(self.original_tag_crops)
            best_nutri = self.select_best_crop(self.original_nutri_crops)

            if best_tag and best_nutri:
                self.display_crop(best_tag[0], self.tag_crop_label)
                tag_info = extract_tag_info(best_tag[0])
                self.ocr_result.append("\nProduct Info:\n" + tag_info)
                tts_queue.put("Product info: " + " ".join(tag_info.split()[:5]))

                self.display_crop(best_nutri[0], self.nutri_crop_label)
                nutri_info = extract_nutri_info(best_nutri[0])
                self.ocr_result.append("\nNutrition Info:\n" + nutri_info)
                tts_queue.put("Nutrition info: " + " ".join(nutri_info.split()[:5]))

                # Switch back to corner mode and disable product switching temporarily
                self.current_model = self.model
                self.can_switch_to_product_mode = False
                self.product_mode_delay_timer.start()
                print("Switching back to corner mode")

                # Reset crops after OCR
                self.tag_crops = []
                self.nutri_crops = []
                self.original_tag_crops = []
                self.original_nutri_crops = []

    def display_frame(self, frame, label):
        """Convert frame to QImage and display it on a QLabel."""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        scaled_qt_image = qt_image.scaled(label.size(), Qt.KeepAspectRatio)
        label.setPixmap(QPixmap.fromImage(scaled_qt_image))

    def detect_objects(self, frame, results, max_crops=10, alpha=0.5):
        """Process YOLO detections and handle crops/OCR/TTS."""

        for result in results:
            boxes = result.boxes.data
            for box in boxes:
                x1, y1, x2, y2, conf, class_id = box[:6]
                class_id = int(class_id)
                label = self.current_model.names[class_id]

                if conf < 0.4:
                    continue

                # Save original crop before drawing
                original_crop = frame[int(y1):int(y2), int(x1):int(x2)]

                if label == 'tag':
                    self.tag_found = True
                    processed, score = process_detections(frame, [box], alpha)
                    if processed is not None:
                        self.tag_crops.append((processed, score))
                        self.original_tag_crops.append((original_crop.copy(), score))
                    if self.frame_count == 0:
                        self.frame_count = 1

                elif label == 'nutri':
                    self.nutri_found = True
                    processed, score = process_detections(frame, [box], alpha)
                    if processed is not None:
                        self.nutri_crops.append((processed, score))
                        self.original_nutri_crops.append((original_crop.copy(), score))
                    if self.frame_count == 0:
                        self.frame_count = 1

                elif label == 'hand':
                    self.hand_detected = True

                elif label in ['snack', 'beverage', 'instant', 'ramen']:
                    if not self.corner_read:
                        tts_queue.put(f"{label} corner")
                        self.read_tts_queue()
                        self.corner_read = True
                        print("corner_read set to True")

                print(f"Detected {label}")

                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                text = f"{label} ({conf:.2f})"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                text_x = int(x1)
                text_y = int(y1) + text_size[1] + 20 if int(y1) + text_size[1] + 20 < frame.shape[0] else int(y1) - 10
                cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                # Display detection frame
                self.display_frame(frame, self.detected_label)

    def display_crop(self, crop, label):
        """Display a cropped image on a QLabel."""
        if isinstance(crop, np.ndarray):
            rgb_image = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            scaled_qt_image = qt_image.scaled(label.size(), Qt.KeepAspectRatio)
            label.setPixmap(QPixmap.fromImage(scaled_qt_image))

    def select_best_crop(self, crops):
        """Select the crop with the highest score."""
        best_crop = None
        for crop, score in crops:
            if best_crop is None or score > best_crop[1]:
                best_crop = (crop, score)
        return best_crop

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoApp()
    ex.show()
    sys.exit(app.exec_())
