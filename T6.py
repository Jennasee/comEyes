import sys
import time
import random
import datetime
import cv2
import numpy as np
import mss
import psutil # For CPU/Memory info
import pyautogui # For screen info
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QPointF, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QFontDatabase
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QGridLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLabel, QTextEdit
import logging
from ultralytics import YOLO
import supervision as sv
import torch
import os

# --- Configurations ---
UPDATE_INTERVAL_MS = 500  # Interval for updating HUD text elements
SCANLINE_SPEED_MS = 15    # Speed of the scanline effect
DETECTION_INTERVAL_MS = 150 # Default interval for running detection
SYSTEM_INFO_INTERVAL_MS = 1000 # Interval for updating system info (CPU, RAM, etc.)
CONFIDENCE_THRESHOLD = 0.4 # Default confidence threshold for detection
NMS_THRESHOLD = 0.3        # Default Non-Maximum Suppression threshold
FONT_NAME = "Press Start 2P" # Preferred retro font
FALLBACK_FONT = "Monospace" # Fallback font if preferred is not found
FONT_SIZE_SMALL = 10
FONT_SIZE_MEDIUM = 12
RED_COLOR = QColor(255, 0, 0) # Primary color for UI elements
TEXT_COLOR = QColor(255, 0, 0) # Color for text

# --- Helper Functions ---
def random_hex(length):
    """Generate a random hexadecimal string of specified length."""
    return ''.join(random.choice('ABCDEF0123456789') for _ in range(length))

def get_gpu_info():
    """Fetches GPU name and memory usage if CUDA is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem, free_mem = torch.cuda.mem_get_info(0)
        used_mem = total_mem - free_mem
        mem_usage = f"{(used_mem / (1024**3)):.1f}/{(total_mem / (1024**3)):.1f} GB"
        return gpu_name, mem_usage
    else:
        return "N/A (CUDA not available)", "N/A"

# --- Custom Logging Handler for QTextEdit ---
class QTextEditLogger(logging.Handler):
    """Sends logging records to a QTextEdit widget."""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        # Use invokeMethod to ensure thread safety when updating GUI from different threads
        QtCore.QMetaObject.invokeMethod(
            self.text_edit,
            "append",
            QtCore.Qt.QueuedConnection,
            QtCore.Q_ARG(str, msg)
        )

# --- Screen Capture Thread ---
class ScreenCaptureThread(QThread):
    """Captures screen frames periodically in a separate thread."""
    frame_ready = Signal(np.ndarray)
    status_update = Signal(str)

    def __init__(self, monitor_spec):
        super().__init__()
        self.monitor_spec = monitor_spec
        self.running = False
        self.sct = None
        self._detection_interval_ms = DETECTION_INTERVAL_MS # Use internal variable

    def run(self):
        self.running = True
        try:
            self.sct = mss.mss()
            logger.info(f"Screen capture started for monitor: {self.monitor_spec}")
            while self.running:
                start_time = time.time()
                try:
                    sct_img = self.sct.grab(self.monitor_spec)
                    frame = np.array(sct_img)
                    # Convert from BGRA to BGR (OpenCV standard)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    self.frame_ready.emit(frame_bgr)
                except mss.ScreenShotError as e:
                    self.status_update.emit(f"Screen capture error: {e}")
                    logger.error(f"Screen capture error: {e}")
                    time.sleep(1) # Wait before retrying
                except Exception as e:
                    self.status_update.emit(f"Unexpected screen capture error: {e}")
                    logger.error(f"Unexpected screen capture error: {e}", exc_info=True)
                    time.sleep(1)

                # Calculate sleep time to maintain the desired interval
                elapsed = time.time() - start_time
                sleep_time = max(0, (self._detection_interval_ms / 1000.0) - elapsed)
                time.sleep(sleep_time)
        finally:
            if self.sct:
                self.sct.close()
            logger.info("Screen capture stopped.")

    def stop(self):
        """Signals the thread to stop."""
        self.running = False
        logger.info("Stopping screen capture thread...")

    @Slot(int)
    def update_detection_interval(self, interval):
        """Updates the interval between frame captures."""
        logger.info(f"Updating capture interval to {interval} ms")
        self._detection_interval_ms = interval

# --- Detection Thread with Ultralytics YOLO ---
class DetectionThread(QThread):
    """Runs object detection on frames in a separate thread."""
    detections_ready = Signal(list) # Emits list of (label, confidence, bbox)
    status_update = Signal(str)     # Emits status messages

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.input_frame = None
        self.running = False
        self.frame_width = 0
        self.frame_height = 0
        self._confidence_threshold = CONFIDENCE_THRESHOLD # Internal variable
        self._nms_threshold = NMS_THRESHOLD           # Internal variable
        self._frame_lock = QtCore.QMutex() # To safely access input_frame

    def load_model(self):
        """Loads the specified YOLO model."""
        self.status_update.emit(f"Loading model {self.model_name}...")
        try:
            self.model = YOLO(self.model_name)
            # Attempt to use GPU if available
            if torch.cuda.is_available():
                self.model.to('cuda')
                device = 'GPU (CUDA)'
            else:
                self.model.to('cpu')
                device = 'CPU'
            logger.info(f"Model '{self.model_name}' loaded successfully on {device}.")
            self.status_update.emit(f"Model '{self.model_name}' loaded on {device}.")
            return True
        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {e}"
            logger.error(error_msg, exc_info=True)
            self.status_update.emit(error_msg)
            return False

    @Slot(np.ndarray)
    def set_frame(self, frame):
        """Receives a new frame for processing."""
        with QtCore.QMutexLocker(self._frame_lock):
            self.input_frame = frame
            # Store frame dimensions once
            if self.frame_width == 0 or self.frame_height == 0:
                self.frame_height, self.frame_width = frame.shape[:2]
                logger.info(f"Received first frame with dimensions: {self.frame_width}x{self.frame_height}")

    def run(self):
        """Main loop for the detection thread."""
        if not self.load_model():
            self.running = False # Stop if model loading failed
            return

        self.running = True
        logger.info("Detection thread started.")
        while self.running:
            frame_to_process = None
            # Safely get the latest frame
            with QtCore.QMutexLocker(self._frame_lock):
                if self.input_frame is not None:
                    frame_to_process = self.input_frame.copy()
                    self.input_frame = None # Consume the frame

            if frame_to_process is not None and self.model is not None:
                start_time = time.time()
                try:
                    # Perform inference
                    results = self.model(
                        frame_to_process,
                        conf=self._confidence_threshold,
                        iou=self._nms_threshold,
                        verbose=False # Reduce console output from YOLO
                    )

                    detections = []
                    # Process results
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = box.conf[0].cpu().numpy()
                            class_id = int(box.cls[0].cpu().numpy())
                            label = self.model.names.get(class_id, f"ID:{class_id}") # Use get for safety
                            detections.append((label, float(confidence), (int(x1), int(y1), int(x2), int(y2))))

                    self.detections_ready.emit(detections)
                    elapsed = time.time() - start_time
                    # logger.debug(f"Detection took {elapsed:.3f}s, found {len(detections)} objects.")

                except Exception as e:
                    error_msg = f"Detection error: {e}"
                    logger.error(error_msg, exc_info=True)
                    self.status_update.emit(error_msg)
                    # Avoid busy-waiting on error
                    self.msleep(50)
            else:
                # Wait briefly if no frame is available
                self.msleep(20)

        logger.info("Detection thread stopped.")

    def stop(self):
        """Signals the thread to stop."""
        self.running = False
        logger.info("Stopping detection thread...")

    @Slot(float)
    def update_confidence_threshold(self, threshold):
        """Updates the confidence threshold for detection."""
        logger.info(f"Updating confidence threshold to {threshold:.2f}")
        self._confidence_threshold = threshold

    @Slot(float)
    def update_nms_threshold(self, threshold):
        """Updates the NMS threshold."""
        logger.info(f"Updating NMS threshold to {threshold:.2f}")
        self._nms_threshold = threshold

# --- Terminator Overlay Widget ---
class TerminatorOverlay(QWidget):
    """The main overlay window displaying HUD elements."""
    def __init__(self, monitor_spec):
        super().__init__()
        self.monitor_spec = monitor_spec
        self.scanline_y = 0
        self.flicker_on = True
        self.power_level = 99.9 # Starting power level
        self.current_detections = []
        self.target_position = None # Current target screen coordinates (x, y)
        self.tracking_status = "SCANNING" # Current tracking status

        # Center crosshair initially
        self.crosshair_position = QPointF(monitor_spec['width'] / 2, monitor_spec['height'] / 2)

        # --- Setup Window ---
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | # Keep on top
            Qt.FramelessWindowHint | # No title bar or borders
            Qt.Tool |                # Prevent appearing in taskbar
            Qt.WindowTransparentForInput # Allow clicks to pass through
        )
        self.setAttribute(Qt.WA_TranslucentBackground) # Make background transparent
        self.setGeometry(
            monitor_spec['left'], monitor_spec['top'],
            monitor_spec['width'], monitor_spec['height']
        )

        # --- Load Font ---
        font_db = QFontDatabase()
        if FONT_NAME in font_db.families():
            self.font_small = QFont(FONT_NAME, FONT_SIZE_SMALL)
            self.font_medium = QFont(FONT_NAME, FONT_SIZE_MEDIUM)
            logger.info(f"Using font: {FONT_NAME}")
        else:
            logger.warning(f"Font '{FONT_NAME}' not found. Using fallback '{FALLBACK_FONT}'.")
            self.font_small = QFont(FALLBACK_FONT, FONT_SIZE_SMALL)
            self.font_medium = QFont(FALLBACK_FONT, FONT_SIZE_MEDIUM)

        # --- Timers ---
        self.scanline_timer = QTimer(self)
        self.scanline_timer.timeout.connect(self.update_scanline)
        self.scanline_timer.start(SCANLINE_SPEED_MS)

        self.hud_update_timer = QTimer(self)
        self.hud_update_timer.timeout.connect(self.update_hud_elements)
        self.hud_update_timer.start(UPDATE_INTERVAL_MS)

        self.crosshair_timer = QTimer(self)
        self.crosshair_timer.timeout.connect(self.update_crosshair)
        self.crosshair_timer.start(16) # Update crosshair smoothly (~60 FPS)

        logger.info("Terminator overlay initialized.")

    def update_scanline(self):
        """Moves the scanline down the screen."""
        self.scanline_y = (self.scanline_y + 5) % self.height()
        self.flicker_on = not self.flicker_on # Toggle flicker effect
        self.update() # Request repaint

    def update_hud_elements(self):
        """Updates dynamic HUD elements like power level."""
        # Simulate power drain
        self.power_level = max(0.0, self.power_level - random.uniform(0.01, 0.05))
        self.update() # Request repaint

    @Slot(list)
    def update_detections(self, detections):
        """Processes new detections and updates the target."""
        self.current_detections = detections # Store for drawing boxes

        if not detections:
            self.target_position = None
            self.tracking_status = "SCANNING"
            # logger.debug("No detections, target lost.")
            return # Exit if no detections

        # Find the detection with the highest confidence
        best_detection = max(detections, key=lambda det: det[1])
        label, confidence, (x1, y1, x2, y2) = best_detection

        # Calculate the center of the best detection's bounding box
        target_center_x = (x1 + x2) / 2
        target_center_y = (y1 + y2) / 2
        self.target_position = (target_center_x, target_center_y)
        self.tracking_status = f"TRACKING: {label.upper()} ({confidence:.1%})"
        # logger.debug(f"Tracking target: {label} at ({target_center_x:.0f}, {target_center_y:.0f})")

        self.update() # Request repaint

    def update_crosshair(self):
        """Moves the crosshair towards the target position."""
        if self.target_position:
            target_x, target_y = self.target_position
            current_x = self.crosshair_position.x()
            current_y = self.crosshair_position.y()

            # Move crosshair smoothly towards the target
            speed = 0.15 # Adjust for desired speed (0.0 to 1.0)
            dx = (target_x - current_x) * speed
            dy = (target_y - current_y) * speed

            # Update crosshair position
            self.crosshair_position.setX(current_x + dx)
            self.crosshair_position.setY(current_y + dy)

            # Small optimization: only repaint if position changed significantly
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                self.update() # Request repaint
        # No need to update if no target

    def paintEvent(self, event):
        """Draws all HUD elements onto the overlay."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing) # Smooth lines/text

        # --- Draw Scanline ---
        if self.flicker_on:
            painter.setPen(QPen(RED_COLOR, 1, Qt.DashLine))
            painter.drawLine(0, self.scanline_y, self.width(), self.scanline_y)

        # --- Draw Border ---
        painter.setPen(QPen(RED_COLOR, 2))
        painter.drawRect(10, 10, self.width() - 20, self.height() - 20) # Inset border

        # --- Draw Text Info (Top Left) ---
        painter.setFont(self.font_small)
        painter.setPen(TEXT_COLOR)
        text_x = 20
        text_y = 30
        line_height = 20
        painter.drawText(text_x, text_y, f"SYS ID: {random_hex(8)}")
        text_y += line_height
        painter.drawText(text_x, text_y, f"TIME: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}") # Milliseconds
        text_y += line_height
        painter.drawText(text_x, text_y, f"PWR: {self.power_level:.1f}%")
        text_y += line_height
        painter.drawText(text_x, text_y, f"STATUS: {self.tracking_status}")

        # --- Draw Detection Boxes ---
        painter.setFont(self.font_small)
        for label, confidence, (x1, y1, x2, y2) in self.current_detections:
            # Draw bounding box
            painter.setPen(QPen(RED_COLOR, 1))
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            # Draw label and confidence
            painter.setPen(TEXT_COLOR)
            painter.drawText(x1, y1 - 5, f"{label} {confidence:.1%}") # Text above box

        # --- Draw Crosshair ---
        painter.setPen(QPen(RED_COLOR, 2))
        size = 25 # Size of the crosshair lines
        x = int(self.crosshair_position.x())
        y = int(self.crosshair_position.y())
        # Draw horizontal and vertical lines centered at crosshair_position
        painter.drawLine(x - size, y, x + size, y)
        painter.drawLine(x, y - size, x, y + size)
        # Optional: Draw a small circle/dot in the center
        # painter.drawEllipse(QPointF(x, y), 3, 3)

        painter.end() # End painting

    def closeEvent(self, event):
        """Clean up timers when the widget is closed."""
        logger.info("Closing overlay widget...")
        self.scanline_timer.stop()
        self.hud_update_timer.stop()
        self.crosshair_timer.stop()
        super().closeEvent(event)

# --- Main Control Window ---
class MainWindow(QMainWindow):
    """Provides controls for starting/stopping and configuring the overlay."""
    # Signals to communicate settings changes to threads
    detection_interval_changed = Signal(int)
    confidence_threshold_changed = Signal(float)
    nms_threshold_changed = Signal(float)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terminator Vision Control Panel")
        self.setGeometry(100, 100, 550, 600) # Position and size

        # Apply basic styling
        self.setStyleSheet("""
            QMainWindow {
                background-color: #111111; /* Dark background */
                color: #ff0000; /* Red text */
            }
            QWidget { /* Apply font to all child widgets */
                font-family: 'Press Start 2P', Monospace;
                font-size: 10pt; /* Slightly larger default font */
            }
            QLabel {
                color: #ff0000;
                padding-top: 5px; /* Add spacing above labels */
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #222222; /* Darker controls */
                color: #ff0000;
                border: 1px solid #ff0000;
                padding: 3px;
                border-radius: 3px; /* Slightly rounded corners */
            }
            QComboBox::drop-down {
                border: 1px solid #ff0000;
            }
            QComboBox QAbstractItemView { /* Style dropdown list */
                background-color: #222222;
                color: #ff0000;
                selection-background-color: #ff0000; /* Red selection */
                selection-color: #000000; /* Black text on selection */
            }
            QPushButton {
                background-color: #ff0000; /* Red button */
                color: #000000; /* Black text */
                border: none;
                padding: 8px 12px; /* More padding */
                margin-top: 10px; /* Space above buttons */
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #cc0000; /* Darker red on hover */
            }
            QPushButton:disabled {
                background-color: #550000; /* Dark red when disabled */
                color: #444444;
            }
            QTextEdit {
                background-color: #000000; /* Black background */
                color: #00ff00; /* Green log text */
                border: 1px solid #ff0000;
                font-family: Monospace; /* Use monospace for logs */
                font-size: 9pt;
                border-radius: 3px;
            }
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget) # Main vertical layout

        # --- Configuration Section ---
        config_layout = QGridLayout() # Use grid for alignment
        self.main_layout.addLayout(config_layout)

        # Monitor Selection
        self.monitor_label = QLabel("Target Display:")
        config_layout.addWidget(self.monitor_label, 0, 0)
        self.monitor_combo = QComboBox()
        config_layout.addWidget(self.monitor_combo, 0, 1)

        # Model Selection
        self.model_label = QLabel("Detection Model:")
        config_layout.addWidget(self.model_label, 1, 0)
        self.model_combo = QComboBox()
        # Add various YOLOv8 models
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.setCurrentText("yolov8n.pt") # Default to nano
        config_layout.addWidget(self.model_combo, 1, 1)

        # Detection Interval
        self.detection_interval_label = QLabel("Detect Interval (ms):")
        config_layout.addWidget(self.detection_interval_label, 2, 0)
        self.detection_interval_spin = QSpinBox()
        self.detection_interval_spin.setMinimum(20) # Min interval
        self.detection_interval_spin.setMaximum(2000) # Max interval
        self.detection_interval_spin.setSingleStep(10)
        self.detection_interval_spin.setValue(DETECTION_INTERVAL_MS)
        config_layout.addWidget(self.detection_interval_spin, 2, 1)

        # Confidence Threshold
        self.confidence_threshold_label = QLabel("Confidence Thresh:")
        config_layout.addWidget(self.confidence_threshold_label, 3, 0)
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setMinimum(0.05)
        self.confidence_threshold_spin.setMaximum(0.95)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.setDecimals(2)
        self.confidence_threshold_spin.setValue(CONFIDENCE_THRESHOLD)
        config_layout.addWidget(self.confidence_threshold_spin, 3, 1)

        # NMS Threshold
        self.nms_threshold_label = QLabel("NMS Thresh:")
        config_layout.addWidget(self.nms_threshold_label, 4, 0)
        self.nms_threshold_spin = QDoubleSpinBox()
        self.nms_threshold_spin.setMinimum(0.1)
        self.nms_threshold_spin.setMaximum(0.9)
        self.nms_threshold_spin.setSingleStep(0.05)
        self.nms_threshold_spin.setDecimals(2)
        self.nms_threshold_spin.setValue(NMS_THRESHOLD)
        config_layout.addWidget(self.nms_threshold_spin, 4, 1)

        # --- Control Buttons ---
        button_layout = QtWidgets.QHBoxLayout() # Horizontal layout for buttons
        self.main_layout.addLayout(button_layout)
        self.start_button = QPushButton("START VISION")
        self.start_button.clicked.connect(self.start_overlay)
        button_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("STOP VISION")
        self.stop_button.clicked.connect(self.stop_overlay)
        self.stop_button.setEnabled(False) # Disabled initially
        button_layout.addWidget(self.stop_button)

        # --- System Information Section ---
        sys_info_layout = QGridLayout()
        self.main_layout.addLayout(sys_info_layout)

        sys_info_layout.addWidget(QLabel("--- System Status ---"), 0, 0, 1, 2, alignment=Qt.AlignCenter)

        self.cpu_label = QLabel("CPU Usage:")
        self.cpu_value = QLabel("N/A")
        sys_info_layout.addWidget(self.cpu_label, 1, 0)
        sys_info_layout.addWidget(self.cpu_value, 1, 1)

        self.mem_label = QLabel("Memory Usage:")
        self.mem_value = QLabel("N/A")
        sys_info_layout.addWidget(self.mem_label, 2, 0)
        sys_info_layout.addWidget(self.mem_value, 2, 1)

        self.gpu_label = QLabel("GPU:")
        self.gpu_value = QLabel("N/A")
        sys_info_layout.addWidget(self.gpu_label, 3, 0)
        sys_info_layout.addWidget(self.gpu_value, 3, 1)

        self.gpu_mem_label = QLabel("GPU Memory:")
        self.gpu_mem_value = QLabel("N/A")
        sys_info_layout.addWidget(self.gpu_mem_label, 4, 0)
        sys_info_layout.addWidget(self.gpu_mem_value, 4, 1)

        self.screen_label = QLabel("Screen Res:")
        self.screen_value = QLabel("N/A")
        sys_info_layout.addWidget(self.screen_label, 5, 0)
        sys_info_layout.addWidget(self.screen_value, 5, 1)

        self.tracking_info_label = QLabel("Tracking:")
        self.tracking_info_value = QLabel("INACTIVE")
        sys_info_layout.addWidget(self.tracking_info_label, 6, 0)
        sys_info_layout.addWidget(self.tracking_info_value, 6, 1)

        # --- Status Bar ---
        self.status_label = QLabel("STATUS: Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid #ff0000;")
        self.main_layout.addWidget(self.status_label)

        # --- Log Output ---
        self.log_label = QLabel("Log Output:")
        self.main_layout.addWidget(self.log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.main_layout.addWidget(self.log_text) # Add log display

        # --- Populate Monitors ---
        try:
            with mss.mss() as sct:
                # Filter out the 'all monitors' option if present
                self.monitors = [m for m in sct.monitors if m.get('width') and m.get('height')]
                if not self.monitors:
                     raise ValueError("No usable monitors found by mss.")
                # Remove the primary 'all' monitor entry if it exists (often index 0)
                if self.monitors[0]['left'] == 0 and self.monitors[0]['top'] == 0:
                    primary_width = pyautogui.size().width
                    primary_height = pyautogui.size().height
                    # Check if the first monitor matches the primary screen size exactly
                    if self.monitors[0]['width'] >= primary_width and self.monitors[0]['height'] >= primary_height:
                         # Heuristic: If the first monitor covers the primary, assume it's the 'all' monitor
                         logger.info("Excluding potential 'all monitors' entry from mss list.")
                         self.monitors = self.monitors[1:]


            if not self.monitors:
                raise ValueError("No individual monitors found after filtering.")

            for i, monitor in enumerate(self.monitors):
                label = f"Display {i+1}: {monitor['width']}x{monitor['height']} @ ({monitor['left']},{monitor['top']})"
                self.monitor_combo.addItem(label, userData=monitor) # Store monitor dict as userData
            logger.info(f"Found {len(self.monitors)} monitor(s).")
            self.monitor_combo.setCurrentIndex(0) # Select first monitor by default

        except Exception as e:
            error_msg = f"Error initializing monitors: {e}"
            logger.error(error_msg, exc_info=True)
            self.status_label.setText(error_msg)
            self.start_button.setEnabled(False)
            self.monitor_combo.addItem("Error: No monitors detected")

        # --- Connect Signals ---
        self.detection_interval_spin.valueChanged.connect(self.on_detection_interval_changed)
        self.confidence_threshold_spin.valueChanged.connect(self.on_confidence_threshold_changed)
        self.nms_threshold_spin.valueChanged.connect(self.on_nms_threshold_changed)

        # --- System Info Timer ---
        self.sys_info_timer = QTimer(self)
        self.sys_info_timer.timeout.connect(self.update_system_info)
        self.sys_info_timer.start(SYSTEM_INFO_INTERVAL_MS)
        self.update_system_info() # Initial update

        # --- Initialize Threads and Overlay ---
        self.capture_thread = None
        self.detection_thread = None
        self.overlay = None

        logger.info("Main window initialized.")

    # --- Slot Methods for Settings Changes ---
    @Slot(int)
    def on_detection_interval_changed(self, value):
        self.detection_interval_changed.emit(value)
        logger.debug(f"GUI emitted detection_interval_changed: {value}")

    @Slot(float)
    def on_confidence_threshold_changed(self, value):
        self.confidence_threshold_changed.emit(value)
        logger.debug(f"GUI emitted confidence_threshold_changed: {value:.2f}")

    @Slot(float)
    def on_nms_threshold_changed(self, value):
        self.nms_threshold_changed.emit(value)
        logger.debug(f"GUI emitted nms_threshold_changed: {value:.2f}")

    # --- Start/Stop Methods ---
    def start_overlay(self):
        """Starts the screen capture, detection, and overlay."""
        monitor_data = self.monitor_combo.currentData()
        if not monitor_data:
            self.update_status("ERROR: Please select a valid monitor.")
            logger.error("Start attempt failed: No monitor selected.")
            return

        monitor_spec = monitor_data
        model_name = self.model_combo.currentText()
        logger.info(f"Starting overlay on monitor: {monitor_spec} with model: {model_name}")
        self.update_status(f"Initializing ({model_name})...")

        # --- Create Threads and Overlay ---
        try:
            self.capture_thread = ScreenCaptureThread(monitor_spec)
            self.detection_thread = DetectionThread(model_name)
            self.overlay = TerminatorOverlay(monitor_spec) # Create overlay instance
        except Exception as e:
            error_msg = f"Failed to create threads/overlay: {e}"
            logger.error(error_msg, exc_info=True)
            self.update_status(f"ERROR: {error_msg}")
            self.stop_overlay() # Clean up any partial setup
            return

        # --- Connect Signals between Components ---
        # Capture -> Detection
        self.capture_thread.frame_ready.connect(self.detection_thread.set_frame)
        # Detection -> Overlay
        self.detection_thread.detections_ready.connect(self.overlay.update_detections)
        # Detection -> Main Window (for tracking status)
        self.detection_thread.detections_ready.connect(self.update_tracking_info)
        # Status Updates -> Main Window
        self.detection_thread.status_update.connect(self.update_status)
        self.capture_thread.status_update.connect(self.update_status)
        # Settings -> Threads
        self.detection_interval_changed.connect(self.capture_thread.update_detection_interval)
        self.confidence_threshold_changed.connect(self.detection_thread.update_confidence_threshold)
        self.nms_threshold_changed.connect(self.detection_thread.update_nms_threshold)

        # --- Emit Initial Settings ---
        # Ensure threads get the current values *before* starting
        self.on_detection_interval_changed(self.detection_interval_spin.value())
        self.on_confidence_threshold_changed(self.confidence_threshold_spin.value())
        self.on_nms_threshold_changed(self.nms_threshold_spin.value())

        # --- Start Threads ---
        # Start detection first (it needs to load the model)
        self.detection_thread.start()
        # Give detection thread a moment to start loading model before capture begins
        QTimer.singleShot(500, self.capture_thread.start)

        # --- Show Overlay ---
        self.overlay.show()

        # --- Update GUI State ---
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.monitor_combo.setEnabled(False) # Prevent changing monitor while running
        self.model_combo.setEnabled(False)   # Prevent changing model while running
        self.update_status("Overlay starting...")
        logger.info("Overlay components started.")

    def stop_overlay(self):
        """Stops the threads and hides the overlay."""
        logger.info("Stopping overlay components...")
        self.update_status("Stopping...")

        # Stop threads safely
        if hasattr(self, 'capture_thread') and self.capture_thread and self.capture_thread.isRunning():
            self.capture_thread.stop()
            if not self.capture_thread.wait(1500): # Wait 1.5 sec
                 logger.warning("Capture thread did not stop gracefully.")
                 self.capture_thread.terminate() # Force terminate if needed

        if hasattr(self, 'detection_thread') and self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            if not self.detection_thread.wait(2500): # Wait 2.5 sec
                logger.warning("Detection thread did not stop gracefully.")
                self.detection_thread.terminate() # Force terminate

        # Close overlay window
        if hasattr(self, 'overlay') and self.overlay:
            self.overlay.close() # Use close() to trigger closeEvent

        # Clean up references
        self.capture_thread = None
        self.detection_thread = None
        self.overlay = None

        # --- Update GUI State ---
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.monitor_combo.setEnabled(True) # Re-enable controls
        self.model_combo.setEnabled(True)
        self.update_status("STATUS: Stopped")
        self.tracking_info_value.setText("INACTIVE")
        logger.info("Overlay components stopped.")

    # --- Slot for Status Updates ---
    @Slot(str)
    def update_status(self, message):
        """Updates the status label and logs the message."""
        # Prepend "STATUS: " if not already present for clarity
        display_message = message if message.lower().startswith("status:") else f"STATUS: {message}"
        self.status_label.setText(display_message)
        # Avoid logging redundant status updates if they come frequently
        # logger.info(message) # Optionally log every status update

    @Slot(list)
    def update_tracking_info(self, detections):
        """Updates the tracking info label based on detections."""
        if self.overlay and self.overlay.isVisible(): # Only update if overlay is active
            self.tracking_info_value.setText(self.overlay.tracking_status)
        else:
            self.tracking_info_value.setText("INACTIVE")

    # --- System Info Update Method ---
    def update_system_info(self):
        """Fetches and displays system information."""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent()
            self.cpu_value.setText(f"{cpu_percent:.1f}%")

            # Memory Usage
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            mem_gb = f"{(mem.used / (1024**3)):.1f} / {(mem.total / (1024**3)):.1f} GB"
            self.mem_value.setText(f"{mem_percent:.1f}% ({mem_gb})")

            # GPU Info (if available)
            gpu_name, gpu_mem = get_gpu_info()
            self.gpu_value.setText(gpu_name)
            self.gpu_mem_value.setText(gpu_mem)

            # Screen Resolution (Primary Monitor)
            try:
                screen_width, screen_height = pyautogui.size()
                self.screen_value.setText(f"{screen_width} x {screen_height}")
            except Exception as e:
                logger.warning(f"Could not get screen size via pyautogui: {e}")
                self.screen_value.setText("N/A")

        except Exception as e:
            logger.error(f"Error updating system info: {e}", exc_info=True)
            # Optionally clear fields or show error indication
            self.cpu_value.setText("Error")
            self.mem_value.setText("Error")
            # Keep GPU/Screen as they might be fetched differently


    # --- Cleanup on Close ---
    def closeEvent(self, event):
        """Ensures threads are stopped when the main window is closed."""
        logger.info("Main window closing. Stopping overlay...")
        self.stop_overlay() # Call the stop function
        self.sys_info_timer.stop() # Stop the system info timer
        super().closeEvent(event) # Proceed with closing

# --- Main Execution ---
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, # Set base level
                        format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()]) # Log to console by default

    logger = logging.getLogger() # Get root logger

    # --- Application Setup ---
    app = QApplication(sys.argv)

    # --- Font Loading ---
    font_path = os.path.join(os.path.dirname(__file__), "PressStart2P-Regular.ttf") # Assuming font file is nearby
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                logger.info(f"Successfully loaded font: {families[0]}")
            else:
                logger.warning(f"Loaded font file '{font_path}' but no families found.")
        else:
            logger.error(f"Failed to load font file: {font_path}")
    else:
        # Check if font is installed system-wide
        font_db = QFontDatabase()
        if FONT_NAME not in font_db.families():
             logger.warning(f"Font file '{font_path}' not found and '{FONT_NAME}' not installed system-wide. Will use fallback.")


    # --- Create and Show Main Window ---
    main_window = MainWindow()

    # Add the QTextEdit logger handler
    log_handler = QTextEditLogger(main_window.log_text)
    logger.addHandler(log_handler)

    main_window.show()

    # Ensure stop_overlay is called when application quits
    app.aboutToQuit.connect(main_window.stop_overlay)

    # --- Run Application ---
    try:
        exit_code = app.exec()
        logger.info(f"Application finished with exit code {exit_code}.")
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Unhandled application error: {e}", exc_info=True)
        sys.exit(1) # Exit with error code
