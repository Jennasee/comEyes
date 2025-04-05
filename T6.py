import sys
import time
import random
import datetime
import cv2
import numpy as np
import mss
import psutil
import pyautogui
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, QTimer, QPointF, QThread, Signal, Slot, QMutex, QMutexLocker, QObject # Added QObject for setter method property access
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QFontDatabase, QImage, QPolygonF
from PySide6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QGridLayout, QComboBox, QSpinBox, QDoubleSpinBox, QPushButton, QLabel, QTextEdit, QCheckBox, QGroupBox # Added QGroupBox
import logging
from ultralytics import YOLO
import torch
import os
import collections # Added for deque

# --- Configurations ---
UPDATE_INTERVAL_MS = 500  # Interval for updating HUD text elements
SCANLINE_SPEED_MS = 15    # Speed of the scanline effect
DETECTION_INTERVAL_MS = 1 # Default interval for running detection
SYSTEM_INFO_INTERVAL_MS = 1000 # Interval for updating system info
CONFIDENCE_THRESHOLD = 0.4 # Default confidence threshold for detection
NMS_THRESHOLD = 0.3        # Default NMS threshold
FONT_NAME = "Press Start 2P"
FALLBACK_FONT = "Monospace"
FONT_SIZE_SMALL = 10
FONT_SIZE_MEDIUM = 12
RED_COLOR = QColor(255, 0, 0)
TEXT_COLOR = QColor(255, 0, 0)
PATH_HISTORY_LENGTH = 50     # Number of past points to store for path tracking
TRAJECTORY_PREDICTION_POINTS = 5 # Number of recent points to use for velocity calculation
TRAJECTORY_PREDICTION_DURATION = 0.5 # Seconds into the future to predict

# --- Helper Functions ---
def random_hex(length):
    """Generate a random hexadecimal string of specified length."""
    return ''.join(random.choice('ABCDEF0123456789') for _ in range(length))

def get_gpu_info():
    """Fetches GPU name and memory usage if CUDA is available."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        try:
            # Use mem_get_info for newer PyTorch versions
            total_mem, free_mem = torch.cuda.mem_get_info(0)
            used_mem = total_mem - free_mem
            mem_usage = f"{(used_mem / (1024**3)):.1f}/{(total_mem / (1024**3)):.1f} GB"
        except AttributeError:
            # Fallback for older PyTorch versions or potential issues
            total_mem = torch.cuda.get_device_properties(0).total_memory
            used_mem = torch.cuda.memory_allocated(0)
            mem_usage = f"{(used_mem / (1024**3)):.1f}/{(total_mem / (1024**3)):.1f} GB (Allocated)"
        except Exception:
             mem_usage = "N/A (Error reading memory)" # Catch other potential errors
        return gpu_name, mem_usage
    return "N/A (CUDA not available)", "N/A"

# --- Custom Logging Handler ---
class QTextEditLogger(logging.Handler):
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    def emit(self, record):
        msg = self.format(record)
        # Use invokeMethod to ensure thread-safety when updating GUI from other threads
        QtCore.QMetaObject.invokeMethod(
            self.text_edit,
            "append",
            QtCore.Qt.QueuedConnection, # Ensure execution in the GUI thread's event loop
            QtCore.Q_ARG(str, msg)
        )

# --- Screen Capture Thread ---
class ScreenCaptureThread(QThread):
    frame_ready = Signal(np.ndarray)
    status_update = Signal(str)

    def __init__(self, monitor_spec):
        super().__init__()
        self.monitor_spec = monitor_spec
        self.running = False
        self.sct = None
        self._detection_interval_ms = DETECTION_INTERVAL_MS
        self._lock = QMutex() # Mutex for thread-safe access to interval

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
                    # Ensure frame has 3 channels (BGR) - some captures might be BGRA
                    if frame.shape[2] == 4:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    elif frame.shape[2] == 3:
                         frame_bgr = frame # Already BGR
                    else:
                         # Handle unexpected frame format if necessary
                         logger.warning(f"Unexpected frame channel count: {frame.shape[2]}")
                         continue # Skip this frame
                    self.frame_ready.emit(frame_bgr)
                except mss.ScreenShotError as e:
                    self.status_update.emit(f"Screen capture error: {e}")
                    logger.error(f"Screen capture error: {e}")
                    time.sleep(1) # Wait before retrying on error
                except Exception as e:
                    self.status_update.emit(f"Unexpected screen capture error: {e}")
                    logger.error(f"Unexpected screen capture error: {e}", exc_info=True)
                    time.sleep(1)

                # Calculate sleep time based on the desired interval
                elapsed = time.time() - start_time
                with QMutexLocker(self._lock):
                    interval_sec = self._detection_interval_ms / 1000.0
                sleep_time = max(0, interval_sec - elapsed)
                # Use QThread's msleep for better precision if needed, but time.sleep is often sufficient
                time.sleep(sleep_time)
        finally:
            if self.sct:
                self.sct.close()
            logger.info("Screen capture stopped.")

    def stop(self):
        self.running = False

    @Slot(int)
    def update_detection_interval(self, interval):
        with QMutexLocker(self._lock):
            logger.info(f"Updating capture interval to {interval} ms")
            self._detection_interval_ms = interval

# --- Detection Thread ---
class DetectionThread(QThread):
    detections_ready = Signal(list) # List of (label, confidence, box_tuple)
    status_update = Signal(str)

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.input_frame = None
        self.running = False
        self._enabled = True
        self._confidence_threshold = CONFIDENCE_THRESHOLD
        self._nms_threshold = NMS_THRESHOLD
        self._frame_lock = QMutex() # Use QMutex for thread safety with frame data

    def load_model(self):
        self.status_update.emit(f"Loading model {self.model_name}...")
        try:
            self.model = YOLO(self.model_name)
            if torch.cuda.is_available():
                self.model.to('cuda')
                device = 'GPU (CUDA)'
            else:
                self.model.to('cpu')
                device = 'CPU'
            logger.info(f"Model '{self.model_name}' loaded on {device}.")
            self.status_update.emit(f"Model '{self.model_name}' loaded on {device}.")
            return True
        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {e}"
            logger.error(error_msg, exc_info=True)
            self.status_update.emit(error_msg)
            return False

    @Slot(np.ndarray)
    def set_frame(self, frame):
        # Safely update the frame using a mutex
        with QMutexLocker(self._frame_lock):
            self.input_frame = frame.copy() # Copy to avoid issues if the source array changes

    def run(self):
        if not self.load_model():
            self.running = False # Stop if model loading fails
            return
        self.running = True
        logger.info("Detection thread started.")
        while self.running:
            if self._enabled:
                frame_to_process = None
                # Safely get the frame to process
                with QMutexLocker(self._frame_lock):
                    if self.input_frame is not None:
                        frame_to_process = self.input_frame
                        self.input_frame = None # Consume the frame

                if frame_to_process is not None and self.model is not None:
                    try:
                        # Perform inference
                        results = self.model(
                            frame_to_process,
                            conf=self._confidence_threshold,
                            iou=self._nms_threshold,
                            verbose=False # Reduce console spam
                        )
                        detections = []
                        # Process results (assuming results[0] contains the detections for the single image)
                        if results and results[0]:
                            boxes = results[0].boxes
                            for box in boxes:
                                # Extract data, ensuring tensors are moved to CPU and converted
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())
                                label = self.model.names.get(class_id, f"ID:{class_id}")
                                detections.append((label, float(confidence), (int(x1), int(y1), int(x2), int(y2))))

                        # Emit the processed detections
                        self.detections_ready.emit(detections)

                    except Exception as e:
                        logger.error(f"Detection error: {e}", exc_info=True)
                        self.status_update.emit(f"Detection error: {e}")
                else:
                    # If no frame or detection disabled, sleep briefly to avoid busy-waiting
                    self.msleep(10) # Sleep for 10 milliseconds
            else:
                # Sleep if detection is disabled
                self.msleep(50)

    def stop(self):
        self.running = False

    @Slot(bool)
    def set_enabled(self, enabled):
        self._enabled = enabled
        logger.info(f"Detection {'enabled' if enabled else 'disabled'}")

    @Slot(float)
    def update_confidence_threshold(self, threshold):
        self._confidence_threshold = threshold
        logger.info(f"Confidence threshold updated to {threshold:.2f}")

    @Slot(float)
    def update_nms_threshold(self, threshold):
        self._nms_threshold = threshold
        logger.info(f"NMS threshold updated to {threshold:.2f}")


# --- Optical Flow Thread ---
class OpticalFlowThread(QThread):
    flow_ready = Signal(list) # List of (start_point_tuple, end_point_tuple)

    def __init__(self):
        super().__init__()
        self.prev_gray = None
        self.prev_points = None
        self.running = False
        self._enabled = False
        self._frame_lock = QMutex()
        self.current_frame = None

    @Slot(np.ndarray)
    def set_frame(self, frame):
        with QMutexLocker(self._frame_lock):
            self.current_frame = frame.copy()

    def run(self):
        self.running = True
        lk_params = dict(winSize=(15, 15),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        feature_params = dict(maxCorners=100,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        logger.info("Optical Flow thread started.")

        while self.running:
            if self._enabled:
                frame = None
                with QMutexLocker(self._frame_lock):
                    if self.current_frame is not None:
                        frame = self.current_frame
                        self.current_frame = None # Consume the frame

                if frame is not None:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    if self.prev_gray is not None and self.prev_points is not None and len(self.prev_points) > 0:
                        # Calculate optical flow
                        next_points, status, err = cv2.calcOpticalFlowPyrLK(
                            self.prev_gray, gray, self.prev_points, None, **lk_params
                        )

                        # Select good points
                        if next_points is not None:
                            good_new = next_points[status == 1]
                            good_old = self.prev_points[status == 1]

                            # Format flow vectors and emit
                            flow_vectors = [(tuple(map(int, p)), tuple(map(int, q))) for p, q in zip(good_old, good_new)]
                            if flow_vectors: # Only emit if there are vectors
                                self.flow_ready.emit(flow_vectors)

                            # Update points for the next iteration
                            self.prev_points = good_new.reshape(-1, 1, 2)
                        else:
                             # If no points tracked, find new features
                             self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

                    else:
                        # Find initial features if none exist or first frame
                        self.prev_points = cv2.goodFeaturesToTrack(gray, mask=None, **feature_params)

                    # Update previous frame and gray image
                    self.prev_gray = gray.copy()

                else:
                     self.msleep(10) # Sleep if no frame available

            else:
                # Reset state when disabled and sleep
                self.prev_gray = None
                self.prev_points = None
                self.msleep(50)
        logger.info("Optical Flow thread stopped.")
        # Clean up state when thread stops
        self.prev_gray = None
        self.prev_points = None


    def stop(self):
        self.running = False

    @Slot(bool)
    def set_enabled(self, enabled):
        self._enabled = enabled
        logger.info(f"Optical Flow {'enabled' if enabled else 'disabled'}")
        if not enabled: # Reset state immediately when disabled
            self.prev_gray = None
            self.prev_points = None


# --- Depth Estimation Thread ---
class DepthEstimationThread(QThread):
    depth_ready = Signal(np.ndarray) # Emits normalized depth map (0-1)
    status_update = Signal(str)      # Added for status updates

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.transform = None # Store the transform
        self.running = False
        self._enabled = False
        self._reload_model = False
        self._frame_lock = QMutex()
        self.current_frame = None

    def load_model(self):
        self.status_update.emit(f"Loading MiDaS model: {self.model_name}...")
        logger.info(f"Attempting to load MiDaS model: {self.model_name}")
        try:
            # Load the MiDaS model from PyTorch Hub
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_name)

            # Select appropriate transform based on model type
            # This is crucial for correct preprocessing
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            if self.model_name == "DPT_Large" or self.model_name == "DPT_Hybrid":
                self.transform = midas_transforms.dpt_transform
            else: # MiDaS_small or other models
                self.transform = midas_transforms.small_transform

            # Set device (GPU if available, else CPU)
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.to(device)
            self.model.eval() # Set model to evaluation mode

            logger.info(f"MiDaS model '{self.model_name}' loaded successfully on {device}.")
            self.status_update.emit(f"MiDaS model '{self.model_name}' loaded on {device}.")
            return True
        except Exception as e:
            error_msg = f"Failed to load MiDaS model '{self.model_name}': {e}"
            logger.error(error_msg, exc_info=True)
            self.status_update.emit(error_msg)
            self.model = None # Ensure model is None if loading failed
            self.transform = None
            return False

    @Slot(np.ndarray)
    def set_frame(self, frame):
        with QMutexLocker(self._frame_lock):
            self.current_frame = frame.copy()

    def run(self):
        self.running = True
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        logger.info("Depth Estimation thread started.")

        while self.running:
            if self._enabled:
                # Load or reload model if needed
                if self.model is None or self._reload_model:
                    if not self.load_model():
                        self.msleep(1000) # Wait before retrying model load
                        continue          # Skip the rest of the loop iteration
                    self._reload_model = False # Reset reload flag

                frame_to_process = None
                with QMutexLocker(self._frame_lock):
                    if self.current_frame is not None:
                        frame_to_process = self.current_frame
                        self.current_frame = None # Consume frame

                if frame_to_process is not None and self.model is not None and self.transform is not None:
                    try:
                        # Preprocess the frame
                        input_batch = self.transform(frame_to_process).to(device)

                        # Perform inference
                        with torch.no_grad():
                            prediction = self.model(input_batch)

                            # Resize prediction to original image size
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=frame_to_process.shape[:2], # (height, width)
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze()

                        # Move prediction to CPU and convert to numpy array
                        depth_map = prediction.cpu().numpy()

                        # Normalize depth map to 0-1 range for visualization
                        depth_min = depth_map.min()
                        depth_max = depth_map.max()
                        if depth_max > depth_min:
                            normalized_depth = (depth_map - depth_min) / (depth_max - depth_min)
                        else:
                            normalized_depth = np.zeros(depth_map.shape, dtype=np.float32)

                        self.depth_ready.emit(normalized_depth)

                    except Exception as e:
                        logger.error(f"Depth estimation error: {e}", exc_info=True)
                        self.status_update.emit(f"Depth estimation error: {e}")
                else:
                    self.msleep(10) # Sleep if no frame or model not ready
            else:
                self.msleep(50) # Sleep if disabled

        logger.info("Depth Estimation thread stopped.")
        # Release resources if necessary (PyTorch handles GPU memory generally)
        self.model = None
        self.transform = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear cache when stopping

    def stop(self):
        self.running = False

    @Slot(bool)
    def set_enabled(self, enabled):
        self._enabled = enabled
        logger.info(f"Depth Estimation {'enabled' if enabled else 'disabled'}")
        if enabled and self.model is None:
            self._reload_model = True # Trigger model load if enabled and not loaded
        elif not enabled:
             # Optionally release model resources when disabled for a long time
             # self.model = None
             # self.transform = None
             # if torch.cuda.is_available(): torch.cuda.empty_cache()
             pass # Keep model loaded for faster re-enabling for now

    @Slot(str)
    def set_model_name(self, model_name):
        # Only trigger reload if the name actually changes
        if self.model_name != model_name:
            logger.info(f"MiDaS model changed to: {model_name}")
            self.model_name = model_name
            self._reload_model = True
            # Ensure the model is released before loading the new one
            self.model = None
            self.transform = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --- Terminator Overlay Widget ---
class TerminatorOverlay(QWidget):
    def __init__(self, monitor_spec):
        super().__init__()
        self.monitor_spec = monitor_spec
        self.scanline_y = 0
        self.flicker_on = True
        self.power_level = 99.9
        self.current_detections = []
        self.target_position = None # Current estimated position QPointF
        self.target_label = ""      # Label of the tracked target
        self.tracking_status = "SCANNING"
        self.flow_vectors = []
        self.depth_qimage = None
        # --- Feature Flags ---
        self._show_detection_boxes = True
        self._show_optical_flow = False
        self._show_depth = False
        self._show_movement_path = False
        self._show_trajectory = False
        # --- End Feature Flags ---
        self.crosshair_position = QPointF(monitor_spec['width'] / 2, monitor_spec['height'] / 2)

        # Store history as deque of (timestamp, QPointF)
        self.target_path_history = collections.deque(maxlen=PATH_HISTORY_LENGTH)

        # Window Flags and Attributes
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(monitor_spec['left'], monitor_spec['top'], monitor_spec['width'], monitor_spec['height'])

        # Font Loading
        font_db = QFontDatabase()
        # Try loading from relative path first
        font_path = os.path.join(os.path.dirname(__file__), "PressStart2P-Regular.ttf")
        if not os.path.exists(font_path):
            # Fallback: try loading just by name (if installed system-wide)
            font_path = "PressStart2P-Regular.ttf"

        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id != -1:
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            if font_families:
                self.font_small = QFont(font_families[0], FONT_SIZE_SMALL)
                self.font_medium = QFont(font_families[0], FONT_SIZE_MEDIUM)
                logger.info(f"Using font: {font_families[0]}")
            else:
                 logger.warning(f"Font '{FONT_NAME}' loaded but no families found. Using '{FALLBACK_FONT}'.")
                 self.font_small = QFont(FALLBACK_FONT, FONT_SIZE_SMALL)
                 self.font_medium = QFont(FALLBACK_FONT, FONT_SIZE_MEDIUM)
        else:
            logger.warning(f"Font '{FONT_NAME}' not found or failed to load from '{font_path}'. Using '{FALLBACK_FONT}'.")
            self.font_small = QFont(FALLBACK_FONT, FONT_SIZE_SMALL)
            self.font_medium = QFont(FALLBACK_FONT, FONT_SIZE_MEDIUM)

        # Timers
        self.scanline_timer = QTimer(self)
        self.scanline_timer.timeout.connect(self.update_scanline)
        self.scanline_timer.start(SCANLINE_SPEED_MS)

        self.hud_update_timer = QTimer(self)
        self.hud_update_timer.timeout.connect(self.update_hud_elements)
        self.hud_update_timer.start(UPDATE_INTERVAL_MS)

        self.crosshair_timer = QTimer(self)
        self.crosshair_timer.timeout.connect(self.update_crosshair)
        self.crosshair_timer.start(16) # ~60 FPS update for smooth crosshair

        logger.info("Terminator overlay initialized.")

    # --- Slots for controlling features ---
    @Slot(bool)
    def set_show_detection_boxes(self, show):
        self._show_detection_boxes = show
        logger.debug(f"Show detection boxes set to: {show}")
        self.update() # Request repaint when state changes

    @Slot(bool)
    def set_show_optical_flow(self, show):
        self._show_optical_flow = show
        logger.debug(f"Show optical flow set to: {show}")
        self.update()

    @Slot(bool)
    def set_show_depth(self, show):
        self._show_depth = show
        logger.debug(f"Show depth set to: {show}")
        self.update()

    @Slot(bool)
    def set_show_movement_path(self, show):
        self._show_movement_path = show
        logger.debug(f"Show movement path set to: {show}")
        self.update()

    @Slot(bool)
    def set_show_trajectory(self, show):
        self._show_trajectory = show
        logger.debug(f"Show trajectory set to: {show}")
        self.update()
    # --- End Slots ---

    def update_scanline(self):
        self.scanline_y = (self.scanline_y + 5) % self.height()
        self.flicker_on = not self.flicker_on
        self.update() # Request repaint

    def update_hud_elements(self):
        self.power_level = max(0.0, self.power_level - random.uniform(0.01, 0.05))
        # No need to call self.update() here, scanline timer already does frequent updates
        # self.update()

    @Slot(list)
    def update_detections(self, detections):
        self.current_detections = detections
        if not detections:
            # No detections, clear target
            if self.target_position is not None: # Only update status if it changed
                self.target_position = None
                self.target_label = ""
                self.tracking_status = "SCANNING"
                self.update() # Request repaint as tracking status changed
            # Optionally clear history: self.target_path_history.clear()
        else:
            # Find the detection with the highest confidence
            best_detection = max(detections, key=lambda det: det[1])
            label, confidence, (x1, y1, x2, y2) = best_detection

            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            new_pos = QPointF(center_x, center_y)

            # Update target info
            self.target_position = new_pos
            self.target_label = label.upper()
            new_tracking_status = f"TRACKING: {self.target_label} ({confidence:.1%})"
            needs_update = self.tracking_status != new_tracking_status
            self.tracking_status = new_tracking_status

            # Add current position and timestamp to history
            current_time = time.time()
            self.target_path_history.append((current_time, new_pos))

            if needs_update:
                self.update() # Request repaint if tracking status changed

        # No general self.update() needed here unless status changed,
        # as drawing happens in paintEvent based on current state.

    @Slot(list)
    def update_flow(self, flow_vectors):
        self.flow_vectors = flow_vectors
        if self._show_optical_flow: # Only update if the flow is visible
            self.update()

    @Slot(np.ndarray)
    def update_depth(self, depth):
        # Convert normalized depth (0-1, float) to a QImage for display
        height, width = depth.shape
        # Create RGBA array: Red color, Alpha based on depth
        depth_rgba = np.zeros((height, width, 4), dtype=np.uint8)
        depth_rgba[:, :, 0] = 255  # Red channel
        # Alpha channel: Invert depth so closer (1) is opaque (255), farther (0) is transparent (0)
        alpha_channel = ((1.0 - depth) * 200).astype(np.uint8) # Scale alpha, max 200 for less intensity
        depth_rgba[:, :, 3] = alpha_channel

        # Create QImage (ensure data is contiguous)
        self.depth_qimage = QImage(depth_rgba.copy().data, width, height, QImage.Format_RGBA8888)
        if self._show_depth: # Only update if the depth map is visible
            self.update()

    def update_crosshair(self):
        moved = False
        # Move crosshair towards the target position smoothly
        if self.target_position:
            target_x = self.target_position.x()
            target_y = self.target_position.y()
            current_x = self.crosshair_position.x()
            current_y = self.crosshair_position.y()

            # Simple linear interpolation for smooth movement
            speed = 0.15 # Adjust for faster/slower crosshair movement
            dx = (target_x - current_x) * speed
            dy = (target_y - current_y) * speed

            # Update crosshair position if moved significantly
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                self.crosshair_position.setX(current_x + dx)
                self.crosshair_position.setY(current_y + dy)
                moved = True

        # Only request update if moved
        if moved:
            self.update()


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 1. Draw Depth Map (if enabled and available)
        if self._show_depth and self.depth_qimage and not self.depth_qimage.isNull():
            painter.setOpacity(0.6) # Make depth map slightly transparent
            painter.drawImage(0, 0, self.depth_qimage)
            painter.setOpacity(1.0) # Reset opacity

        # 2. Draw Scanline (if flicker is on)
        if self.flicker_on:
            painter.setPen(QPen(RED_COLOR, 1, Qt.DashLine))
            painter.drawLine(0, self.scanline_y, self.width(), self.scanline_y)

        # 3. Draw Outer Rectangle
        painter.setPen(QPen(RED_COLOR, 2))
        painter.drawRect(10, 10, self.width() - 20, self.height() - 20)

        # 4. Draw HUD Text
        painter.setFont(self.font_small)
        painter.setPen(TEXT_COLOR)
        text_x, text_y, line_height = 20, 30, 20
        painter.drawText(text_x, text_y, f"SYS ID: {random_hex(8)}")
        text_y += line_height
        # Use system time for consistency
        painter.drawText(text_x, text_y, f"TIME: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        text_y += line_height
        painter.drawText(text_x, text_y, f"PWR: {self.power_level:.1f}%")
        text_y += line_height
        painter.drawText(text_x, text_y, f"STATUS: {self.tracking_status}")

        # 5. Draw Detection Boxes (if enabled)
        if self._show_detection_boxes:
            painter.setFont(self.font_small) # Ensure correct font
            for label, confidence, (x1, y1, x2, y2) in self.current_detections:
                painter.setPen(QPen(RED_COLOR, 1))
                painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                painter.setPen(TEXT_COLOR)
                # Position text above the box
                painter.drawText(x1, y1 - 5, f"{label.upper()} {confidence:.1%}")

        # 6. Draw Movement Path (if enabled and history exists)
        if self._show_movement_path and len(self.target_path_history) > 1:
            painter.setPen(QPen(QColor(255, 100, 100, 180), 1.5)) # Semi-transparent red path
            points = [p for t, p in self.target_path_history]
            # Draw lines connecting consecutive points using QPolygonF for potential efficiency
            polygon = QPolygonF(points)
            painter.drawPolyline(polygon)
            # for i in range(len(points) - 1):
            #     painter.drawLine(points[i], points[i+1]) # Keep if polyline looks bad

        # 7. Draw Trajectory Prediction (if enabled and possible)
        if self._show_trajectory and len(self.target_path_history) >= TRAJECTORY_PREDICTION_POINTS:
            # Get the last few points for velocity calculation
            recent_history = list(self.target_path_history)[-TRAJECTORY_PREDICTION_POINTS:]
            if len(recent_history) >= 2:
                # Calculate average velocity
                start_time, start_pos = recent_history[0]
                end_time, end_pos = recent_history[-1]

                # Simple velocity: difference between last and first point in window
                dt = end_time - start_time
                if dt > 0.01: # Avoid division by zero or near-zero
                    vx = (end_pos.x() - start_pos.x()) / dt
                    vy = (end_pos.y() - start_pos.y()) / dt

                    # Predict future position
                    predict_x = end_pos.x() + vx * TRAJECTORY_PREDICTION_DURATION
                    predict_y = end_pos.y() + vy * TRAJECTORY_PREDICTION_DURATION
                    predicted_pos = QPointF(predict_x, predict_y)

                    # Draw the prediction line
                    painter.setPen(QPen(QColor(0, 255, 0, 200), 2, Qt.DotLine)) # Green dotted line
                    painter.drawLine(end_pos, predicted_pos)
                    # Draw a small circle at the predicted end point
                    painter.setBrush(QColor(0, 255, 0, 200)) # Fill the circle
                    painter.drawEllipse(predicted_pos, 3, 3)
                    painter.setBrush(Qt.NoBrush) # Reset brush


        # 8. Draw Optical Flow (if enabled)
        if self._show_optical_flow:
            painter.setPen(QPen(QColor(255, 165, 0, 150), 1)) # Orange, semi-transparent
            painter.setBrush(QColor(255, 165, 0, 180)) # Brush for end points
            for p1_tuple, p2_tuple in self.flow_vectors:
                 p1 = QPointF(*p1_tuple)
                 p2 = QPointF(*p2_tuple)
                 painter.drawLine(p1, p2)
                 # Draw small circle at the end point for directionality
                 painter.drawEllipse(p2, 2, 2)
            painter.setBrush(Qt.NoBrush) # Reset brush


        # 9. Draw Crosshair (always draw last, on top)
        painter.setPen(QPen(RED_COLOR, 2))
        crosshair_size = 20 # Size of the crosshair lines
        x = int(self.crosshair_position.x())
        y = int(self.crosshair_position.y())
        # Draw horizontal and vertical lines
        painter.drawLine(x - crosshair_size, y, x + crosshair_size, y)
        painter.drawLine(x, y - crosshair_size, x, y + crosshair_size)
        # Optional: Draw a small center dot
        # painter.drawPoint(x, y)

        painter.end() # End painting

    def closeEvent(self, event):
        # Stop timers when the widget is closed
        self.scanline_timer.stop()
        self.hud_update_timer.stop()
        self.crosshair_timer.stop()
        logger.info("Terminator overlay closed.")
        super().closeEvent(event)

# --- Main Control Window ---
class MainWindow(QMainWindow):
    # Define signals for parameter changes
    detection_interval_changed = Signal(int)
    confidence_threshold_changed = Signal(float)
    nms_threshold_changed = Signal(float)
    # Signals for enabling/disabling features
    enable_detection_changed = Signal(bool)
    show_boxes_changed = Signal(bool)
    enable_optical_flow_changed = Signal(bool)
    enable_depth_changed = Signal(bool)
    show_path_changed = Signal(bool)        # New signal
    show_trajectory_changed = Signal(bool)  # New signal
    midas_model_changed = Signal(str)       # New signal

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Terminator Vision Control Panel")
        self.setGeometry(100, 100, 650, 800) # Increased size slightly
        self.setStyleSheet("""
            QMainWindow { background-color: #111111; color: #ff0000; }
            QWidget { font-family: 'Press Start 2P', Monospace; font-size: 9pt; } /* Slightly smaller default */
            QLabel { color: #ff0000; padding-top: 6px; padding-bottom: 2px;}
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #222222; color: #ff0000; border: 1px solid #ff0000;
                padding: 4px; border-radius: 3px; font-size: 9pt;
            }
            QCheckBox { color: #ff0000; margin-top: 8px; }
            QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #ff0000; background-color: #222; }
            QCheckBox::indicator:checked { background-color: #ff0000; }
            QComboBox::drop-down { border: 1px solid #ff0000; width: 15px; }
            QComboBox QAbstractItemView { background-color: #222222; color: #ff0000; selection-background-color: #ff0000; selection-color: #000000; }
            QPushButton {
                background-color: #ff0000; color: #000000; border: none;
                padding: 10px 15px; margin-top: 12px; border-radius: 4px;
                font-size: 10pt; font-weight: bold;
            }
            QPushButton:hover { background-color: #cc0000; }
            QPushButton:disabled { background-color: #550000; color: #444444; }
            QTextEdit {
                background-color: #050505; color: #00ff00; border: 1px solid #ff0000;
                font-family: Monospace; font-size: 8pt; border-radius: 3px; padding: 5px;
            }
            QGroupBox { color: #ffaaaa; border: 1px solid #440000; margin-top: 10px; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px 0 5px; }
            QGridLayout { margin: 5px; spacing: 8px;}
        """)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setSpacing(10) # Add spacing between main sections

        # --- Configuration Section ---
        config_group = QGroupBox("Configuration") # Use QGroupBox
        config_layout = QGridLayout(config_group)
        self.main_layout.addWidget(config_group)

        row = 0
        # Monitor Selection
        self.monitor_label = QLabel("Target Display:")
        config_layout.addWidget(self.monitor_label, row, 0)
        self.monitor_combo = QComboBox()
        config_layout.addWidget(self.monitor_combo, row, 1)
        row += 1

        # Detection Model
        self.model_label = QLabel("Detection Model:")
        config_layout.addWidget(self.model_label, row, 0)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"])
        self.model_combo.setCurrentText("yolov8n.pt")
        config_layout.addWidget(self.model_combo, row, 1)
        row += 1

        # Detection Interval
        self.detection_interval_label = QLabel("Detect Interval (ms):")
        config_layout.addWidget(self.detection_interval_label, row, 0)
        self.detection_interval_spin = QSpinBox()
        self.detection_interval_spin.setMinimum(1) # Min interval 1ms
        self.detection_interval_spin.setMaximum(1000) # Max 1 sec
        self.detection_interval_spin.setSingleStep(10)
        self.detection_interval_spin.setValue(DETECTION_INTERVAL_MS)
        config_layout.addWidget(self.detection_interval_spin, row, 1)
        row += 1

        # Confidence Threshold
        self.confidence_threshold_label = QLabel("Confidence Thresh:")
        config_layout.addWidget(self.confidence_threshold_label, row, 0)
        self.confidence_threshold_spin = QDoubleSpinBox()
        self.confidence_threshold_spin.setMinimum(0.05)
        self.confidence_threshold_spin.setMaximum(0.95)
        self.confidence_threshold_spin.setSingleStep(0.05)
        self.confidence_threshold_spin.setValue(CONFIDENCE_THRESHOLD)
        config_layout.addWidget(self.confidence_threshold_spin, row, 1)
        row += 1

        # NMS Threshold
        self.nms_threshold_label = QLabel("NMS Thresh:")
        config_layout.addWidget(self.nms_threshold_label, row, 0)
        self.nms_threshold_spin = QDoubleSpinBox()
        self.nms_threshold_spin.setMinimum(0.1)
        self.nms_threshold_spin.setMaximum(0.9)
        self.nms_threshold_spin.setSingleStep(0.05)
        self.nms_threshold_spin.setValue(NMS_THRESHOLD)
        config_layout.addWidget(self.nms_threshold_spin, row, 1)
        row += 1

        # MiDaS Model
        self.midas_model_label = QLabel("Depth Model (MiDaS):")
        config_layout.addWidget(self.midas_model_label, row, 0)
        self.midas_model_combo = QComboBox()
        self.midas_model_combo.addItems(["MiDaS_small", "DPT_Hybrid", "DPT_Large"]) # Common MiDaS models
        self.midas_model_combo.setCurrentText("MiDaS_small")
        config_layout.addWidget(self.midas_model_combo, row, 1)
        row += 1

        # --- Feature Toggles Section ---
        features_group = QGroupBox("Features") # Use QGroupBox
        features_layout = QVBoxLayout(features_group)
        self.main_layout.addWidget(features_group)

        self.enable_detection_checkbox = QCheckBox("Enable Object Detection")
        self.enable_detection_checkbox.setChecked(True)
        features_layout.addWidget(self.enable_detection_checkbox)

        self.show_boxes_checkbox = QCheckBox("Show Detection Boxes")
        self.show_boxes_checkbox.setChecked(True)
        features_layout.addWidget(self.show_boxes_checkbox)

        self.enable_optical_flow_checkbox = QCheckBox("Enable Optical Flow")
        self.enable_optical_flow_checkbox.setChecked(False) # Default off
        features_layout.addWidget(self.enable_optical_flow_checkbox)

        self.enable_depth_checkbox = QCheckBox("Enable Depth Estimation")
        self.enable_depth_checkbox.setChecked(False) # Default off
        features_layout.addWidget(self.enable_depth_checkbox)

        self.show_path_checkbox = QCheckBox("Show Movement Path") # New Checkbox
        self.show_path_checkbox.setChecked(False) # Default off
        features_layout.addWidget(self.show_path_checkbox)

        self.show_trajectory_checkbox = QCheckBox("Show Trajectory Prediction") # New Checkbox
        self.show_trajectory_checkbox.setChecked(False) # Default off
        features_layout.addWidget(self.show_trajectory_checkbox)


        # --- Control Buttons ---
        button_layout = QtWidgets.QHBoxLayout() # Use QHBoxLayout for side-by-side buttons
        self.main_layout.addLayout(button_layout)
        self.start_button = QPushButton("START VISION")
        button_layout.addWidget(self.start_button)
        self.stop_button = QPushButton("STOP VISION")
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)

        # --- System Status Section ---
        sys_info_group = QGroupBox("System Status") # Use QGroupBox
        sys_info_layout = QGridLayout(sys_info_group)
        self.main_layout.addWidget(sys_info_group)

        row = 0
        self.cpu_label = QLabel("CPU Usage:")
        self.cpu_value = QLabel("N/A")
        sys_info_layout.addWidget(self.cpu_label, row, 0)
        sys_info_layout.addWidget(self.cpu_value, row, 1)
        row += 1
        self.mem_label = QLabel("Memory Usage:")
        self.mem_value = QLabel("N/A")
        sys_info_layout.addWidget(self.mem_label, row, 0)
        sys_info_layout.addWidget(self.mem_value, row, 1)
        row += 1
        self.gpu_label = QLabel("GPU:")
        self.gpu_value = QLabel("N/A")
        sys_info_layout.addWidget(self.gpu_label, row, 0)
        sys_info_layout.addWidget(self.gpu_value, row, 1)
        row += 1
        self.gpu_mem_label = QLabel("GPU Memory:")
        self.gpu_mem_value = QLabel("N/A")
        sys_info_layout.addWidget(self.gpu_mem_label, row, 0)
        sys_info_layout.addWidget(self.gpu_mem_value, row, 1)
        row += 1
        self.screen_label = QLabel("Screen Res:")
        self.screen_value = QLabel("N/A")
        sys_info_layout.addWidget(self.screen_label, row, 0)
        sys_info_layout.addWidget(self.screen_value, row, 1)
        row += 1
        self.tracking_info_label = QLabel("Tracking:")
        self.tracking_info_value = QLabel("INACTIVE")
        sys_info_layout.addWidget(self.tracking_info_label, row, 0)
        sys_info_layout.addWidget(self.tracking_info_value, row, 1)
        row += 1
        self.status_label = QLabel("STATUS: Ready")
        self.status_label.setStyleSheet("padding: 5px; border-top: 1px solid #ff0000; margin-top: 5px;")
        sys_info_layout.addWidget(self.status_label, row, 0, 1, 2) # Span across both columns

        # --- Log Output ---
        self.log_label = QLabel("Log Output:")
        self.main_layout.addWidget(self.log_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.main_layout.addWidget(self.log_text, 1) # Allow log to stretch vertically

        # --- Initialization ---
        self.monitors = []
        try:
            with mss.mss() as sct:
                all_monitors = sct.monitors
                # Filter out the 'All' monitor and ensure width/height are present
                self.monitors = [m for m in all_monitors if m.get('width') and m.get('height') and m.get('left') is not None and m.get('top') is not None]

                # Attempt to identify and exclude the primary monitor if it's the first one listed
                if len(self.monitors) > 1 and self.monitors[0]['left'] == 0 and self.monitors[0]['top'] == 0:
                     try:
                         primary_width, primary_height = pyautogui.size()
                         # If the first monitor matches primary size, assume it's primary and remove it
                         if self.monitors[0]['width'] == primary_width and self.monitors[0]['height'] == primary_height:
                             logger.info("Excluding potential primary monitor (Display 1) from target list.")
                             self.monitors = self.monitors[1:]
                         elif len(all_monitors) > 1 and all_monitors[0]['width'] >= primary_width and all_monitors[0]['height'] >= primary_height:
                             # Handle the 'All monitors' case if it wasn't filtered perfectly
                             logger.info("Excluding 'All Monitors' entry.")
                             self.monitors = self.monitors[1:]

                     except Exception as e:
                         logger.warning(f"Could not get primary screen size via pyautogui: {e}. Monitor filtering might be inaccurate.")


            if not self.monitors:
                 logger.warning("No suitable secondary monitors found. Using all available monitors.")
                 self.monitors = [m for m in all_monitors if m.get('width') and m.get('height')] # Fallback

            if not self.monitors:
                 raise ValueError("No usable monitors detected by MSS.")

            for i, monitor in enumerate(self.monitors):
                label = f"Display {i+1}: {monitor['width']}x{monitor['height']} @ ({monitor['left']},{monitor['top']})"
                self.monitor_combo.addItem(label, userData=monitor) # Store monitor dict as userData

            if self.monitor_combo.count() > 0:
                self.monitor_combo.setCurrentIndex(0)
            else:
                 self.update_status("ERROR: No monitors available for selection.")
                 self.start_button.setEnabled(False)

        except Exception as e:
            logger.error(f"Error initializing monitors: {e}", exc_info=True)
            self.update_status(f"Error initializing monitors: {e}")
            self.start_button.setEnabled(False)

        # --- Connect Signals and Slots ---
        self.start_button.clicked.connect(self.start_overlay)
        self.stop_button.clicked.connect(self.stop_overlay)

        # Configuration changes
        self.detection_interval_spin.valueChanged.connect(self.detection_interval_changed.emit)
        self.confidence_threshold_spin.valueChanged.connect(self.confidence_threshold_changed.emit)
        self.nms_threshold_spin.valueChanged.connect(self.nms_threshold_changed.emit)
        self.midas_model_combo.currentTextChanged.connect(self.midas_model_changed.emit)

        # Feature toggles
        self.enable_detection_checkbox.stateChanged.connect(lambda state: self.enable_detection_changed.emit(state == Qt.Checked))
        self.show_boxes_checkbox.stateChanged.connect(lambda state: self.show_boxes_changed.emit(state == Qt.Checked))
        self.enable_optical_flow_checkbox.stateChanged.connect(lambda state: self.enable_optical_flow_changed.emit(state == Qt.Checked))
        self.enable_depth_checkbox.stateChanged.connect(lambda state: self.enable_depth_changed.emit(state == Qt.Checked))
        self.show_path_checkbox.stateChanged.connect(lambda state: self.show_path_changed.emit(state == Qt.Checked)) # Connect new checkbox
        self.show_trajectory_checkbox.stateChanged.connect(lambda state: self.show_trajectory_changed.emit(state == Qt.Checked)) # Connect new checkbox

        # System Info Timer
        self.sys_info_timer = QTimer(self)
        self.sys_info_timer.timeout.connect(self.update_system_info)
        self.sys_info_timer.start(SYSTEM_INFO_INTERVAL_MS)
        self.update_system_info() # Initial update

        # Thread and Overlay references
        self.capture_thread = None
        self.detection_thread = None
        self.optical_flow_thread = None
        self.depth_estimation_thread = None
        self.overlay = None

    def start_overlay(self):
        # Get selected monitor spec
        monitor_data = self.monitor_combo.currentData()
        if not monitor_data or not isinstance(monitor_data, dict):
            self.update_status("ERROR: No valid monitor selected.")
            logger.error("Invalid monitor data selected.")
            return
        monitor_spec = monitor_data

        # Get model names
        model_name = self.model_combo.currentText()
        midas_model_name = self.midas_model_combo.currentText()

        # --- Create Threads and Overlay ---
        logger.info("Creating threads and overlay...")
        self.update_status("Initializing...")
        QApplication.processEvents() # Update GUI

        try:
            self.overlay = TerminatorOverlay(monitor_spec)
            self.capture_thread = ScreenCaptureThread(monitor_spec)
            self.detection_thread = DetectionThread(model_name)
            self.optical_flow_thread = OpticalFlowThread()
            self.depth_estimation_thread = DepthEstimationThread(midas_model_name)

            # --- Connect Signals from Threads to Overlay/UI ---
            # Capture -> Processing Threads
            self.capture_thread.frame_ready.connect(self.detection_thread.set_frame)
            self.capture_thread.frame_ready.connect(self.optical_flow_thread.set_frame)
            self.capture_thread.frame_ready.connect(self.depth_estimation_thread.set_frame)

            # Processing Threads -> Overlay
            self.detection_thread.detections_ready.connect(self.overlay.update_detections)
            self.optical_flow_thread.flow_ready.connect(self.overlay.update_flow)
            self.depth_estimation_thread.depth_ready.connect(self.overlay.update_depth)

            # Status Updates -> UI
            self.capture_thread.status_update.connect(self.update_status)
            self.detection_thread.status_update.connect(self.update_status)
            self.depth_estimation_thread.status_update.connect(self.update_status) # Connect depth status
            # Update tracking status label from overlay
            # Use a lambda that checks if overlay exists before accessing its attribute
            self.detection_thread.detections_ready.connect(
                lambda dets: self.tracking_info_value.setText(self.overlay.tracking_status if self.overlay else "INACTIVE")
            )


            # --- Connect UI Controls to Threads/Overlay ---
            # Connect parameter changes
            self.detection_interval_changed.connect(self.capture_thread.update_detection_interval)
            self.confidence_threshold_changed.connect(self.detection_thread.update_confidence_threshold)
            self.nms_threshold_changed.connect(self.detection_thread.update_nms_threshold)
            self.midas_model_changed.connect(self.depth_estimation_thread.set_model_name)

            # Connect feature enables/disables to Threads
            self.enable_detection_changed.connect(self.detection_thread.set_enabled)
            self.enable_optical_flow_changed.connect(self.optical_flow_thread.set_enabled)
            self.enable_depth_changed.connect(self.depth_estimation_thread.set_enabled)

            # Connect feature visibility toggles to Overlay Slots (CORRECTED)
            self.show_boxes_changed.connect(self.overlay.set_show_detection_boxes)
            self.enable_optical_flow_changed.connect(self.overlay.set_show_optical_flow) # Link enable/show for flow
            self.enable_depth_changed.connect(self.overlay.set_show_depth) # Link enable/show for depth
            self.show_path_changed.connect(self.overlay.set_show_movement_path)
            self.show_trajectory_changed.connect(self.overlay.set_show_trajectory)


            # --- Start Threads ---
            logger.info("Starting threads...")
            self.detection_thread.start()
            self.optical_flow_thread.start()
            self.depth_estimation_thread.start()
            # Start capture thread last, after others are ready to receive frames
            self.capture_thread.start()

            # --- Set Initial States ---
            # Trigger initial state updates based on checkboxes
            # These will now call the slots in the overlay or threads
            self.enable_detection_changed.emit(self.enable_detection_checkbox.isChecked())
            self.show_boxes_changed.emit(self.show_boxes_checkbox.isChecked())
            self.enable_optical_flow_changed.emit(self.enable_optical_flow_checkbox.isChecked())
            self.enable_depth_changed.emit(self.enable_depth_checkbox.isChecked())
            self.show_path_changed.emit(self.show_path_checkbox.isChecked())
            self.show_trajectory_changed.emit(self.show_trajectory_checkbox.isChecked())
            # Trigger initial parameter updates
            self.detection_interval_changed.emit(self.detection_interval_spin.value())
            self.confidence_threshold_changed.emit(self.confidence_threshold_spin.value())
            self.nms_threshold_changed.emit(self.nms_threshold_spin.value())
            self.midas_model_changed.emit(self.midas_model_combo.currentText())


            # --- Show Overlay and Update UI ---
            self.overlay.show()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            # Disable config options while running
            self.monitor_combo.setEnabled(False)
            self.model_combo.setEnabled(False)
            self.midas_model_combo.setEnabled(False) # Disable MiDaS combo too
            self.update_status("Overlay Activated.")
            logger.info("Overlay started successfully.")

        except Exception as e:
            logger.error(f"Failed to start overlay: {e}", exc_info=True)
            self.update_status(f"ERROR starting overlay: {e}")
            # Clean up any partially created objects
            self.stop_overlay()


    def stop_overlay(self):
        logger.info("Stopping overlay and threads...")
        self.update_status("Stopping...")
        QApplication.processEvents() # Update UI

        # Stop threads safely
        threads = [self.capture_thread, self.detection_thread, self.optical_flow_thread, self.depth_estimation_thread]
        for thread in threads:
            if thread and thread.isRunning():
                try:
                    thread.stop()
                    if not thread.wait(2000): # Wait up to 2 seconds
                         logger.warning(f"Thread {thread.__class__.__name__} did not stop gracefully, terminating.")
                         thread.terminate() # Force terminate if needed
                         thread.wait() # Wait after terminating
                except Exception as e:
                    logger.error(f"Error stopping thread {thread.__class__.__name__}: {e}")


        # Close overlay window
        if self.overlay:
            try:
                # Disconnect signals to avoid issues during shutdown
                # (Optional but can prevent late signals causing errors)
                self.show_boxes_changed.disconnect(self.overlay.set_show_detection_boxes)
                self.enable_optical_flow_changed.disconnect(self.overlay.set_show_optical_flow)
                self.enable_depth_changed.disconnect(self.overlay.set_show_depth)
                self.show_path_changed.disconnect(self.overlay.set_show_movement_path)
                self.show_trajectory_changed.disconnect(self.overlay.set_show_trajectory)

                self.overlay.close()
            except RuntimeError as e:
                 logger.warning(f"Minor error disconnecting signals during stop: {e}") # Catch potential disconnect errors if already disconnected
            except Exception as e:
                logger.error(f"Error closing overlay: {e}")

        # Clear references
        self.capture_thread = None
        self.detection_thread = None
        self.optical_flow_thread = None
        self.depth_estimation_thread = None
        self.overlay = None

        # Update UI state
        self.start_button.setEnabled(self.monitor_combo.count() > 0) # Only enable if monitors exist
        self.stop_button.setEnabled(False)
        self.monitor_combo.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.midas_model_combo.setEnabled(True)
        self.update_status("STATUS: Stopped")
        self.tracking_info_value.setText("INACTIVE")
        logger.info("Overlay stopped.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache() # Clear GPU cache after stopping


    @Slot(str)
    def update_status(self, message):
        # Ensure status updates happen on the main thread
        if QThread.currentThread() != self.thread():
             # If called from another thread, invoke method on main thread
             QtCore.QMetaObject.invokeMethod(self, "update_status", Qt.QueuedConnection, QtCore.Q_ARG(str, message))
        else:
            # Already on the main thread, update directly
            self.status_label.setText(f"STATUS: {message}")


    def update_system_info(self):
        # Update system info labels (runs in main thread via QTimer)
        try:
            cpu_percent = psutil.cpu_percent()
            self.cpu_value.setText(f"{cpu_percent:.1f}%")
            mem = psutil.virtual_memory()
            self.mem_value.setText(f"{mem.percent:.1f}% ({(mem.used / (1024**3)):.1f} / {(mem.total / (1024**3)):.1f} GB)")
        except Exception as e:
            logger.warning(f"Could not get CPU/Mem info: {e}")
            self.cpu_value.setText("Error")
            self.mem_value.setText("Error")

        try:
            gpu_name, gpu_mem = get_gpu_info()
            self.gpu_value.setText(gpu_name)
            self.gpu_mem_value.setText(gpu_mem)
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
            self.gpu_value.setText("Error")
            self.gpu_mem_value.setText("Error")

        try:
            screen_width, screen_height = pyautogui.size()
            self.screen_value.setText(f"{screen_width} x {screen_height}")
        except Exception as e:
            logger.warning(f"Could not get Screen info via pyautogui: {e}")
            self.screen_value.setText("Error")


    def closeEvent(self, event):
        # Ensure overlay and threads are stopped when closing the main window
        self.stop_overlay()
        self.sys_info_timer.stop()
        logger.info("Main window closed.")
        super().closeEvent(event)

# --- Main Execution ---
if __name__ == "__main__":
    # Setup logging
    log_format = '%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)
    logger = logging.getLogger()

    # Create Qt Application
    app = QApplication(sys.argv)

    # Attempt to load the custom font
    # Ensure the font file "PressStart2P-Regular.ttf" is in the same directory
    # or provide the correct path.
    font_path = os.path.join(os.path.dirname(__file__), "PressStart2P-Regular.ttf")
    if os.path.exists(font_path):
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id == -1:
            logger.warning(f"Failed to load font: {font_path}")
    else:
        logger.warning(f"Font file not found: {font_path}. Using fallback.")


    # Create and show the main window
    main_window = MainWindow()

    # Add the QTextEdit logger handler
    log_handler = QTextEditLogger(main_window.log_text)
    logger.addHandler(log_handler)

    main_window.show()

    # Connect cleanup function to application exit
    app.aboutToQuit.connect(main_window.stop_overlay) # Ensure cleanup on quit

    # Start the Qt event loop
    sys.exit(app.exec())
