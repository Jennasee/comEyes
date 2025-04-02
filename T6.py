import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk  # Import themed widgets
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk
import time
from mss import mss
import torch
import sys
import traceback
import supervision as sv

class VisionSimulator:
    def __init__(self, root):
        """
        Initializes the Vision Simulator application.

        Args:
            root (tk.Tk): The main Tkinter window.
        """
        self.root = root
        self.root.title("Vision Simulator - Screen Capture")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.geometry("1000x800") # Set a default window size

        # --- Style ---
        style = ttk.Style(self.root)
        style.theme_use('clam') # Use a modern theme ('clam', 'alt', 'default', 'classic')

        # --- Screen Capture Initialization ---
        try:
            self.sct = mss()
            # Get primary monitor dimensions dynamically
            monitor_info = self.sct.monitors[1] # Index 1 is usually the primary monitor
            self.monitor = {
                "left": monitor_info["left"],
                "top": monitor_info["top"],
                "width": monitor_info["width"],
                "height": monitor_info["height"]
            }
        except Exception as e:
            messagebox.showerror("Screen Capture Error", f"Failed to initialize screen capture: {e}")
            sys.exit(1)

        # --- Hardware Checks ---
        self.available_gpus = []
        if torch.cuda.is_available():
            self.available_gpus = [f"GPU {i}: {torch.cuda.get_device_name(i)}" for i in range(torch.cuda.device_count())]
        if not self.available_gpus:
            messagebox.showwarning("Hardware Warning", "No CUDA-enabled GPU found. Processing will run on CPU (if supported by models) or fail.")
            self.device = torch.device("cpu") # Fallback to CPU
        else:
            self.device = None # Will be set based on user selection

        self.cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if not self.cuda_available:
             messagebox.showwarning("OpenCV Warning", "OpenCV was not built with CUDA support. Optical flow might be slower (CPU).")

        # --- Model Initialization (Deferred) ---
        self.yolo_model = None
        self.depth_model = None
        self.depth_transform = None

        # --- Processing Variables ---
        self.prev_frame_gray_cpu = None
        self.prev_frame_gray_gpu = None
        self.gpu_flow_calculator = None
        self.fps_last_time = time.time()
        self.fps_counter = 0
        self.optical_flow_hsv = np.zeros((self.monitor["height"], self.monitor["width"], 3), dtype=np.uint8)
        self.frame_count = 0
        self.last_detections = None
        self.last_depth_colormap = None
        self.is_running = False
        self.update_id = None

        # --- GUI Creation ---
        self.create_gui()

    def create_gui(self):
        """Sets up the Tkinter GUI with themed widgets and grid layout."""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid columns for responsiveness
        main_frame.columnconfigure(0, weight=3) # Video display area
        main_frame.columnconfigure(1, weight=1) # Controls area
        main_frame.rowconfigure(0, weight=1)    # Make video row expandable

        # --- Video Display Area ---
        video_frame = ttk.LabelFrame(main_frame, text="Live Feed", padding="5")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        video_frame.rowconfigure(0, weight=1)
        video_frame.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_frame, text="Select settings and click Start", anchor=tk.CENTER)
        self.video_label.grid(row=0, column=0, sticky="nsew")

        # --- Controls Area ---
        controls_frame = ttk.Frame(main_frame, padding="5")
        controls_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))

        # --- Hardware Selection ---
        hw_frame = ttk.LabelFrame(controls_frame, text="Hardware", padding="10")
        hw_frame.pack(fill=tk.X, pady=(0, 10))
        hw_frame.columnconfigure(1, weight=1) # Make combobox expand

        ttk.Label(hw_frame, text="Processing Device:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.gpu_var = tk.StringVar()
        gpu_options = self.available_gpus if self.available_gpus else ["CPU (No CUDA GPU found)"]
        self.gpu_dropdown = ttk.Combobox(hw_frame, textvariable=self.gpu_var, values=gpu_options, state="readonly")
        if self.available_gpus:
            self.gpu_var.set(gpu_options[0])
        else:
            self.gpu_var.set(gpu_options[0])
            self.gpu_dropdown.config(state="disabled")
        self.gpu_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # --- Model Selection ---
        model_frame = ttk.LabelFrame(controls_frame, text="Models", padding="10")
        model_frame.pack(fill=tk.X, pady=(0, 10))
        model_frame.columnconfigure(1, weight=1) # Make comboboxes expand

        # Object Detection
        ttk.Label(model_frame, text="Object Detection:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.obj_model_var = tk.StringVar(value="yolov5s")
        self.obj_model_dropdown = ttk.Combobox(model_frame, textvariable=self.obj_model_var,
                                               values=["yolov5n", "yolov5s", "yolov5m", "yolov8n", "yolov8s"],
                                               state="readonly")
        self.obj_model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Depth Estimation
        ttk.Label(model_frame, text="Depth Estimation:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.depth_model_var = tk.StringVar(value="MiDaS_small") # Default to smaller model
        self.depth_model_dropdown = ttk.Combobox(model_frame, textvariable=self.depth_model_var,
                                                 values=["MiDaS_small", "DPT_Hybrid", "DPT_Large"],
                                                 state="readonly")
        self.depth_model_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        # --- Processing Settings ---
        settings_frame = ttk.LabelFrame(controls_frame, text="Processing Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1) # Make scales expand

        # Frame Skip
        ttk.Label(settings_frame, text="Frame Skip:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.skip_var = tk.IntVar(value=3)
        self.skip_scale = ttk.Scale(settings_frame, from_=1, to=30, orient=tk.HORIZONTAL, variable=self.skip_var, command=lambda v: self.skip_label.config(text=f"{int(float(v))}"))
        self.skip_scale.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.skip_label = ttk.Label(settings_frame, text=f"{self.skip_var.get()}")
        self.skip_label.grid(row=0, column=2, padx=5, pady=5)


        # YOLO resolution scale
        ttk.Label(settings_frame, text="YOLO Scale:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.yolo_scale_var = tk.DoubleVar(value=0.5) # Default to lower res
        self.yolo_scale = ttk.Scale(settings_frame, from_=0.2, to=1.0, orient=tk.HORIZONTAL, variable=self.yolo_scale_var, command=lambda v: self.yolo_scale_label.config(text=f"{float(v):.1f}"))
        self.yolo_scale.grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        self.yolo_scale_label = ttk.Label(settings_frame, text=f"{self.yolo_scale_var.get():.1f}")
        self.yolo_scale_label.grid(row=1, column=2, padx=5, pady=5)


        # --- Feature Toggles ---
        features_frame = ttk.LabelFrame(controls_frame, text="Enable Features", padding="10")
        features_frame.pack(fill=tk.X, pady=(0, 10))

        self.yolo_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(features_frame, text="Object Detection", variable=self.yolo_enabled, command=self._toggle_widget_state).pack(anchor=tk.W, padx=5, pady=2)

        self.optical_flow_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(features_frame, text="Optical Flow", variable=self.optical_flow_enabled).pack(anchor=tk.W, padx=5, pady=2)

        self.depth_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(features_frame, text="Depth Estimation", variable=self.depth_enabled, command=self._toggle_widget_state).pack(anchor=tk.W, padx=5, pady=2)

        # --- Control Buttons ---
        button_frame = ttk.Frame(controls_frame, padding="5")
        button_frame.pack(fill=tk.X, pady=(10, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.start_button = ttk.Button(button_frame, text="Start", command=self.start_processing)
        self.start_button.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # --- Status Bar ---
        status_frame = ttk.Frame(self.root, padding=(5, 2))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_label = ttk.Label(status_frame, text="Status: Idle")
        self.status_label.pack(side=tk.LEFT)

        self.fps_label = ttk.Label(status_frame, text="FPS: 0.0")
        self.fps_label.pack(side=tk.RIGHT)

        # Initial state update for widgets
        self._toggle_widget_state()


    def _toggle_widget_state(self):
        """Enable/disable model/scale widgets based on checkboxes."""
        # Object detection widgets
        obj_state = tk.NORMAL if self.yolo_enabled.get() else tk.DISABLED
        self.obj_model_dropdown.config(state=obj_state if obj_state == tk.NORMAL else "readonly")
        self.yolo_scale.config(state=obj_state)

        # Depth estimation widgets
        depth_state = tk.NORMAL if self.depth_enabled.get() else tk.DISABLED
        self.depth_model_dropdown.config(state=depth_state if depth_state == tk.NORMAL else "readonly")


    def start_processing(self):
        """Initialize models and start the processing loop."""
        if self.is_running:
            return

        # --- Set Device ---
        if self.available_gpus:
            selected_gpu = self.gpu_var.get()
            try:
                gpu_index = int(selected_gpu.split(":")[0].split()[1])
                self.device = torch.device(f"cuda:{gpu_index}")
                status_msg = f"Using {selected_gpu}"
            except (IndexError, ValueError) as e:
                 messagebox.showerror("GPU Error", f"Invalid GPU selection: {selected_gpu}. Error: {e}")
                 return
        else:
            self.device = torch.device("cpu")
            status_msg = "Using CPU"
        self.status_label.config(text=f"Status: Initializing models on {self.device}...")
        self.root.update_idletasks() # Force GUI update

        # --- Disable Controls ---
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._set_controls_state(tk.DISABLED)

        # --- Load Models ---
        if not self.initialize_models():
            self.stop_processing() # Re-enable controls if loading fails
            return # Stop if models failed to load

        # --- Setup Processing Variables ---
        self.setup_variables()

        # --- Start Loop ---
        self.is_running = True
        self.status_label.config(text=f"Status: Running ({status_msg})")
        self.video_label.config(text="")  # Clear placeholder text
        self.update()  # Start the processing loop

    def stop_processing(self):
        """Stops the processing loop."""
        if not self.is_running:
            return

        self.is_running = False
        if self.update_id:
            self.root.after_cancel(self.update_id)
            self.update_id = None

        # --- Re-enable Controls ---
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self._set_controls_state(tk.NORMAL)
        self._toggle_widget_state() # Re-apply enable/disable based on checkboxes

        # --- Clear State ---
        self.prev_frame_gray_cpu = None
        self.prev_frame_gray_gpu = None # Release GPU memory if allocated
        self.last_detections = None
        self.last_depth_colormap = None
        self.video_label.config(image='') # Clear image
        self.video_label.config(text="Select settings and click Start")
        self.status_label.config(text="Status: Idle")
        self.fps_label.config(text="FPS: 0.0")
        print("Processing stopped.")


    def _set_controls_state(self, state):
        """Enable or disable all control widgets."""
        widgets_to_toggle = [
            self.gpu_dropdown, self.obj_model_dropdown, self.depth_model_dropdown,
            self.skip_scale, self.yolo_scale
        ]
        # Also toggle checkboxes
        for child in self.root.winfo_children():
             if isinstance(child, ttk.Frame): # Look in frames
                 for sub_child in child.winfo_children():
                     if isinstance(sub_child, ttk.LabelFrame):
                         for widget in sub_child.winfo_children():
                              if isinstance(widget, (ttk.Combobox, ttk.Scale, ttk.Checkbutton)):
                                   try:
                                       widget.config(state=state)
                                   except tk.TclError: # Handle widgets that don't have state (like Labels)
                                       pass
                     elif isinstance(sub_child, (ttk.Combobox, ttk.Scale, ttk.Checkbutton)):
                          try:
                              sub_child.config(state=state)
                          except tk.TclError:
                              pass

        # Special handling for combobox readonly state if disabling
        if state == tk.DISABLED:
            self.obj_model_dropdown.config(state="disabled")
            self.depth_model_dropdown.config(state="disabled")
            if self.available_gpus:
                self.gpu_dropdown.config(state="disabled")
        else: # Re-enable based on checkboxes
             self._toggle_widget_state()
             if self.available_gpus:
                 self.gpu_dropdown.config(state="readonly")


    def initialize_models(self):
        """
        Loads object detection and depth estimation models onto the selected device.

        Returns:
            bool: True if models loaded successfully, False otherwise.
        """
        success = True
        try:
            # --- Object Detection Model ---
            if self.yolo_enabled.get():
                obj_model_name = self.obj_model_var.get()
                print(f"Loading object detection model: {obj_model_name}...")
                # Ensure cache directory exists or handle potential issues
                torch.hub.set_dir(torch.hub.get_dir()) # Use default cache dir
                if "yolov5" in obj_model_name:
                    self.yolo_model = torch.hub.load('ultralytics/yolov5', obj_model_name, pretrained=True, trust_repo=True)
                elif "yolov8" in obj_model_name:
                     # Yolov8 loading might differ, adjust if needed based on ultralytics library
                     # This assumes a similar torch.hub interface or requires ultralytics pip package
                     from ultralytics import YOLO # Requires `pip install ultralytics`
                     self.yolo_model = YOLO(f'{obj_model_name}.pt') # Load weights file
                     # Note: YOLOv8 might return results differently than YOLOv5
                     # The 'from_yolov5' function in supervision might need adjustment or replacement
                     # For simplicity, we'll assume a compatible results object for now.
                     # If errors occur in process_frame, this is a likely cause.

                else:
                    raise ValueError(f"Unsupported object detection model: {obj_model_name}")

                self.yolo_model.to(self.device) # Move model to device
                if hasattr(self.yolo_model, 'conf'): # yolov5 style
                     self.yolo_model.conf = 0.4
                if hasattr(self.yolo_model, 'eval'): # Common PyTorch method
                    self.yolo_model.eval()
                print("Object detection model loaded.")
            else:
                self.yolo_model = None # Ensure it's None if not enabled

            # --- Depth Estimation Model ---
            if self.depth_enabled.get():
                depth_model_name = self.depth_model_var.get()
                print(f"Loading depth estimation model: {depth_model_name}...")
                # Ensure cache directory exists
                torch.hub.set_dir(torch.hub.get_dir())
                self.depth_model = torch.hub.load("intel-isl/MiDaS", depth_model_name, trust_repo=True)
                self.depth_model.to(self.device)
                self.depth_model.eval()
                # Load appropriate transform based on model type
                transform_name = 'transforms'
                if "DPT" in depth_model_name:
                    self.depth_transform = torch.hub.load("intel-isl/MiDaS", f"{transform_name}.dpt_transform", trust_repo=True)
                elif "MiDaS_small" in depth_model_name:
                     self.depth_transform = torch.hub.load("intel-isl/MiDaS", f"{transform_name}.small_transform", trust_repo=True)
                else: # Generic MiDaS transform as fallback
                    self.depth_transform = torch.hub.load("intel-isl/MiDaS", f"{transform_name}.midas_transform", trust_repo=True)

                print("Depth estimation model loaded.")
            else:
                self.depth_model = None # Ensure it's None if not enabled
                self.depth_transform = None

        except Exception as e:
            error_msg = f"Failed to load models: {str(e)}\n\n{traceback.format_exc()}"
            messagebox.showerror("Model Loading Error", error_msg)
            print(error_msg)
            # Clean up partially loaded models
            self.yolo_model = None
            self.depth_model = None
            self.depth_transform = None
            success = False

        return success


    def setup_variables(self):
        """Initialize or reset processing variables before starting."""
        self.prev_frame_gray_cpu = None
        self.prev_frame_gray_gpu = None
        self.fps_last_time = time.time()
        self.fps_counter = 0
        # Re-initialize HSV matrix based on current monitor dimensions if needed
        self.optical_flow_hsv = np.zeros((self.monitor["height"], self.monitor["width"], 3), dtype=np.uint8)
        self.optical_flow_hsv[..., 1] = 255 # Set saturation to max
        self.frame_count = 0
        self.last_detections = None
        self.last_depth_colormap = None

        # Initialize GPU-based optical flow if CUDA is available and enabled
        if self.optical_flow_enabled.get() and self.cuda_available:
            try:
                self.prev_frame_gray_gpu = cv2.cuda_GpuMat()
                # Parameters for Farneback can be tuned
                self.gpu_flow_calculator = cv2.cuda_FarnebackOpticalFlow.create(
                    numLevels=5, pyrScale=0.5, fastPyramids=False, winSize=13,
                    numIters=10, polyN=5, polySigma=1.1, flags=0
                )
                print("Using GPU for Optical Flow.")
            except cv2.error as e:
                 print(f"Warning: Could not initialize CUDA Optical Flow ({e}). Falling back to CPU.")
                 self.cuda_available = False # Disable CUDA flow for this session
                 self.gpu_flow_calculator = None
                 self.prev_frame_gray_gpu = None
        else:
            self.gpu_flow_calculator = None
            self.prev_frame_gray_gpu = None
            if self.optical_flow_enabled.get():
                 print("Using CPU for Optical Flow.")


    def update(self):
        """Main loop to capture and process frames."""
        if not self.is_running:
            return

        start_frame_time = time.time()
        processed_frame = None # Initialize
        try:
            # --- Capture Frame ---
            frame_bgr = np.array(self.sct.grab(self.monitor))[:, :, :3] # Grab BGR
            if frame_bgr.size == 0:
                print("Warning: Captured empty frame.")
                # Schedule next update and skip processing this frame
                self.update_id = self.root.after(10, self.update) # Wait a bit longer
                return

            # Ensure contiguous array for OpenCV compatibility
            frame_bgr = np.ascontiguousarray(frame_bgr)

            # --- Process Frame ---
            processed_frame = self.process_frame(frame_bgr)

            # --- Display Frame ---
            # Convert final processed frame (BGR) to RGB for PIL/Tkinter
            img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)

            # Resize image to fit the label while maintaining aspect ratio (optional but recommended)
            label_w = self.video_label.winfo_width()
            label_h = self.video_label.winfo_height()
            if label_w > 1 and label_h > 1: # Check if widget size is available
                img_pil.thumbnail((label_w, label_h), Image.Resampling.LANCZOS)

            img_tk = ImageTk.PhotoImage(image=img_pil)

            # Update label
            self.video_label.img = img_tk  # Keep reference to avoid garbage collection
            self.video_label.configure(image=img_tk)

        except mss.ScreenShotError as e:
             print(f"Screen capture error: {e}")
             # Potentially stop processing or try to re-initialize sct
             self.stop_processing()
             messagebox.showerror("Capture Error", "Screen capture failed. Stopping.")
             return
        except Exception as e:
            print(f"Error during frame processing or display: {e}")
            traceback.print_exc()
            # Optionally stop processing on error, or just log and continue
            # self.stop_processing()
            # return

        # --- Update FPS ---
        self.fps_counter += 1
        current_time = time.time()
        elapsed_time = current_time - self.fps_last_time
        if elapsed_time >= 1.0:
            fps = self.fps_counter / elapsed_time
            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.fps_last_time = current_time
            self.fps_counter = 0

        # --- Schedule Next Update ---
        # Calculate delay for roughly 60fps target, considering processing time
        # processing_time_ms = int((time.time() - start_frame_time) * 1000)
        # delay = max(1, 16 - processing_time_ms) # Target ~60 FPS (16ms per frame)
        delay = 1 # Run as fast as possible
        self.update_id = self.root.after(delay, self.update)


    def process_frame(self, frame_bgr):
        """
        Processes a single frame using enabled techniques.

        Args:
            frame_bgr (np.ndarray): The input frame in BGR format.

        Returns:
            np.ndarray: The processed frame in BGR format.
        """
        self.frame_count += 1
        processed_frame = frame_bgr.copy() # Work on a copy
        skip_frames = self.skip_var.get()
        yolo_scale = self.yolo_scale_var.get()
        perform_expensive_ops = (self.frame_count % skip_frames == 0)

        # --- Object Detection ---
        if self.yolo_enabled.get() and self.yolo_model:
            if perform_expensive_ops:
                # Prepare frame for YOLO (RGB, potential resize)
                if yolo_scale < 1.0:
                    proc_height = int(self.monitor["height"] * yolo_scale)
                    proc_width = int(self.monitor["width"] * yolo_scale)
                    yolo_input_frame = cv2.resize(frame_bgr, (proc_width, proc_height), interpolation=cv2.INTER_LINEAR)
                else:
                    yolo_input_frame = frame_bgr
                yolo_input_frame_rgb = cv2.cvtColor(yolo_input_frame, cv2.COLOR_BGR2RGB)

                # Inference
                with torch.no_grad():
                    # Use AMP for potential speedup on compatible GPUs
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type=='cuda'):
                        results = self.yolo_model(yolo_input_frame_rgb) # Pass RGB frame

                # Process results (assuming supervision `from_yolov5` or similar works for yolov8)
                try:
                    # This part might need adjustment based on the exact format of YOLOv8 results
                    if isinstance(self.yolo_model, torch.nn.Module): # Heuristic for yolov5
                         detections = sv.Detections.from_yolov5(results)
                    else: # Assume ultralytics YOLO object for yolov8
                         # Need to adapt this based on ultralytics result object structure
                         # Example: results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls
                         # This is a placeholder - requires checking ultralytics documentation
                         boxes = results[0].boxes.xyxy.cpu().numpy()
                         conf = results[0].boxes.conf.cpu().numpy()
                         cls = results[0].boxes.cls.cpu().numpy().astype(int)
                         detections = sv.Detections(xyxy=boxes, confidence=conf, class_id=cls)


                    # Scale detections back if resized
                    if yolo_scale < 1.0:
                        scale_x = self.monitor["width"] / proc_width
                        scale_y = self.monitor["height"] / proc_height
                        detections.xyxy[:, [0, 2]] *= scale_x
                        detections.xyxy[:, [1, 3]] *= scale_y
                    self.last_detections = detections
                except Exception as e:
                    print(f"Error processing YOLO results: {e}") # Log error but continue
                    self.last_detections = None # Reset detections on error

            # Annotate frame with the latest valid detections
            if self.last_detections is not None and len(self.last_detections) > 0:
                # Customize annotators if needed
                box_annotator = sv.BoxAnnotator(thickness=1, text_thickness=1, text_scale=0.4)
                label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER, text_scale=0.4, text_thickness=1)

                labels = [
                    f"{self.yolo_model.names[class_id]} {confidence:.2f}"
                    for _, _, confidence, class_id, _
                    in self.last_detections
                ]
                processed_frame = box_annotator.annotate(scene=processed_frame, detections=self.last_detections)
                processed_frame = label_annotator.annotate(scene=processed_frame, detections=self.last_detections, labels=labels)


        # --- Optical Flow ---
        if self.optical_flow_enabled.get():
            # Convert current frame to grayscale
            gray_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            flow = None
            if self.cuda_available and self.gpu_flow_calculator and self.prev_frame_gray_gpu:
                # --- GPU Optical Flow ---
                gpu_current_gray = cv2.cuda_GpuMat()
                gpu_current_gray.upload(gray_bgr)

                if not self.prev_frame_gray_gpu.empty():
                    gpu_flow_field = self.gpu_flow_calculator.calc(self.prev_frame_gray_gpu, gpu_current_gray, None)
                    flow = gpu_flow_field.download() # Download result back to CPU

                # Update previous frame on GPU for next iteration
                self.prev_frame_gray_gpu.upload(gray_bgr) # More efficient than reassigning

            elif self.prev_frame_gray_cpu is not None:
                # --- CPU Optical Flow ---
                flow = cv2.calcOpticalFlowFarneback(self.prev_frame_gray_cpu, gray_bgr, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

            # If flow was calculated, visualize it
            if flow is not None:
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                self.optical_flow_hsv[..., 0] = ang * 180 / np.pi / 2 # Hue from angle
                # self.optical_flow_hsv[..., 1] = 255 # Saturation (already set)
                self.optical_flow_hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX) # Value from magnitude
                flow_rgb = cv2.cvtColor(self.optical_flow_hsv, cv2.COLOR_HSV2BGR)
                # Blend flow visualization with the processed frame
                processed_frame = cv2.addWeighted(processed_frame, 0.7, flow_rgb, 0.3, 0)

            # Store current grayscale frame for next iteration (CPU)
            self.prev_frame_gray_cpu = gray_bgr


        # --- Depth Estimation ---
        if self.depth_enabled.get() and self.depth_model and self.depth_transform:
            if perform_expensive_ops:
                # Prepare frame for MiDaS (RGB)
                depth_input_frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                # Transform and move to device
                input_batch = self.depth_transform(depth_input_frame_rgb).to(self.device)

                with torch.no_grad():
                    with torch.amp.autocast(device_type=self.device.type, enabled=self.device.type=='cuda'):
                        prediction = self.depth_model(input_batch)

                    # Resize prediction to original frame size
                    prediction = torch.nn.functional.interpolate(
                        prediction.unsqueeze(1),
                        size=(self.monitor["height"], self.monitor["width"]),
                        mode="bicubic", # Use bicubic for smoother results
                        align_corners=False,
                    ).squeeze()

                # Process depth map (normalize, convert to uint8, apply colormap)
                depth_map = prediction.cpu().numpy()
                depth_map_normalized = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX) # Normalize to 0-1 range
                depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)
                self.last_depth_colormap = cv2.applyColorMap(depth_map_uint8, cv2.COLORMAP_MAGMA) # Or COLORMAP_JET, etc.

            # Blend depth map with the processed frame using the latest colormap
            if self.last_depth_colormap is not None:
                processed_frame = cv2.addWeighted(processed_frame, 0.6, self.last_depth_colormap, 0.4, 0)

        return processed_frame


    def on_close(self):
        """Clean up resources on window close."""
        print("Closing application...")
        self.stop_processing() # Ensure the loop is stopped
        if self.sct:
            self.sct.close() # Close mss context
            print("Screen capture closed.")
        # No explicit cv2.destroyAllWindows() needed as Tkinter manages the window
        self.root.destroy()
        print("Application closed.")
        # Ensure script exits cleanly, especially if threads were involved (though not in this version)
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = VisionSimulator(root)
    root.mainloop()
