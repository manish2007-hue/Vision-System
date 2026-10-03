import sys
import os
import time
import threading
import numpy as np
import cv2
import urllib.request
import base64
from flask import Flask, render_template, Response, jsonify, request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vision")))

from neural_lighting_restorer import NeuralLightingRestorer
from advanced_detector import AdvancedShapeDetector
from pipeline import VisionPipeline
from sensor import get_lux

app = Flask(__name__)

# Core AI & Vision Engines
pipeline = VisionPipeline(min_area=350)
restorer = pipeline.restorer
shape_detector = pipeline.detector

ESP32_STREAM_URL = "http://10.187.226.123:81/stream"

class ESP32DedicatedCamera:
    """
    STRICTLY ESP32-CAM ONLY. Zero laptop webcam.
    Dedicated high-speed thread keeping only the latest frame (zero-lag buffer size = 1).
    """
    def __init__(self, stream_url):
        self.stream_url = stream_url
        self.latest_frame = None
        self.lock = threading.Lock()
        self.is_connected = False
        self.fps = 0.0
        self.is_running = True
        self.thread = threading.Thread(target=self._stream_worker, daemon=True)
        self.thread.start()

    def _stream_worker(self):
        while self.is_running:
            try:
                print(f"[ESP32Camera] Connecting to {self.stream_url} ...")
                req = urllib.request.Request(self.stream_url)
                stream = urllib.request.urlopen(req, timeout=3)
                bytes_data = bytes()
                self.is_connected = True
                print("[ESP32Camera] Connected to ESP32-CAM stream successfully!")

                last_t = time.time()
                frames = 0

                while self.is_running:
                    chunk = stream.read(2048)
                    if not chunk:
                        break
                    bytes_data += chunk
                    a = bytes_data.find(b"\xff\xd8")
                    b = bytes_data.find(b"\xff\xd9")

                    if a != -1 and b != -1:
                        jpg = bytes_data[a:b+2]
                        bytes_data = bytes_data[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                        if frame is not None:
                            with self.lock:
                                self.latest_frame = frame

                            frames += 1
                            now = time.time()
                            if now - last_t >= 1.0:
                                self.fps = round(frames / (now - last_t), 1)
                                frames = 0
                                last_t = now

                stream.close()
            except Exception as e:
                self.is_connected = False
                with self.lock:
                    self.latest_frame = None
                time.sleep(1.0)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.fps, True
        return None, 0.0, False

cam_reader = ESP32DedicatedCamera(ESP32_STREAM_URL)

telemetry = {
    "lux": 250.0,
    "mode": "NORMALIZED",
    "fps": 0.0,
    "source": "ESP32-CAM",
    "connected": False,
    "latency_ms": 0,
    "shapes": [],
    "count": 0,
    "timestamp": "--:--:--"
}

def poll_sensor():
    global telemetry
    while True:
        try:
            lux = get_lux(timeout=0.4)
            if lux is not None and lux > 0:
                telemetry["lux"] = round(lux, 1)
            else:
                telemetry["lux"] = None
        except Exception:
            telemetry["lux"] = None
        time.sleep(0.35)

threading.Thread(target=poll_sensor, daemon=True).start()

def generate_frames():
    global telemetry
    prev_t = time.time()

    while True:
        frame, cam_fps, connected = cam_reader.get_frame()

        if not connected or frame is None:
            # Standby screen waiting specifically for ESP32-CAM
            standby = np.zeros((240, 720, 3), dtype=np.uint8)
            cv2.putText(standby, "WAITING FOR ESP32-CAM HARDWARE...", (140, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.putText(standby, f"Target: {ESP32_STREAM_URL}", (170, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
            cv2.putText(standby, "Check 5V and GND power wires on camera module!", (130, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
            
            telemetry["connected"] = False
            telemetry["fps"] = 0.0
            telemetry["source"] = "WAITING FOR ESP32-CAM"
            
            ret, buf = cv2.imencode(".jpg", standby)
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.2)
            continue

        frame = cv2.resize(frame, (320, 240))
        t_start = time.time()
        
        # Sensor reading with automatic image-based fallback
        raw_lux = telemetry.get("lux")
        if raw_lux is None or raw_lux <= 0:
            gray_eval = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_eval = float(np.mean(gray_eval))
            lux_for_engine = round(mean_eval * 2.2, 1)
            lux_display = f"{int(lux_for_engine)} (Est)"
        else:
            lux_for_engine = raw_lux
            lux_display = f"{raw_lux} lx"

        # 1. Process frame through Illumination-Invariant Vision Pipeline
        annotated, shapes = pipeline.process_frame(frame, lux_for_engine)
        restored_bgr = pipeline.last_restored
        gain_heatmap = pipeline.last_heatmap
        lighting_state = pipeline.last_mode

        # Telemetry
        latency = int((time.time() - t_start) * 1000)
        curr_t = time.time()
        display_fps = round(1.0 / max(0.001, (curr_t - prev_t)), 1)
        prev_t = curr_t

        telemetry["connected"] = True
        telemetry["fps"] = display_fps
        telemetry["lux"] = lux_display
        telemetry["mode"] = lighting_state
        telemetry["source"] = "ESP32-CAM (WiFi)"
        telemetry["latency_ms"] = latency
        telemetry["shapes"] = shapes
        telemetry["count"] = len(shapes)
        telemetry["timestamp"] = time.strftime("%H:%M:%S")

        # 4. Construct 3-Panel Inspection View: [Raw | Restored | Localization]
        raw_display = frame.copy()
        cv2.putText(raw_display, "1. RAW ESP32-CAM", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
        cv2.putText(restored_bgr, f"2. AI RESTORED ({lighting_state})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(annotated, f"3. LOCALIZED ({len(shapes)})", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

        composite = np.hstack([raw_display, restored_bgr, annotated])

        ret, buffer = cv2.imencode(".jpg", composite, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ret:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def status():
    return jsonify(telemetry)

# ========================================================
# IMAGE RESEARCH STUDIO API ROUTES (SOFTWARE-ONLY BENCHMARK)
# ========================================================
TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_images"))

def _b64_encode(cv_img):
    ret, buf = cv2.imencode(".jpg", cv_img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if ret:
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")
    return ""

@app.route("/api/images", methods=["GET"])
def list_images():
    os.makedirs(TEST_DIR, exist_ok=True)
    files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    return jsonify({"images": sorted(files)})

@app.route("/api/upload", methods=["POST"])
def upload_image():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = f"custom_{int(time.time())}_{file.filename}"
    save_path = os.path.join(TEST_DIR, filename)
    file.save(save_path)
    return jsonify({"success": True, "filename": filename})

@app.route("/api/process_image", methods=["POST"])
def process_single_image():
    data = request.get_json() or {}
    image_name = data.get("image_name", "airpod_shadow.png")

    img_path = os.path.join(TEST_DIR, image_name)
    if not os.path.exists(img_path):
        return jsonify({"error": f"Image '{image_name}' not found"}), 404

    frame = cv2.imread(img_path)
    if frame is None:
        return jsonify({"error": "Failed to decode image"}), 400

    h, w = frame.shape[:2]
    t0 = time.time()

    # Process through the complete 4-stage Object-First Vision Pipeline
    annotated_frame, detections = pipeline.process_frame(frame)
    dt_ms = round((time.time() - t0) * 1000, 1)

    shadow_overlay = pipeline.last_shadow_overlay if pipeline.last_shadow_overlay is not None else frame
    shadow_free = pipeline.last_shadow_free if pipeline.last_shadow_free is not None else frame
    restored_bgr = pipeline.last_restored if pipeline.last_restored is not None else frame
    heatmap = pipeline.last_heatmap if pipeline.last_heatmap is not None else frame

    gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_res = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2GRAY)
    mean_raw = round(float(np.mean(gray_raw)), 1)
    mean_res = round(float(np.mean(gray_res)), 1)
    std_raw = round(float(np.std(gray_raw)), 1)
    std_res = round(float(np.std(gray_res)), 1)

    return jsonify({
        "success": True,
        "image_name": image_name,
        "width": w,
        "height": h,
        "raw_b64": _b64_encode(frame),
        "shadow_overlay_b64": _b64_encode(shadow_overlay),
        "shadow_free_b64": _b64_encode(shadow_free),
        "restored_b64": _b64_encode(restored_bgr),
        "annotated_b64": _b64_encode(annotated_frame),
        "heatmap_b64": _b64_encode(heatmap),
        "shapes": detections,
        "metrics": {
            "latency_ms": dt_ms,
            "fps": round(1000.0 / max(0.1, dt_ms), 1),
            "mean_before": mean_raw,
            "mean_after": mean_res,
            "std_before": std_raw,
            "std_after": std_res,
            "lighting_mode": pipeline.last_mode,
            "resolution": f"{w}x{h}"
        }
    })

if __name__ == "__main__":
    print("[Industrial Vision Server] Running on http://localhost:5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

