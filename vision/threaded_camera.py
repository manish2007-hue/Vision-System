import threading
import time
import urllib.request
import numpy as np
import cv2

class ThreadedCamera:
    """
    High-speed, zero-latency threaded camera reader for ESP32-CAM MJPEG stream.
    Always maintains ONLY the newest frame in memory (queue size = 1),
    completely eliminating network buffer lag.
    """
    def __init__(self, stream_url="http://10.144.252.123:81/stream", fallback_cam=0):
        self.stream_url = stream_url
        self.fallback_cam = fallback_cam
        self.latest_frame = None
        self.is_running = True
        self.lock = threading.Lock()
        self.fps = 0.0
        self.source_type = "ESP32-CAM"
        self.connected = False

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.is_running:
            try:
                req = urllib.request.Request(self.stream_url)
                stream = urllib.request.urlopen(req, timeout=4)
                bytes_data = bytes()
                self.connected = True
                self.source_type = "ESP32-CAM"
                print(f"[ThreadedCamera] Connected to {self.stream_url}")

                last_fps_time = time.time()
                frames_counted = 0

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

                            frames_counted += 1
                            now = time.time()
                            if now - last_fps_time >= 1.0:
                                self.fps = round(frames_counted / (now - last_fps_time), 1)
                                frames_counted = 0
                                last_fps_time = now

                stream.close()
            except Exception as e:
                self.connected = False
                # Try fallback to local webcam if ESP32 stream is offline
                print(f"[ThreadedCamera] Stream disconnected ({e}). Retrying...")
                time.sleep(1.0)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy(), self.fps, self.source_type
        return None, 0.0, "OFFLINE"

    def stop(self):
        self.is_running = False
