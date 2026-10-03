# AI-Powered Adaptive Vision System
### A Software-First Solution for Lighting-Invariant Shape Recognition in Industrial Robotics

**Industry Partner:** SVR Robotics Pvt Ltd, Pune  
**Project Branch:** AI & Robotics  

---

## 📌 Project Overview

Industrial vision systems frequently fail under changing factory lighting conditions (morning sunlight, moving shadows, overhead LED flicker, glare). Expensive hardware fixes (such as ₹5-20 lakh Cognex/Keyence cameras or bulky dome lights) are cost-prohibitive for small enterprises and educational robots.

This project delivers a **software-first adaptive vision system** that runs on low-cost hardware (e.g., ₹550 ESP32-CAM or any standard USB webcam) and makes the computer vision pipeline immune to illumination shifts.

### 🌟 Key Technical Innovations
1. **Stage 1 — Lighting Normalization (CLAHE):** Converts frame to LAB color space and applies Contrast Limited Adaptive Histogram Equalization solely on the $L$ (Lightness) channel to balance illumination locally.
2. **Stage 2 — Shadow Compensation:** Uses HSV color space analysis to detect shadow penumbras and dynamically scales shadow pixel brightness using the non-shadow to shadow luminance ratio.
3. **Stage 3 — Triple-Method Voting Engine:** Resolves the notorious **Hexagon vs. Circle ambiguity** reported by SVR Robotics through weighted consensus voting:
   - **Method 1:** Polygon Vertex Count (`approxPolyDP`) + Corner Angle Estimation (~60° Triangle, ~90° Square, ~120° Hexagon).
   - **Method 2:** Circularity Metric ($\frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$), providing decisive separation: Circle ($\ge 0.875$) vs Hexagon ($0.805 - 0.875$) vs Square ($0.70 - 0.805$) vs Triangle ($0.45 - 0.70$).
   - **Method 3:** Hu Moments Invariant Pattern Matching against canonical shape models (scale, rotation, and translation invariant).

---

## 📁 Repository Directory Structure

```
C:\Users\AJINKYA JOSHI\Desktop\CEP\
│
├── firmware\                 ← Arduino / ESP32 source code
│   ├── esp32cam\
│   │   └── esp32cam.ino      ← WiFi MJPEG video streaming server (OV2640)
│   └── esp32_bh1750\
│       └── esp32_bh1750.ino  ← BH1750 I2C ambient Lux reader & serial streamer
│
├── vision\                   ← Core Python computer vision pipeline
│   ├── pipeline.py           ← Multi-stage adaptive processing & annotation
│   ├── shape_detector.py     ← Triple-method voting classifier & angle solver
│   └── sensor.py             ← BH1750 serial interface & simulation fallback
│
├── dashboard\                ← Interactive Flask web application
│   ├── app.py                ← Web server, MJPEG streaming & REST APIs
│   ├── templates\
│   │   └── index.html        ← Dark industrial UI with live telemetry & controls
│   └── static\
│       └── style.css         ← Modern glassmorphism robotics stylesheet
│
├── test_images\              ← Benchmark test scenes (harsh shadows, low light, glare)
├── results\                  ← Saved comparison snapshots and verification benchmarks
├── generate_test_images.py   ← Automated synthetic test image generator
├── test_pipeline.py          ← Benchmark verification script
└── README.md                 ← Project documentation & user guide
```

---

## 🚀 Quickstart Guide

### 1. Run the Verification Benchmark
To run the vision pipeline across all benchmark scenes (harsh shadows, low light, flashlight glare, hexagon vs circle challenge):

```powershell
python test_pipeline.py
```
This processes all test images and outputs high-resolution side-by-side comparison images into the `results/` folder.

### 2. Launch the Interactive Web Dashboard
To start the live web dashboard:

```powershell
python dashboard/app.py
```
Open your browser at: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

#### Dashboard Features:
- **Live View Switcher:** Toggle between *Side-by-Side (Raw vs AI)*, *AI Processed*, and *Raw Camera Feed*.
- **Live Lux Gauge:** Real-time Lux readings from BH1750 sensor with environmental condition tagging.
- **Pipeline Controls:** Toggle CLAHE normalization and Shadow Compensation dynamically to observe immediate before/after differences.
- **Shape Classification Cards:** Displays shape tag, confidence score (%), circularity ratio, vertex count, and edge corner angles in degrees.
- **Source Selector:** Switch between synthetic test scenes, USB webcam, and ESP32-CAM WiFi stream.
- **Snapshot Capture:** Saves timestamped annotated results directly to `results/`.

---

## 🔌 Hardware Setup

### 1. ESP32-CAM (AI-Thinker OV2640)
1. Open `firmware/esp32cam/esp32cam.ino` in Arduino IDE.
2. Select Board: **AI Thinker ESP32-CAM**.
3. Update `ssid` and `password` with your WiFi credentials.
4. Upload using FTDI programmer (connect GPIO 0 to GND during flash).
5. Open Serial Monitor at 115200 baud to view the assigned IP address (e.g., `http://192.168.1.50/stream`).

### 2. BH1750 Ambient Light Sensor (ESP32 DevKit V1)
1. Wiring:
   - `VCC` $\rightarrow$ 3.3V
   - `GND` $\rightarrow$ GND
   - `SDA` $\rightarrow$ GPIO 21
   - `SCL` $\rightarrow$ GPIO 22
2. Open `firmware/esp32_bh1750/esp32_bh1750.ino` in Arduino IDE.
3. Select Board: **ESP32 Dev Module**.
4. Upload sketch. The board automatically transmits JSON Lux telemetry over Serial at 115200 baud to `vision/sensor.py`.
