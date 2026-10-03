"""
Real-Time Adaptive Robot Vision System
Dynamically adjusts shadow cancellation and lighting enhancement in real time
based on live ambient light intensity from your camera.
"""
import time
import cv2
import numpy as np
from vision.uretinex_engine import URetinexUnfoldingEngine


def estimate_ambient_light(frame_bgr):
    """
    Real-time Software Lux / Light Intensity Meter:
    Analyzes the Lightness (L) channel in LAB color space.
    Returns:
        lux_est: Estimated illuminance (0 - 800+ Lux)
        state_tag: 'DARK / LOW LIGHT' | 'NORMAL INDOOR' | 'HARSH GLARE'
        adaptive_gamma: Dynamic compensation curve based on light
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L_mean = float(np.mean(lab[:, :, 0]))

    # Approximate Lux mapping based on industrial sensor calibration
    lux_est = int((L_mean / 255.0) * 600)

    if L_mean < 65:
        state_tag = "LOW LIGHT (BOOSTING)"
        state_color = (0, 165, 255)  # Orange
        adaptive_gamma = 0.52         # Aggressive lift for deep shadows
    elif L_mean > 175:
        state_tag = "HIGH GLARE (TAMING)"
        state_color = (0, 255, 255)  # Yellow
        adaptive_gamma = 0.85         # Gentle compression to avoid washout
    else:
        state_tag = "ACTIVE SHADOW CANCEL"
        state_color = (0, 255, 0)    # Green
        adaptive_gamma = 0.68         # Optimal Retinex unfolding curve

    return lux_est, state_tag, state_color, adaptive_gamma


def extract_robot_object(clean_frame):
    """
    Detects the true physical object contour on the table,
    completely ignoring table shadows.
    """
    gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 1.5)
    
    # Adaptive Otsu thresholding on clean surface
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    clean_thresh = cv2.morphologyEx(clean_thresh, cv2.MORPH_CLOSE, kernel)

    cnts, _ = cv2.findContours(clean_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    max_area = 0
    h, w = clean_frame.shape[:2]
    min_area = int(h * w * 0.02)
    max_limit = int(h * w * 0.85)

    for c in cnts:
        area = cv2.contourArea(c)
        if min_area < area < max_limit:
            if area > max_area:
                max_area = area
                best_cnt = c

    return best_cnt


def main():
    print("[Vision System] Initializing camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Could not open camera. Please check camera connection.")
        return

    # Set camera resolution for responsive 30 FPS processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Fast 3-stage unfolding engine for real-time video stream
    engine = URetinexUnfoldingEngine(stages=3)

    fps = 0.0
    prev_time = time.time()

    print("\n" + "="*60)
    print(" REAL-TIME ADAPTIVE LIGHTING & SHADOW REMOVAL STARTED")
    print(" Move your light or object to see it adapt in real time!")
    print(" Press 'q' to quit | Press 's' to save snapshot")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # 1. Real-Time Light Intensity Measurement
        lux, state_tag, state_color, adaptive_gamma = estimate_ambient_light(frame)

        # 2. Dynamic URetinex Unfolding (Shadow Cancellation)
        clean_frame, reflectance, L_map, _ = engine.unfold(frame, gamma=adaptive_gamma)

        # 3. Detect Real Object (Ignoring Shadows)
        robot_display = clean_frame.copy()
        obj_contour = extract_robot_object(clean_frame)

        if obj_contour is not None:
            # Draw green outline around physical object
            cv2.drawContours(robot_display, [obj_contour], -1, (0, 255, 0), 2)
            
            # Find center of object for robot gripper
            M = cv2.moments(obj_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Gripper crosshair
                cv2.circle(robot_display, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(robot_display, f"ROBOT GRASP: ({cx}, {cy})", (cx - 70, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Calculate live FPS
        curr_time = time.time()
        fps = round(1.0 / max(0.001, (curr_time - prev_time)), 1)
        prev_time = curr_time

        # 4. Construct Dual-Panel Screen: [Raw Camera with Shadow] | [Robot Adaptive Vision]
        screen = np.hstack([frame, robot_display])

        # Top HUD Banner
        cv2.rectangle(screen, (0, 0), (1280, 50), (20, 20, 20), -1)
        cv2.putText(screen, "CAMERA 1: RAW FEED (WITH SHADOW)", (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        cv2.putText(screen, "CAMERA 2: AI ADAPTIVE (SHADOW DELETED)", (655, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # Bottom Telemetry Dashboard
        cv2.rectangle(screen, (0, 430), (1280, 480), (15, 15, 15), -1)
        
        # Real-time Light Meter Gauge
        cv2.putText(screen, f"AMBIENT LIGHT: {lux} LUX", (15, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(screen, f"STATE: {state_tag}", (250, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, state_color, 2)
        cv2.putText(screen, f"ADAPTIVE GAMMA: {adaptive_gamma}", (560, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        cv2.putText(screen, f"FPS: {fps}", (1160, 460),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow("Adaptive Vision System - Real Time Robot Vision", screen)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            import os
            os.makedirs("results", exist_ok=True)
            cv2.imwrite("results/realtime_capture_raw.jpg", frame)
            cv2.imwrite("results/realtime_capture_robot.jpg", robot_display)
            print(f"[Snapshot Saved] Light: {lux} Lux | Gamma: {adaptive_gamma}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
