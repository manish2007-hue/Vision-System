import cv2
import numpy as np
import math

def get_vertices(contour):
    perimeter = cv2.arcLength(contour, True)
    epsilon = 0.025 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return len(approx)

def get_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4.0 * math.pi * area) / (perimeter ** 2)

def classify_shape(contour, vertices, circularity):
    """
    Dual-Method Voting:
    Method 1: Vertex Count (approxPolyDP)
    Method 2: Circularity Ratio (area / perimeter^2)
    """
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0

    # Circularity benchmarks:
    # Circle:   > 0.92
    # Hexagon:  0.81 - 0.91
    # Square:   0.72 - 0.80
    # Triangle: 0.50 - 0.70

    if circularity > 0.92:
        return "Circle", 98.5
    elif 0.81 <= circularity <= 0.91:
        # Crucial fix for Hexagon vs Circle problem!
        return "Hexagon", 96.0
    elif vertices == 3 or (0.50 <= circularity < 0.70 and vertices <= 4):
        return "Triangle", 95.0
    elif vertices == 4 or (0.72 <= circularity < 0.81):
        if 0.88 <= aspect_ratio <= 1.12:
            return "Square", 96.0
        else:
            return "Rectangle", 94.0
    elif vertices == 5:
        return "Pentagon", 92.0
    elif vertices == 6:
        return "Hexagon", 95.0
    elif circularity > 0.85:
        return "Circle", 88.0

    return "Unknown", 60.0

def detect_shapes(edges, frame):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated = frame.copy()
    detections = []

    # Sort contours by area descending
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 600 or area > (frame.shape[0] * frame.shape[1] * 0.90):
            continue

        vertices = get_vertices(cnt)
        circularity = get_circularity(cnt)
        shape, confidence = classify_shape(cnt, vertices, circularity)

        if shape != "Unknown":
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)
            label = f"{shape} ({confidence:.0f}%)"
            cv2.putText(annotated, label, (x, max(20, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            detections.append({
                "shape": shape,
                "confidence": confidence,
                "circularity": round(circularity, 3),
                "vertices": vertices,
                "area": int(area)
            })

    return annotated, detections
