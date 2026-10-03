"""
Synthetic Test Image Generator for Adaptive Vision System
CEP Project: AI-Powered Adaptive Vision System

Generates high-fidelity industrial benchmark test scenes with geometric shapes
under harsh real-world lighting conditions:
 1. Normal Lighting (Baseline)
 2. Harsh Directional Shadow (Simulating SVR sunlight through window)
 3. Low Light (Dim workshop environment)
 4. Harsh Glare / Flashlight (Overexposed reflection)
 5. Hexagon vs Circle Discriminator (The core SVR challenge)
"""

import math
import os
import cv2
import numpy as np


def draw_regular_polygon(img, center, radius, sides, color, angle_offset_deg=0):
    pts = []
    angle_offset_rad = math.radians(angle_offset_deg)
    for i in range(sides):
        theta = i * (2 * math.pi / sides) + angle_offset_rad
        x = int(center[0] + radius * math.cos(theta))
        y = int(center[1] + radius * math.sin(theta))
        pts.append([x, y])
    cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], color)
    return pts


def create_base_scene(width=800, height=600):
    """Creates an industrial workspace tabletop with four distinct shapes."""
    # Warm grey tabletop surface with subtle texture
    scene = np.ones((height, width, 3), dtype=np.uint8) * 220
    # Add slight texture/noise
    noise = np.random.normal(0, 4, scene.shape).astype(np.int16)
    scene = np.clip(scene.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 1. Triangle (Vibrant Red 3D printed part)
    tri_center = (160, 220)
    tri_pts = []
    for i in range(3):
        theta = i * (2 * math.pi / 3) - (math.pi / 2)
        x = int(tri_center[0] + 75 * math.cos(theta))
        y = int(tri_center[1] + 75 * math.sin(theta))
        tri_pts.append([x, y])
    cv2.fillPoly(scene, [np.array(tri_pts, dtype=np.int32)], (35, 35, 205))

    # 2. Square (Royal Blue 3D printed part)
    sq_center = (400, 220)
    side = 65
    cv2.rectangle(
        scene,
        (sq_center[0] - side, sq_center[1] - side),
        (sq_center[0] + side, sq_center[1] + side),
        (205, 75, 35),
        -1,
    )

    # 3. Hexagon (Emerald Green 3D printed part - SVR challenge)
    hex_center = (640, 220)
    draw_regular_polygon(scene, hex_center, 75, 6, (40, 175, 45), angle_offset_deg=30)

    # 4. Circle (Bright Amber/Orange 3D printed part)
    circ_center = (400, 440)
    cv2.circle(scene, circ_center, 70, (25, 150, 235), -1)

    return scene


def apply_harsh_shadow(img):
    """Casts a diagonal harsh shadow simulating sunlight from a window."""
    h, w = img.shape[:2]
    shadow_mask = np.zeros((h, w), dtype=np.float32)

    # Diagonal shadow line across the scene (e.g., cutting through square and hexagon)
    pts = np.array([[0, 100], [w, 350], [w, h], [0, h]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [pts], 0.70)  # 70% reduction in brightness in shadow area

    # Blur the shadow boundary to mimic realistic penumbra
    shadow_mask = cv2.GaussianBlur(shadow_mask, (35, 35), 0)

    # Apply shadow attenuation
    b, g, r = cv2.split(img.astype(np.float32))
    b = b * (1.0 - shadow_mask)
    g = g * (1.0 - shadow_mask)
    r = r * (1.0 - shadow_mask)

    shadowed = cv2.merge([b, g, r])
    return np.clip(shadowed, 0, 255).astype(np.uint8)


def apply_low_light(img):
    """Simulates low ambient lighting (dim room, night inspection)."""
    # Reduce brightness to 25% and add camera sensor noise
    dimmed = (img.astype(np.float32) * 0.28).astype(np.uint8)
    noise = np.random.normal(0, 10, dimmed.shape).astype(np.int16)
    noisy = np.clip(dimmed.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_harsh_glare(img):
    """Simulates overexposure / harsh flashlight from one side."""
    h, w = img.shape[:2]
    glare_map = np.zeros((h, w), dtype=np.float32)

    # Spotlight center at top-left
    spot_center = (200, 180)
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.hypot(x_coords - spot_center[0], y_coords - spot_center[1])
    glare_map = np.clip(1.0 - dist / 400.0, 0.0, 1.0).astype(np.float32)

    glare_map = cv2.GaussianBlur(glare_map, (41, 41), 0)
    boosted = img.astype(np.float32) + (glare_map[:, :, np.newaxis] * 160.0)
    return np.clip(boosted, 0, 255).astype(np.uint8)


def create_hexagon_vs_circle_challenge(width=800, height=480):
    """Specialized scene placing Hexagon and Circle under identical shadow for direct comparison."""
    scene = np.ones((height, width, 3), dtype=np.uint8) * 215
    # Emerald Green Hexagon on left
    draw_regular_polygon(scene, (260, 240), 90, 6, (40, 175, 45), angle_offset_deg=0)
    # Royal Blue Circle on right with exact same apparent radius
    cv2.circle(scene, (540, 240), 90, (205, 75, 35), -1)

    # Diagonal shadow slicing right across both shapes
    shadow_mask = np.zeros((height, width), dtype=np.float32)
    pts = np.array([[100, 0], [w_poly := width, 250], [width, height], [0, height]], dtype=np.int32)
    cv2.fillPoly(shadow_mask, [pts], 0.65)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (25, 25), 0)

    b, g, r = cv2.split(scene.astype(np.float32))
    b = b * (1.0 - shadow_mask)
    g = g * (1.0 - shadow_mask)
    r = r * (1.0 - shadow_mask)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


def generate_all(output_dir="test_images"):
    os.makedirs(output_dir, exist_ok=True)

    base = create_base_scene()
    cv2.imwrite(os.path.join(output_dir, "normal_shapes.jpg"), base)

    shadowed = apply_harsh_shadow(base.copy())
    cv2.imwrite(os.path.join(output_dir, "harsh_shadow_shapes.jpg"), shadowed)

    low_light = apply_low_light(base.copy())
    cv2.imwrite(os.path.join(output_dir, "low_light_shapes.jpg"), low_light)

    glare = apply_harsh_glare(base.copy())
    cv2.imwrite(os.path.join(output_dir, "harsh_glare_shapes.jpg"), glare)

    hex_vs_circ = create_hexagon_vs_circle_challenge()
    cv2.imwrite(os.path.join(output_dir, "hexagon_vs_circle_challenge.jpg"), hex_vs_circ)

    print(f"Generated 5 benchmark test images in '{output_dir}/'")


if __name__ == "__main__":
    generate_all()
