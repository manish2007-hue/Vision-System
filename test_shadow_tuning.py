"""
Specialized Test & Tuning Script for Real-World Objects & Shadow Removal
Focus: Clean removal of shadow without leaving halo borders, and detecting
the true outer boundary of the object (e.g. rounded rectangle earbuds case).
"""
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vision"))

from shadow_remover import RetinexShadowRemover
from neural_lighting_restorer import MultiScaleRetinexRestorer
from advanced_detector import TripleVotingShapeDetector

def test_shadow_and_object():
    img_path = "test_images/airpod_shadow.png"
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"Loaded image: {w}x{h}")

    # 1. Advanced Shadow Detection & Seamless Inpainting / Relighting
    # Let's inspect the background color surrounding the shadow
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    
    # We notice the shadow is the dark curved crescent
    # L channel Otsu finds dark regions
    L_blur = cv2.GaussianBlur(L, (7, 7), 2.0)
    otsu_val, _ = cv2.threshold(L_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Shadow mask
    shadow_mask = np.zeros((h, w), dtype=np.uint8)
    shadow_mask[L_blur < int(otsu_val * 0.88)] = 255
    
    # Exclude image borders
    border = 20
    shadow_mask[:border, :] = 0
    shadow_mask[-border:, :] = 0
    shadow_mask[:, :border] = 0
    shadow_mask[:, -border:] = 0
    
    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # To eliminate the artificial bright halo border:
    # 1. Dilate shadow mask slightly so it covers the penumbra transition completely
    dilated_mask = cv2.dilate(shadow_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations=2)
    
    # Sample the clean background around the shadow to get target illumination
    # Create ring around shadow
    outer_ring = cv2.dilate(dilated_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)), iterations=1)
    bg_sample_mask = cv2.subtract(outer_ring, dilated_mask)
    
    # Calculate median background color
    bg_bgr = [np.median(img[:, :, c][bg_sample_mask > 0]) for c in range(3)]
    print("Estimated ambient background BGR around shadow:", bg_bgr)
    
    # Relight shadow area smoothly towards background color or inpaint
    # Telea inpainting or seamless illumination adjustment
    shadow_free = img.copy()
    
    # Inpaint the shadow region seamlessly using Navier-Stokes / Telea
    # This replaces the cast shadow on the desk directly with the desk texture!
    inpainted = cv2.inpaint(img, dilated_mask, inpaintRadius=9, flags=cv2.INPAINT_TELEA)
    
    # Smooth blend between inpainted and original
    blend_mask = cv2.GaussianBlur(dilated_mask.astype(np.float32) / 255.0, (21, 21), 7)
    blend_mask_3d = blend_mask[:, :, np.newaxis]
    shadow_free = (inpainted.astype(np.float32) * blend_mask_3d + img.astype(np.float32) * (1.0 - blend_mask_3d)).astype(np.uint8)

    # 2. Object Boundary Detection (Targeting the earbuds case)
    # The earbuds case is a bright object in the center
    # Otsu or gradient magnitude
    gray = cv2.cvtColor(shadow_free, cv2.COLOR_BGR2GRAY)
    
    # High-pass or bilateral
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Finding the main case contour
    # Canny on shadow_free
    edges = cv2.Canny(filtered, 20, 60)
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, iterations=2)
    
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contour closest to the center with significant area
    annotated = shadow_free.copy()
    case_cnt = None
    max_score = 0
    center_y, center_x = h // 2, w // 2
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.05) or area > (h * w * 0.70):
            continue
        # Solidity & bounding box
        x, y, bw, bh = cv2.boundingRect(cnt)
        dist_to_center = np.hypot(x + bw/2 - center_x, y + bh/2 - center_y)
        score = area / (dist_to_center + 100)
        if score > max_score:
            max_score = score
            case_cnt = cnt
            
    if case_cnt is not None:
        # Approximate outline
        peri = cv2.arcLength(case_cnt, True)
        approx = cv2.approxPolyDP(case_cnt, 0.015 * peri, True)
        cv2.drawContours(annotated, [case_cnt], -1, (0, 255, 0), 3)
        rect = cv2.minAreaRect(case_cnt)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        cv2.drawContours(annotated, [box], 0, (0, 215, 255), 2)
        
        (cx, cy), (rw, rh), angle = rect
        label = f"Earbuds Case (Rounded Rect) [{rw:.0f}x{rh:.0f}]"
        cv2.putText(annotated, label, (int(cx) - 130, int(cy) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        cv2.circle(annotated, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        print(f"Case detected at centroid ({cx:.1f}, {cy:.1f}), size: {rw:.1f}x{rh:.1f}, angle: {angle:.1f} deg")
    else:
        print("Main case contour not isolated with standard Canny.")
        
    out_dir = "test_images/pipeline_results"
    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_dir, "test_inpainted_shadow_free.jpg"), shadow_free)
    cv2.imwrite(os.path.join(out_dir, "test_object_annotated.jpg"), annotated)
    cv2.imwrite(os.path.join(out_dir, "test_dilated_mask.png"), dilated_mask)
    print("Saved test results to pipeline_results/")

if __name__ == "__main__":
    test_shadow_and_object()
