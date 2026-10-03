"""
Quick test: Run the full vision pipeline on the airpod shadow image.
Tests shadow removal, Retinex enhancement, and shape detection.
"""
import sys
import os
import time

# Add vision directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vision"))

import cv2
import numpy as np

print("=" * 60)
print("CEP Vision Pipeline - Full Integration Test")
print("=" * 60)

# Test 1: Import all modules
print("\n[1] Importing modules...")
try:
    from shadow_remover import RetinexShadowRemover
    print("  ✓ RetinexShadowRemover imported")
except Exception as e:
    print(f"  ✗ shadow_remover import failed: {e}")
    sys.exit(1)

try:
    from neural_lighting_restorer import MultiScaleRetinexRestorer
    print("  ✓ MultiScaleRetinexRestorer imported")
except Exception as e:
    print(f"  ✗ neural_lighting_restorer import failed: {e}")
    sys.exit(1)

try:
    from advanced_detector import TripleVotingShapeDetector
    print("  ✓ TripleVotingShapeDetector imported")
except Exception as e:
    print(f"  ✗ advanced_detector import failed: {e}")
    sys.exit(1)

try:
    from pipeline import VisionPipeline
    print("  ✓ VisionPipeline imported")
except Exception as e:
    print(f"  ✗ pipeline import failed: {e}")
    sys.exit(1)

# Test 2: Load airpod image
print("\n[2] Loading test image...")
img_path = "test_images/airpod_shadow.png"
if not os.path.exists(img_path):
    print(f"  ✗ Image not found: {img_path}")
    sys.exit(1)

img = cv2.imread(img_path)
print(f"  ✓ Loaded {img_path}: {img.shape[1]}x{img.shape[0]}")

# Test 3: Shadow removal standalone
print("\n[3] Testing Shadow Removal...")
t0 = time.time()
remover = RetinexShadowRemover()
shadow_free, shadow_mask, shadow_overlay, obj_contour = remover.process(img)
dt = (time.time() - t0) * 1000
shadow_pixels = cv2.countNonZero(shadow_mask) if shadow_mask is not None else 0
total_pixels = img.shape[0] * img.shape[1]
print(f"  ✓ Shadow detection: {shadow_pixels}/{total_pixels} pixels ({100*shadow_pixels/total_pixels:.1f}%)")
print(f"  ✓ Processing time: {dt:.1f} ms")

# Test 4: Retinex restoration standalone
print("\n[4] Testing Retinex Restoration...")
t0 = time.time()
restorer = MultiScaleRetinexRestorer()
restored, heatmap, mode = restorer.restore(img)
dt = (time.time() - t0) * 1000
gray_before = float(np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
gray_after = float(np.mean(cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)))
print(f"  ✓ Lighting state: {mode}")
print(f"  ✓ Mean brightness: {gray_before:.1f} → {gray_after:.1f}")
print(f"  ✓ Processing time: {dt:.1f} ms")

# Test 5: Full pipeline
print("\n[5] Testing Full Pipeline (Shadow → Retinex → Edges → Detection)...")
pipeline = VisionPipeline(min_area=500)

# Warmup
_ = pipeline.process_frame(img)

# Timed run
t0 = time.time()
annotated, detections = pipeline.process_frame(img)
dt = (time.time() - t0) * 1000

print(f"  ✓ Pipeline latency: {dt:.1f} ms ({1000/max(1,dt):.1f} FPS)")
print(f"  ✓ Lighting mode: {pipeline.last_mode}")
print(f"  ✓ Shapes detected: {len(detections)}")
for i, d in enumerate(detections):
    print(f"    [{i+1}] {d['shape']} ({d['confidence']:.0f}%) at {d['centroid']}")

# Save all outputs
print("\n[6] Saving results to test_images/pipeline_results/...")
out_dir = "test_images/pipeline_results"
os.makedirs(out_dir, exist_ok=True)

cv2.imwrite(os.path.join(out_dir, "1_original.jpg"), img)
cv2.imwrite(os.path.join(out_dir, "2_shadow_mask.jpg"), shadow_mask)
cv2.imwrite(os.path.join(out_dir, "3_shadow_overlay.jpg"), shadow_overlay)
cv2.imwrite(os.path.join(out_dir, "4_shadow_free.jpg"), pipeline.last_shadow_free)
cv2.imwrite(os.path.join(out_dir, "5_ai_restored.jpg"), pipeline.last_restored)
cv2.imwrite(os.path.join(out_dir, "6_illumination_map.jpg"), pipeline.last_heatmap)
cv2.imwrite(os.path.join(out_dir, "7_shape_detected.jpg"), annotated)

# Create 4-panel composite
four_panel = pipeline.create_four_panel(img)
cv2.imwrite(os.path.join(out_dir, "8_four_panel_composite.jpg"), four_panel)

print(f"  ✓ Saved 8 output images to {out_dir}/")

print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
