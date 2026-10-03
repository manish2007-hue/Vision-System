"""
Benchmark and Verification Script for Vision Pipeline
CEP Project: AI-Powered Adaptive Vision System

Processes all benchmark test images in test_images/, applies:
 1. Raw vs Processed comparison
 2. Triple-method voting shape identification
 3. Output verification in results/ directory
"""

import os
import cv2
from vision.pipeline import VisionPipeline


def run_benchmark():
    os.makedirs("results", exist_ok=True)
    pipeline = VisionPipeline()

    test_files = [
        "normal_shapes.jpg",
        "harsh_shadow_shapes.jpg",
        "low_light_shapes.jpg",
        "harsh_glare_shapes.jpg",
        "hexagon_vs_circle_challenge.jpg",
    ]

    print("=" * 70)
    print("AI-POWERED ADAPTIVE VISION SYSTEM: BENCHMARK TEST RUN")
    print("=" * 70)

    for filename in test_files:
        path = os.path.join("test_images", filename)
        if not os.path.exists(path):
            print(f"[SKIP] {filename} not found.")
            continue

        raw = cv2.imread(path)
        annotated, detections = pipeline.process_frame(raw)
        side_by_side = pipeline.create_side_by_side(raw, annotated)

        # Save annotated result
        out_name = f"result_{filename}"
        out_path = os.path.join("results", out_name)
        cv2.imwrite(out_path, side_by_side)

        print(f"\n[SCENE: {filename}]")
        print(f"  Detected Shapes Count: {len(detections)}")
        for i, d in enumerate(detections, 1):
            angles_str = ", ".join(f"{a}°" for a in d.get("angles", [])[:6])
            print(
                f"   {i}. Shape: {d['shape']:<10} | Confidence: {d['confidence']}% | "
                f"Circularity: {d['circularity']:<5} | Vertices: {d['vertices']} | "
                f"Votes: (Vtx={d['votes']['vertex_method']}, Circ={d['votes']['circularity_method']}, Hu={d['votes']['hu_moments_method']})"
            )
            if angles_str:
                print(f"      Internal Angles: [{angles_str}]")

    print("\n" + "=" * 70)
    print("Benchmark complete! All annotated comparisons saved in 'results/' directory.")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
