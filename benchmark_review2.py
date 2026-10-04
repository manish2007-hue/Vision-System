"""
Review-2 Automated Benchmark Evaluation & Testing Suite
Community Engagement Project (CEP) - AI & Robotics
Partner: SVR Robotics Pvt. Ltd., Pune

Outputs:
1. Automated execution across all 7 benchmark lighting scenes
2. Telemetry logging: Latency (ms), FPS, Shadow Reduction (%), Luminance Gain, Voting Confidence
3. High-resolution presentation graphs saved to results/review2_graphs/
4. Metrics export to results/review2_metrics.csv and review2_metrics.json
5. Composite 4-panel visual comparisons saved to results/review2_visuals/
"""

import os
import sys
import time
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Headless backend for server/script execution
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Ensure vision module is on path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from vision.pipeline import VisionPipeline
from vision.neural_lighting_restorer import MultiScaleRetinexRestorer
from vision.shadow_remover import RetinexShadowRemover


def evaluate_shadow_metrics(raw_bgr, shadow_mask, shadow_free_bgr):
    """
    Computes rigorous shadow reduction and luminance recovery metrics.
    """
    if shadow_mask is None or cv2.countNonZero(shadow_mask) == 0:
        return {
            "shadow_area_px": 0,
            "shadow_area_pct": 0.0,
            "shadow_reduction_pct": 0.0,
            "luminance_recovery_pct": 0.0,
            "ambient_lum": float(np.mean(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY))),
            "shadow_lum_before": 0.0,
            "shadow_lum_after": 0.0
        }

    h, w = raw_bgr.shape[:2]
    total_px = h * w
    shadow_px = int(cv2.countNonZero(shadow_mask))
    shadow_pct = round((shadow_px / total_px) * 100.0, 2)

    raw_gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    free_gray = cv2.cvtColor(shadow_free_bgr, cv2.COLOR_BGR2GRAY)

    # Ambient surface luminance (non-shadow pixels)
    non_shadow_mask = cv2.bitwise_not(shadow_mask)
    if cv2.countNonZero(non_shadow_mask) > 0:
        ambient_lum = float(np.median(raw_gray[non_shadow_mask > 0]))
    else:
        ambient_lum = 200.0

    # Luminance inside shadow region before vs after
    shadow_lum_before = float(np.mean(raw_gray[shadow_mask > 0]))
    shadow_lum_after = float(np.mean(free_gray[shadow_mask > 0]))

    # Luminance recovery percentage
    lum_deficit = max(1.0, ambient_lum - shadow_lum_before)
    lum_gain = max(0.0, shadow_lum_after - shadow_lum_before)
    luminance_recovery = min(100.0, round((lum_gain / lum_deficit) * 100.0, 1))

    # Shadow pixel reduction (pixels brought within 15% of ambient)
    shadow_thresh = ambient_lum - 20.0
    recovered_px = int(np.sum(free_gray[shadow_mask > 0] >= shadow_thresh))
    shadow_reduction_pct = min(100.0, round((recovered_px / max(1, shadow_px)) * 100.0, 1))

    return {
        "shadow_area_px": shadow_px,
        "shadow_area_pct": shadow_pct,
        "shadow_reduction_pct": shadow_reduction_pct,
        "luminance_recovery_pct": luminance_recovery,
        "ambient_lum": round(ambient_lum, 1),
        "shadow_lum_before": round(shadow_lum_before, 1),
        "shadow_lum_after": round(shadow_lum_after, 1)
    }


def compute_rms_contrast(gray_img):
    """Calculates RMS (Root Mean Square) contrast."""
    norm = gray_img.astype(np.float32) / 255.0
    return float(np.sqrt(np.mean((norm - np.mean(norm)) ** 2)))


def run_comprehensive_benchmark():
    os.makedirs(os.path.join(BASE_DIR, "results", "review2_graphs"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "results", "review2_visuals"), exist_ok=True)

    pipeline = VisionPipeline()

    test_scenarios = [
        {
            "id": "SCENE_1_NORMAL",
            "name": "Normal Lighting (Baseline)",
            "file": "normal_shapes.jpg",
            "condition": "Ideal (350 lux)",
            "lighting_cat": "Baseline"
        },
        {
            "id": "SCENE_2_SHADOW",
            "name": "Harsh Directional Shadow",
            "file": "harsh_shadow_shapes.jpg",
            "condition": "High Contrast Shadow (110 lux)",
            "lighting_cat": "Shadow"
        },
        {
            "id": "SCENE_3_LOWLIGHT",
            "name": "Extreme Low-Light",
            "file": "low_light_shapes.jpg",
            "condition": "Severe Underexposure (< 40 lux)",
            "lighting_cat": "Low-Light"
        },
        {
            "id": "SCENE_4_GLARE",
            "name": "Specular Spotlight Glare",
            "file": "harsh_glare_shapes.jpg",
            "condition": "Overexposed Glare (> 650 lux)",
            "lighting_cat": "Glare"
        },
        {
            "id": "SCENE_5_CHALLENGE",
            "name": "Hexagon vs Circle Ambiguity",
            "file": "hexagon_vs_circle_challenge.jpg",
            "condition": "SVR Classification Challenge",
            "lighting_cat": "Discrimination"
        },
        {
            "id": "SCENE_6_EARBUDS",
            "name": "Real Physical Earbuds Workpiece",
            "file": "airpod_shadow.png",
            "condition": "Curved Plastic on Table with Penumbra",
            "lighting_cat": "Physical Workpiece"
        },
        {
            "id": "SCENE_7_BREADBOARD",
            "name": "Real Physical Breadboard Workpiece",
            "file": "custom_1790937489_WhatsApp Image 2026-10-02 at 3.52.18 PM.jpeg",
            "condition": "High-Density Internal Pins + Cast Shadow",
            "lighting_cat": "Physical Workpiece"
        }
    ]

    benchmark_records = []
    hist_data = {}

    print("=" * 80)
    print("AI-POWERED ADAPTIVE VISION SYSTEM: REVIEW-2 BENCHMARK & EVALUATION RUN")
    print("Industrial Partner: SVR Robotics Pvt. Ltd., Pune")
    print("=" * 80)

    for item in test_scenarios:
        filename = item["file"]
        img_path = os.path.join(BASE_DIR, "test_images", filename)

        if not os.path.exists(img_path):
            print(f"[SKIP] File not found: {filename}")
            continue

        frame = cv2.imread(img_path)
        h, w = frame.shape[:2]

        # ── Micro-Benchmarking Individual Pipeline Stages ──
        # 1. Warm-up
        _ = pipeline.process_frame(frame)

        # 2. Timed Stage 1: Shadow Detection & Removal
        t0 = time.perf_counter()
        shadow_free, shadow_mask, shadow_overlay, obj_cnt = pipeline.shadow_remover.process(frame)
        t_stage1 = (time.perf_counter() - t0) * 1000.0

        # 3. Timed Stage 2: Retinex Illumination Normalization
        t0 = time.perf_counter()
        restored_bgr, heatmap, lighting_mode = pipeline.restorer.restore(shadow_free)
        t_stage2 = (time.perf_counter() - t0) * 1000.0

        # 4. Timed Stage 3 & 4: Adaptive Edge Extraction & Shape Detection
        t0 = time.perf_counter()
        annotated_frame, detections = pipeline.process_frame(frame)
        t_total = (time.perf_counter() - t0) * 1000.0
        t_detection = max(1.0, t_total - t_stage1 - t_stage2)

        fps = round(1000.0 / max(0.1, t_total), 1)

        # ── Compute Shadow and Luminance Metrics ──
        shadow_metrics = evaluate_shadow_metrics(frame, shadow_mask, shadow_free)

        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        restored_gray = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)

        mean_lum_raw = round(float(np.mean(raw_gray)), 1)
        mean_lum_res = round(float(np.mean(restored_gray)), 1)
        std_lum_raw = round(float(np.std(raw_gray)), 1)
        std_lum_res = round(float(np.std(restored_gray)), 1)

        rms_raw = round(compute_rms_contrast(raw_gray), 3)
        rms_res = round(compute_rms_contrast(restored_gray), 3)

        # Shapes summary
        num_shapes = len(detections)
        avg_confidence = round(float(np.mean([d["confidence"] for d in detections])), 1) if detections else 0.0
        shapes_detected_str = ", ".join([f"{d['shape']} ({d['confidence']:.0f}%)" for d in detections])

        # Store record
        rec = {
            "id": item["id"],
            "name": item["name"],
            "condition": item["condition"],
            "category": item["lighting_cat"],
            "resolution": f"{w}x{h}",
            "latency_total_ms": round(t_total, 1),
            "latency_shadow_ms": round(t_stage1, 1),
            "latency_retinex_ms": round(t_stage2, 1),
            "latency_detection_ms": round(t_detection, 1),
            "fps": fps,
            "lighting_state": lighting_mode,
            "shadow_reduction_pct": shadow_metrics["shadow_reduction_pct"],
            "luminance_recovery_pct": shadow_metrics["luminance_recovery_pct"],
            "lum_raw": mean_lum_raw,
            "lum_restored": mean_lum_res,
            "lum_std_raw": std_lum_raw,
            "lum_std_restored": std_lum_res,
            "rms_contrast_raw": rms_raw,
            "rms_contrast_restored": rms_res,
            "num_shapes": num_shapes,
            "avg_confidence": avg_confidence,
            "shapes_detail": shapes_detected_str,
            "detections": detections
        }
        benchmark_records.append(rec)

        # Cache histograms for distribution plot
        hist_data[item["id"]] = {
            "name": item["name"],
            "raw": raw_gray.flatten(),
            "restored": restored_gray.flatten()
        }

        # ── Save Visual 4-Panel Composites ──
        four_panel = pipeline.create_four_panel(frame)
        out_visual_path = os.path.join(BASE_DIR, "results", "review2_visuals", f"eval_{item['id']}.jpg")
        cv2.imwrite(out_visual_path, four_panel)

        print(f"\n[{item['id']}] {item['name']}")
        print(f"  Condition: {item['condition']} | Resolution: {w}x{h}")
        print(f"  Latency: Total = {t_total:.1f} ms (Shadow: {t_stage1:.1f}ms, Retinex: {t_stage2:.1f}ms, Detection: {t_detection:.1f}ms) -> {fps} FPS")
        print(f"  Lighting Mode: {lighting_mode} | Mean Brightness: {mean_lum_raw} -> {mean_lum_res}")
        if shadow_metrics["shadow_area_px"] > 0:
            print(f"  Shadow Reduction: {shadow_metrics['shadow_reduction_pct']}% | Luminance Recovery: {shadow_metrics['luminance_recovery_pct']}%")
        print(f"  Shapes Detected ({num_shapes}): {shapes_detected_str} [Mean Conf: {avg_confidence}%]")

    # ── Export Data Files ──
    # JSON
    json_path = os.path.join(BASE_DIR, "results", "review2_metrics.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_records, f, indent=2)

    # CSV
    csv_path = os.path.join(BASE_DIR, "results", "review2_metrics.csv")
    csv_fields = [
        "id", "name", "condition", "category", "resolution",
        "latency_total_ms", "latency_shadow_ms", "latency_retinex_ms", "latency_detection_ms",
        "fps", "lighting_state", "shadow_reduction_pct", "luminance_recovery_pct",
        "lum_raw", "lum_restored", "rms_contrast_raw", "rms_contrast_restored",
        "num_shapes", "avg_confidence", "shapes_detail"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(benchmark_records)

    print("\n" + "=" * 80)
    print("Generating High-Resolution Publication-Quality Graphs for Review-2...")
    print("=" * 80)
    generate_presentation_graphs(benchmark_records, hist_data)

    print("\n[SUCCESS] Benchmark Suite Complete!")
    print(f"  - CSV Log: {csv_path}")
    print(f"  - JSON Log: {json_path}")
    print(f"  - Graphs:  results/review2_graphs/")
    print(f"  - Visuals: results/review2_visuals/")


def generate_presentation_graphs(records, hist_data):
    """Generates 4 high-resolution plots for Review-2 presentation slides."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

    # ── GRAPH 1: Latency Breakdown & FPS by Lighting Scenario ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1.2]})

    scene_labels = [r["name"].replace(" (Baseline)", "").replace(" Ambiguity", "") for r in records]
    y_pos = np.arange(len(records))

    s1 = [r["latency_shadow_ms"] for r in records]
    s2 = [r["latency_retinex_ms"] for r in records]
    s3 = [r["latency_detection_ms"] for r in records]

    # Stacked horizontal bar chart
    ax1.barh(y_pos, s1, color="#ef4444", alpha=0.85, label="Stage 1: Shadow Removal (ms)")
    ax1.barh(y_pos, s2, left=s1, color="#3b82f6", alpha=0.85, label="Stage 2: Retinex MSRCR (ms)")
    s12 = [s1[i] + s2[i] for i in range(len(s1))]
    ax1.barh(y_pos, s3, left=s12, color="#10b981", alpha=0.85, label="Stage 3 & 4: Voting & Edges (ms)")

    # Latency numbers at ends
    for i in range(len(records)):
        tot = records[i]["latency_total_ms"]
        ax1.text(tot + 1.5, y_pos[i], f"{tot:.1f} ms", va="center", fontsize=9, fontweight="bold", color="#1e293b")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(scene_labels, fontsize=10, fontweight="bold")
    ax1.invert_yaxis()
    ax1.set_xlabel("Processing Latency (Milliseconds)", fontsize=11, fontweight="bold")
    ax1.set_title("Pipeline Stage Latency Breakdown Across Lighting Test Cases", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="lower right", frameon=True, fontsize=9)
    ax1.set_xlim(0, max([r["latency_total_ms"] for r in records]) * 1.32)
    ax1.legend(loc="upper right", frameon=True, fontsize=9)

    # Subplot 2: Real-Time FPS Gauge
    fps_vals = [r["fps"] for r in records]
    bar_colors = ["#10b981" if f >= 2.0 else "#f59e0b" for f in fps_vals]
    ax2.barh(y_pos, fps_vals, color=bar_colors, alpha=0.85)

    for i in range(len(records)):
        f = fps_vals[i]
        ax2.text(f + 0.08, y_pos[i], f"{f:.1f}", va="center", fontsize=9, fontweight="bold", color="#1e293b")

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])  # Share with ax1
    ax2.invert_yaxis()
    ax2.set_xlabel("Effective Frame Rate (FPS)", fontsize=11, fontweight="bold")
    ax2.set_title("Software Pipeline Throughput", fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlim(0, max(fps_vals) * 1.25)

    plt.tight_layout()
    graph1_path = os.path.join(BASE_DIR, "results", "review2_graphs", "1_latency_fps_benchmark.png")
    plt.savefig(graph1_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved {graph1_path}")

    # ── GRAPH 2: Shadow Reduction % & Luminance Recovery ──
    shadow_records = [r for r in records if r["shadow_reduction_pct"] > 0 or "SHADOW" in r["id"] or "EARBUDS" in r["id"] or "BREADBOARD" in r["id"]]
    if shadow_records:
        fig, ax = plt.subplots(figsize=(11, 6))
        names = [r["name"].replace(" (Baseline)", "").replace("Real Physical ", "") for r in shadow_records]
        x = np.arange(len(shadow_records))
        width = 0.35

        red_pcts = [r["shadow_reduction_pct"] for r in shadow_records]
        rec_pcts = [r["luminance_recovery_pct"] for r in shadow_records]

        rects1 = ax.bar(x - width/2, red_pcts, width, label="Shadow Pixel Attenuation (%)", color="#6366f1", alpha=0.9)
        rects2 = ax.bar(x + width/2, rec_pcts, width, label="Luminance Recovery Ratio (%)", color="#06b6d4", alpha=0.9)

        # Add data labels
        for rect in rects1:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel("Effectiveness Percentage (%)", fontsize=11, fontweight="bold")
        ax.set_title("Shadow Elimination & Luminance Restoration Performance", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=9, fontweight="bold", rotation=22, ha="right")
        ax.set_ylim(0, 118)
        ax.axhline(y=90.0, color="#10b981", linestyle=":", linewidth=1.5, label="Target Industrial Threshold (90%)")
        ax.legend(loc="upper right", frameon=True, fontsize=10)

        plt.tight_layout()
        graph2_path = os.path.join(BASE_DIR, "results", "review2_graphs", "2_shadow_reduction_performance.png")
        plt.savefig(graph2_path, dpi=300)
        plt.close()
        print(f"  [OK] Saved {graph2_path}")

    # ── GRAPH 3: Dynamic Range & Illumination Distributions ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Low light scenario distribution
    if "SCENE_3_LOWLIGHT" in hist_data:
        d = hist_data["SCENE_3_LOWLIGHT"]
        ax1.hist(d["raw"], bins=50, range=(0, 255), density=True, color="#ef4444", alpha=0.55, label="Raw Low-Light (<40 lux)")
        ax1.hist(d["restored"], bins=50, range=(0, 255), density=True, color="#10b981", alpha=0.65, label="After Retinexformer Enhancement")
        ax1.set_title("Low-Light Scene: Lifting Crushed Shadows", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Pixel Luminance (0=Black, 255=White)", fontsize=10)
        ax1.set_ylabel("Pixel Probability Density", fontsize=10)
        ax1.legend(loc="upper right", frameon=True)

    # Glare scenario distribution
    if "SCENE_4_GLARE" in hist_data:
        d = hist_data["SCENE_4_GLARE"]
        ax2.hist(d["raw"], bins=50, range=(0, 255), density=True, color="#f59e0b", alpha=0.55, label="Raw Glare Exposure (>650 lux)")
        ax2.hist(d["restored"], bins=50, range=(0, 255), density=True, color="#3b82f6", alpha=0.65, label="After Adaptive Glare Suppression")
        ax2.set_title("Specular Glare Scene: Highlight Compression", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Pixel Luminance (0=Black, 255=White)", fontsize=10)
        ax2.set_ylabel("Pixel Probability Density", fontsize=10)
        ax2.legend(loc="upper left", frameon=True)

    plt.suptitle("Illumination Map & Retinex Histogram Normalization Impact", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    graph3_path = os.path.join(BASE_DIR, "results", "review2_graphs", "3_illumination_distribution_shift.png")
    plt.savefig(graph3_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved {graph3_path}")

    # ── GRAPH 4: Triple-Voting Consensus & Classification Confidence ──
    fig, ax = plt.subplots(figsize=(11, 5.5))
    confs = [r["avg_confidence"] for r in records]
    cats = [r["name"].replace(" (Baseline)", "").replace(" Ambiguity", "") for r in records]

    colors = ["#10b981" if c >= 95 else "#3b82f6" for c in confs]
    bars = ax.bar(cats, confs, color=colors, width=0.55, alpha=0.88)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Mean Recognition Confidence (%)", fontsize=11, fontweight="bold")
    ax.set_title("Triple-Voting Shape Classifier Accuracy Across Adverse Lighting Conditions", fontsize=12, fontweight="bold", pad=12)
    ax.set_ylim(0, 115)
    ax.axhline(y=95.0, color="#10b981", linestyle="--", linewidth=1.5, label="95% High-Precision Robotic Threshold")
    ax.legend(loc="lower right", frameon=True)
    plt.xticks(rotation=20, ha="right", fontsize=9, fontweight="bold")

    plt.tight_layout()
    graph4_path = os.path.join(BASE_DIR, "results", "review2_graphs", "4_voting_confidence_consensus.png")
    plt.savefig(graph4_path, dpi=300)
    plt.close()
    print(f"  [OK] Saved {graph4_path}")

    # ── GRAPH 5: 4x4 Review-2 Visual Showcase Matrix ──
    generate_showcase_matrix()


def generate_showcase_matrix():
    """Constructs a high-impact 4x4 visual matrix comparing all stages across 4 key scenes."""
    pipeline = VisionPipeline()
    key_cases = [
        {"title": "1. Harsh Directional Shadow", "file": "harsh_shadow_shapes.jpg"},
        {"title": "2. Extreme Low-Light (<40 lux)", "file": "low_light_shapes.jpg"},
        {"title": "3. Specular Glare (>650 lux)", "file": "harsh_glare_shapes.jpg"},
        {"title": "4. Real Workpiece (Earbuds Case)", "file": "airpod_shadow.png"}
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 14))
    col_titles = [
        "Raw Degraded Input",
        "Stage 1: Shadow / Illum Map",
        "Stage 2: Retinex Normalized",
        "Stage 3 & 4: Consensus Shape HUD"
    ]

    for col_idx, ct in enumerate(col_titles):
        axes[0, col_idx].set_title(ct, fontsize=12, fontweight="bold", pad=10)

    for row_idx, case in enumerate(key_cases):
        path = os.path.join(BASE_DIR, "test_images", case["file"])
        if not os.path.exists(path):
            continue
        raw = cv2.imread(path)
        ann, det = pipeline.process_frame(raw)

        p1 = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        p2_img = pipeline.last_shadow_overlay if pipeline.last_shadow_overlay is not None else pipeline.last_heatmap
        p2 = cv2.cvtColor(p2_img, cv2.COLOR_BGR2RGB)
        p3 = cv2.cvtColor(pipeline.last_restored, cv2.COLOR_BGR2RGB)
        p4 = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)

        for col_idx, img_disp in enumerate([p1, p2, p3, p4]):
            ax = axes[row_idx, col_idx]
            ax.imshow(img_disp)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(case["title"], fontsize=11, fontweight="bold", labelpad=8)

    plt.suptitle("Adaptive Vision System: 4-Stage Illumination Invariance Showcase", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    matrix_path = os.path.join(BASE_DIR, "results", "review2_graphs", "5_review2_visual_showcase_matrix.png")
    plt.savefig(matrix_path, dpi=250)
    plt.close()
    print(f"  [OK] Saved {matrix_path}")


if __name__ == "__main__":
    run_comprehensive_benchmark()
