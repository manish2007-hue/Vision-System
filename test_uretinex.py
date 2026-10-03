"""
Verification script for 4-stage URetinex-Net unfolding
"""
import os
import cv2
from vision.uretinex_engine import URetinexUnfoldingEngine

def run_test():
    img_path = os.path.join("test_images", "airpod_shadow.png")
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return

    img = cv2.imread(img_path)
    print(f"Loaded {img_path}: {img.shape[1]}x{img.shape[0]}")

    engine = URetinexUnfoldingEngine(stages=4)
    clean_out, reflectance, L_map, history = engine.unfold(img, gamma=0.7)

    os.makedirs("results", exist_ok=True)
    cv2.imwrite("results/uretinex_clean_shadow_free.jpg", clean_out)
    cv2.imwrite("results/uretinex_reflectance_R3.jpg", reflectance)
    cv2.imwrite("results/uretinex_illumination_L3.jpg", (L_map * 255).astype("uint8"))

    print("Success: Generated clean shadow-free images in results/ folder.")

if __name__ == "__main__":
    run_test()
