import cv2
import numpy as np

class LightingEngine:
    """
    Advanced Illumination-Invariant Vision Engine.
    Implements Retinex Theory (Land, 1977) and Chromaticity Space Gradient Filtering
    (Finlayson et al., 2006) to suppress physical shadows, glare, and ambient shifts.
    """
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process(self, frame, lux):
        """
        Takes raw BGR frame and ambient Lux.
        Returns:
            - normalized_bgr: Illumination-normalized image (shadows lifted, glare suppressed)
            - invariant_edges: Clean edge map containing ONLY physical object geometry
            - mode: "DARK" | "NORMAL" | "GLARE"
        """
        # 1. Classify Lux state
        if lux < 80:
            mode = "DARK"
        elif lux <= 500:
            mode = "NORMAL"
        else:
            mode = "GLARE"

        # 2. Retinex Illumination-Reflectance Decomposition
        # Separate illumination L (low frequency) from reflectance R (high frequency)
        float_img = frame.astype(np.float32) + 1.0
        # Estimate illumination map using wide spatial Gaussian filter
        illumination = cv2.GaussianBlur(float_img, (31, 31), 15.0)
        # Reflectance R = Image / Illumination
        reflectance = float_img / (illumination + 1e-5)
        reflectance = cv2.normalize(reflectance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 3. Chromaticity Gradient Calculation (Shadow-Proof Edges)
        # Compute normalized color ratios: r = R/(R+G+B), g = G/(R+G+B)
        # Shadows alter intensity (R+G+B) but leave (r, g) invariant!
        b, g, r = cv2.split(float_img)
        intensity_sum = b + g + r + 1e-5
        norm_r = (r / intensity_sum) * 255.0
        norm_g = (g / intensity_sum) * 255.0

        norm_r = norm_r.astype(np.uint8)
        norm_g = norm_g.astype(np.uint8)

        # Sobel gradients on Chromaticity
        grad_r_x = cv2.Sobel(norm_r, cv2.CV_32F, 1, 0, ksize=3)
        grad_r_y = cv2.Sobel(norm_r, cv2.CV_32F, 0, 1, ksize=3)
        mag_r = cv2.magnitude(grad_r_x, grad_r_y)

        grad_g_x = cv2.Sobel(norm_g, cv2.CV_32F, 1, 0, ksize=3)
        grad_g_y = cv2.Sobel(norm_g, cv2.CV_32F, 0, 1, ksize=3)
        mag_g = cv2.magnitude(grad_g_x, grad_g_y)

        chroma_edge = cv2.addWeighted(mag_r, 0.5, mag_g, 0.5, 0)
        chroma_edge = cv2.normalize(chroma_edge, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # 4. Adaptive Mode Processing
        gray_reflectance = cv2.cvtColor(reflectance, cv2.COLOR_BGR2GRAY)

        if mode == "DARK":
            # Boost local dark areas
            enhanced = self.clahe.apply(gray_reflectance)
            denoised = cv2.bilateralFilter(enhanced, 7, 75, 75)
            canny = cv2.Canny(denoised, 20, 60)
            # Combine reflectance Canny with chromaticity edges
            invariant_edges = cv2.bitwise_or(canny, cv2.threshold(chroma_edge, 25, 255, cv2.THRESH_BINARY)[1])

        elif mode == "NORMAL":
            denoised = cv2.GaussianBlur(gray_reflectance, (5, 5), 1.0)
            canny = cv2.Canny(denoised, 35, 100)
            invariant_edges = cv2.bitwise_or(canny, cv2.threshold(chroma_edge, 35, 255, cv2.THRESH_BINARY)[1])

        else: # GLARE / BRIGHT
            # In bright light, suppress specular highlights
            _, mask_glare = cv2.threshold(gray_reflectance, 240, 255, cv2.THRESH_BINARY)
            denoised = cv2.bilateralFilter(gray_reflectance, 9, 80, 80)
            canny = cv2.Canny(denoised, 50, 150)
            # Mask out glare reflections
            canny = cv2.bitwise_and(canny, cv2.bitwise_not(mask_glare))
            invariant_edges = cv2.bitwise_or(canny, cv2.threshold(chroma_edge, 45, 255, cv2.THRESH_BINARY)[1])

        # Morphological refinement: close tiny boundary gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        invariant_edges = cv2.morphologyEx(invariant_edges, cv2.MORPH_CLOSE, kernel)

        return reflectance, invariant_edges, mode
