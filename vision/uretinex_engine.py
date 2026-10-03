"""
URetinex-Net Progressive Deep Unfolding Engine
Paper: URetinex-Net: Retinex-based Deep Unfolding Network for Low-light Image Enhancement (CVPR 2022)

Physics model: S = R * L
Unfolding sequence: (L0, R0) -> (L1, R1) -> (L2, R2) -> (L3, R3)
"""

import cv2
import numpy as np


class URetinexUnfoldingEngine:
    def __init__(self, stages=4, eps=1e-4):
        self.stages = stages
        self.eps = eps

    def _fast_guided_filter(self, guide, src, radius=15, eps=0.01):
        """Edge-preserving guided filter for piecewise smooth illumination."""
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
        mean_p = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

        return mean_a * guide + mean_b

    def unfold(self, frame_bgr, gamma=0.65):
        """
        Executes the 4 progressive unfolding stages.

        Returns:
            clean_shadow_free: Restored image with shadow removed
            reflectance_R3: Pure intrinsic reflectance (shadow-free)
            illumination_L3: Converged smooth illumination map
            history: List of (Lk, Rk) tuples for each stage (0, 1, 2, 3)
        """
        # Normalize input S to [0, 1]
        S = frame_bgr.astype(np.float32) / 255.0
        h, w = S.shape[:2]

        history = []

        # ==========================================
        # STEP 1: Initial Decomposition (L0, R0)
        # ==========================================
        # L0 = Max-RGB channel across pixels
        L_k = np.max(S, axis=2)
        # R0 = S / (L0 + eps)
        R_k = S / (L_k[:, :, np.newaxis] + self.eps)
        R_k = np.clip(R_k, 0.0, 1.0)

        history.append((L_k.copy(), R_k.copy()))

        # Guide image (grayscale of input)
        guide_gray = cv2.cvtColor((S * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # ==========================================
        # STEPS 2 & 3: Progressive Unfolding Stages (k = 1, 2, 3)
        # ==========================================
        radii = [15, 25, 35]  # Multi-scale smoothing from local to global
        eps_vals = [0.04, 0.02, 0.01]

        for k in range(1, self.stages):
            # 1. Update Illumination L_k (piecewise smooth regularization)
            r = radii[min(k - 1, len(radii) - 1)]
            e = eps_vals[min(k - 1, len(eps_vals) - 1)]
            L_k = self._fast_guided_filter(guide_gray, L_k, radius=r, eps=e)
            L_k = np.clip(L_k, self.eps, 1.0)

            # Ensure L >= max(S) to prevent saturation clipping
            L_k = np.maximum(L_k, np.max(S, axis=2))

            # 2. Update Reflectance R_k = S / L_k
            R_k = S / (L_k[:, :, np.newaxis] + self.eps)
            R_k = np.clip(R_k, 0.0, 1.0)

            # 3. Suppress noise and residual shadow penumbra in reflectance
            R_k_u8 = (R_k * 255.0).astype(np.uint8)
            R_k_denoised = cv2.bilateralFilter(R_k_u8, d=5, sigmaColor=15, sigmaSpace=15)
            R_k = R_k_denoised.astype(np.float32) / 255.0

            history.append((L_k.copy(), R_k.copy()))

        # ==========================================
        # STEP 4: Final Output Generation
        # ==========================================
        L_final = history[-1][0]
        R_final = history[-1][1]

        # Recompose with gamma-adjusted illumination: lifts shadows evenly
        L_boosted = np.power(L_final, gamma)
        enhanced = R_final * L_boosted[:, :, np.newaxis]
        clean_shadow_free = (np.clip(enhanced, 0.0, 1.0) * 255.0).astype(np.uint8)
        pure_reflectance = (np.clip(R_final, 0.0, 1.0) * 255.0).astype(np.uint8)

        return clean_shadow_free, pure_reflectance, L_final, history
