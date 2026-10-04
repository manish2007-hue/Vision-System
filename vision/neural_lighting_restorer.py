"""
Multi-Scale Retinex with Color Restoration Engine
Community Engagement Project (CEP) - SE Robotics & AI

Research Paper Foundations:
1. Guo et al. (IEEE TIP 2017) - LIME: Low-Light Image Enhancement via Illumination Map Estimation
2. Jobson et al. (1997) - A Multiscale Retinex for Bridging the Gap Between Color Images and the Human Observation of Scenes
3. Cai et al. (ICCV 2023) - Retinexformer: One-stage Retinex-based Transformer
4. Wu et al. (CVPR 2022) - URetinex-Net: Retinex-Based Deep Unfolding Network
5. Li et al. (IEEE TIP 2018) - Structure-Revealing Low-Light Image Enhancement via Robust Retinex Model

Key Equations:
- LIME Illumination Map:  T_init(x) = max_{c} I_c(x)                    [Guo Eq.3]
- Retinex Decomposition:  I(x) = R(x) * T(x)                            [Land 1977]
- MSR:  R_msr = sum_n { w_n * [log(I) - log(G_n * I)] }                 [Jobson 1997]
- Color Restoration:  C_i = beta * log(alpha * I_i / sum(I_j))           [MSRCR]
- Light-Up Map:  I_enh = I * L_hat, where L_hat = (target/T)^gamma      [Cai 2023]
"""

import cv2
import numpy as np


class MultiScaleRetinexRestorer:
    """
    Adaptive Multi-Scale Retinex with Color Restoration (MSRCR) combined with
    LIME illumination map estimation and Retinexformer-inspired light-up map.

    The engine automatically adjusts its processing intensity based on
    scene analysis — applying aggressive enhancement only when needed
    (dark scenes) and gentle processing for well-lit scenes.
    """

    def __init__(self, scales=(15, 80, 200)):
        self.scales = scales
        self.eps = 1e-6

    def _guided_filter(self, guide, src, r, eps):
        """
        Fast guided filter for edge-preserving illumination smoothing.
        Preserves structure boundaries while smoothing flat regions.
        Reference: He et al. "Guided Image Filtering" (ECCV 2010)
        """
        mean_I = cv2.boxFilter(guide, cv2.CV_32F, (r, r))
        mean_p = cv2.boxFilter(src, cv2.CV_32F, (r, r))
        mean_Ip = cv2.boxFilter(guide * src, cv2.CV_32F, (r, r))
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(guide * guide, cv2.CV_32F, (r, r))
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, (r, r))
        mean_b = cv2.boxFilter(b, cv2.CV_32F, (r, r))

        return mean_a * guide + mean_b

    def _multi_scale_retinex(self, img_f):
        """
        Multi-Scale Retinex (Jobson et al. 1997)

        MSR(x,y) = sum_n { w_n * [log(I(x,y)) - log(G_n * I(x,y))] }

        Uses 3 Gaussian scales to capture local, medium, and global contrast.
        """
        log_img = np.log(img_f + 1.0)
        msr = np.zeros_like(img_f)
        w = 1.0 / len(self.scales)

        for sigma in self.scales:
            blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
            msr += w * (log_img - np.log(blur + 1.0))

        return msr

    def _color_restoration(self, img_f, alpha=125.0, beta=46.0):
        """
        Color Restoration Factor (MSRCR - Jobson et al. 1997)

        C_i(x,y) = beta * log(alpha * I_i(x,y) / sum_j(I_j(x,y)))

        Prevents the grayish appearance that raw MSR produces.
        """
        img_sum = np.sum(img_f, axis=2, keepdims=True) + self.eps
        C = beta * (np.log(alpha * img_f + 1.0) - np.log(img_sum + 1.0))
        return C

    def _retinex_lightup(self, img_f, T_smooth, target=0.75, gamma=0.65):
        """
        Retinexformer-Inspired Multiplicative Light-Up (Cai et al. ICCV 2023)

        Instead of dividing by illumination (amplifies noise), multiply:
        I_enhanced = I * L_hat
        where L_hat = (target / T_norm)^gamma

        This is more stable and produces fewer artifacts than division.
        """
        T_norm = np.clip(T_smooth, self.eps, 1.0)
        L_hat = np.power(target / T_norm, gamma)
        # Clamp light-up factor to prevent extreme amplification
        L_hat = np.clip(L_hat, 0.5, 3.0)
        enhanced = img_f * L_hat[:, :, np.newaxis]
        return np.clip(enhanced, 0.0, 1.0)

    def restore(self, frame, lux_reading=None):
        """
        Process a single BGR frame through the adaptive Retinex pipeline.

        Args:
            frame: Input BGR image (uint8)
            lux_reading: Optional ambient lux from BH1750 sensor

        Returns:
            restored_bgr: Enhanced BGR image (uint8)
            gain_heatmap: False-color illumination map for inspection
            lighting_state: 'DARK_BOOST' | 'NORMALIZED' | 'GLARE_SUPPRESSION'
        """
        if frame is None or frame.size == 0:
            return frame, None, "OFFLINE"

        h, w = frame.shape[:2]
        img_f = frame.astype(np.float32) / 255.0

        # ── Scene Analysis ──
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        mean_L = float(np.mean(lab[:, :, 0]))

        valid_lux = None
        if lux_reading is not None and isinstance(lux_reading, (int, float)) and lux_reading > 0:
            valid_lux = float(lux_reading)

        if valid_lux is not None:
            is_dark = valid_lux < 70 or mean_L < 55
            is_bright = valid_lux > 500 or mean_L > 185
        else:
            is_dark = mean_L < 70
            is_bright = mean_L > 185

        if is_dark:
            lighting_state = "DARK_BOOST"
            lightup_target = 0.85
            lightup_gamma = 0.50
            msrcr_gain = 5.0       # Strong MSRCR for dark images
            clahe_clip = 3.0
            color_boost = 1.15
            denoise_strength = 0.6
        elif is_bright:
            lighting_state = "GLARE_SUPPRESSION"
            lightup_target = 0.55
            lightup_gamma = 1.1
            msrcr_gain = 1.5
            clahe_clip = 1.0
            color_boost = 1.0
            denoise_strength = 0.1
        else:
            lighting_state = "NORMALIZED"
            lightup_target = 0.70
            lightup_gamma = 0.80
            msrcr_gain = 2.5
            clahe_clip = 1.5
            color_boost = 1.05
            denoise_strength = 0.2

        # ── 1. LIME Illumination Map (Guo et al. Eq. 3, 4, 7) ──
        T_init = np.max(img_f, axis=2)  # Max-RGB initial estimate

        # Dual filtering: bilateral for edge preservation + guided for structure
        T_u8 = (T_init * 255.0).astype(np.uint8)
        # Downsample for speed
        scale = max(1, min(h, w) // 320)
        if scale > 1:
            T_small = cv2.resize(T_u8, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        else:
            T_small = T_u8
        T_bilateral = cv2.bilateralFilter(T_small, d=7, sigmaColor=30, sigmaSpace=30)
        if scale > 1:
            T_bilateral = cv2.resize(T_bilateral, (w, h), interpolation=cv2.INTER_LINEAR)
        T_bilateral_f = T_bilateral.astype(np.float32) / 255.0

        # Guided filter for structure-aware refinement
        T_smooth = self._guided_filter(T_init, T_bilateral_f, r=15, eps=0.01)
        T_smooth = np.clip(T_smooth, self.eps, 1.0)

        # Ensure T >= max(I) everywhere (Guo et al. Eq. 4 - prevents saturation)
        T_smooth = np.maximum(T_smooth, T_init)

        # ── 2. Retinexformer Light-Up Map ──
        enhanced = self._retinex_lightup(img_f, T_smooth, lightup_target, lightup_gamma)

        # ── 3. Multi-Scale Retinex in CIE-LAB Luminance Space (3x Fast Pass) ──
        # Operating on L-channel prevents costly 3-channel Gaussian loops and preserves color fidelity
        L_enh = lab[:, :, 0].astype(np.float32) / 255.0
        # Multi-scale downscale proxy for high-resolution images
        msr_scale = max(1, min(h, w) // 280)
        if msr_scale > 1:
            L_small = cv2.resize(L_enh, (w // msr_scale, h // msr_scale), interpolation=cv2.INTER_AREA)
        else:
            L_small = L_enh

        log_L = np.log(L_small + 1.0)
        msr_L_small = np.zeros_like(L_small)
        w_msr = 1.0 / len(self.scales)
        for sigma in self.scales:
            s_scaled = max(1.0, sigma / float(msr_scale))
            blur = cv2.GaussianBlur(L_small, (0, 0), s_scaled)
            msr_L_small += w_msr * (log_L - np.log(blur + 1.0))

        if msr_scale > 1:
            msr_L = cv2.resize(msr_L_small, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            msr_L = msr_L_small

        # Normalize MSR luminance response
        mn_l, mx_l = msr_L.min(), msr_L.max()
        if mx_l > mn_l:
            msr_L_norm = (msr_L - mn_l) / (mx_l - mn_l)
        else:
            msr_L_norm = L_enh

        # Blend Retinexformer light-up with MSR luminance
        if lighting_state == "DARK_BOOST":
            alpha_msr = 0.35
            blend_gain = (1.0 - alpha_msr) + alpha_msr * msr_L_norm[:, :, np.newaxis]
            blend = np.clip(enhanced * blend_gain * msrcr_gain * 0.4, 0.0, 1.0)
        else:
            alpha_msr = 0.12 if lighting_state == "NORMALIZED" else 0.06
            blend_gain = (1.0 - alpha_msr) + alpha_msr * msr_L_norm[:, :, np.newaxis]
            blend = np.clip(enhanced * blend_gain, 0.0, 1.0)


        # ── 4. CLAHE on L Channel (Adaptive Local Contrast) ──
        blend_u8 = (blend * 255.0).astype(np.uint8)
        lab_enh = cv2.cvtColor(blend_u8, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        lab_enh[:, :, 0] = clahe.apply(lab_enh[:, :, 0])
        blend_u8 = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)

        # ── 5. Illumination-Weighted Denoising ──
        if denoise_strength > 0.05:
            denoised = cv2.bilateralFilter(blend_u8, d=5, sigmaColor=20, sigmaSpace=20)
            # More denoising in darker areas
            weight = (1.0 - T_smooth) * denoise_strength
            weight_3d = weight[:, :, np.newaxis]
            blend_u8 = (blend_u8.astype(np.float32) * (1.0 - weight_3d) +
                        denoised.astype(np.float32) * weight_3d).astype(np.uint8)

        # ── 6. Unsharp Masking (Edge Crispness) ──
        gauss = cv2.GaussianBlur(blend_u8, (0, 0), 2.0)
        sharp = cv2.addWeighted(blend_u8, 1.3, gauss, -0.3, 0)

        # ── 7. Color Boost in HSV ──
        if color_boost > 1.0:
            hsv = cv2.cvtColor(sharp, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * color_boost, 0, 255)
            sharp = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # ── 8. Illumination Gain Heatmap ──
        norm_t = np.clip(T_smooth * 255.0, 0, 255).astype(np.uint8)
        gain_heatmap = cv2.applyColorMap(norm_t, cv2.COLORMAP_TURBO)

        return sharp, gain_heatmap, lighting_state


# Backwards compatibility alias
NeuralLightingRestorer = MultiScaleRetinexRestorer
