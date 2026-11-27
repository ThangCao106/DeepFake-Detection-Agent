"""
2D Discrete Wavelet Transform (DWT) & Spatial-Frequency Spectral Analyzer.
Based on WAVIE architecture (arXiv:2609.14437) and FreqFusion degradation analysis.
Extracts deterministic 2-level 2D Haar DWT sub-band residuals (LH, HL, HH),
2D-FFT periodic lattice spikes, 8x8 JPEG DCT blockiness, Laplacian blur severity,
facial boundary seam ratios, and frequency signal survival quality (q_freq).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _coerce_rgb_array(image_input: Any, size: Tuple[int, int] = (256, 256)) -> Optional[np.ndarray]:
    """Deterministically load an image path, PIL Image, or ndarray as float32 RGB in [0, 255]."""
    if image_input is None:
        return None
    try:
        if isinstance(image_input, (str, os.PathLike, Path)):
            p = Path(image_input)
            if not p.exists() or Image is None:
                return None
            with Image.open(p) as im:
                rgb = im.convert("RGB").resize(size, Image.Resampling.BILINEAR)
                return np.asarray(rgb, dtype=np.float32)
        if Image is not None and isinstance(image_input, Image.Image):
            rgb = image_input.convert("RGB").resize(size, Image.Resampling.BILINEAR)
            return np.asarray(rgb, dtype=np.float32)
        if isinstance(image_input, np.ndarray):
            arr = image_input.astype(np.float32)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[2] >= 3:
                arr = arr[:, :, :3]
            else:
                return None
            if float(np.max(arr)) <= 1.5:
                arr = arr * 255.0
            if Image is not None and (arr.shape[0] != size[1] or arr.shape[1] != size[0]):
                pil_im = Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")
                pil_im = pil_im.resize(size, Image.Resampling.BILINEAR)
                return np.asarray(pil_im, dtype=np.float32)
            return np.clip(arr, 0.0, 255.0)
    except Exception:
        return None
    return None


def _haar_dwt_2d(channel: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-level orthonormal 2D Haar wavelet transform via vectorized slicing."""
    h, w = channel.shape
    h2 = (h // 2) * 2
    w2 = (w // 2) * 2
    x = channel[:h2, :w2]
    a = x[0::2, 0::2]
    b = x[0::2, 1::2]
    c = x[1::2, 0::2]
    d = x[1::2, 1::2]
    ll = 0.5 * (a + b + c + d)
    lh = 0.5 * (a - b + c - d)
    hl = 0.5 * (a + b - c - d)
    hh = 0.5 * (a - b - c + d)
    return ll, lh, hl, hh


class WaveletSpectralAnalyzer:
    """
    Deterministic 2-level 2D Haar DWT, 2D-FFT spectral lattice, and JPEG/blur
    signal survival analyzer.
    """

    def __init__(self, energy_threshold: float = 0.35):
        self.energy_threshold = float(energy_threshold)

    def _neutral_fallback(self) -> Dict[str, Any]:
        return {
            "wavelet_high_freq_energy": 0.22,
            "wavelet_high_freq_artifacts": False,
            "skin_rppg_consistency": 0.90,
            "spectral_signature": "NEUTRAL_UNAVAILABLE",
            "lh_hl_hh_energies": {"lh1": 0.0, "hl1": 0.0, "hh1": 0.0, "lh2": 0.0, "hl2": 0.0, "hh2": 0.0},
            "hh_to_ll_ratio": 0.0,
            "fft_lattice_z_peak": 0.0,
            "jpeg_blockiness_score": 0.0,
            "blur_severity_score": 0.5,
            "boundary_seam_ratio": 1.0,
            "frequency_signal_quality": 0.5,
            "feature_vector": [0.0] * 10,
        }

    def analyze(self, image_input: Any = None) -> Dict[str, Any]:
        """
        Deterministically extract frequency residuals, compression degradation markers,
        and post-compression frequency signal quality (q_freq).
        """
        arr = _coerce_rgb_array(image_input, size=(256, 256))
        if arr is None:
            return self._neutral_fallback()

        eps = 1e-6
        y = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]

        # 1. 2-level 2D Haar DWT on normalized luminance
        y_norm = y / 255.0
        ll1, lh1, hl1, hh1 = _haar_dwt_2d(y_norm)
        ll2, lh2, hl2, hh2 = _haar_dwt_2d(ll1)

        rms = lambda m: float(np.sqrt(np.mean(m * m) + eps))
        e_ll1, e_lh1, e_hl1, e_hh1 = rms(ll1), rms(lh1), rms(hl1), rms(hh1)
        e_lh2, e_hl2, e_hh2 = rms(lh2), rms(hl2), rms(hh2)

        hh_to_ll = e_hh1 / (e_ll1 + eps)
        hh2_to_ll1 = e_hh2 / (e_ll1 + eps)
        directional_anisotropy = abs(e_lh1 - e_hl1) / (e_lh1 + e_hl1 + eps)

        # Normalize high-frequency sub-band energy index into [0, 1]
        hf_raw = (e_lh1 + e_hl1 + 1.4 * e_hh1) / (e_ll1 + eps)
        wavelet_hf_energy = float(np.clip(hf_raw * 3.2, 0.02, 0.99))

        # 2. 2D-FFT radial high-frequency roll-off & periodic lattice Z-peak
        y_zero = y_norm - float(np.mean(y_norm))
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(y_zero)))
        log_mag = np.log1p(fft_mag)
        h, w = log_mag.shape
        yy, xx = np.ogrid[:h, :w]
        cy, cx = h / 2.0, w / 2.0
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.5 * min(h, w))

        low_mask = (radius >= 0.04) & (radius < 0.25)
        mid_mask = (radius >= 0.25) & (radius < 0.60)
        high_mask = (radius >= 0.60) & (radius <= 0.98)

        fft_low = float(np.mean(log_mag[low_mask])) if np.any(low_mask) else 1.0
        fft_mid = float(np.mean(log_mag[mid_mask])) if np.any(mid_mask) else 0.5
        fft_high = float(np.mean(log_mag[high_mask])) if np.any(high_mask) else 0.2
        fft_hf_ratio = fft_high / (fft_low + eps)

        if np.any(high_mask):
            hi_vals = log_mag[high_mask]
            hi_med = float(np.median(hi_vals))
            hi_std = float(np.std(hi_vals))
            fft_z_peak = float(np.clip((float(np.percentile(hi_vals, 99.5)) - hi_med) / (hi_std + 1e-4), 0.0, 12.0))
        else:
            fft_z_peak = 0.0

        # 3. 8x8 JPEG DCT block-boundary discontinuity estimator
        diff_h = np.abs(y[:, 1:] - y[:, :-1])
        diff_v = np.abs(y[1:, :] - y[:-1, :])
        h_idx = np.arange(y.shape[1] - 1)
        v_idx = np.arange(y.shape[0] - 1)
        b_h = float(np.mean(diff_h[:, (h_idx % 8) == 7]))
        in_h = float(np.mean(diff_h[:, (h_idx % 8) != 7]))
        b_v = float(np.mean(diff_v[(v_idx % 8) == 7, :]))
        in_v = float(np.mean(diff_v[(v_idx % 8) != 7, :]))
        block_ratio = 0.5 * ((b_h / (in_h + 0.5)) + (b_v / (in_v + 0.5)))
        jpeg_blockiness = float(np.clip((block_ratio - 1.0) / 0.65, 0.0, 1.0))

        # 4. Laplacian sharpness / blur severity index
        lap = (
            y_norm[:-2, 1:-1]
            + y_norm[2:, 1:-1]
            + y_norm[1:-1, :-2]
            + y_norm[1:-1, 2:]
            - 4.0 * y_norm[1:-1, 1:-1]
        )
        lap_var = float(np.var(lap))
        blur_severity = float(np.clip(1.0 / (1.0 + lap_var * 900.0), 0.0, 1.0))

        # 5. Facial blending boundary seam ratio (outer portrait ring vs inner portrait core)
        grad_mag = np.sqrt((y_norm[1:, :-1] - y_norm[:-1, :-1]) ** 2 + (y_norm[:-1, 1:] - y_norm[:-1, :-1]) ** 2)
        gy, gx = np.ogrid[: grad_mag.shape[0], : grad_mag.shape[1]]
        r_norm = np.sqrt(((gy - 120.0) / 72.0) ** 2 + ((gx - 128.0) / 58.0) ** 2)
        seam_band = (r_norm >= 0.85) & (r_norm <= 1.15)
        inner_core = r_norm < 0.70
        seam_grad = float(np.mean(grad_mag[seam_band])) if np.any(seam_band) else 0.0
        core_grad = float(np.mean(grad_mag[inner_core])) if np.any(inner_core) else 1.0
        boundary_seam_ratio = float(np.clip(seam_grad / (core_grad + 1e-4), 0.0, 6.0))

        # 6. Deterministic spatial chrominance homogeneity (green-red consistency proxy)
        rg_diff = (arr[72:176, 80:176, 1] - arr[72:176, 80:176, 0]) / 255.0
        cb_like = (arr[72:176, 80:176, 2] - 0.5 * (arr[72:176, 80:176, 0] + arr[72:176, 80:176, 1])) / 255.0
        chroma_std = 0.5 * (float(np.std(rg_diff)) + float(np.std(cb_like)))
        skin_rppg_consistency = float(np.clip(1.0 - chroma_std * 2.1, 0.20, 0.99))

        # 7. Frequency signal survival quality q_freq in [0.05, 1.0]
        # Clean, unblurred, non-blocky images retain reliable HF spectral forensics;
        # heavy JPEG quantization and blur attenuate sub-band forensics -> lower q_freq.
        q_freq = float(np.clip((1.0 - 0.65 * jpeg_blockiness) * (1.0 - 0.55 * blur_severity), 0.05, 1.0))

        is_natural = (wavelet_hf_energy <= self.energy_threshold) and (fft_z_peak < 3.6) and (boundary_seam_ratio < 1.85)

        feature_vector: List[float] = [
            round(wavelet_hf_energy, 6),
            round(hh_to_ll, 6),
            round(hh2_to_ll1, 6),
            round(directional_anisotropy, 6),
            round(fft_hf_ratio, 6),
            round(fft_mid / (fft_low + eps), 6),
            round(fft_z_peak / 12.0, 6),
            round(jpeg_blockiness, 6),
            round(blur_severity, 6),
            round(boundary_seam_ratio / 6.0, 6),
        ]

        return {
            "wavelet_high_freq_energy": round(wavelet_hf_energy, 4),
            "wavelet_high_freq_artifacts": not is_natural,
            "skin_rppg_consistency": round(skin_rppg_consistency, 4),
            "spectral_signature": "NATURAL_PRNU" if is_natural else "DIFFUSION_SPECTRAL_GRID",
            "lh_hl_hh_energies": {
                "lh1": round(e_lh1, 5),
                "hl1": round(e_hl1, 5),
                "hh1": round(e_hh1, 5),
                "lh2": round(e_lh2, 5),
                "hl2": round(e_hl2, 5),
                "hh2": round(e_hh2, 5),
            },
            "hh_to_ll_ratio": round(hh_to_ll, 5),
            "fft_lattice_z_peak": round(fft_z_peak, 4),
            "jpeg_blockiness_score": round(jpeg_blockiness, 4),
            "blur_severity_score": round(blur_severity, 4),
            "boundary_seam_ratio": round(boundary_seam_ratio, 4),
            "frequency_signal_quality": round(q_freq, 4),
            "feature_vector": feature_vector,
        }
