"""
Ophthalmic Corneal Specular Reflection & Periocular Consistency Analyzer.
Based on CVIU 2022 (Matern et al.) and Light2Lie (NDSS 2026) optical reflection physics.
Deterministically extracts bilateral eye ROIs, corneal highlight asymmetry,
Fresnel specular contrast, eye-ROI pixel area, ambient luminance, glasses/mask
occlusion severity, and eye-branch reliability quality (q_eye).
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


_FACE_CASCADE = None
_EYE_CASCADE = None


def _get_cascades():
    global _FACE_CASCADE, _EYE_CASCADE
    if cv2 is None:
        return None, None
    if _FACE_CASCADE is None:
        try:
            f_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
            e_xml = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
            if os.path.exists(f_xml):
                _FACE_CASCADE = cv2.CascadeClassifier(f_xml)
            if os.path.exists(e_xml):
                _EYE_CASCADE = cv2.CascadeClassifier(e_xml)
        except Exception:
            _FACE_CASCADE, _EYE_CASCADE = None, None
    return _FACE_CASCADE, _EYE_CASCADE


def _load_rgb_native(image_input: Any) -> Optional[np.ndarray]:
    """Load input path, PIL Image, or ndarray as uint8/float32 RGB at native portrait resolution."""
    if image_input is None:
        return None
    try:
        if isinstance(image_input, (str, os.PathLike, Path)):
            p = Path(image_input)
            if not p.exists() or Image is None:
                return None
            with Image.open(p) as im:
                return np.asarray(im.convert("RGB"), dtype=np.float32)
        if Image is not None and isinstance(image_input, Image.Image):
            return np.asarray(image_input.convert("RGB"), dtype=np.float32)
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
            return np.clip(arr, 0.0, 255.0)
    except Exception:
        return None
    return None


def _resize_bilinear(patch: np.ndarray, size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    if Image is not None:
        pil_p = Image.fromarray(np.clip(patch, 0.0, 255.0).astype(np.uint8), mode="RGB")
        return np.asarray(pil_p.resize(size, Image.Resampling.BILINEAR), dtype=np.float32)
    h, w, _ = patch.shape
    yi = np.linspace(0, max(0, h - 1), size[1]).astype(int)
    xi = np.linspace(0, max(0, w - 1), size[0]).astype(int)
    return patch[np.ix_(yi, xi)]


def _extract_specular_stats(eye_rgb: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Compute top-2% luminance specular highlight centroid (cx, cy), Fresnel contrast,
    area fraction, mean luminance, and saturated glare fraction for one eye ROI.
    """
    eps = 1e-6
    gray = (0.299 * eye_rgb[:, :, 0] + 0.587 * eye_rgb[:, :, 1] + 0.114 * eye_rgb[:, :, 2]) / 255.0
    h, w = gray.shape
    p98 = float(np.percentile(gray, 97.5))
    mask = gray >= max(p98, 0.40)
    if not np.any(mask):
        mask = gray >= float(np.percentile(gray, 92.0))
    if np.any(mask):
        ys, xs = np.where(mask)
        weights = gray[ys, xs] + eps
        cx = float(np.sum(xs * weights) / np.sum(weights)) / max(1.0, float(w - 1))
        cy = float(np.sum(ys * weights) / np.sum(weights)) / max(1.0, float(h - 1))
        peak_val = float(np.mean(gray[mask]))
        area_frac = float(np.mean(mask))
    else:
        cx, cy = 0.5, 0.5
        peak_val = float(np.max(gray))
        area_frac = 0.02

    med_val = float(np.median(gray))
    fresnel_contrast = float(np.clip((peak_val - med_val) / (med_val + 0.08), 0.0, 3.0))
    glare_frac = float(np.mean(gray > 0.90))
    return cx, cy, fresnel_contrast, area_frac, float(np.mean(gray)), glare_frac


class CornealReflectionAnalyzer:
    """
    Deterministic bilateral eye-ROI & optical corneal reflection analyzer.
    """

    def __init__(self, asymmetry_threshold: float = 0.35):
        self.asymmetry_threshold = float(asymmetry_threshold)

    def _neutral_fallback(self) -> Dict[str, Any]:
        return {
            "corneal_asymmetry_score": 0.12,
            "corneal_reflection_symmetry": 0.88,
            "optical_gaze_consistent": True,
            "reflection_status": "NEUTRAL_UNAVAILABLE",
            "threshold": self.asymmetry_threshold,
            "fresnel_specular_contrast": 0.0,
            "bilateral_illumination_asymmetry": 0.0,
            "periocular_color_drift": 0.0,
            "eye_roi_pixels": 0,
            "eye_luminance": 0.5,
            "occlusion_glare_score": 0.5,
            "bilateral_eyes_detected": False,
            "detection_mode": "unavailable",
            "corneal_signal_quality": 0.25,
            "feature_vector": [0.0] * 10,
        }

    def analyze(self, image_input: Any = None) -> Dict[str, Any]:
        """
        Deterministically extract left/right ocular geometry, Fresnel highlight consistency,
        periocular RGB drift, resolution/luminance/occlusion covariates, and quality q_eye.
        """
        rgb = _load_rgb_native(image_input)
        if rgb is None or rgb.shape[0] < 16 or rgb.shape[1] < 16:
            return self._neutral_fallback()

        h, w, _ = rgb.shape
        fx, fy, fw, fh = int(w * 0.18), int(h * 0.14), max(16, int(w * 0.64)), max(16, int(h * 0.72))
        bilateral_detected = False
        detection_mode = "anthropomorphic_roi"
        left_box = None
        right_box = None

        face_cascade, eye_cascade = _get_cascades()
        if cv2 is not None and face_cascade is not None and not face_cascade.empty():
            gray_u8 = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
            faces = face_cascade.detectMultiScale(gray_u8, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
            if len(faces) > 0:
                fx, fy, fw, fh = [int(v) for v in max(faces, key=lambda b: b[2] * b[3])]
                if eye_cascade is not None and not eye_cascade.empty():
                    upper_face = gray_u8[fy : fy + max(12, int(fh * 0.62)), fx : fx + fw]
                    eyes = eye_cascade.detectMultiScale(upper_face, scaleFactor=1.08, minNeighbors=4, minSize=(14, 14))
                    if len(eyes) >= 2:
                        eyes_sorted = sorted(eyes, key=lambda b: b[0])
                        ex1, ey1, ew1, eh1 = [int(v) for v in eyes_sorted[0]]
                        ex2, ey2, ew2, eh2 = [int(v) for v in eyes_sorted[-1]]
                        if ex2 > ex1 + ew1 * 0.4:
                            left_box = (fx + ex1, fy + ey1, max(8, ew1), max(8, eh1))
                            right_box = (fx + ex2, fy + ey2, max(8, ew2), max(8, eh2))
                            bilateral_detected = True
                            detection_mode = "haar_cascade"

        if left_box is None or right_box is None:
            left_box = (
                fx + int(fw * 0.16),
                fy + int(fh * 0.24),
                max(8, int(fw * 0.27)),
                max(8, int(fh * 0.20)),
            )
            right_box = (
                fx + int(fw * 0.57),
                fy + int(fh * 0.24),
                max(8, int(fw * 0.27)),
                max(8, int(fh * 0.20)),
            )

        def _crop(box: Tuple[int, int, int, int]) -> Tuple[np.ndarray, int]:
            bx, by, bw, bh = box
            x0 = max(0, min(w - 4, bx))
            y0 = max(0, min(h - 4, by))
            x1 = max(x0 + 4, min(w, bx + bw))
            y1 = max(y0 + 4, min(h, by + bh))
            crop = rgb[y0:y1, x0:x1, :]
            return crop, int(crop.shape[0] * crop.shape[1])

        l_raw, l_px = _crop(left_box)
        r_raw, r_px = _crop(right_box)
        eye_roi_pixels = l_px + r_px

        l_eye = _resize_bilinear(l_raw, (48, 48))
        r_eye = _resize_bilinear(r_raw, (48, 48))

        lx, ly, lfresnel, larea, llum, lglare = _extract_specular_stats(l_eye)
        rx, ry, rfresnel, rarea, rlum, rglare = _extract_specular_stats(r_eye)

        # Centroid offset asymmetry (mirror-normalized dx and vertical dy)
        centroid_dist = float(np.sqrt((lx - rx) ** 2 + (ly - ry) ** 2)) / np.sqrt(2.0)
        fresnel_mean = 0.5 * (lfresnel + rfresnel)
        fresnel_asym = abs(lfresnel - rfresnel) / (lfresnel + rfresnel + 0.15)
        area_asym = abs(larea - rarea) / (larea + rarea + 0.02)
        illum_asym = abs(llum - rlum) / (llum + rlum + 0.10)

        # Periocular color-temperature drift & inner/outer cheek-forehead contrast (Light2Lie / SBI seam cue)
        peri_mean_l = np.mean(l_eye, axis=(0, 1))
        peri_mean_r = np.mean(r_eye, axis=(0, 1))
        color_drift = float(np.linalg.norm(peri_mean_l - peri_mean_r) / 255.0)

        # Lower-face mask / glasses glare occlusion estimator
        glare_ratio = max(lglare, rglare)
        eye_gray = 0.5 * (
            (0.299 * l_eye[:, :, 0] + 0.587 * l_eye[:, :, 1] + 0.114 * l_eye[:, :, 2])
            + (0.299 * r_eye[:, :, 0] + 0.587 * r_eye[:, :, 1] + 0.114 * r_eye[:, :, 2])
        ) / 255.0
        eye_edge_energy = float(np.mean(np.abs(eye_gray[1:, :] - eye_gray[:-1, :])))

        # Lower face region for mask/chin occlusion detection
        lf_y0 = min(h - 4, fy + int(fh * 0.58))
        lf_y1 = min(h, fy + int(fh * 0.94))
        lf_x0 = max(0, fx + int(fw * 0.22))
        lf_x1 = min(w, fx + int(fw * 0.78))
        lower_face = rgb[lf_y0:lf_y1, lf_x0:lf_x1, :]
        if lower_face.size > 0:
            lf_std = float(np.mean(np.std(lower_face / 255.0, axis=(0, 1))))
            # Very uniform or saturated surgical mask / heavy shadow lowers textural skin variance
            mask_indicator = float(np.clip((0.11 - lf_std) / 0.08, 0.0, 1.0)) if lf_std < 0.11 else 0.0
        else:
            mask_indicator = 0.0

        glare_occlusion = float(np.clip((glare_ratio - 0.04) / 0.22, 0.0, 1.0))
        low_detail_occlusion = float(np.clip((0.028 - eye_edge_energy) / 0.025, 0.0, 1.0)) if eye_edge_energy < 0.028 else 0.0
        occlusion_glare_score = float(np.clip(0.55 * glare_occlusion + 0.30 * mask_indicator + 0.25 * low_detail_occlusion, 0.0, 1.0))

        eye_luminance = float(np.clip(0.5 * (llum + rlum), 0.0, 1.0))

        # Corneal asymmetry composite score in [0, 1]
        corneal_asym = float(
            np.clip(
                0.42 * centroid_dist
                + 0.28 * fresnel_asym
                + 0.18 * area_asym
                + 0.12 * illum_asym,
                0.01,
                0.98,
            )
        )
        symmetry = float(np.clip(1.0 - corneal_asym, 0.0, 1.0))
        is_consistent = corneal_asym < self.asymmetry_threshold

        # Signal quality q_eye in [0.05, 1.0]: penalizes tiny eye ROIs, extreme dark/bright exposure, and occlusion/glare
        res_factor = float(np.clip(eye_roi_pixels / 1200.0, 0.15, 1.0))
        light_factor = float(np.clip(1.0 - abs(eye_luminance - 0.50) * 1.45, 0.20, 1.0))
        det_factor = 1.0 if bilateral_detected else 0.72
        occ_factor = float(np.clip(1.0 - 0.72 * occlusion_glare_score, 0.12, 1.0))
        q_eye = float(np.clip(res_factor * light_factor * det_factor * occ_factor, 0.05, 1.0))

        # Horizontal bilateral patch L1 & inner-ocular Laplacian roughness
        l_mirror = r_eye[:, ::-1, :] / 255.0
        l_norm = l_eye / 255.0
        bilateral_patch_l1 = float(np.mean(np.abs(l_norm - l_mirror)))
        eye_lap = float(
            np.std(
                eye_gray[:-2, 1:-1]
                + eye_gray[2:, 1:-1]
                + eye_gray[1:-1, :-2]
                + eye_gray[1:-1, 2:]
                - 4.0 * eye_gray[1:-1, 1:-1]
            )
        )

        feature_vector: List[float] = [
            round(corneal_asym, 6),
            round(centroid_dist, 6),
            round(fresnel_asym, 6),
            round(fresnel_mean / 3.0, 6),
            round(area_asym, 6),
            round(illum_asym, 6),
            round(color_drift, 6),
            round(bilateral_patch_l1, 6),
            round(eye_lap, 6),
            round(occlusion_glare_score, 6),
        ]

        return {
            "corneal_asymmetry_score": round(corneal_asym, 4),
            "corneal_reflection_symmetry": round(symmetry, 4),
            "optical_gaze_consistent": bool(is_consistent),
            "reflection_status": "AUTHENTIC_LIGHTING" if is_consistent else "SYNTHETIC_ASYMMETRY",
            "threshold": self.asymmetry_threshold,
            "fresnel_specular_contrast": round(fresnel_mean, 4),
            "bilateral_illumination_asymmetry": round(illum_asym, 4),
            "periocular_color_drift": round(color_drift, 4),
            "eye_roi_pixels": int(eye_roi_pixels),
            "eye_luminance": round(eye_luminance, 4),
            "occlusion_glare_score": round(occlusion_glare_score, 4),
            "bilateral_eyes_detected": bool(bilateral_detected),
            "detection_mode": detection_mode,
            "corneal_signal_quality": round(q_eye, 4),
            "feature_vector": feature_vector,
        }
