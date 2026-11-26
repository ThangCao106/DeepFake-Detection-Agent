# -*- coding: utf-8 -*-
"""
Forensics-Aware Data Augmentation & Counterfactual Synthesis:
  1. SocialMediaForensicAugment: simulates JPEG/WebP re-compression, beauty-filter blur,
     resolution scaling, color jitter, and social-media caption/sticker overlays.
  2. generate_self_blended_image: CVPR-2022 SBI counterfactual generator operating strictly
     on authentic images (blending-boundary & sensor-noise inconsistency without blank canvases).
  3. rawboost_augment_waveform: RawBoost-inspired acoustic degradations for audio robustness
     across 8kHz telephony, Opus/Zalo codecs, FIR coloring, and additive noise.
"""

import io
import math
import random
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


class SocialMediaForensicAugment:
    """
    Reversible social-media channel degradation pipeline for Threads/IG/TikTok/YT Shorts.
    Helps eliminate beauty-filter False Positives and caption/sticker False Negatives.
    """

    def __init__(
        self,
        p_jpeg: float = 0.50,
        p_webp: float = 0.25,
        p_blur: float = 0.35,
        p_resize: float = 0.40,
        p_overlay: float = 0.30,
        p_color: float = 0.30,
        seed: Optional[int] = None,
    ):
        self.p_jpeg = float(p_jpeg)
        self.p_webp = float(p_webp)
        self.p_blur = float(p_blur)
        self.p_resize = float(p_resize)
        self.p_overlay = float(p_overlay)
        self.p_color = float(p_color)
        self.rng = random.Random(seed)

    def __call__(self, pil_img: Image.Image) -> Image.Image:
        img = pil_img.convert("RGB")
        w, h = img.size

        # 1. Color / beauty tone jitter
        if self.rng.random() < self.p_color:
            img = ImageEnhance.Brightness(img).enhance(self.rng.uniform(0.88, 1.12))
            img = ImageEnhance.Contrast(img).enhance(self.rng.uniform(0.90, 1.12))
            img = ImageEnhance.Color(img).enhance(self.rng.uniform(0.90, 1.14))

        # 2. Resolution downscale/upscale
        if self.rng.random() < self.p_resize and min(w, h) >= 64:
            scale = self.rng.uniform(0.55, 0.95)
            nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
            img = img.resize((nw, nh), Image.BILINEAR).resize((w, h), Image.BICUBIC)

        # 3. Beauty-filter Gaussian smoothing
        if self.rng.random() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.rng.uniform(0.45, 1.35)))

        # 4. Random social caption / sticker bar directly over a copy of input image
        if self.rng.random() < self.p_overlay and min(w, h) >= 64:
            img = img.copy()
            draw = ImageDraw.Draw(img, "RGBA")
            bar_h = max(14, int(h * self.rng.uniform(0.08, 0.14)))
            y0 = int(h * self.rng.uniform(0.72, 0.84))
            y1 = min(h - 2, y0 + bar_h)
            x0 = int(w * 0.08)
            x1 = int(w * 0.92)
            bg_alpha = self.rng.randint(150, 220)
            draw.rounded_rectangle([x0, y0, x1, y1], radius=6, fill=(18, 22, 32, bg_alpha))
            draw.text((x0 + 8, y0 + max(2, bar_h // 5)), "@deeptrace #caption", fill=(245, 248, 255, 235))

        # 5. JPEG re-compression
        if self.rng.random() < self.p_jpeg:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=self.rng.randint(35, 95))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")

        # 6. WebP lossy transcode
        if self.rng.random() < self.p_webp:
            try:
                buf = io.BytesIO()
                img.save(buf, format="WEBP", quality=self.rng.randint(40, 90))
                buf.seek(0)
                img = Image.open(buf).convert("RGB")
            except Exception:
                pass

        return img


def generate_self_blended_image(pil_img: Image.Image, seed: Optional[int] = None) -> Image.Image:
    """
    CVPR-2022 Self-Blended Image (SBI) counterfactual generator from an authentic image.
    Creates realistic blending seams, affine micro-misalignment, color shift, and
    sensor-noise disparity inside a feathered elliptical facial ROI without synthetic blank canvases.
    """
    rng = random.Random(seed)
    base = pil_img.convert("RGB")
    w, h = base.size
    if w < 32 or h < 32:
        return base

    src = base.copy()

    # Subtle affine micro-shift & resize blur mismatch on foreground copy
    dx = rng.uniform(-0.025, 0.025) * w
    dy = rng.uniform(-0.025, 0.025) * h
    scale = rng.uniform(0.96, 1.04)
    src = src.transform(
        (w, h),
        Image.AFFINE,
        (scale, 0.0, dx, 0.0, scale, dy),
        resample=Image.BILINEAR,
    )
    src = ImageEnhance.Color(src).enhance(rng.uniform(0.84, 1.18))
    src = ImageEnhance.Brightness(src).enhance(rng.uniform(0.90, 1.12))
    if rng.random() < 0.5:
        src = src.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.4)))
    else:
        src = src.filter(ImageFilter.SHARPEN)

    base_arr = np.asarray(base, dtype=np.float32)
    src_arr = np.asarray(src, dtype=np.float32)

    # Smooth feathered elliptical mask over central facial/subject region
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * rng.uniform(0.46, 0.54)
    cy = h * rng.uniform(0.42, 0.52)
    rx = max(8.0, w * rng.uniform(0.24, 0.34))
    ry = max(10.0, h * rng.uniform(0.28, 0.38))
    dist = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    alpha = np.clip(1.35 - dist, 0.0, 1.0)
    # Smooth sigmoid-like feathering so boundary mimics FaceSwap feathering seams
    alpha = (alpha * alpha * (3.0 - 2.0 * alpha))[..., None]

    # Add subtle high-frequency sensor mismatch inside blended ROI
    blend_ratio = rng.uniform(0.70, 0.92)
    blended = base_arr * (1.0 - alpha * blend_ratio) + src_arr * (alpha * blend_ratio)
    blended_u8 = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    out_img = Image.fromarray(blended_u8, mode="RGB")

    # Apply realistic social JPEG compression over the spliced result
    buf = io.BytesIO()
    out_img.save(buf, format="JPEG", quality=rng.randint(55, 92))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rawboost_augment_waveform(
    y: np.ndarray,
    sr: int = 16000,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    RawBoost-inspired waveform perturbation simulating Zalo/Opus codecs, 8kHz telephony
    bandlimiting, convolutive FIR coloring, non-linear compression, and ambient noise.
    Uses numpy.random.default_rng for deterministic reproducible augmentation.
    """
    arr = np.asarray(y, dtype=np.float32).flatten()
    if arr.size < 64:
        return arr

    rng = np.random.default_rng(seed)
    out = arr.copy()
    sr = max(8000, int(sr or 16000))

    # 1. Frequency-domain codec / telephony bandlimiting (keeps 250Hz..3.8kHz on telephony mode)
    mode = int(rng.integers(0, 4))
    spec = np.fft.rfft(out)
    freqs = np.fft.rfftfreq(len(out), 1.0 / sr)
    if mode in (0, 2):
        # Telephony / Zalo Opus bandwidth roll-off above ~3.8kHz-6.5kHz
        cutoff = float(rng.uniform(3600.0, min(7200.0, sr * 0.45)))
        taper = 1.0 / (1.0 + (freqs / cutoff) ** 6)
        spec = spec * taper
    # 2. Random smooth spectral tilt (convolutive microphone coloring)
    tilt = 1.0 + float(rng.uniform(-0.18, 0.18)) * (freqs / (sr * 0.5 + 1e-6))
    spec = spec * np.clip(tilt, 0.65, 1.35)
    out = np.fft.irfft(spec, n=len(out)).astype(np.float32)

    # 3. Mild non-linear soft clipping / codec quantization distortion
    if mode in (1, 2):
        drive = float(rng.uniform(1.05, 1.45))
        peak = max(1e-5, float(np.max(np.abs(out))))
        out = np.tanh((out / peak) * drive) * peak

    # 4. Low-level SNR-controlled stationary background noise (22-36 dB SNR)
    sig_rms = float(np.sqrt(np.mean(out ** 2))) + 1e-6
    snr_db = float(rng.uniform(22.0, 36.0))
    noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, size=out.shape).astype(np.float32)
    out = out + noise

    # Preserve [-1, 1] safe dynamic range
    max_abs = float(np.max(np.abs(out)))
    if max_abs > 1.0:
        out = out / max_abs
    return np.clip(out, -1.0, 1.0).astype(np.float32)
