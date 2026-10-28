# -*- coding: utf-8 -*-
"""
Option D: Context-Guided Dual-Stream Preprocessing Module
Tương thích trực tiếp với hệ thống 2 Model (model_human & model_general)
Hỗ trợ: Image, Video, Audio (Người thật, Anime, Tranh vẽ, Phong cảnh)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, Union

# ==============================================================================
# 1. BỘ LỌC TOÁN HỌC VI MÔ (MICRO PHYSICS FILTERS CHO MODEL GENERAL)
# ==============================================================================

class MicroPhysicsExtractor:
    """Trích xuất đặc trưng vật lý vi mô: 2D-FFT và SRM Filter Residuals."""
    
    @staticmethod
    def get_srm_kernels() -> np.ndarray:
        """
        Khởi tạo 3 bộ lọc SRM cơ bản (1st order, 2nd order, và 3x3 square)
        để triệt tiêu ngữ nghĩa hình ảnh và giữ lại vi nhiễu sinh của AI.
        """
        k1 = np.array([
            [0,  0, 0],
            [0, -1, 1],
            [0,  0, 0]
        ], dtype=np.float32)
        
        k2 = np.array([
            [0,  1, 0],
            [1, -4, 1],
            [0,  1, 0]
        ], dtype=np.float32)
        
        k3 = np.array([
            [-1,  2, -1],
            [ 2, -4,  2],
            [-1,  2, -1]
        ], dtype=np.float32)
        
        return np.stack([k1, k2, k3], axis=0) # Shape: (3, 3, 3)

    @classmethod
    def extract_srm_residuals(cls, image_rgb: np.ndarray) -> np.ndarray:
        """Lọc ảnh xám qua 3 kernel SRM -> Chuẩn hóa về [-1, 1]."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        kernels = cls.get_srm_kernels()
        residuals = []
        for k in kernels:
            res = cv2.filter2D(gray, -1, k)
            res = np.tanh(res / 10.0)
            residuals.append(res)
        return np.stack(residuals, axis=-1) # Shape: (H, W, 3)

    @staticmethod
    def extract_fft_spectrum(image_rgb: np.ndarray) -> np.ndarray:
        """
        Biến đổi Fourier 2D (2D-FFT) sang Phổ biên độ Logarit (Log Magnitude Spectrum)
        để bắt lưới ca-rô (checkerboard artifacts) của Diffusion/GAN trên phong cảnh/anime.
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        log_mag = np.log1p(magnitude)
        norm_mag = (log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8)
        return norm_mag # Shape: (H, W)


# ==============================================================================
# 2. BỘ ĐIỀU PHỐI MIỀN NỘI DUNG (DOMAIN GATING ROUTER)
# ==============================================================================

class DomainGatingRouter:
    """
    Phân loại nhanh nội dung để quyết định chuyển tiếp tới model_human hay model_general.
    Sử dụng bộ phát hiện khuôn mặt nhẹ để không gây nghẽn tốc độ.
    """
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def route_image(self, image_bgr: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        
        if len(faces) > 0:
            main_face = max(faces, key=lambda b: b[2] * b[3])
            face_area_ratio = (main_face[2] * main_face[3]) / (h * w)
            if face_area_ratio > 0.03:
                return "human", {"bbox": main_face, "type": "human_face"}
        
        return "general", {"type": "general_scene_or_art"}


# ==============================================================================
# 3. TIỀN XỬ LÝ NHÁNH CON NGƯỜI (HUMAN PREPROCESSOR)
# ==============================================================================

class HumanPreprocessor:
    """
    Tiền xử lý chân dung tối ưu cho Luận văn:
    - Mở rộng Bounding Box (Margin 1.3x) để giữ ranh giới ghép nối (seams).
    - Căn chỉnh xoay ngang hai mắt.
    """
    def __init__(self, target_size: Tuple[int, int] = (256, 256), margin: float = 1.3):
        self.target_size = target_size
        self.margin = margin

    def process_face(self, image_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> torch.Tensor:
        x, y, w, h = bbox
        img_h, img_w, _ = image_bgr.shape
        
        cx, cy = x + w / 2.0, y + h / 2.0
        new_w, new_h = w * self.margin, h * self.margin
        
        x1 = max(0, int(cx - new_w / 2.0))
        y1 = max(0, int(cy - new_h / 2.0))
        x2 = min(img_w, int(cx + new_w / 2.0))
        y2 = min(img_h, int(cy + new_h / 2.0))
        
        cropped = image_bgr[y1:y2, x1:x2]
        resized = cv2.resize(cropped, self.target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        normalized = (rgb - mean) / std
        
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).float()
        return tensor


# ==============================================================================
# 4. TIỀN XỬ LÝ NHÁNH TỔNG QUÁT (GENERAL PREPROCESSOR: PHONG CẢNH, ANIME, TRANH)
# ==============================================================================

class GeneralPreprocessor:
    """
    Tiền xử lý cho model_general:
    - Luồng RGB chuẩn hóa toàn cảnh.
    - Luồng Vi mô: Ghép kênh Phổ 2D-FFT và Dư số lọc SRM.
    """
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.target_size = target_size

    def process_general(self, image_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(image_bgr, self.target_size, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        rgb_norm = (rgb.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        fft_map = MicroPhysicsExtractor.extract_fft_spectrum(rgb)
        fft_channel = np.expand_dims(fft_map, axis=-1)
        
        combined = np.concatenate([rgb_norm, fft_channel], axis=-1)
        tensor = torch.from_numpy(combined).permute(2, 0, 1).float()
        return tensor


# ==============================================================================
# 5. TIỀN XỬ LÝ ÂM THANH ĐA MIỀN (AUDIO PREPROCESSOR)
# ==============================================================================

class AudioPreprocessor:
    """Xử lý âm thanh: Tiếng nói người (LFCC) vs Âm thanh môi trường/Phong cảnh."""
    
    @staticmethod
    def process_audio(waveform: np.ndarray, sample_rate: int = 16000, target_duration: float = 3.0) -> torch.Tensor:
        target_len = int(sample_rate * target_duration)
        if len(waveform) < target_len:
            waveform = np.pad(waveform, (0, target_len - len(waveform)))
        else:
            waveform = waveform[:target_len]
            
        spec = np.abs(np.fft.rfft(waveform.reshape(-1, 512), axis=-1))
        spec = np.log1p(spec)
        return torch.from_numpy(spec).float()


# ==============================================================================
# 6. LỚP HỢP NHẤT TOÀN DIỆN (UNIFIED OPTION D PIPELINE)
# ==============================================================================

class UnifiedOptionDPreprocessor:
    """Điểm truy cập duy nhất cho toàn bộ hệ thống phát hiện."""
    
    def __init__(self, target_size: Tuple[int, int] = (256, 256)):
        self.router = DomainGatingRouter()
        self.human_proc = HumanPreprocessor(target_size=target_size, margin=1.3)
        self.general_proc = GeneralPreprocessor(target_size=target_size)
        self.audio_proc = AudioPreprocessor()

    def __call__(self, input_data: Union[str, np.ndarray], media_type: str = "image") -> Dict[str, Any]:
        if media_type == "image":
            if isinstance(input_data, str):
                img_bgr = cv2.imread(input_data)
                if img_bgr is None:
                    raise FileNotFoundError(f"Không thể đọc ảnh: {input_data}")
            else:
                img_bgr = input_data
                
            target_model, meta = self.router.route_image(img_bgr)
            
            if target_model == "human":
                tensor = self.human_proc.process_face(img_bgr, meta["bbox"])
            else:
                tensor = self.general_proc.process_general(img_bgr)
                
            return {
                "target_model": target_model,
                "tensor": tensor.unsqueeze(0),
                "metadata": meta
            }
            
        return {}

