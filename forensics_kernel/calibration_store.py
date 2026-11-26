# -*- coding: utf-8 -*-
"""
Shared Calibration & Weight Telemetry Store (Stdlib-Only).
Single source of truth for reading/validating image, video, audio, and
Vietnamese demographic calibration JSONs, 2-parameter Platt scaling (a*z + b),
Split Conformal abstention bands, and SHA-256 weight binding across
Core (Flask Port 5000) and Observability Cockpit (FastAPI Port 8787).
"""

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_custom_vit_dir(explicit_dir: Optional[str] = None) -> Path:
    """Resolve the active custom_vit_model directory."""
    if explicit_dir and os.path.exists(explicit_dir):
        return Path(explicit_dir)
    candidates = [
        PROJECT_ROOT / "custom_vit_model",
        Path("/app/custom_vit_model"),
        PROJECT_ROOT / "web_ui" / "custom_vit_model",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def load_json_safe(path: Path, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load a JSON file returning defaults cleanly on missing/corrupt files."""
    out = dict(defaults or {})
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                out.update(data)
        except Exception:
            pass
    return out


def compute_file_sha256(path: Path, prefix_len: int = 16) -> str:
    """Compute SHA-256 hex digest of a weight file without excessive memory allocation."""
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    digest = h.hexdigest()
    return digest[:prefix_len] if prefix_len > 0 else digest


def compute_active_model_sha256(custom_dir: Optional[str] = None) -> str:
    """Compute canonical SHA-256 identifier of active `custom_vit_model/model.safetensors`."""
    cdir = resolve_custom_vit_dir(custom_dir)
    st_path = cdir / "model.safetensors"
    if st_path.exists():
        return compute_file_sha256(st_path, prefix_len=16)
    pt_path = cdir / "weights_general.pt"
    if pt_path.exists():
        return compute_file_sha256(pt_path, prefix_len=16)
    return ""


def apply_platt_2param(
    prob_or_logit: float,
    params: Optional[Dict[str, Any]] = None,
    is_logit: bool = False,
) -> float:
    """
    Apply calibrated 2-parameter Platt Scaling: sigmoid(a * z + b).
    Falls back cleanly to single-parameter Temperature Scaling sigmoid(z / T) when
    only `temperature_scaling_T` or `temperature_T` is present.
    """
    p = params or {}
    val = float(prob_or_logit)
    if not is_logit:
        clipped = min(1.0 - 1e-6, max(1e-6, val))
        z = math.log(clipped / (1.0 - clipped))
    else:
        z = val

    if "platt_a" in p and p["platt_a"] is not None:
        a = float(p["platt_a"])
        b = float(p.get("platt_b", 0.0) or 0.0)
        arg = max(-30.0, min(30.0, a * z + b))
        return round(1.0 / (1.0 + math.exp(-arg)), 6)

    t_val = float(p.get("temperature_scaling_T", p.get("temperature_T", 1.6178)) or 1.6178)
    t_val = max(0.1, t_val)
    arg = max(-30.0, min(30.0, z / t_val))
    return round(1.0 / (1.0 + math.exp(-arg)), 6)


def evaluate_conformal_band(
    score: float,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Split Conformal Prediction uncertainty band detector.
    Samples falling inside [conformal_low, conformal_high] are escalated to the Triad Council
    and flagged for Active Learning human review rather than forced into overconfident binary flips.
    """
    p = params or {}
    tau = float(p.get("optimal_threshold_tau", p.get("tau_video", p.get("tau_audio", 0.8118))) or 0.8118)
    c_low = float(p.get("conformal_low", max(0.25, tau - 0.14)) or max(0.25, tau - 0.14))
    c_high = float(p.get("conformal_high", min(0.95, tau + 0.10)) or min(0.95, tau + 0.10))
    if c_low > c_high:
        c_low, c_high = c_high, c_low
    s_val = float(score)
    uncertain = c_low <= s_val <= c_high
    return {
        "in_uncertain_band": uncertain,
        "conformal_low": round(c_low, 4),
        "conformal_high": round(c_high, 4),
        "score": round(s_val, 4),
        "action": "ESCALATE_TO_TRIAD_COUNCIL" if uncertain else "CONFIDENT_DECISION",
    }


DEFAULT_IMAGE_PARAMS: Dict[str, Any] = {
    "temperature_scaling_T": 1.6178,
    "platt_a": 0.6181,
    "platt_b": 0.0,
    "optimal_threshold_tau": 0.8118,
    "conformal_low": 0.67,
    "conformal_high": 0.90,
    "model_sha256": "",
    "peft_method": "dora",
    "calibration_status": "DEFAULT_BASELINE",
}

DEFAULT_VIDEO_PARAMS: Dict[str, Any] = {
    "w_peak": 0.55,
    "w_avg": 0.30,
    "w_jitter": 0.15,
    "weight_general_video": 0.60,
    "weight_human_video": 0.40,
    "tau_video": 0.68,
    "temperature_T": 1.6178,
    "platt_a": 0.6181,
    "platt_b": 0.0,
    "conformal_low": 0.54,
    "conformal_high": 0.78,
    "model_sha256": "",
    "min_corroborated_ratio": 0.30,
    "multi_frame_max_prob_gate": 0.80,
    "deepfake_frame_ratio_gate": 0.35,
    "ai_art_avg_comp_gate": 0.45,
    "calibration_status": "DEFAULT_BASELINE",
}

DEFAULT_AUDIO_PARAMS: Dict[str, Any] = {
    "w_vocoder": 0.45,
    "w_pitch": 0.35,
    "w_phase": 0.20,
    "tau_audio": 0.8118,
    "temperature_T": 1.6178,
    "platt_a": 0.6181,
    "platt_b": 0.0,
    "conformal_low": 0.66,
    "conformal_high": 0.90,
    "model_sha256": "",
    "vocoder_gate": 0.72,
    "pitch_gate": 0.55,
    "calibration_status": "DEFAULT_AUDIO",
}


def load_all_modality_params(custom_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return synchronized image, video, audio, and vn_demographic parameter dicts."""
    cdir = resolve_custom_vit_dir(custom_dir)
    img_p = load_json_safe(cdir / "calibration_params.json", DEFAULT_IMAGE_PARAMS)
    vid_defaults = dict(DEFAULT_VIDEO_PARAMS)
    vid_defaults["tau_video"] = float(img_p.get("optimal_threshold_tau", vid_defaults["tau_video"]))
    vid_defaults["temperature_T"] = float(img_p.get("temperature_scaling_T", vid_defaults["temperature_T"]))
    vid_p = load_json_safe(cdir / "video_calibration_params.json", DEFAULT_VIDEO_PARAMS)
    aud_p = load_json_safe(cdir / "audio_calibration_params.json", DEFAULT_AUDIO_PARAMS)
    vn_p = load_json_safe(cdir / "vietnamese_demographic_params.json", {})
    return {
        "image": img_p,
        "video": vid_p,
        "audio": aud_p,
        "vn_demographic": vn_p,
    }


def inspect_weights_health(custom_dir: Optional[str] = None) -> Dict[str, Any]:
    """Verify physical presence and size of trained model artifacts mounted in runtime."""
    cdir = resolve_custom_vit_dir(custom_dir)
    human_dir = PROJECT_ROOT / "human_vit_model"
    if not human_dir.exists() and Path("/app/human_vit_model").exists():
        human_dir = Path("/app/human_vit_model")

    artifacts = {
        "vit_safetensors": cdir / "model.safetensors",
        "general_weights_pt": cdir / "weights_general.pt",
        "human_weights_pt": human_dir / "weights_human.pt",
        "audio_fusionnet_pt": cdir / "audio_fusionnet.pt",
        "video_temporal_mil_pt": cdir / "video_temporal_mil.pt",
        "audio_ssl_lora_pt": cdir / "audio_ssl_lora.pt",
        "ppo_meta_controller_pt": cdir / "ppo_meta_controller.pt",
        "ewc_fisher_state_pt": cdir / "ewc_fisher_state.pt",
        "calibration_json": cdir / "calibration_params.json",
        "video_calibration_json": cdir / "video_calibration_params.json",
        "audio_calibration_json": cdir / "audio_calibration_params.json",
        "vn_demographic_json": cdir / "vietnamese_demographic_params.json",
    }
    report: Dict[str, Any] = {
        "custom_vit_dir": str(cdir),
        "active_model_sha256": compute_active_model_sha256(str(cdir)),
        "artifacts": {},
        "all_critical_present": True,
    }
    critical = ("vit_safetensors", "audio_fusionnet_pt", "ppo_meta_controller_pt", "calibration_json", "video_calibration_json")
    for name, path in artifacts.items():
        exists = path.exists() and path.stat().st_size > 0
        size_mb = round(path.stat().st_size / (1024 * 1024), 3) if exists else 0.0
        report["artifacts"][name] = {"exists": exists, "size_mb": size_mb, "path": str(path)}
        if name in critical and not exists:
            report["all_critical_present"] = False
    return report
