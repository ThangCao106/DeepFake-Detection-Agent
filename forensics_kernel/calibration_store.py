# -*- coding: utf-8 -*-
"""
Shared Calibration & Weight Telemetry Store.
Single source of truth for reading/validating image, video, audio, and
Vietnamese demographic calibration JSONs and model checkpoint health across
Core (Flask Port 5000) and Observability Cockpit (FastAPI Port 8787).
"""

import json
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


DEFAULT_IMAGE_PARAMS: Dict[str, Any] = {
    "temperature_scaling_T": 1.6178,
    "optimal_threshold_tau": 0.8118,
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
        "ppo_meta_controller_pt": cdir / "ppo_meta_controller.pt",
        "calibration_json": cdir / "calibration_params.json",
        "video_calibration_json": cdir / "video_calibration_params.json",
        "audio_calibration_json": cdir / "audio_calibration_params.json",
        "vn_demographic_json": cdir / "vietnamese_demographic_params.json",
    }
    report: Dict[str, Any] = {"custom_vit_dir": str(cdir), "artifacts": {}, "all_critical_present": True}
    critical = ("vit_safetensors", "audio_fusionnet_pt", "ppo_meta_controller_pt", "calibration_json", "video_calibration_json")
    for name, path in artifacts.items():
        exists = path.exists() and path.stat().st_size > 0
        size_mb = round(path.stat().st_size / (1024 * 1024), 3) if exists else 0.0
        report["artifacts"][name] = {"exists": exists, "size_mb": size_mb, "path": str(path)}
        if name in critical and not exists:
            report["all_critical_present"] = False
    return report
