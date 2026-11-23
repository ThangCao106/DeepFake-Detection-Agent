# -*- coding: utf-8 -*-
"""
Modular Flask Blueprint for Colab GPU -> Production Server Closed-Loop Weight Sync (/api/colab/callback).
Supports updating image/video/audio calibration JSONs AND optionally pulling full checkpoints
(model.safetensors, audio_fusionnet.pt, ppo_meta_controller.pt) from Hugging Face Hub on callback.
"""

import json
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from flask import Blueprint, jsonify, request

from forensics_kernel.calibration_store import resolve_custom_vit_dir

logger = logging.getLogger(__name__)


def create_colab_blueprint(reload_detector_fn, emit_fn):
    bp = Blueprint("colab_routes", __name__)

    @bp.route('/api/colab/callback', methods=['POST'])
    def colab_callback():
        try:
            data = request.get_json(force=True) or {}
            logger.info(f"[colab_bridge] Received tuning callback from Colab: {data}")
            custom_dir = resolve_custom_vit_dir()
            custom_dir.mkdir(parents=True, exist_ok=True)

            # 1. Update Image calibration
            calib_file = custom_dir / "calibration_params.json"
            existing_img = {}
            if calib_file.exists():
                try:
                    existing_img = json.loads(calib_file.read_text(encoding="utf-8"))
                except Exception:
                    pass
            calib_data = {
                **existing_img,
                "temperature_scaling_T": float(data.get("temperature_scaling_T", existing_img.get("temperature_scaling_T", 1.6178))),
                "optimal_threshold_tau": float(data.get("optimal_threshold_tau", existing_img.get("optimal_threshold_tau", 0.8118))),
                "calibration_status": "CALIBRATED_FROM_COLAB_GPU",
                "accuracy": data.get("out_of_distribution_accuracy", existing_img.get("accuracy")),
                "dataset_trained": data.get("dataset_trained", existing_img.get("dataset_trained")),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            calib_file.write_text(json.dumps(calib_data, indent=2), encoding="utf-8")

            # 2. Optional Video / Audio calibration payload from Colab MultiModality AutoTrainer
            if isinstance(data.get("video_calibration_params"), dict):
                vid_file = custom_dir / "video_calibration_params.json"
                vid_cur = json.loads(vid_file.read_text(encoding="utf-8")) if vid_file.exists() else {}
                vid_cur.update(data["video_calibration_params"])
                vid_cur["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                vid_file.write_text(json.dumps(vid_cur, indent=2), encoding="utf-8")

            if isinstance(data.get("audio_calibration_params"), dict):
                aud_file = custom_dir / "audio_calibration_params.json"
                aud_cur = json.loads(aud_file.read_text(encoding="utf-8")) if aud_file.exists() else {}
                aud_cur.update(data["audio_calibration_params"])
                aud_cur["updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                aud_file.write_text(json.dumps(aud_cur, indent=2), encoding="utf-8")

            # 3. Optional background HF Hub checkpoint pull when Colab publishes new safetensors / .pt weights
            hf_repo_id = (data.get("hf_repo_id") or "").strip()
            hf_sync_started = False
            if hf_repo_id:
                hf_sync_started = True

                def _pull_and_reload(repo_id: str, dest: Path):
                    try:
                        from huggingface_hub import snapshot_download
                        snapshot_download(
                            repo_id=repo_id,
                            local_dir=str(dest),
                            token=os.environ.get("HF_TOKEN") or None,
                            allow_patterns=["*.json", "*.safetensors", "*.pt"],
                        )
                        logger.info(f"[colab_bridge] Pulled checkpoints from HF Hub '{repo_id}' -> {dest}")
                        reload_detector_fn(full_reinit=True)
                        emit_fn('colab_sync_received', {"hf_repo_id": repo_id, "status": "HF_WEIGHTS_RELOADED"})
                    except Exception as err:
                        logger.warning(f"[colab_bridge] HF snapshot pull failed for {repo_id}: {err}")

                threading.Thread(target=_pull_and_reload, args=(hf_repo_id, custom_dir), daemon=True).start()

            reload_detector_fn(full_reinit=False)
            emit_fn('colab_sync_received', calib_data)

            return jsonify({
                "status": "success",
                "message": "Colab tuning parameters ingested & detector hot-reloaded",
                "calibrated": calib_data,
                "hf_checkpoint_sync_started": hf_sync_started,
            }), 200
        except Exception as e:
            logger.error(f"[colab_bridge] Callback error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    return bp
