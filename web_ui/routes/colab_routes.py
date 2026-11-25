# -*- coding: utf-8 -*-
"""
Modular Flask Blueprint for Colab GPU -> Production Server Closed-Loop Weight Sync (/api/colab/callback).
Supports updating image/video/audio/vn_demographic calibration JSONs AND securely pulling full checkpoints
(model.safetensors, audio_fusionnet.pt, ppo_meta_controller.pt) from whitelisted Hugging Face Hub repos.
"""

import hmac
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from flask import Blueprint, jsonify, request

from forensics_kernel.calibration_store import resolve_custom_vit_dir

logger = logging.getLogger(__name__)

TRUSTED_HF_REPO_PATTERN = re.compile(r"^(ThangCao106|ThangCao)/[A-Za-z0-9._-]+$")


def _is_trusted_hf_repo(repo_id: str) -> bool:
    repo_id = (repo_id or "").strip()
    if not repo_id:
        return False
    configured_repo = (os.environ.get("CUSTOM_VIT_HF_REPO") or "").strip()
    if configured_repo and repo_id == configured_repo:
        return True
    return bool(TRUSTED_HF_REPO_PATTERN.match(repo_id))


def _verify_callback_auth(req, payload: dict) -> bool:
    """
    Verify COLAB_CALLBACK_SECRET when configured on the production server.
    Accepts header `X-Colab-Secret`, `Authorization: Bearer <secret>`, or body `callback_secret`.
    If COLAB_CALLBACK_SECRET is not set on server, allows callback (protected by hf_repo_id whitelist).
    """
    expected = (os.environ.get("COLAB_CALLBACK_SECRET") or "").strip()
    if not expected:
        return True

    provided = (req.headers.get("X-Colab-Secret") or "").strip()
    if not provided:
        auth_hdr = (req.headers.get("Authorization") or "").strip()
        if auth_hdr.lower().startswith("bearer "):
            provided = auth_hdr[7:].strip()
    if not provided and isinstance(payload, dict):
        provided = str(payload.get("callback_secret") or "").strip()

    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _download_hf_snapshot_resilient(repo_id: str, dest: Path) -> None:
    """Download checkpoints from Hugging Face Hub, falling back to anonymous access if server HF_TOKEN is expired."""
    from huggingface_hub import snapshot_download

    allow_patterns = ["*.json", "*.safetensors", "*.pt"]
    env_token = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest),
            token=env_token if env_token else False,
            allow_patterns=allow_patterns,
        )
    except Exception as first_err:
        if env_token:
            logger.warning(
                f"[colab_bridge] Authenticated HF download failed for '{repo_id}' ({first_err}); retrying anonymously (token=False)..."
            )
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(dest),
                token=False,
                allow_patterns=allow_patterns,
            )
        else:
            raise


def create_colab_blueprint(reload_detector_fn, emit_fn):
    bp = Blueprint("colab_routes", __name__)

    @bp.route('/api/colab/callback', methods=['POST'])
    def colab_callback():
        try:
            data = request.get_json(force=True, silent=True) or {}
            if not _verify_callback_auth(request, data):
                logger.warning("[colab_bridge] Rejected unauthorized /api/colab/callback request.")
                return jsonify({"status": "error", "message": "Unauthorized Colab callback secret"}), 401

            hf_repo_id = (data.get("hf_repo_id") or "").strip()
            if hf_repo_id and not _is_trusted_hf_repo(hf_repo_id):
                logger.warning(f"[colab_bridge] Blocked untrusted hf_repo_id '{hf_repo_id}'.")
                return jsonify({
                    "status": "error",
                    "message": f"Untrusted hf_repo_id '{hf_repo_id}'. Allowed namespaces: ThangCao106/*, ThangCao/*, or CUSTOM_VIT_HF_REPO.",
                }), 403

            now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            logger.info(f"[colab_bridge] Received tuning callback from Colab (hf_repo_id={hf_repo_id or 'none'})")
            custom_dir = resolve_custom_vit_dir()
            custom_dir.mkdir(parents=True, exist_ok=True)

            # 1. Update Image calibration (clamped within physically valid bounds)
            calib_file = custom_dir / "calibration_params.json"
            existing_img = {}
            if calib_file.exists():
                try:
                    existing_img = json.loads(calib_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            raw_t = float(data.get("temperature_scaling_T", existing_img.get("temperature_scaling_T", 1.6178)))
            raw_tau = float(data.get("optimal_threshold_tau", existing_img.get("optimal_threshold_tau", 0.8118)))
            clamped_t = round(min(5.0, max(0.25, raw_t)), 4)
            clamped_tau = round(min(0.98, max(0.20, raw_tau)), 4)

            calib_data = {
                **existing_img,
                "temperature_scaling_T": clamped_t,
                "optimal_threshold_tau": clamped_tau,
                "calibration_status": "CALIBRATED_FROM_COLAB_GPU",
                "accuracy": data.get("out_of_distribution_accuracy", data.get("accuracy", existing_img.get("accuracy"))),
                "auc_score": data.get("auc_score", existing_img.get("auc_score")),
                "eer_score": data.get("eer_score", existing_img.get("eer_score")),
                "ece_score": data.get("ece_score", existing_img.get("ece_score")),
                "dataset_trained": data.get("dataset_trained", existing_img.get("dataset_trained")),
                "timestamp": now_iso,
            }
            calib_file.write_text(json.dumps(calib_data, indent=2), encoding="utf-8")

            # 2. Optional Video / Audio / Vietnamese Demographic calibration payloads
            if isinstance(data.get("video_calibration_params"), dict):
                vid_file = custom_dir / "video_calibration_params.json"
                vid_cur = json.loads(vid_file.read_text(encoding="utf-8")) if vid_file.exists() else {}
                vid_cur.update(data["video_calibration_params"])
                vid_cur["updated_at"] = now_iso
                vid_file.write_text(json.dumps(vid_cur, indent=2), encoding="utf-8")

            if isinstance(data.get("audio_calibration_params"), dict):
                aud_file = custom_dir / "audio_calibration_params.json"
                aud_cur = json.loads(aud_file.read_text(encoding="utf-8")) if aud_file.exists() else {}
                aud_cur.update(data["audio_calibration_params"])
                aud_cur["updated_at"] = now_iso
                aud_file.write_text(json.dumps(aud_cur, indent=2), encoding="utf-8")

            if isinstance(data.get("vn_demographic_params"), dict):
                vn_file = custom_dir / "vietnamese_demographic_params.json"
                vn_cur = json.loads(vn_file.read_text(encoding="utf-8")) if vn_file.exists() else {}
                vn_cur.update(data["vn_demographic_params"])
                vn_cur["updated_at"] = now_iso
                vn_file.write_text(json.dumps(vn_cur, indent=2), encoding="utf-8")

            # 3. Optional background HF Hub checkpoint pull when Colab publishes new safetensors / .pt weights
            hf_sync_started = False
            if hf_repo_id:
                hf_sync_started = True

                def _pull_and_reload(repo_id: str, dest: Path):
                    try:
                        _download_hf_snapshot_resilient(repo_id=repo_id, dest=dest)
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
