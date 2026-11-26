# -*- coding: utf-8 -*-
"""
Modular Flask Blueprint for Closed-Loop Feedback, Human Review Queue, and Multimodal Pentests.
Registered by web_ui/app.py with shared detector/socketio callbacks injected via create_feedback_blueprint.
Strict Governance:
  - Public /api/feedback submissions are unconditionally quarantined in /dfagent (PENDING) and
    NEVER modify calibration params, model weights, or LoRA checkpoints until an admin approves
    them in /dfagent -> Human-in-the-Loop User Review Queue.
"""

import hmac
import logging
import os

logger = logging.getLogger(__name__)


def _is_trusted_internal_pentest(req, data: dict) -> bool:
    """Allow direct AUTO_PENTEST logging only when explicitly authorized by internal secret."""
    source_type = str(data.get("source_type") or "HUMAN_REVIEW").upper().strip()
    if source_type != "AUTO_PENTEST":
        return False
    expected = (os.environ.get("INTERNAL_PENTEST_SECRET") or "").strip()
    if not expected:
        return False
    provided = (req.headers.get("X-Internal-Pentest-Secret") or "").strip()
    if not provided:
        return False
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def create_feedback_blueprint(get_detector_fn, emit_fn, execute_short_video_fn):
    from flask import Blueprint, jsonify, request
    bp = Blueprint("feedback_routes", __name__)

    @bp.route('/api/feedback', methods=['POST'])
    @bp.route('/benchmark/api/feedback', methods=['POST'])
    @bp.route('/dfagent/api/feedback', methods=['POST'])
    @bp.route('/dfagent/benchmark/api/feedback', methods=['POST'])
    def api_submit_feedback():
        try:
            data = request.get_json(silent=True) or {}
            true_label = data.get("true_label")
            if not true_label:
                return jsonify({"status": "error", "message": "Missing required field: true_label ('Realism' or 'Deepfake')"}), 400

            from benchmark.feedback_weight_updater import FeedbackWeightUpdater
            active_detector = get_detector_fn()

            # HARD GOVERNANCE GATE: Public submissions strictly go to /dfagent Review Queue (PENDING).
            if _is_trusted_internal_pentest(request, data):
                source_type = "AUTO_PENTEST"
                review_status = str(data.get("review_status") or "AUTO_LOGGED").upper()
                auto_calib = bool(data.get("auto_update_calibration", False))
                trigger_lora = bool(data.get("trigger_lora_update", False))
            else:
                source_type = "HUMAN_REVIEW"
                review_status = "PENDING"
                auto_calib = False
                trigger_lora = False

            res = FeedbackWeightUpdater.submit_feedback(
                true_label=true_label,
                certificate_id=data.get("certificate_id") or data.get("cert_id"),
                platform=data.get("platform"),
                media_type=data.get("media_type", "video"),
                file_path=data.get("file_path") or data.get("filepath"),
                source_url=data.get("source_url"),
                predicted_label=data.get("predicted_label"),
                composite_risk_score=data.get("composite_risk_score"),
                feature_telemetry=data.get("feature_telemetry"),
                user_note=data.get("user_note", ""),
                auto_update_calibration=auto_calib,
                trigger_lora_update=trigger_lora,
                detector=active_detector,
                source_type=source_type,
                review_status=review_status,
            )
            calib = res.get("calibration_update") or {}
            if calib.get("accepted") and calib.get("after_params"):
                emit_fn('optimal_weights_applied', calib["after_params"])
            return jsonify(res), 200
        except Exception as e:
            logger.error(f"Feedback submission error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/feedback/history', methods=['GET'])
    @bp.route('/benchmark/api/feedback/history', methods=['GET'])
    @bp.route('/dfagent/api/feedback/history', methods=['GET'])
    @bp.route('/dfagent/benchmark/api/feedback/history', methods=['GET'])
    def api_feedback_history():
        try:
            from benchmark.deepfake_memory import get_recent_feedback, get_human_review_stats
            from benchmark.feedback_weight_updater import FeedbackWeightUpdater
            limit = int(request.args.get("limit", 50))
            source_type = request.args.get("source_type")
            review_status = request.args.get("review_status")
            media_type = request.args.get("media_type")
            return jsonify({
                "status": "success",
                "events": get_recent_feedback(
                    limit=limit,
                    media_type=media_type,
                    source_type=source_type,
                    review_status=review_status,
                ),
                "review_stats": get_human_review_stats(),
                "active_video_params": FeedbackWeightUpdater.load_video_calibration_params(),
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/feedback/review-queue', methods=['GET'])
    @bp.route('/dfagent/api/feedback/review-queue', methods=['GET'])
    def api_human_review_queue():
        """Dedicated Human-in-the-Loop queue listing real user review requests cleanly separated from auto-pentests."""
        try:
            from benchmark.deepfake_memory import get_recent_feedback, get_human_review_stats
            limit = int(request.args.get("limit", 100))
            status_filter = request.args.get("review_status")
            events = get_recent_feedback(
                limit=limit,
                source_type="HUMAN_REVIEW",
                review_status=status_filter if status_filter and status_filter != "ALL" else None,
            )
            return jsonify({
                "status": "success",
                "stats": get_human_review_stats(),
                "events": events,
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/feedback/review-queue/<feedback_id>/decision', methods=['POST'])
    @bp.route('/dfagent/api/feedback/review-queue/<feedback_id>/decision', methods=['POST'])
    def api_human_review_decision(feedback_id: str):
        """Approve or dismiss a user review case and optionally trigger calibration / LoRA weight update."""
        try:
            from benchmark.deepfake_memory import update_feedback_review_status, get_human_review_stats, get_db_connection
            from benchmark.feedback_weight_updater import FeedbackWeightUpdater

            data = request.get_json(silent=True) or {}
            decision = str(data.get("decision") or "").upper().strip()
            reviewer_note = str(data.get("reviewer_note") or "")
            trigger_calib = bool(data.get("trigger_calibration", False))
            trigger_lora = bool(data.get("trigger_lora", False))

            allowed = ("APPROVE_REAL", "APPROVE_FAKE", "DISMISS", "TRIGGER_CALIBRATION", "APPROVE_AND_UPDATE_WEIGHTS")
            if decision not in allowed:
                return jsonify({"status": "error", "message": f"Unsupported decision: {decision}"}), 400

            conn = get_db_connection()
            row = conn.execute(
                "SELECT media_type, true_label FROM user_feedback_events WHERE feedback_id = ?",
                (feedback_id,),
            ).fetchone()
            conn.close()
            if not row:
                return jsonify({"status": "error", "message": f"Feedback '{feedback_id}' not found"}), 404

            row_d = dict(row)
            mtype = str(row_d.get("media_type", "video")).strip().lower()
            existing_true = str(row_d.get("true_label") or "Deepfake")

            true_label = None
            new_status = "DISMISSED"
            if decision == "APPROVE_REAL":
                true_label = "Realism"
                new_status = "APPROVED_REAL"
            elif decision == "APPROVE_FAKE":
                true_label = "Deepfake"
                new_status = "APPROVED_FAKE"
            elif decision in ("TRIGGER_CALIBRATION", "APPROVE_AND_UPDATE_WEIGHTS"):
                true_label = data.get("true_label") or existing_true
                new_status = "APPROVED_CALIBRATED"
                trigger_calib = True

            ok = update_feedback_review_status(
                feedback_id=feedback_id,
                review_status=new_status,
                reviewer_note=reviewer_note,
                true_label=true_label,
            )
            if not ok:
                return jsonify({"status": "error", "message": f"Feedback '{feedback_id}' update failed"}), 500

            FeedbackWeightUpdater.sync_manifest_approval_status(
                feedback_id=feedback_id,
                review_status=new_status,
                true_label=true_label,
            )

            calib_res = None
            lora_res = None
            active_detector = get_detector_fn()
            if trigger_calib:
                if mtype == "audio":
                    calib_res = FeedbackWeightUpdater.run_audio_calibration_update(
                        detector=active_detector,
                        trigger_source=f"dfagent_review:{feedback_id}:{new_status}",
                    )
                elif mtype == "image":
                    calib_res = FeedbackWeightUpdater.run_image_calibration_update(
                        detector=active_detector,
                        trigger_source=f"dfagent_review:{feedback_id}:{new_status}",
                    )
                else:
                    calib_res = FeedbackWeightUpdater.run_fast_calibration_update(
                        detector=active_detector,
                        trigger_source=f"dfagent_review:{feedback_id}:{new_status}",
                    )
                if isinstance(calib_res, dict) and calib_res.get("accepted") and calib_res.get("after_params"):
                    emit_fn('optimal_weights_applied', calib_res["after_params"])

            if trigger_lora and decision != "DISMISS":
                lora_res = FeedbackWeightUpdater.run_slow_lora_weight_update(detector=active_detector)

            return jsonify({
                "status": "success",
                "feedback_id": feedback_id,
                "review_status": new_status,
                "calibration_update": calib_res,
                "lora_update": lora_res,
                "stats": get_human_review_stats(),
            }), 200
        except Exception as e:
            logger.error(f"Review decision error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/feedback/review-job/status', methods=['GET'])
    @bp.route('/dfagent/api/feedback/review-job/status', methods=['GET'])
    def api_feedback_review_job_status():
        """Return latest Feedback Review & Triage job history + queue stats."""
        try:
            from benchmark.deepfake_memory import get_human_review_stats, get_latest_feedback_review_jobs
            return jsonify({
                "status": "success",
                "stats": get_human_review_stats(),
                "jobs": get_latest_feedback_review_jobs(limit=15),
            }), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/feedback/review-job/run', methods=['POST'])
    @bp.route('/dfagent/api/feedback/review-job/run', methods=['POST'])
    def api_run_feedback_review_job():
        """Run Feedback Review & AI Pre-Triage job on demand from /dfagent."""
        try:
            data = request.get_json(silent=True) or {}
            apply_weights = bool(data.get("apply_approved_weights", False))
            from scripts.review_feedback_queue_job import run_feedback_review_job
            result = run_feedback_review_job(
                apply_approved_weights=apply_weights,
                trigger_source="dfagent_ui_manual",
                detector=get_detector_fn(),
            )
            return jsonify({"status": "success", **result}), 200
        except Exception as e:
            logger.error(f"Feedback review job error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/pentest/short-video', methods=['GET', 'POST'])
    @bp.route('/benchmark/api/pentest/short-video', methods=['GET', 'POST'])
    @bp.route('/dfagent/api/pentest/short-video', methods=['GET', 'POST'])
    @bp.route('/dfagent/benchmark/api/pentest/short-video', methods=['GET', 'POST'])
    def api_short_video_pentest():
        try:
            if request.method == 'GET':
                from benchmark.deepfake_memory import get_latest_short_video_pentest
                return jsonify({"status": "success", "runs": get_latest_short_video_pentest(limit=20)}), 200

            data = request.get_json(silent=True) or {}
            res = execute_short_video_fn(
                platforms=data.get("platforms"),
                attacks=data.get("attacks"),
                manifest_path=data.get("manifest_path"),
                download_urls=bool(data.get("download_urls", True)),
                auto_feedback=bool(data.get("auto_feedback", True)),
                trigger_lora=bool(data.get("trigger_lora", False)),
            )
            if res.get("skipped"):
                return jsonify({"status": "skipped", "message": "Short-video pentest job already running"}), 202
            return jsonify({
                "status": "success",
                "run_id": res["run_id"],
                "overall_metrics": res["overall_metrics"],
                "by_platform": res["by_platform"],
                "by_attack_vector": res["by_attack_vector"],
                "misclassified_count": res["misclassified_count"],
                "report_md_path": res["report_md_path"],
            }), 200
        except Exception as e:
            logger.error(f"Short-video pentest API error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    @bp.route('/api/pentest/audio', methods=['GET', 'POST'])
    @bp.route('/benchmark/api/pentest/audio', methods=['GET', 'POST'])
    @bp.route('/dfagent/api/pentest/audio', methods=['GET', 'POST'])
    @bp.route('/dfagent/benchmark/api/pentest/audio', methods=['GET', 'POST'])
    def api_audio_pentest():
        try:
            if request.method == 'GET':
                from benchmark.deepfake_memory import get_latest_audio_pentest
                return jsonify({"status": "success", "runs": get_latest_audio_pentest(limit=20)}), 200

            from benchmark.audio_pentest import run_audio_pentest
            data = request.get_json(silent=True) or {}
            res = run_audio_pentest(
                channels=data.get("channels"),
                attacks=data.get("attacks"),
                manifest_path=data.get("manifest_path"),
                include_cache=bool(data.get("include_cache", True)),
                auto_feedback=bool(data.get("auto_feedback", False)),
                detector=get_detector_fn(),
            )
            return jsonify({
                "status": "success",
                "run_id": res["run_id"],
                "overall_metrics": res["overall_metrics"],
                "by_channel": res["by_channel"],
                "by_attack_vector": res["by_attack_vector"],
                "misclassified_count": res["misclassified_count"],
                "report_md_path": res["report_md_path"],
            }), 200
        except Exception as e:
            logger.error(f"Audio pentest API error: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

    return bp
