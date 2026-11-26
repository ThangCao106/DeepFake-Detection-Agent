# -*- coding: utf-8 -*-
"""
Scheduled & On-Demand Feedback Review & AI Pre-Triage Job (/dfagent Queue Auditor).

Strictly preserves the Human-in-the-Loop Governance Gate:
  - Audits PENDING `HUMAN_REVIEW` disputes from SQLite (and optionally live `/dfagent/api/feedback/review-queue`)
  - Checks duplicate spam, telemetry corroboration, and risk-vs-claim consistency
  - Writes advisory AI Triage columns (`ai_triage_recommendation`, `ai_triage_score`, `ai_triage_reason`)
  - NEVER approves/dismisses `PENDING` rows automatically and NEVER mutates weights from `PENDING` items
  - Optionally applies approved weights (`--apply-approved-weights`) strictly using rows already
    approved on `/dfagent` (`APPROVED_REAL`, `APPROVED_FAKE`, `APPROVED_CALIBRATED`)
  - Writes JSON + Markdown audit reports and logs run history in SQLite `feedback_review_jobs`
"""

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def triage_single_feedback_item(
    ev: Dict[str, Any],
    seen_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Compute non-destructive AI Pre-Triage recommendation for a single PENDING human review item.
    Returns: {"recommendation": str, "score": float, "reason": str}
    """
    cert_id = str(ev.get("certificate_id") or "").strip()
    src_url = str(ev.get("source_url") or "").strip()
    true_lbl = str(ev.get("true_label") or "").strip()
    pred_lbl = str(ev.get("predicted_label") or "").strip()
    fb_type = str(ev.get("feedback_type") or "").upper()
    risk = float(ev.get("composite_risk_score") or 0.5)
    note = str(ev.get("user_note") or "").strip()

    tel = ev.get("feature_telemetry") or {}
    if isinstance(tel, str):
        try:
            tel = json.loads(tel)
        except Exception:
            tel = {}

    dedup_key = f"{cert_id or src_url}::{true_lbl}".lower()
    if seen_keys is not None and (cert_id or src_url):
        if dedup_key in seen_keys:
            return {
                "recommendation": "RECOMMEND_DISMISS_SPAM",
                "score": 0.92,
                "reason": "Trùng lặp phản ánh (cùng Certificate/URL và nhãn đề xuất).",
            }
        seen_keys.add(dedup_key)

    max_fp = float(tel.get("max_fake_prob", risk))
    jitter = float(tel.get("motion_jitter", 0.0))
    vocoder = float(tel.get("vocoder_anomaly_score", 0.0))
    file_exists = bool(ev.get("file_path") and os.path.exists(str(ev.get("file_path"))))

    if fb_type == "CONFIRMED" or pred_lbl.lower() == true_lbl.lower():
        return {
            "recommendation": "RECOMMEND_DISMISS_SPAM",
            "score": 0.88,
            "reason": "Nhãn đề xuất trùng với kết quả mô hình đã dự đoán.",
        }

    # Case 1: FALSE_POSITIVE (model flagged Fake, user insists Real)
    if fb_type == "FALSE_POSITIVE" or "real" in true_lbl.lower():
        if risk >= 0.96 and not note and not file_exists:
            return {
                "recommendation": "RECOMMEND_DISMISS_SPAM",
                "score": 0.84,
                "reason": f"Rủi ro gốc rất cao ({risk*100:.1f}%) nhưng thiếu ghi chú & tệp kiểm chứng.",
            }
        if 0.45 <= risk <= 0.80 and (note or file_exists or cert_id):
            return {
                "recommendation": "RECOMMEND_APPROVE_REAL",
                "score": 0.79,
                "reason": f"Ca vùng biên ({risk*100:.1f}%) có ngữ cảnh hợp lệ; đề xuất duyệt REAL nếu ảnh/video tự nhiên.",
            }
        return {
            "recommendation": "NEEDS_MANUAL_INSPECTION",
            "score": 0.58,
            "reason": f"User báo FP khi risk={risk*100:.1f}%; cần Admin mở /dfagent kiểm tra trực quan.",
        }

    # Case 2: FALSE_NEGATIVE (model predicted Real, user insists Deepfake/AI)
    if max_fp >= 0.55 or jitter >= 0.14 or vocoder >= 0.65:
        return {
            "recommendation": "RECOMMEND_APPROVE_FAKE",
            "score": 0.86,
            "reason": (
                f"Telemetry ẩn ghi nhận dị thường (peak={max_fp:.2f}, jitter={jitter:.2f}, vocoder={vocoder:.2f}); "
                "đề xuất Admin duyệt FAKE."
            ),
        }
    if risk <= 0.08 and not note and not file_exists:
        return {
            "recommendation": "RECOMMEND_DISMISS_SPAM",
            "score": 0.81,
            "reason": f"Rủi ro rất thấp ({risk*100:.1f}%) và không có ghi chú/bằng chứng kèm theo.",
        }
    return {
        "recommendation": "NEEDS_MANUAL_INSPECTION",
        "score": 0.60,
        "reason": f"User báo FN (risk={risk*100:.1f}%); cần Admin đối chiếu Certificate Vault trước khi duyệt.",
    }


def _fetch_remote_pending_reviews(api_base: str, limit: int = 100) -> List[Dict[str, Any]]:
    url = f"{api_base.rstrip('/')}/dfagent/api/feedback/review-queue?review_status=PENDING&limit={int(limit)}"
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "DeepTrace-FeedbackReviewJob/1.0"},
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        events = payload.get("events") if isinstance(payload, dict) else []
        return events if isinstance(events, list) else []
    except Exception:
        return []


def run_feedback_review_job(
    db_path: Optional[str] = None,
    api_base: Optional[str] = None,
    apply_approved_weights: bool = False,
    trigger_source: str = "scheduled_job",
    output_dir: str = "report_colab/feedback_reviews",
    detector=None,
) -> Dict[str, Any]:
    """
    Execute the Feedback Review & Pre-Triage Audit Job.
    NEVER mutates weights from PENDING feedback; only /dfagent-approved events can update weights.
    """
    from benchmark.deepfake_memory import (
        get_approved_feedback_for_weights,
        get_human_review_stats,
        get_recent_feedback,
        save_feedback_review_job,
        update_feedback_triage,
    )

    now_utc = datetime.now(timezone.utc)
    now_iso = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
    date_str = now_utc.strftime("%Y-%m-%d")
    job_run_id = f"FRJ-{now_utc.strftime('%Y%m%d%H%M%S')}"

    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    pending_local = get_recent_feedback(
        limit=200,
        source_type="HUMAN_REVIEW",
        review_status="PENDING",
        db_path=db_path,
    )
    remote_pending: List[Dict[str, Any]] = []
    if api_base:
        remote_pending = _fetch_remote_pending_reviews(api_base, limit=150)

    seen_fids: Set[str] = set()
    audit_items: List[Dict[str, Any]] = []
    for ev in pending_local + remote_pending:
        fid = str(ev.get("feedback_id") or "")
        if not fid or fid in seen_fids:
            continue
        seen_fids.add(fid)
        audit_items.append(ev)

    seen_keys: Set[str] = set()
    rec_approve = 0
    rec_dismiss = 0
    manual_insp = 0
    triaged_records: List[Dict[str, Any]] = []

    for ev in audit_items:
        fid = str(ev.get("feedback_id"))
        triage = triage_single_feedback_item(ev, seen_keys=seen_keys)
        rec = triage["recommendation"]
        if rec in ("RECOMMEND_APPROVE_REAL", "RECOMMEND_APPROVE_FAKE"):
            rec_approve += 1
        elif rec == "RECOMMEND_DISMISS_SPAM":
            rec_dismiss += 1
        else:
            manual_insp += 1

        update_feedback_triage(
            feedback_id=fid,
            recommendation=rec,
            score=triage["score"],
            reason=triage["reason"],
            reviewed_at=now_iso,
            db_path=db_path,
        )
        triaged_records.append({
            "feedback_id": fid,
            "media_type": ev.get("media_type", "unknown"),
            "platform": ev.get("platform", "unknown"),
            "predicted_label": ev.get("predicted_label", ""),
            "true_label": ev.get("true_label", ""),
            "risk": ev.get("composite_risk_score", 0.0),
            "review_status": ev.get("review_status", "PENDING"),
            **triage,
        })

    approved_events = get_approved_feedback_for_weights(limit=250, db_path=db_path)
    weights_applied = False
    applied_updates: Dict[str, Any] = {}

    if apply_approved_weights and approved_events:
        try:
            from benchmark.feedback_weight_updater import FeedbackWeightUpdater
            modalities = {str(e.get("media_type", "video")).lower() for e in approved_events}
            if "video" in modalities:
                applied_updates["video"] = FeedbackWeightUpdater.run_fast_calibration_update(
                    detector=detector, trigger_source=f"review_job:{job_run_id}", db_path=db_path
                )
            if "audio" in modalities:
                applied_updates["audio"] = FeedbackWeightUpdater.run_audio_calibration_update(
                    detector=detector, trigger_source=f"review_job:{job_run_id}", db_path=db_path
                )
            if "image" in modalities:
                applied_updates["image"] = FeedbackWeightUpdater.run_image_calibration_update(
                    detector=detector, trigger_source=f"review_job:{job_run_id}", db_path=db_path
                )
            weights_applied = any(isinstance(v, dict) and v.get("accepted") for v in applied_updates.values())
        except Exception as exc:
            applied_updates["error"] = str(exc)

    stats = get_human_review_stats(db_path=db_path)
    json_path = out_path / "latest_feedback_review_job.json"
    md_path = out_path / f"feedback_review_report_{date_str}.md"

    summary = {
        "job_run_id": job_run_id,
        "timestamp_utc": now_iso,
        "trigger_source": trigger_source,
        "approval_gate_enforced": True,
        "pending_audited": len(triaged_records),
        "recommended_approve": rec_approve,
        "recommended_dismiss": rec_dismiss,
        "manual_inspection": manual_insp,
        "approved_ready_for_weights": len(approved_events),
        "weights_applied": weights_applied,
        "applied_updates": applied_updates,
        "queue_stats": stats,
        "triaged_items": triaged_records,
        "report_json_path": str(json_path),
        "report_md_path": str(md_path),
    }

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        f"# DeepTrace Feedback Review & Pre-Triage Report (`{job_run_id}`)",
        f"- **Timestamp (UTC)**: `{now_iso}` | **Trigger**: `{trigger_source}`",
        "- **Governance Policy**: Public feedback strictly enters `/dfagent` (`PENDING`); weights NEVER change without Admin approval.",
        f"- **Pending Audited**: `{len(triaged_records)}` (`Approve={rec_approve}`, `Dismiss={rec_dismiss}`, `Manual={manual_insp}`)",
        f"- **Already Approved for Weights**: `{len(approved_events)}` | **Weights Applied This Run**: `{weights_applied}`",
        "",
        "## Pending Queue AI Pre-Triage Summary",
        "| Feedback ID | Media / Platform | Model -> Claim | Recommendation | Score | Reason |",
        "|:---|:---|:---|:---|---:|:---|",
    ]
    if not triaged_records:
        md_lines.append("| *(none)* | - | - | `QUEUE_CLEAR` | 1.00 | Không có phản ánh PENDING cần pre-triage. |")
    else:
        for item in triaged_records:
            md_lines.append(
                f"| `{item['feedback_id']}` | `{item['media_type']}/{item['platform']}` | "
                f"{item['predicted_label']} -> {item['true_label']} | `{item['recommendation']}` | "
                f"{item['score']:.2f} | {item['reason']} |"
            )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    save_feedback_review_job(summary, db_path=db_path)
    print(
        f"[FeedbackReviewJob] {job_run_id}: audited={len(triaged_records)} pending "
        f"(rec_approve={rec_approve}, rec_dismiss={rec_dismiss}, manual={manual_insp}), "
        f"approved_ready={len(approved_events)}, weights_applied={weights_applied}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="DeepTrace /dfagent Feedback Review & Pre-Triage Job")
    parser.add_argument("--db-path", type=str, default=None, help="Optional SQLite path")
    parser.add_argument("--api-base", type=str, default=None, help="Optional live base URL (e.g. https://deeptrace.io.vn)")
    parser.add_argument("--apply-approved-weights", action="store_true", help="Apply calibration ONLY for /dfagent-approved feedback")
    parser.add_argument("--trigger-source", type=str, default="scheduled_job")
    parser.add_argument("--output-dir", type=str, default="report_colab/feedback_reviews")
    args = parser.parse_args()

    run_feedback_review_job(
        db_path=args.db_path,
        api_base=args.api_base,
        apply_approved_weights=args.apply_approved_weights,
        trigger_source=args.trigger_source,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
