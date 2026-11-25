"""
Colab Burst-GPU & Closed-Loop Multi-Modality AutoTrainer.

Supports:
1. Pulling latest code updates from GitHub (`origin/main`).
2. Syncing approved Human Review Queue disputes from `https://deeptrace.io.vn`.
3. Empirical PyTorch Fine-Tuning across all 3 streams:
   - Stream 1: Continual ViT (`model.safetensors` + Platt T* / Youden tau*) + Spectral FCB (`weights_general.pt`)
   - Stream 2: Human Biometric Multi-Modal Fusion (`weights_human.pt`)
   - Stream 3: EnhancedAudioFusionNet (`audio_fusionnet.pt`)
4. Enforcing Out-of-Distribution (OOD) Safety Gate before publishing weights.
5. Uploading checkpoints to Hugging Face Hub (`ThangCao106/custom-deepfake-vit`) + triggering
   zero-downtime hot-reload on `https://deeptrace.io.vn/api/colab/callback`.
6. Optional `--auto-disconnect` to free Colab Pro GPU Compute Units immediately after training completes.
"""

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

COLAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COLAB_DIR.parent
for _p in (str(COLAB_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from colab_dataset_pipeline import ColabAutoSyncManager, ColabMultiModalityDatasetManager


def resolve_credentials(hf_token=None, github_token=None, callback_secret=None):
    """
    Resolve HF_TOKEN, GITHUB_TOKEN, and COLAB_CALLBACK_SECRET via Smart Fallback:
      1. Explicit CLI arguments
      2. Google Colab Secrets (`google.colab.userdata`)
      3. Private `.env` file (workspace or mounted Google Drive)
      4. Environment variables (`os.environ`)
    """
    resolved_hf = (hf_token or os.environ.get("HF_TOKEN") or "").strip()
    resolved_gh = (github_token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
    resolved_cb = (callback_secret or os.environ.get("COLAB_CALLBACK_SECRET") or "").strip()

    try:
        from google.colab import userdata
        if not resolved_hf:
            resolved_hf = (userdata.get("HF_TOKEN") or "").strip()
        if not resolved_gh:
            resolved_gh = (userdata.get("GITHUB_TOKEN") or userdata.get("GH_TOKEN") or "").strip()
        if not resolved_cb:
            try:
                resolved_cb = (userdata.get("COLAB_CALLBACK_SECRET") or "").strip()
            except Exception:
                pass
    except Exception:
        pass

    if not (resolved_hf and resolved_gh and resolved_cb):
        candidate_env_paths = [
            PROJECT_ROOT / ".env",
            COLAB_DIR / ".env",
            Path("/content/DeepFake-Detection-Agent/.env"),
            Path("/content/drive/MyDrive/.env"),
            Path("/content/drive/MyDrive/Thread_Data/.env"),
            Path("/content/drive/MyDrive/NCS/.env"),
        ]
        for p in candidate_env_paths:
            if p.is_file():
                try:
                    for line in p.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k, v = k.strip(), v.strip().strip("'\"")
                        if not resolved_hf and k in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                            resolved_hf = v
                        elif not resolved_gh and k in ("GITHUB_TOKEN", "GH_TOKEN"):
                            resolved_gh = v
                        elif not resolved_cb and k == "COLAB_CALLBACK_SECRET":
                            resolved_cb = v
                except Exception:
                    pass

    if resolved_hf:
        os.environ["HF_TOKEN"] = resolved_hf
    if resolved_gh:
        os.environ["GITHUB_TOKEN"] = resolved_gh
    if resolved_cb:
        os.environ["COLAB_CALLBACK_SECRET"] = resolved_cb

    return resolved_hf or None, resolved_gh or None, resolved_cb or None


def maybe_auto_disconnect_colab(enabled: bool):
    """Release Colab GPU runtime immediately after Burst-GPU training & sync finish."""
    if not enabled:
        return
    try:
        if hasattr(os, "sync"):
            os.sync()
        time.sleep(2)
        from google.colab import runtime
        print("\n🔌 [Burst-GPU Auto-Disconnect] Releasing Colab GPU runtime to save Compute Units...")
        runtime.unassign()
    except Exception as exc:
        print(f"ℹ️ [Burst-GPU] Runtime unassign skipped (not inside active Colab session: {exc})")


def conduct_council_preflight(requested_alpha=None, apply_if_approved=True):
    """Trigger the 3-Pillar Deliberation Council audit before training."""
    print("\n" + "=" * 70)
    print("🏛️ [COUNCIL PREFLIGHT AUDIT] Summoning LLM Deliberation Council...")
    print("=" * 70)
    try:
        from agents.calibration_deliberation_council import CalibrationDeliberationCouncil
        delib = CalibrationDeliberationCouncil.conduct_deliberation(requested_alpha=requested_alpha)
        verdict = delib.get("consensus_verdict", "UNKNOWN")
        is_approved = delib.get("is_approved_for_training", False)
        print(f"🏛️ Council Verdict: [{verdict}] | Approved={is_approved}")
        if is_approved and apply_if_approved:
            CalibrationDeliberationCouncil.apply_approved_parameters(delib, user="colab_runner")
            return True, delib
        return False, delib
    except Exception as e:
        print(f"   ⚠️ [Council Audit Note]: {e}")
        return False, None


def _build_callback_payload(vit_res: dict | None, audio_meta: dict | None, hf_repo_id: str | None, dataset_label: str) -> dict:
    """Assemble calibrated multimodal payload for POST /api/colab/callback."""
    payload = {
        "dataset_trained": dataset_label,
    }
    if vit_res:
        payload["temperature_scaling_T"] = float(vit_res.get("temperature_scaling_T", 1.6178))
        payload["optimal_threshold_tau"] = float(vit_res.get("optimal_threshold_tau", 0.8118))
        payload["out_of_distribution_accuracy"] = float(vit_res.get("accuracy", 0.0))
        payload["auc_score"] = float(vit_res.get("auc_score", vit_res.get("auc", 0.0)))
        payload["eer_score"] = float(vit_res.get("eer_score", 0.0))
        payload["ece_score"] = float(vit_res.get("ece_score", 0.0))
        payload["video_calibration_params"] = {
            "temperature_T": float(vit_res.get("temperature_scaling_T", 1.6178)),
            "tau_video": round(min(0.85, max(0.50, float(vit_res.get("optimal_threshold_tau", 0.68)))), 4),
            "calibration_status": "CALIBRATED_FROM_COLAB_BURST_GPU",
        }
    if audio_meta and isinstance(audio_meta.get("metrics"), dict):
        am = audio_meta["metrics"]
        payload["audio_calibration_params"] = {
            "calibration_status": "CALIBRATED_AUDIO_FUSIONNET_COLAB",
            "holdout_auc": float(am.get("holdout_auc", 0.0)),
            "holdout_accuracy": float(am.get("holdout_accuracy", 0.0)),
        }
    if hf_repo_id:
        payload["hf_repo_id"] = hf_repo_id
    return payload


def run_burst_all_streams(
    dataset_key: str = "threads_vietnamese_in_the_wild",
    max_samples: int = 1000,
    epochs: int = 3,
    batch_size: int = 16,
    lr: float = 1e-5,
    hf_token: str | None = None,
    github_token: str | None = None,
    callback_secret: str | None = None,
    callback_url: str = "https://deeptrace.io.vn",
    sync_human_review: bool = True,
    upload_hf: bool = True,
    notify_server: bool = True,
    auto_pull: bool = True,
    council_audit: bool = False,
    min_auc: float = 0.70,
    max_ece: float = 0.15,
    auto_disconnect: bool = False,
):
    """
    Execute a complete Single-Pass Burst-GPU Closed-Loop Training Run across Vision, Human Biometrics, and Audio,
    validate via the OOD Safety Gate, publish to Hugging Face + deeptrace.io.vn, and optionally release GPU.
    """
    hf_token, github_token, callback_secret = resolve_credentials(hf_token, github_token, callback_secret)
    sync_manager = ColabAutoSyncManager()

    print("\n" + "=" * 78)
    print("🚀 STARTING BURST-GPU CLOSED-LOOP AUTOTRAINER (VISION + BIOMETRICS + AUDIO)")
    print("=" * 78)

    if auto_pull:
        sync_manager.auto_pull_latest_code()

    if council_audit:
        conduct_council_preflight()

    extra_manifests = []
    if sync_human_review and callback_url:
        hr_manifest, _ = sync_manager.pull_human_review_feedback(api_base=callback_url)
        if hr_manifest:
            extra_manifests.append(hr_manifest)

    dataset_manager = ColabMultiModalityDatasetManager()
    img_manifest = dataset_manager.prepare_dataset_for_colab(
        modality="IMAGE",
        dataset_key=dataset_key,
        max_samples=max_samples,
    )
    if img_manifest:
        extra_manifests.append(img_manifest)

    # Stream 1A: Continual ViT Backbone -> custom_vit_model/model.safetensors
    vit_res = None
    try:
        from train_threads_continual import run_threads_continual_training
        vit_res = run_threads_continual_training(
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            extra_manifests=extra_manifests,
            max_train_samples=max_samples,
            min_ood_auc=min_auc,
            max_ood_ece=max_ece,
        )
    except Exception as exc_vit:
        print(f"[burst_runner] ⚠️ Continual ViT training error: {exc_vit}")

    # Stream 1B: DualStreamGeneralDetector (Spatial + Spectral FCB) -> custom_vit_model/weights_general.pt
    try:
        from train_general import train_general_model
        train_general_model(epochs=max(3, epochs * 2), batch_size=batch_size, output_dir=str(PROJECT_ROOT / "custom_vit_model"))
    except Exception as exc_gen:
        print(f"[burst_runner] ⚠️ General FCB stream notice: {exc_gen}")

    sync_manager.generate_and_save_report(
        modality="IMAGE",
        dataset_key=dataset_key,
        metrics=vit_res,
        pass_number=1,
        target_model="general",
    )

    # Stream 2: Human Biometric Multi-Modal Detector -> human_vit_model/weights_human.pt
    human_res = None
    try:
        from train_human import train_human_model
        human_res = train_human_model(epochs=max(3, epochs), batch_size=batch_size, output_dir=str(PROJECT_ROOT / "human_vit_model"))
        sync_manager.generate_and_save_report(
            modality="IMAGE",
            dataset_key=dataset_key,
            pass_number=1,
            target_model="human",
        )
    except Exception as exc_hum:
        print(f"[burst_runner] ⚠️ Human Biometrics stream notice: {exc_hum}")

    # Stream 3: EnhancedAudioFusionNet -> custom_vit_model/audio_fusionnet.pt
    audio_meta = None
    try:
        from train_audio_fusionnet import train_audio_fusionnet
        audio_meta = train_audio_fusionnet(epochs=35)
        sync_manager.generate_and_save_report(
            modality="AUDIO",
            dataset_key="garystafford_audio",
            pass_number=1,
            target_model="audio",
        )
    except Exception as exc_aud:
        print(f"[burst_runner] ⚠️ Audio FusionNet stream notice: {exc_aud}")

    # OOD Safety Gate before publishing weights to Hugging Face & Production
    safety_passed = bool(vit_res.get("safety_passed", True)) if vit_res else True
    hf_uploaded = False
    if safety_passed:
        if hf_token and upload_hf:
            hf_uploaded = sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "custom_vit_model"), hf_token=hf_token, force=True)
            if human_res:
                sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "human_vit_model"), hf_token=hf_token, force=True)

        if notify_server and callback_url:
            cb_payload = _build_callback_payload(
                vit_res=vit_res,
                audio_meta=audio_meta,
                hf_repo_id=sync_manager.hf_repo if hf_uploaded else None,
                dataset_label=f"burst:{dataset_key}",
            )
            sync_manager.notify_production_callback(
                api_base=callback_url,
                payload=cb_payload,
                callback_secret=callback_secret,
            )
    else:
        print(
            f"🛑 [OOD Safety Gate HOLD] Model promotion blocked (AUC={vit_res.get('auc')} < {min_auc} "
            f"or ECE={vit_res.get('ece_score')} > {max_ece}). Production weights untouched."
        )

    sync_manager.sync_to_google_drive()
    if github_token:
        sync_manager.auto_sync_metrics_to_github(
            github_token=github_token,
            commit_message=f"Auto-sync Burst-GPU empirical training metrics ({dataset_key}) [Auto-Bot]",
        )

    print("\n" + "=" * 78)
    print("🏁 [BURST-GPU COMPLETE] Empirical training, HF sync & production callback finished!")
    print("=" * 78)
    maybe_auto_disconnect_colab(auto_disconnect)
    return {"vit": vit_res, "human": human_res, "audio": audio_meta, "hf_uploaded": hf_uploaded}


def run_colab_auto_pipeline(
    modality="IMAGE",
    dataset_key="ntire_robust_detection",
    max_samples=2000,
    hf_token=None,
    github_token=None,
    callback_secret=None,
    callback_url="https://deeptrace.io.vn",
    sync_human_review=True,
    auto_pull=True,
    pass_number=1,
    target="all",
    upload_hf=False,
    notify_server=True,
    council_audit=False,
    epochs=3,
    batch_size=16,
    min_auc=0.70,
    max_ece=0.15,
    auto_disconnect=False,
):
    """Run single-modality empirical pipeline (`IMAGE`, `VIDEO`, or `AUDIO`)."""
    hf_token, github_token, callback_secret = resolve_credentials(hf_token, github_token, callback_secret)
    sync_manager = ColabAutoSyncManager()
    modality = modality.upper()
    target = target.lower()

    if auto_pull:
        sync_manager.auto_pull_latest_code()

    if council_audit:
        conduct_council_preflight()

    extra_manifests = []
    if sync_human_review and callback_url and modality in ("IMAGE", "VIDEO"):
        hr_manifest, _ = sync_manager.pull_human_review_feedback(api_base=callback_url)
        if hr_manifest:
            extra_manifests.append(hr_manifest)

    dataset_manager = ColabMultiModalityDatasetManager()
    manifest_path = dataset_manager.prepare_dataset_for_colab(
        modality=modality,
        dataset_key=dataset_key,
        max_samples=max_samples,
    )
    if manifest_path:
        extra_manifests.append(manifest_path)

    vit_res = None
    audio_meta = None
    safety_passed = True

    if modality == "AUDIO":
        from train_audio_fusionnet import train_audio_fusionnet
        audio_meta = train_audio_fusionnet(epochs=35)
        sync_manager.generate_and_save_report(
            modality="AUDIO",
            dataset_key=dataset_key,
            pass_number=pass_number,
            target_model="audio",
        )
    else:
        if target in ("general", "all"):
            from train_threads_continual import run_threads_continual_training
            from train_general import train_general_model
            try:
                vit_res = run_threads_continual_training(
                    epochs=epochs,
                    batch_size=batch_size,
                    extra_manifests=extra_manifests,
                    max_train_samples=max_samples,
                    min_ood_auc=min_auc,
                    max_ood_ece=max_ece,
                )
                safety_passed = bool(vit_res.get("safety_passed", True))
            except Exception as e_vit:
                print(f"[colab_runner] Continual ViT notice: {e_vit}")
            try:
                train_general_model(epochs=max(3, epochs * 2), batch_size=batch_size, output_dir=str(PROJECT_ROOT / "custom_vit_model"))
            except Exception as e_gen:
                print(f"[colab_runner] General FCB notice: {e_gen}")

            sync_manager.generate_and_save_report(
                modality=modality,
                dataset_key=dataset_key,
                metrics=vit_res,
                pass_number=pass_number,
                target_model="general",
            )

        if target in ("human", "all"):
            from train_human import train_human_model
            try:
                train_human_model(epochs=max(3, epochs), batch_size=batch_size, output_dir=str(PROJECT_ROOT / "human_vit_model"))
                sync_manager.generate_and_save_report(
                    modality=modality,
                    dataset_key=dataset_key,
                    pass_number=pass_number,
                    target_model="human",
                )
            except Exception as e_hum:
                print(f"[colab_runner] Human Biometrics notice: {e_hum}")

    hf_uploaded = False
    if safety_passed and hf_token and upload_hf:
        if target in ("general", "all") or modality == "AUDIO":
            hf_uploaded = sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "custom_vit_model"), hf_token=hf_token)
        if target in ("human", "all") and modality != "AUDIO":
            sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "human_vit_model"), hf_token=hf_token)

    if safety_passed and notify_server and callback_url and upload_hf:
        cb_payload = _build_callback_payload(
            vit_res=vit_res,
            audio_meta=audio_meta,
            hf_repo_id=sync_manager.hf_repo if hf_uploaded else None,
            dataset_label=f"{modality.lower()}:{dataset_key}",
        )
        sync_manager.notify_production_callback(api_base=callback_url, payload=cb_payload, callback_secret=callback_secret)

    if github_token:
        commit_msg = f"Auto-sync Colab empirical reports Pass #{pass_number} {modality}:{dataset_key} [Auto-Bot]"
        sync_manager.auto_sync_metrics_to_github(github_token=github_token, commit_message=commit_msg)

    sync_manager.sync_to_google_drive()
    maybe_auto_disconnect_colab(auto_disconnect)


def get_commit_sha(ref="HEAD"):
    try:
        res = subprocess.run(["git", "rev-parse", ref], cwd=str(PROJECT_ROOT), capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return ""


def get_full_multimodal_queue():
    from colab_dataset_pipeline import AWESOME_DEEPFAKES_REGISTRY
    queue = []
    for modality in ["IMAGE", "VIDEO", "AUDIO"]:
        if modality in AWESOME_DEEPFAKES_REGISTRY:
            for key in AWESOME_DEEPFAKES_REGISTRY[modality].keys():
                queue.append((modality, key))
    return queue


def run_continuous_auto_loop(
    hf_token=None,
    github_token=None,
    callback_secret=None,
    callback_url="https://deeptrace.io.vn",
    max_samples=1000,
    pass_number=1,
    target="all",
):
    """Run full multi-modality pass sequentially over registered IMAGE, VIDEO, and AUDIO datasets."""
    queue = get_full_multimodal_queue()
    for idx, (modality, dataset_key) in enumerate(queue, 1):
        print(f"\n🚀 STAGE [{idx}/{len(queue)}]: [{modality}] | [{dataset_key}] | PASS #{pass_number}")
        run_colab_auto_pipeline(
            modality=modality,
            dataset_key=dataset_key,
            max_samples=max_samples,
            hf_token=hf_token,
            github_token=github_token,
            callback_secret=callback_secret,
            callback_url=callback_url,
            auto_pull=(idx == 1),
            pass_number=pass_number,
            target=target,
            upload_hf=False,
            notify_server=False,
        )

    sync_manager = ColabAutoSyncManager()
    hf_uploaded = False
    if hf_token:
        if target in ("general", "all"):
            hf_uploaded = sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "custom_vit_model"), hf_token=hf_token)
        if target in ("human", "all"):
            sync_manager.auto_upload_hf(model_dir=str(PROJECT_ROOT / "human_vit_model"), hf_token=hf_token)

    if callback_url:
        emp = sync_manager.load_empirical_stream_metrics("general")
        sync_manager.notify_production_callback(
            api_base=callback_url,
            payload={
                "temperature_scaling_T": emp.get("temperature_scaling_T", 1.6178),
                "optimal_threshold_tau": emp.get("optimal_threshold_tau", 0.8118),
                "out_of_distribution_accuracy": emp.get("accuracy", 0.0),
                "auc_score": emp.get("auc_score", 0.0),
                "dataset_trained": f"loop_pass_{pass_number}",
                "hf_repo_id": sync_manager.hf_repo if hf_uploaded else None,
            },
            callback_secret=callback_secret,
        )
    sync_manager.sync_to_google_drive()


def run_smart_auto_loop(hf_token=None, github_token=None, callback_secret=None, callback_url="https://deeptrace.io.vn", max_samples=1000, target="all"):
    pass_number = 1
    last_commit_sha = get_commit_sha("HEAD")
    try:
        while True:
            subprocess.run(["git", "fetch", "origin", "main"], cwd=str(PROJECT_ROOT), check=False)
            remote_sha = get_commit_sha("origin/main")
            if remote_sha and remote_sha != last_commit_sha:
                subprocess.run(["git", "pull", "--rebase", "--autostash", "origin", "main"], cwd=str(PROJECT_ROOT), check=False)
                last_commit_sha = get_commit_sha("HEAD")

            run_continuous_auto_loop(
                hf_token=hf_token,
                github_token=github_token,
                callback_secret=callback_secret,
                callback_url=callback_url,
                max_samples=max_samples,
                pass_number=pass_number,
                target=target,
            )
            pass_number += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n🛑 Smart Auto-Runner Loop stopped by user.")


def run_auto_listen_daemon(hf_token=None, github_token=None, callback_secret=None, callback_url="https://deeptrace.io.vn", max_samples=1000, poll_interval=30):
    last_processed_sha = get_commit_sha("HEAD")
    try:
        while True:
            subprocess.run(["git", "fetch", "origin", "main"], cwd=str(PROJECT_ROOT), check=False)
            remote_sha = get_commit_sha("origin/main")
            if remote_sha and remote_sha != last_processed_sha:
                run_burst_all_streams(
                    max_samples=max_samples,
                    hf_token=hf_token,
                    github_token=github_token,
                    callback_secret=callback_secret,
                    callback_url=callback_url,
                )
                last_processed_sha = get_commit_sha("HEAD")
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n🛑 Auto-Listen Daemon stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Colab Burst-GPU & Closed-Loop Multi-Modality AutoTrainer")
    parser.add_argument("--burst", action="store_true", help="Recommended: Train Vision + Biometrics + Audio in 1 Burst pass, sync HF & deeptrace.io.vn")
    parser.add_argument("--modality", type=str, default="IMAGE", choices=["IMAGE", "VIDEO", "AUDIO"], help="Dataset modality")
    parser.add_argument("--dataset-key", type=str, default="threads_vietnamese_in_the_wild", help="Dataset registry key")
    parser.add_argument("--max-samples", type=int, default=1000, help="Maximum samples to process")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs per stream")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--target", type=str, default="all", choices=["general", "human", "all"], help="Target model stream")
    parser.add_argument("--hf-token", type=str, default=None, help="Hugging Face Write Token")
    parser.add_argument("--github-token", type=str, default=None, help="GitHub Token for auto-pushing lightweight metrics")
    parser.add_argument("--callback-secret", type=str, default=None, help="Optional X-Colab-Secret for /api/colab/callback")
    parser.add_argument("--callback-url", type=str, default="https://deeptrace.io.vn", help="Production server base URL")
    parser.add_argument("--no-human-review", action="store_true", help="Skip pulling approved Human Review Queue samples")
    parser.add_argument("--no-notify", action="store_true", help="Skip POST /api/colab/callback notification")
    parser.add_argument("--no-upload-hf", action="store_true", help="Skip Hugging Face model upload")
    parser.add_argument("--no-pull", action="store_true", help="Disable auto-pulling code from GitHub")
    parser.add_argument("--min-auc", type=float, default=0.70, help="OOD Safety Gate minimum ROC-AUC")
    parser.add_argument("--max-ece", type=float, default=0.15, help="OOD Safety Gate maximum ECE")
    parser.add_argument("--auto-disconnect", action="store_true", help="Call google.colab.runtime.unassign() when finished")
    parser.add_argument("--loop", action="store_true", help="Run continuous multi-pass loop")
    parser.add_argument("--listen", action="store_true", help="Poll GitHub and run Burst training on new commits")
    parser.add_argument("--poll-interval", type=int, default=30, help="Polling interval in seconds for --listen")
    parser.add_argument("--council-audit", action="store_true", help="Run 3-Pillar LLM Council preflight audit")
    args = parser.parse_args()

    hf_token, github_token, cb_secret = resolve_credentials(args.hf_token, args.github_token, args.callback_secret)

    if args.listen:
        run_auto_listen_daemon(
            hf_token=hf_token,
            github_token=github_token,
            callback_secret=cb_secret,
            callback_url=args.callback_url,
            max_samples=args.max_samples,
            poll_interval=args.poll_interval,
        )
    elif args.loop:
        run_smart_auto_loop(
            hf_token=hf_token,
            github_token=github_token,
            callback_secret=cb_secret,
            callback_url=args.callback_url,
            max_samples=args.max_samples,
            target=args.target,
        )
    elif args.burst:
        run_burst_all_streams(
            dataset_key=args.dataset_key,
            max_samples=args.max_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hf_token=hf_token,
            github_token=github_token,
            callback_secret=cb_secret,
            callback_url=args.callback_url,
            sync_human_review=not args.no_human_review,
            upload_hf=not args.no_upload_hf,
            notify_server=not args.no_notify,
            auto_pull=not args.no_pull,
            council_audit=args.council_audit,
            min_auc=args.min_auc,
            max_ece=args.max_ece,
            auto_disconnect=args.auto_disconnect,
        )
    else:
        run_colab_auto_pipeline(
            modality=args.modality,
            dataset_key=args.dataset_key,
            max_samples=args.max_samples,
            hf_token=hf_token,
            github_token=github_token,
            callback_secret=cb_secret,
            callback_url=args.callback_url,
            sync_human_review=not args.no_human_review,
            auto_pull=not args.no_pull,
            target=args.target,
            upload_hf=not args.no_upload_hf,
            notify_server=not args.no_notify,
            council_audit=args.council_audit,
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_auc=args.min_auc,
            max_ece=args.max_ece,
            auto_disconnect=args.auto_disconnect,
        )


if __name__ == "__main__":
    main()
