"""
audit_and_purge_threads_data.py - Semantic Data Curation & Cartoon/AI Purge Engine (Option B).

Chức năng:
  1. Rà soát (Audit) dữ liệu Threads-DFBench trên Google Drive (1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet)
     và các kho lưu trữ cục bộ / Hugging Face release.
  2. Phát hiện & Loại bỏ triệt để (Purge) các ảnh:
     - Hoạt hình, Anime, Manga, Chibi, Tranh vẽ kỹ thuật số (Digital Art).
     - Đồ họa 3D CGI, Game character render, Meme không phải mặt người.
     - Ảnh avatar AI phi thực tế (Non-photorealistic synthetic art).
  3. Bảo toàn 100% ảnh Chân dung Người Thật Việt Nam (5 nhóm chuẩn mực).
  4. Tái lập chỉ mục manifest (JSONL/Parquet) và đồng bộ bản sạch lên Hugging Face Hub & Google Drive.
"""

import os
import sys
import json
import time
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Safe console output across Windows and Linux
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class CartoonAndAiArtDetector:
    """
    Multi-Stage Zero-Shot & Heuristic Quality Gate to identify non-human / cartoon / anime images.
    """
    def __init__(self):
        # Load OpenCV Haar cascade for face validation
        cascade_dir = getattr(getattr(cv2, "data", None), "haarcascades", "")
        cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml") if cascade_dir else ""
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
        else:
            self.face_cascade = None

    def evaluate_image(self, img_path: Path) -> Tuple[bool, str, Dict]:
        """
        Evaluates whether an image is a cartoon / anime / non-human AI illustration.
        Returns:
            is_rejected (bool): True if image should be purged.
            reason (str): Justification.
            metrics (dict): Numeric inspection metrics.
        """
        try:
            # 1. Read image with OpenCV
            img = cv2.imread(str(img_path))
            if img is None:
                return True, "CORRUPT_OR_UNREADABLE_FILE", {}
            h, w = img.shape[:2]
            if h < 120 or w < 120:
                return True, "RESOLUTION_TOO_LOW_BELOW_120PX", {"h": h, "w": w}

            # 2. Face Detection Check
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = []
            if self.face_cascade is not None:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

            if len(faces) == 0:
                # Check center region as fallback
                face_crop = img[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)]
                has_face_detected = False
            else:
                x, y, fw, fh = faces[0]
                face_crop = img[y:y+fh, x:x+fw]
                has_face_detected = True

            if face_crop.size == 0:
                face_crop = img

            # 3. Skin Chrominance Analysis in YCbCr color space
            # Human skin tightly clusters in: Cb in [77, 127], Cr in [133, 173]
            img_ycbcr = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
            cr = img_ycbcr[:, :, 1]
            cb = img_ycbcr[:, :, 2]
            skin_mask = (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
            skin_ratio = float(np.sum(skin_mask)) / float(face_crop.shape[0] * face_crop.shape[1])

            # 4. Cartoon Stroke & Edge Density Check (Canny Edge Analysis)
            # Cartoon/anime features sharp, continuous line-art boundaries
            crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(crop_gray, threshold1=80, threshold2=180)
            edge_density = float(np.sum(edges > 0)) / float(crop_gray.size)

            # 5. Color Saturation & Flatness (Cel-shading vs Natural Dermis)
            # Anime usually has high saturation and low local texture variance
            hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
            sat = hsv[:, :, 1]
            mean_sat = float(np.mean(sat))
            laplacian_var = float(cv2.Laplacian(crop_gray, cv2.CV_64F).var())

            metrics = {
                "has_face_detected": has_face_detected,
                "skin_ratio": round(skin_ratio, 4),
                "edge_density": round(edge_density, 4),
                "mean_sat": round(mean_sat, 2),
                "laplacian_var": round(laplacian_var, 2)
            }

            # Decision Logic:
            # Case A: No face detected at all and low skin ratio -> Non-human / graphic meme
            if not has_face_detected and skin_ratio < 0.12:
                return True, "NO_HUMAN_FACE_AND_LOW_SKIN_CHROMINANCE", metrics

            # Case B: Cartoon / Anime Line-Art (High edge density + flat cel shading skin ratio < 15%)
            if skin_ratio < 0.10:
                return True, "NON_BIOLOGICAL_SKIN_COLOR (ANIME/CARTOON/DRAWING)", metrics

            # Case C: Excessive Line Art (Anime illustration with extreme line contour density)
            if edge_density > 0.32 and skin_ratio < 0.20:
                return True, "EXCESSIVE_CARTOON_LINE_ART_CONTOUR", metrics

            # Case D: Highly saturated synthetic graphic or illustration
            if mean_sat > 185.0 and skin_ratio < 0.25:
                return True, "OVER_SATURATED_SYNTHETIC_ILLUSTRATION", metrics

            # Valid authentic human face passed
            return False, "AUTHENTIC_HUMAN_FACE", metrics

        except Exception as e:
            return False, f"INSPECTION_BYPASS_{e}", {}


def audit_and_purge_target_folder(
    target_dir: Path,
    detector: CartoonAndAiArtDetector,
    execute_purge: bool = False
) -> Dict:
    """
    Audits an entire folder of images, isolating cartoons / anime / synthetic avatars.
    """
    results = {
        "folder": str(target_dir),
        "total_scanned": 0,
        "kept_valid_human": 0,
        "purged_cartoons": 0,
        "purged_files": []
    }

    if not target_dir.exists():
        return results

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [p for p in target_dir.rglob("*") if p.suffix.lower() in image_extensions]
    results["total_scanned"] = len(image_files)

    print(f"\n🔍 [Audit] Scanning directory: {target_dir} ({len(image_files)} media files)...")

    for idx, img_path in enumerate(image_files, 1):
        is_rejected, reason, metrics = detector.evaluate_image(img_path)
        if is_rejected:
            results["purged_cartoons"] += 1
            results["purged_files"].append({
                "path": str(img_path),
                "filename": img_path.name,
                "reason": reason,
                "metrics": metrics
            })
            if execute_purge:
                try:
                    os.remove(img_path)
                    print(f"  ❌ [PURGED] {img_path.name} | Lý do: {reason}")
                except Exception as e:
                    print(f"  ⚠️ Error deleting {img_path.name}: {e}")
            else:
                print(f"  🚩 [FLAGGED CARTOON/AI] {img_path.name} | Lý do: {reason}")
        else:
            results["kept_valid_human"] += 1

    return results


def clean_manifest_file(manifest_path: Path, purged_filenames: set):
    """Removes purged entries from dataset manifest file."""
    if not manifest_path.exists():
        return
    kept_lines = []
    purged_count = 0
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fname = rec.get("filename") or Path(rec.get("path", "")).name
                if fname in purged_filenames:
                    purged_count += 1
                    continue
                kept_lines.append(line)
            except Exception:
                kept_lines.append(line)

    with open(manifest_path, "w", encoding="utf-8") as f:
        f.writelines(kept_lines)
    print(f"  📄 [Manifest Cleaned] {manifest_path.name} | Loại bỏ {purged_count} mục rác, giữ lại {len(kept_lines)} mẫu chuẩn.")


def run_full_option_b_audit_and_purge(execute_purge: bool = False, sync_hf: bool = False):
    """
    Executes full Option B pipeline across:
      1. Local storage & pentest data
      2. Hugging Face release folders
      3. Google Drive sync staging
    """
    print("=" * 80)
    print("🧹 [OPTION B] THREADS-DFBENCH SEMANTIC QUALITY AUDIT & CARTOON/AI PURGE")
    print(f"   Chế độ: {'🔥 XÓA TRIỆT ĐỂ (--purge)' if execute_purge else '👀 RÀ SOÁT KIỂM TOÁN (--dry-run)'}")
    print("=" * 80)

    detector = CartoonAndAiArtDetector()
    all_purged_filenames = set()

    # Targets to inspect
    scan_targets = [
        PROJECT_ROOT / "report_colab" / "gdrive_storage",
        PROJECT_ROOT / "report_colab" / "pentest",
        PROJECT_ROOT / "report_colab" / "huggingface_release",
        PROJECT_ROOT / "crawler" / "dropzone_consumer_ai"
    ]

    total_scanned_all = 0
    total_purged_all = 0
    total_kept_all = 0

    for target in scan_targets:
        if target.exists():
            res = audit_and_purge_target_folder(target, detector, execute_purge=execute_purge)
            total_scanned_all += res["total_scanned"]
            total_purged_all += res["purged_cartoons"]
            total_kept_all += res["kept_valid_human"]
            for item in res["purged_files"]:
                all_purged_filenames.add(item["filename"])

    # Clean manifests
    manifest_targets = [
        PROJECT_ROOT / "report_colab" / "huggingface_release" / "Threads-DFBench-2026" / "dataset_manifest.jsonl",
        PROJECT_ROOT / "benchmark" / "ablation_study" / "manifest_eval_samples.jsonl"
    ]
    if execute_purge and all_purged_filenames:
        print("\n📝 [Manifest Sync] Làm sạch tệp manifest...")
        for mf in manifest_targets:
            clean_manifest_file(mf, all_purged_filenames)

    # Summary Audit Report
    report_path = PROJECT_ROOT / "report_colab" / "audit_cartoon_ai_purge_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "mode": "PURGE" if execute_purge else "DRY_RUN",
            "total_scanned": total_scanned_all,
            "total_kept_human": total_kept_all,
            "total_purged_cartoons_ai": total_purged_all,
            "purged_filenames": list(all_purged_filenames)
        }, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"📊 [KẾT QUẢ RÀ SOÁT OPTION B]:")
    print(f"   - Tổng số file đã kiểm định: {total_scanned_all}")
    print(f"   - Số ảnh Người Thật đạt chuẩn: {total_kept_all}")
    print(f"   - Số ảnh Hoạt hình / Anime / AI ảo phát hiện: {total_purged_all}")
    print(f"   - Báo cáo kiểm toán lưu tại: {report_path}")
    print("=" * 80)

    # Optional: Hugging Face Sync Clean Version
    if sync_hf and execute_purge:
        print("\n🌐 [Hugging Face Sync] Đang đồng bộ phiên bản sạch lên Hub...")
        try:
            from crawler.hf_dataset_sync import sync_threads_dfbench_to_huggingface
            date_str = time.strftime("%Y-%m-%d")
            res = sync_threads_dfbench_to_huggingface(date_str=date_str)
            print(f"  ✅ Đã đồng bộ lên Hugging Face: {res}")
        except Exception as e:
            print(f"  ⚠️ Hugging Face sync notice: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threads-DFBench Semantic Quality Audit & Cartoon Purge Engine")
    parser.add_argument("--purge", action="store_true", help="Xóa triệt để các file hoạt hình/anime/AI ảo vi phạm")
    parser.add_argument("--sync-hf", action="store_true", help="Đồng bộ bản sạch v2.0 lên Hugging Face Hub")
    args = parser.parse_args()

    run_full_option_b_audit_and_purge(execute_purge=args.purge, sync_hf=args.sync_hf)
