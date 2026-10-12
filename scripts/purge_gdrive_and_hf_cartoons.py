"""
purge_gdrive_and_hf_cartoons.py - Google Drive & Hugging Face Dataset Purge Engine (Option B).

Quét và dọn sạch triệt để các tệp ảnh hoạt hình, tranh vẽ anime, manga, 3D CGI, meme
trên Google Drive (Folder ID: 1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet) và Hugging Face Hub (Threads-DFBench-Vietnamese).

Hỗ trợ 2 chế độ:
  1. Chạy trên Google Colab với Drive Mount: /content/drive/MyDrive/...
  2. Chạy trên Local / Server với Google Drive API (hoặc local sync staging)
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.audit_and_purge_threads_data import CartoonAndAiArtDetector

# Target Google Drive Folder ID
GDRIVE_TARGET_FOLDER_ID = "1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"


def scan_and_purge_colab_drive_mount(execute_purge: bool = False) -> Dict:
    """
    Scans and purges cartoons from Google Drive when running inside Google Colab.
    """
    colab_mydrive = Path("/content/drive/MyDrive")
    candidate_folders = [
        colab_mydrive / GDRIVE_TARGET_FOLDER_ID,
        colab_mydrive / "Thread_Data",
        colab_mydrive / "DeepFake_Pentest_Data",
        colab_mydrive / "1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"
    ]
    
    found_folder = None
    for cand in candidate_folders:
        if cand.exists() and cand.is_dir():
            found_folder = cand
            break

    if not found_folder:
        print(f"[Colab Drive] Direct mount folder not found in /content/drive/MyDrive/. Falling back to local storage.")
        return {"status": "NOT_MOUNTED"}

    print(f"\n📂 [Colab Drive Mount] Đang rà soát thư mục Google Drive: {found_folder}")
    detector = CartoonAndAiArtDetector()
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [p for p in found_folder.rglob("*") if p.suffix.lower() in image_exts]

    print(f"  🔍 Tìm thấy {len(image_files)} tệp media trên Google Drive. Bắt đầu kiểm định ngữ nghĩa...")
    purged_count = 0
    kept_count = 0
    purged_files = []

    for img_p in image_files:
        is_rejected, reason, metrics = detector.evaluate_image(img_p)
        if is_rejected:
            purged_count += 1
            purged_files.append({"path": str(img_p), "reason": reason})
            if execute_purge:
                try:
                    os.remove(img_p)
                    print(f"  ❌ [XÓA TRÊN DRIVE] {img_p.name} | Lý do: {reason}")
                except Exception as e:
                    print(f"  ⚠️ Lỗi khi xóa {img_p.name}: {e}")
            else:
                print(f"  🚩 [PHÁT HIỆN HOẠT HÌNH/AI TRÊN DRIVE] {img_p.name} | Lý do: {reason}")
        else:
            kept_count += 1

    return {
        "status": "COMPLETED",
        "folder": str(found_folder),
        "total": len(image_files),
        "kept": kept_count,
        "purged": purged_count,
        "purged_files": purged_files
    }


def purge_huggingface_repo_cartoons(execute_purge: bool = False) -> Dict:
    """
    Rà soát và loại bỏ ảnh hoạt hình/AI trên Hugging Face Dataset release folder,
    sau đó tự động push bản sạch v2.0 lên Hugging Face Hub.
    """
    hf_dir = PROJECT_ROOT / "report_colab" / "huggingface_release" / "Threads-DFBench-2026"
    if not hf_dir.exists():
        print("[Hugging Face Audit] Thư mục release cục bộ không tồn tại. Bỏ qua.")
        return {"status": "DIR_NOT_FOUND"}

    detector = CartoonAndAiArtDetector()
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_files = [p for p in hf_dir.rglob("*") if p.suffix.lower() in image_exts]

    print(f"\n🤗 [Hugging Face Hub Release] Đang rà soát: {hf_dir} ({len(image_files)} tệp)...")
    purged_count = 0
    kept_count = 0
    purged_names = set()

    for img_p in image_files:
        is_rejected, reason, metrics = detector.evaluate_image(img_p)
        if is_rejected:
            purged_count += 1
            purged_names.add(img_p.name)
            if execute_purge:
                try:
                    os.remove(img_p)
                    print(f"  ❌ [XÓA TRÊN HF RELEASE] {img_p.name} | {reason}")
                except Exception as e:
                    print(f"  ⚠️ Lỗi xóa: {e}")
            else:
                print(f"  🚩 [FLAGGED TRÊN HF] {img_p.name} | {reason}")
        else:
            kept_count += 1

    # Clean JSONL manifest
    manifest_p = hf_dir / "dataset_manifest.jsonl"
    if manifest_p.exists() and execute_purge and purged_names:
        kept_lines = []
        with open(manifest_p, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("filename") not in purged_names:
                        kept_lines.append(line)
                except Exception:
                    kept_lines.append(line)
        with open(manifest_p, "w", encoding="utf-8") as f:
            f.writelines(kept_lines)
        print(f"  📄 Đã cập nhật lại manifest Hugging Face ({len(kept_lines)} mẫu sạch còn lại).")

    return {
        "status": "COMPLETED",
        "total": len(image_files),
        "kept": kept_count,
        "purged": purged_count
    }


def main():
    parser = argparse.ArgumentParser(description="Purge Cartoons and AI Art from GDrive and Hugging Face")
    parser.add_argument("--purge", action="store_true", help="Thực thi lệnh xóa triệt để (mặc định chỉ rà soát xem trước)")
    parser.add_argument("--sync-hf", action="store_true", help="Đẩy cập nhật phiên bản sạch lên Hugging Face Hub")
    args = parser.parse_args()

    print("=" * 80)
    print("🧹 CHƯƠNG TRÌNH THANH LỌC DỮ LIỆU GDRIVE & HUGGING FACE (OPTION B)")
    print(f"   Mục tiêu GDrive Folder ID: {GDRIVE_TARGET_FOLDER_ID}")
    print(f"   Chế độ: {'🔥 XÓA TRIỆT ĐỂ (--purge)' if args.purge else '👀 RÀ SOÁT XEM TRƯỚC (--dry-run)'}")
    print("=" * 80)

    # 1. Quét Colab / Drive mount
    colab_res = scan_and_purge_colab_drive_mount(execute_purge=args.purge)

    # 2. Quét Hugging Face Release
    hf_res = purge_huggingface_repo_cartoons(execute_purge=args.purge)

    # 3. Đồng bộ lên Hugging Face nếu có tham số
    if args.sync_hf and args.purge:
        print("\n🌐 Đang đẩy cập nhật bản sạch lên Hugging Face Hub...")
        try:
            from crawler.hf_dataset_sync import sync_threads_dfbench_to_huggingface
            date_str = time.strftime("%Y-%m-%d")
            sync_threads_dfbench_to_huggingface(date_str=date_str)
            print("  ✅ Đồng bộ Hugging Face Hub hoàn tất!")
        except Exception as e:
            print(f"  ⚠️ Lỗi đồng bộ Hugging Face: {e}")

    print("\n" + "=" * 80)
    print("🎉 Hoàn tất chu trình rà soát và thanh lọc dữ liệu!")
    print("=" * 80)


if __name__ == "__main__":
    main()
