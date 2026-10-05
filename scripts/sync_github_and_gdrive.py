"""
Script Đồng Bộ Kép: Tự động Push GitHub & Đóng gói Cập nhật Google Drive
Thư mục Drive mục tiêu: https://drive.google.com/drive/folders/1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet
"""
import os
import sys
import subprocess
import webbrowser
from pathlib import Path

# Đảm bảo UTF-8 console output trên Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GDRIVE_FOLDER_ID = "1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"
GDRIVE_URL = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

def run_cmd(cmd, cwd=PROJECT_ROOT):
    print(f"▶️ Đang chạy: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.stdout and res.stdout.strip():
        print(res.stdout.strip())
    if res.stderr and res.stderr.strip() and res.returncode != 0:
        print(f"⚠️ [Stderr]: {res.stderr.strip()}")
    return res.returncode == 0

def sync_github(commit_msg=None):
    print("\n" + "=" * 65)
    print("🐙 BƯỚC 1: ĐỒNG BỘ MÃ NGUỒN LÊN GITHUB")
    print("=" * 65)
    
    if not commit_msg:
        commit_msg = "feat: auto-sync codebase updates for Google Colab & Google Drive"
        
    run_cmd("git add -A")
    run_cmd(f'git commit -m "{commit_msg}"')
    
    print("🚀 Đang đẩy commit lên GitHub (origin/main)...")
    ok = run_cmd("git push origin main")
    if not ok:
        print("🔄 Đang thử rebase với nhánh remote...")
        run_cmd("git pull --rebase origin main")
        ok = run_cmd("git push origin main")
        
    if ok:
        print("✅ [GitHub] Đã đồng bộ commit mới nhất lên GitHub thành công!")
    else:
        print("⚠️ [GitHub] Không thể tự động push. Vui lòng kiểm tra lại token hoặc mạng.")
    return ok

def sync_gdrive():
    print("\n" + "=" * 65)
    print("📁 BƯỚC 2: ĐÓNG GÓI BẢN CẬP NHẬT CHO GOOGLE DRIVE")
    print("=" * 65)
    
    package_script = PROJECT_ROOT / "package_for_gdrive.py"
    if package_script.exists():
        py_exec = sys.executable or "python"
        run_cmd(f'"{py_exec}" "{package_script}"')
    
    zip_path = PROJECT_ROOT / "DeepFake-Detection-Agent.zip"
    if zip_path.exists():
        size_mb = round(zip_path.stat().st_size / (1024 * 1024), 2)
        print(f"\n📦 Tệp nén sẵn sàng: {zip_path.name} ({size_mb} MB)")
        print(f"🔗 Google Drive Folder Đích: {GDRIVE_URL}")
        print("\n👉 HƯỚNG DẪN ĐẨY LÊN GOOGLE DRIVE:")
        print(f"  1. Kéo thả tệp '{zip_path.name}' vào thư mục Google Drive tại liên kết trên.")
        print("  2. Máy ảo Colab sẽ tự động nhận diện và nạp mã nguồn tức thì!")

def main():
    commit_msg = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        commit_msg = sys.argv[1]
        
    print(f"🌟 KHỞI CHẠY ĐỒNG BỘ KÉP (GITHUB & GOOGLE DRIVE FOLDER {GDRIVE_FOLDER_ID})")
    sync_github(commit_msg)
    sync_gdrive()
    print("\n" + "=" * 65)
    print("🎉 HOÀN TẤT ĐỒNG BỘ KÉP!")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
