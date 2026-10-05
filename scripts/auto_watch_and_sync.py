"""
Auto Watch & Sync Daemon: Tự động phát hiện thay đổi mã nguồn và đẩy lên GitHub & Google Drive
Sử dụng: py -3 scripts/auto_watch_and_sync.py
"""
import os
import sys
import time
import subprocess
from pathlib import Path

# Đảm bảo UTF-8 console output trên Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IGNORE_PATTERNS = {
    ".git", "__pycache__", ".venv", "venv", ".pytest_cache", 
    "benchmark_data", "colab_datasets", "uploads", ".cache"
}
WATCH_EXTENSIONS = {".py", ".ipynb", ".md", ".sh", ".json", ".txt", ".yaml", ".yml"}

def get_snapshot():
    """Lấy danh sách thời gian sửa đổi (mtime) của các tệp mã nguồn."""
    snapshot = {}
    for root, dirs, files in os.walk(PROJECT_ROOT):
        dirs[:] = [d for d in dirs if d not in IGNORE_PATTERNS]
        for f in files:
            ext = Path(f).suffix.lower()
            if ext in WATCH_EXTENSIONS and not f.startswith("~"):
                fp = Path(root) / f
                try:
                    snapshot[str(fp)] = fp.stat().st_mtime
                except OSError:
                    pass
    return snapshot

def trigger_sync():
    """Thực hiện đẩy tự động lên GitHub."""
    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n⚡ [{now_str}] Phát hiện thay đổi mã nguồn! Đang tự động đồng bộ...")
    
    # 1. Git add & commit
    subprocess.run("git add -A", shell=True, cwd=PROJECT_ROOT, capture_output=True)
    res = subprocess.run(
        f'git commit -m "auto-sync: update codebase at {now_str}"',
        shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    
    if "nothing to commit" in res.stdout or "nothing to commit" in res.stderr:
        print("ℹ️ Không có thay đổi mới cần commit.")
        return
        
    print(f"📦 [Git Commit] {res.stdout.strip().splitlines()[0] if res.stdout.strip() else 'Committed'}")
    
    # 2. Git push
    print("🚀 Đang tự động đẩy lên GitHub (origin/main)...")
    push_res = subprocess.run("git push origin main", shell=True, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if push_res.returncode == 0:
        print(f"✅ [GitHub Auto-Sync] Đã đẩy commit mới nhất lên GitHub thành công!")
        print("💡 Trên Google Colab, hệ thống sẽ tự kéo code và tự động cập nhật vào Google Drive folder.")
    else:
        # Thử rebase nếu có xung đột
        subprocess.run("git pull --rebase origin main", shell=True, cwd=PROJECT_ROOT, capture_output=True)
        push_retry = subprocess.run("git push origin main", shell=True, cwd=PROJECT_ROOT, capture_output=True)
        if push_retry.returncode == 0:
            print("✅ [GitHub Auto-Sync] Đã rebase và đẩy thành công!")
        else:
            print(f"⚠️ [Git Push Error]: {push_res.stderr.strip()}")

def main():
    print("=" * 65)
    print("👀 KHỞI ĐỘNG AUTO-WATCH & SYNC DAEMON (DEEPFAKE-DETECTION-AGENT)")
    print(f"📂 Theo dõi thư mục: {PROJECT_ROOT}")
    print("⚡ Cơ chế: Bất cứ khi nào bạn bấm Ctrl + S lưu file, hệ thống sẽ tự động")
    print("   Commit ➔ Push GitHub ➔ Tự kích hoạt Colab cập nhật Google Drive.")
    print("🛑 Nhấn Ctrl + C để dừng bất kỳ lúc nào.")
    print("=" * 65 + "\n")
    
    last_snapshot = get_snapshot()
    
    try:
        while True:
            time.sleep(3) # Kiểm tra mỗi 3 giây
            current_snapshot = get_snapshot()
            
            # Kiểm tra xem có file nào được sửa hoặc tạo mới không
            changed = False
            for fp, mtime in current_snapshot.items():
                if fp not in last_snapshot or mtime > last_snapshot[fp]:
                    changed = True
                    break
                    
            if not changed and len(current_snapshot) != len(last_snapshot):
                changed = True
                
            if changed:
                time.sleep(1) # Chờ 1 giây để IDE ghi xong file
                trigger_sync()
                last_snapshot = get_snapshot()
                print("\n👀 Đang tiếp tục theo dõi các thay đổi tiếp theo...")
                
    except KeyboardInterrupt:
        print("\n👋 Đã dừng Auto-Watch Daemon.")

if __name__ == "__main__":
    main()
