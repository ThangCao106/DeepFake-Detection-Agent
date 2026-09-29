"""
Colab Free Auto-Scheduler & Heartbeat Runner (Option C).

Designed specifically for Google Colab Free accounts to synchronize training with the
GitHub Actions 4h Pentest cycle:
- GitHub Actions Pentest runs at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC (~15 mins).
- Training triggers 30 minutes after Pentest completion: at 00:45, 04:45, 08:45, 12:45, 16:45, 20:45 UTC.
- Includes active stdout heartbeats to prevent Colab Free idle timeout disconnections.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Auto-resolve repository root and guarantee execution in root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    os.chdir(str(PROJECT_ROOT))
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
except Exception:
    pass

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Target schedule hours in UTC (runs at minute 45 of each 4-hour cycle)
SCHEDULE_HOURS_UTC = [0, 4, 8, 12, 16, 20]
TARGET_MINUTE = 45


def get_next_run_time(now_utc: datetime) -> datetime:
    """Calculate the next upcoming run time at HH:45 UTC for the 4h schedule."""
    today = now_utc.date()
    candidates = []

    # Check candidates for today and tomorrow
    for day_offset in [0, 1]:
        d = today + timedelta(days=day_offset)
        for h in SCHEDULE_HOURS_UTC:
            candidate = datetime(d.year, d.month, d.day, h, TARGET_MINUTE, 0, tzinfo=timezone.utc)
            if candidate > now_utc:
                candidates.append(candidate)

    candidates.sort()
    return candidates[0]


def format_countdown(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}h {m:02d}m {s:02d}s"


def run_training_pipeline(args_list: list):
    """Execute git pull and the dual-stream Colab training script."""
    print("\n" + "=" * 75)
    print("🚀 [Colab Auto-Scheduler] KÍCH HOẠT CHU KỲ HUẤN LUYỆN TỰ ĐỘNG (TRIGGERED)")
    print(f"⏰ Thời gian bắt đầu (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)

    # 1. Pull latest code from GitHub
    print("\n📦 [Bước 1/3] Cập nhật mã nguồn mới nhất từ GitHub (git pull origin main)...")
    try:
        res = subprocess.run(["git", "pull", "origin", "main"], capture_output=True, text=True, timeout=60)
        print(res.stdout.strip())
        if res.returncode != 0 and res.stderr:
            print(f"⚠️ Cảnh báo git pull: {res.stderr.strip()}")
    except Exception as e_git:
        print(f"⚠️ Bỏ qua git pull: {e_git}")

    # 2. Execute colab_sync_to_gdrive.py
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "colab_sync_to_gdrive.py")
    ] + args_list

    print(f"\n🧠 [Bước 2/3] Thực thi tiến trình huấn luyện Dual-Stream & Sync GDrive/HuggingFace:")
    print(f"   Lệnh: {' '.join(cmd)}")
    start_time = time.time()

    try:
        # Run and stream stdout directly
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
        proc.wait()
        elapsed = time.time() - start_time
        if proc.returncode == 0:
            print(f"\n✅ [Bước 3/3] Huấn luyện & Đồng bộ hoàn tất thành công! (Thời gian chạy: {elapsed/60:.1f} phút)")
        else:
            print(f"\n❌ [Bước 3/3] Quá trình kết thúc với mã lỗi {proc.returncode} sau {elapsed/60:.1f} phút.")
    except Exception as e_train:
        print(f"\n❌ Lỗi thực thi huấn luyện: {e_train}")

    print("=" * 75 + "\n")


def start_scheduler_loop(training_args: list, run_now_first: bool = False):
    """Main scheduler loop with anti-idle heartbeats."""
    print("=" * 75)
    print("🤖 GOOGLE COLAB FREE AUTO-SCHEDULER (OPTION C)")
    print(" đồng bộ với GitHub Actions Threads Pentest (Lệch pha +30 phút)")
    print(f"📅 Các mốc chạy cố định (UTC): {', '.join([f'{h:02d}:45' for h in SCHEDULE_HOURS_UTC])}")
    vietnam_hours = [(h + 7) % 24 for h in SCHEDULE_HOURS_UTC]
    print(f"🇻🇳 Các mốc chạy giờ Việt Nam: {', '.join([f'{h:02d}:45' for h in sorted(vietnam_hours)])}")
    print("=" * 75)

    if run_now_first:
        print("\n⚡ Cờ --run-now được bật: Chạy ngay 1 lượt đầu tiên trước khi vào chế độ chờ...")
        run_training_pipeline(training_args)

    heartbeat_counter = 0

    while True:
        now_utc = datetime.now(timezone.utc)
        next_run = get_next_run_time(now_utc)
        time_to_wait = (next_run - now_utc).total_seconds()

        print(f"\n⏳ Lượt chạy tiếp theo lúc: {next_run.strftime('%Y-%m-%d %H:%M:%S UTC')} (Giờ VN: {(next_run + timedelta(hours=7)).strftime('%H:%M:%S')})")
        print(f"⏱️ Đếm ngược: {format_countdown(time_to_wait)} còn lại...")

        # Sleep in small 30-second steps to emit heartbeats and keep Colab runtime alive
        while time_to_wait > 0:
            sleep_step = min(30.0, time_to_wait)
            time.sleep(sleep_step)
            now_utc = datetime.now(timezone.utc)
            time_to_wait = (next_run - now_utc).total_seconds()

            heartbeat_counter += 1
            if heartbeat_counter % 10 == 0:  # Every 5 minutes
                remaining_str = format_countdown(max(0, time_to_wait))
                print(f"💓 [Colab Heartbeat Active] Đang chờ đợt Pentest kế tiếp... Còn {remaining_str} (UTC: {now_utc.strftime('%H:%M:%S')})")
                sys.stdout.flush()

        # Trigger training
        run_training_pipeline(training_args)
        time.sleep(10)  # Brief pause before next cycle calculation


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Colab Free Auto-Scheduler (Option C)")
    parser.add_argument("--run-now", action="store_true", help="Chạy ngay 1 lượt đầu tiên trước khi vào lịch chờ")
    parser.add_argument("--images", type=int, default=1000, help="Số lượng ảnh crawl")
    parser.add_argument("--videos", type=int, default=500, help="Số lượng video crawl")
    parser.add_argument("--audios", type=int, default=500, help="Số lượng audio crawl")
    parser.add_argument("--samples", type=int, default=25, help="Số mẫu trực quan kiểm định")
    parser.add_argument("--epochs", type=int, default=3, help="Số epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước batch")
    parser.add_argument("--lr", type=float, default=0.0001, help="Tốc độ học")
    parser.add_argument("--train-both-models", action="store_true", default=True, help="Huấn luyện cả Model 1 và Model 2")

    parser.add_argument("--job-batch", "--job-time", dest="job_batch", type=str, default="auto", help="Option C 4h batch subfolder (default: 'auto')")

    args = parser.parse_args()

    training_flags = [
        "--images", str(args.images),
        "--videos", str(args.videos),
        "--audios", str(args.audios),
        "--samples", str(args.samples),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--job-batch", str(args.job_batch)
    ]
    if args.train_both_models:
        training_flags.append("--train-both-models")

    start_scheduler_loop(training_flags, run_now_first=args.run_now)
