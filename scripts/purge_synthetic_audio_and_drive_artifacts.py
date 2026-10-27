"""
Option B Purge Agent: Synthetic Audio & Corrupt Drive Artifact Purger
Identifies and purges:
  1. Synthetic sine-wave audio files (0.5*sin(140) + 0.3*sin(520) + 0.15*sin(1600))
  2. Corrupted batch directories (e.g., batch_01_10h03 where 01_raw_originals has 0 real images)
  3. Legacy AI cartoon/avatar seed duplicates (human_face_sample_*, threads_20260921_*, threads_20260922_*)
Maintains clean 100% authentic Vietnamese Threads media on Google Drive and local storage.
"""
import os
import re
import sys
import json
import shutil
import wave
import numpy as np
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEARCH_CANDIDATE_PATHS = [
    # Google Colab Drive Mount Paths
    Path("/content/drive/MyDrive/1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6"),
    Path("/content/drive/MyDrive/Thread_Data"),
    Path("/content/drive/MyDrive/1W6R3QYzTI6GJAD2lByKoVaQnXfURcLO0"),
    Path("/content/drive/MyDrive/DeepFake_Pentest_Data"),
    Path("/content/drive/MyDrive/1MipTWpXt0o0xz2IyUc_pTvMzo0tcXRlr"),
    Path("/content/drive/MyDrive/1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"),
    Path("/content/drive/Shareddrives/Thread_Data"),
    Path("/content/drive/Shareddrives/1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6"),
    # Local Storage Paths
    PROJECT_ROOT / "report_colab" / "gdrive_storage",
    PROJECT_ROOT / "report_colab" / "pentest",
    PROJECT_ROOT / "report_colab" / "temp_media_samples",
]

LEGACY_FILENAME_PATTERNS = [
    re.compile(r"^human_face_sample_.*"),
    re.compile(r"^faceswap_sample_.*"),
    re.compile(r"^openweight_sample_.*"),
    re.compile(r"^threads_20260921_.*"),
    re.compile(r"^threads_20260922_.*"),
]


def is_synthetic_sine_audio(wav_path: Path) -> bool:
    """
    Detects if a WAV file was synthetically synthesized using pure sine wave formula:
      audio = 0.5 * np.sin(2 * np.pi * 140 * t) + 0.3 * np.sin(2 * np.pi * 520 * t) + 0.15 * np.sin(2 * np.pi * 1600 * t)
    A genuine human speech recording has continuous spectral dispersion and natural harmonics,
    whereas the synthetic test tone matches the mathematical formula exactly.
    """
    try:
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            if framerate != 16000 or nframes < 16000:
                # If not standard 16kHz or too short, test naming
                return False
            frames = wf.readframes(min(32000, nframes))
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

        num_check = len(data)
        t = np.linspace(0, num_check / framerate, num_check, endpoint=False)
        expected_synth = (
            0.5 * np.sin(2 * np.pi * 140 * t)
            + 0.3 * np.sin(2 * np.pi * 520 * t)
            + 0.15 * np.sin(2 * np.pi * 1600 * t)
        )
        # Calculate mean absolute error between file samples and synthetic formula
        mae = float(np.mean(np.abs(data - expected_synth)))
        if mae < 0.05:
            return True
        return False
    except Exception:
        return False


def purge_synthetic_and_corrupt_artifacts(extra_paths=None, dry_run=False):
    """
    Scans Google Drive storage and local repository, removing synthetic audio files,
    legacy AI avatar duplicates, and corrupted batch directories.
    """
    print("=" * 75)
    print("🧹 [Option B Purge Agent] Purging Synthetic Data & Corrupted Batches...")
    print(f"   Target Action: {'[DRY RUN]' if dry_run else '[PERMANENT DELETE]'}")
    print("=" * 75)

    all_paths = list(SEARCH_CANDIDATE_PATHS)
    if extra_paths:
        for ep in extra_paths:
            if ep and Path(ep) not in all_paths:
                all_paths.append(Path(ep))

    stats = {
        "purged_synthetic_audios": 0,
        "purged_legacy_files": 0,
        "purged_corrupt_batches": 0,
        "bytes_freed": 0,
        "scanned_dirs": 0
    }

    for base_path in all_paths:
        if not base_path.exists():
            continue

        stats["scanned_dirs"] += 1
        print(f"📁 [Scanning Storage] {base_path}")

        # 1. First pass: Detect and purge individual synthetic audios and legacy files
        for root, dirs, files in os.walk(str(base_path), topdown=False):
            r_path = Path(root)
            for fname in files:
                f_path = r_path / fname
                is_legacy = any(pat.match(fname) for pat in LEGACY_FILENAME_PATTERNS)
                is_synth_audio = False

                if fname.lower().endswith(".wav"):
                    # Check if file has synthetic sine signature
                    if is_synthetic_sine_audio(f_path):
                        is_synth_audio = True

                if is_legacy or is_synth_audio:
                    try:
                        sz = f_path.stat().st_size
                        if not dry_run:
                            f_path.unlink()
                        stats["bytes_freed"] += sz
                        if is_synth_audio:
                            stats["purged_synthetic_audios"] += 1
                            print(f"  🗑️ [Purged Synthetic Sine Audio] {f_path.relative_to(base_path)}")
                        else:
                            stats["purged_legacy_files"] += 1
                            print(f"  🗑️ [Purged Legacy AI Avatar File] {f_path.relative_to(base_path)}")
                    except Exception as err:
                        print(f"  ⚠️ Error removing {fname}: {err}")

        # 2. Second pass: Detect and purge corrupted batch directories
        # Any batch directory (e.g. batch_01_10h03) where 01_raw_originals has 0 JPG images
        for root, dirs, files in os.walk(str(base_path), topdown=False):
            r_path = Path(root)
            if r_path.name.startswith("batch_"):
                raw_dir = r_path / "01_raw_originals"
                if raw_dir.exists():
                    jpg_images = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.jpeg")) + list(raw_dir.glob("*.png"))
                    if len(jpg_images) == 0:
                        print(f"  ❌ [Detected Corrupted Batch] {r_path.name}: 01_raw_originals has 0 human photos!")
                        try:
                            sz_batch = sum(f.stat().st_size for f in r_path.rglob("*") if f.is_file())
                            if not dry_run:
                                shutil.rmtree(str(r_path), ignore_errors=True)
                            stats["purged_corrupt_batches"] += 1
                            stats["bytes_freed"] += sz_batch
                            print(f"  🗑️ [Purged Corrupted Batch Folder] Removed {r_path.name} entirely ({round(sz_batch/1024, 1)} KB).")
                        except Exception as e:
                            print(f"  ⚠️ Error deleting batch folder {r_path.name}: {e}")

    # 3. Clean up stale registry files to prevent [GDrive Dedup Block] from suppressing real photos
    for base_path in all_paths:
        if base_path.exists():
            reg_file = base_path / "perceptual_dhash_registry.json"
            if reg_file.exists():
                try:
                    # Clean registry if it points to purged legacy files
                    with open(reg_file, "r", encoding="utf-8") as rf:
                        reg_data = json.load(rf)
                    cleaned_hashes = {}
                    for h_val, meta in reg_data.get("hashes", {}).items():
                        fname = meta.get("filename", "")
                        if not any(pat.match(fname) for pat in LEGACY_FILENAME_PATTERNS) and not fname.startswith("fake_sample_"):
                            cleaned_hashes[h_val] = meta
                    reg_data["hashes"] = cleaned_hashes
                    reg_data["total_registered"] = len(cleaned_hashes)
                    if not dry_run:
                        with open(reg_file, "w", encoding="utf-8") as wf:
                            json.dump(reg_data, wf, indent=2)
                        print(f"  ✅ [Cleaned Registry] Updated {reg_file.name} ({len(cleaned_hashes)} authentic entries).")
                except Exception as e_reg:
                    print(f"  ⚠️ Notice cleaning registry {reg_file}: {e_reg}")

    freed_mb = round(stats["bytes_freed"] / (1024 ** 2), 2)
    print("=" * 75)
    print(f"✨ [Purge Summary] Completed across {stats['scanned_dirs']} storage locations:")
    print(f"   - Synthetic Sine-Wave Audios Removed: {stats['purged_synthetic_audios']}")
    print(f"   - Legacy AI Avatar / Cartoon Files Removed: {stats['purged_legacy_files']}")
    print(f"   - Corrupted Empty-Image Batches Purged: {stats['purged_corrupt_batches']}")
    print(f"   - Disk Space Freed: {freed_mb} MB")
    print("=" * 75)
    return stats


if __name__ == "__main__":
    purge_synthetic_and_corrupt_artifacts()
