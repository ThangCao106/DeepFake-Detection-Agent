"""
Option C Purge Engine: Eliminate Synthetic AI Talking-Head Videos & Artificial Audio Seeds
from Google Drive and Local Storage across all past dates (2026-09 to 2026-10).

Target Media Eliminated:
  1. Synthetic AI Talking-Head Videos:
     - Matching exact SHA256 of the 5 AI avatar seed videos (deb185bb, af4d7ba7, c228cb07, 066e4cfd, 4879688e)
     - Filenames matching real_video_vn_*, real_vid_*_<hash>, or synthesized face-animation MP4s
  2. Synthetic & Artificial Audio Seeds:
     - Matching exact SHA256 of vn_speech_seed_01..10 (363653ce, b457b94c, c869a29e, etc.)
     - Synthetic sine-wave test tones (0.5*sin(140) + 0.3*sin(520) + 0.15*sin(1600))
  3. Associated Manifest and Registry references cleaned in-place.

Usage:
  # Local scan & purge:
  py -3.13 scripts/purge_synthetic_video_and_audio_gdrive.py

  # On Google Colab (Direct Mount):
  !python scripts/purge_synthetic_video_and_audio_gdrive.py --drive-mount /content/drive/MyDrive/Thread_Data

  # Dry run (inspect without deleting):
  python scripts/purge_synthetic_video_and_audio_gdrive.py --dry-run
"""

import os
import sys
import re
import json
import time
import wave
import hashlib
import argparse
from pathlib import Path
from typing import List, Set, Dict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Known SHA256 Hashes of the 5 Synthetic Face Animation Videos
KNOWN_SYNTHETIC_VIDEO_SHAS = {
    "deb185bbf3112c596140ed948b1ad057f7a1d11e7a556c670daba0cd7a976e0d",  # vn_id_official talking head
    "af4d7ba78b279693e170bd3a70f7ca68932ac1e871ffe0326dc4bc681f350f15",  # vn_mature_aging talking head
    "c228cb0770a48bf2714664a496c103dcecebaad4edc42466c4c9e19c31199639",  # vn_occlusion talking head
    "066e4cfd3e73ef19a6b2c26ca45bccf95efa23462898603ed1786e0cf9d87c86",  # vn_selfie talking head
    "4879688e7ecd8a412bc41c6d7ff74bfa84dbdea255bc7396c80579adbd2676e5",  # vn_street talking head
}

# Known SHA256 8-char prefixes
KNOWN_SYNTH_VID_PREFIXES = {"deb185bb", "af4d7ba7", "c228cb07", "066e4cfd", "4879688e"}

# Known SHA256 Hashes of the 10 Artificial Audio Seeds
KNOWN_SYNTHETIC_AUDIO_SHAS = {
    "363653cec3a4050281cd9c873602c245f5977385108c4b53e2240a9a9d41a6c9",
    "b457b94ce88e8e0434ed3ce08dbdecfc8b1b13eb1b90e921da3c03bd0546ba89",
    "c869a29e305ff5db90ac90e8ec3acdece6c6602020fe5120110975148d001884",
    "0ce89a7ccb2a6e5ccf5f9dc04430f9eba0f44b73506f8c4cfe131a09212e9145",
    "72655ee3133482f2eb78216e926d0c1b37b77ba7adeec84123daad45a6158824",
    "fe99f3dc3d4485ed44aebd1fab81f9d2ec0822bc8c1aab0a5d8fcf5dfeb7e00a",
    "5813b73b0b184ee39b7901e20db88edc64c9aed8a75a3a1d30c5a14c9c20c005",
    "742b7b9b382f0db0dd37a0d61d01b15f25fce59b3e4525e13dfd87e5e88c306a",
    "5e094e3fb5d3739d5044178402d7223495c7b0b5b9636ac6899c9e6d118a07fb",
    "7561f8cec273e5aed94ddef194cfb27c95d49a64ea696309be60e9f3349b2997",
}

KNOWN_SYNTH_AUD_PREFIXES = {
    "363653ce", "b457b94c", "c869a29e", "0ce89a7c", "72655ee3",
    "fe99f3dc", "5813b73b", "742b7b9b", "5e094e3f", "7561f8ce"
}

# Candidate Paths across Colab and Local environment
CANDIDATE_SEARCH_PATHS = [
    # Google Colab Mount Targets
    Path("/content/drive/MyDrive/Thread_Data"),
    Path("/content/drive/MyDrive/1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"),
    Path("/content/drive/MyDrive/1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6"),
    Path("/content/drive/MyDrive/1W6R3QYzTI6GJAD2lByKoVaQnXfURcLO0"),
    Path("/content/drive/MyDrive/DeepFake_Pentest_Data"),
    Path("/content/drive/MyDrive/1MipTWpXt0o0xz2IyUc_pTvMzo0tcXRlr"),
    Path("/content/drive/Shareddrives/Thread_Data"),
    # Local Storage Targets
    PROJECT_ROOT / "report_colab" / "gdrive_storage",
    PROJECT_ROOT / "report_colab" / "pentest",
    PROJECT_ROOT / "report_colab" / "temp_media_samples",
]


def compute_sha256(file_path: Path) -> str:
    """Computes SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_synthetic_sine_audio(wav_path: Path) -> bool:
    """Checks if audio matches pure synthetic sine formula."""
    try:
        import numpy as np
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            if framerate != 16000 or nframes < 16000:
                return False
            frames = wf.readframes(min(32000, nframes))
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

        num_check = len(data)
        t = np.linspace(0, num_check / framerate, num_check, endpoint=False)
        expected = (
            0.5 * np.sin(2 * np.pi * 140 * t)
            + 0.3 * np.sin(2 * np.pi * 520 * t)
            + 0.15 * np.sin(2 * np.pi * 1600 * t)
        )
        mae = float(np.mean(np.abs(data - expected)))
        return mae < 0.05
    except Exception:
        return False


def is_synthetic_talking_head_video(mp4_path: Path) -> bool:
    """Checks if a video file is a synthetic AI talking head video or legacy seed."""
    fname = mp4_path.name.lower()
    
    # 1. Direct name patterns
    if (
        fname.startswith("real_video_vn_") 
        or fname.startswith("real_vid_") 
        or "vn_id_official" in fname
        or re.match(r"^threads_\d{8}_vid_\d+\.mp4$", fname)
        or re.match(r"^faceswap_threads_\d{8}_vid_\d+\.mp4$", fname)
        or re.match(r"^openweight_threads_\d{8}_vid_\d+\.mp4$", fname)
    ):
        return True
        
    for pfx in KNOWN_SYNTH_VID_PREFIXES:
        if pfx in fname:
            return True
            
    # 2. SHA256 match
    try:
        f_sha = compute_sha256(mp4_path)
        if f_sha in KNOWN_SYNTHETIC_VIDEO_SHAS or f_sha[:8] in KNOWN_SYNTH_VID_PREFIXES:
            return True
    except Exception:
        pass
        
    return False


def is_synthetic_or_seed_audio(wav_path: Path) -> bool:
    """Checks if an audio file is an artificial seed, legacy fake, or synthetic sine."""
    fname = wav_path.name.lower()
    
    if (
        fname.startswith("vn_speech_seed_") 
        or fname.startswith("real_aud_")
        or re.match(r"^threads_\d{8}_aud_\d+\.wav$", fname)
    ):
        return True
        
    for pfx in KNOWN_SYNTH_AUD_PREFIXES:
        if pfx in fname:
            return True
            
    try:
        f_sha = compute_sha256(wav_path)
        if f_sha in KNOWN_SYNTHETIC_AUDIO_SHAS or f_sha[:8] in KNOWN_SYNTH_AUD_PREFIXES:
            return True
    except Exception:
        pass
        
    if is_synthetic_sine_audio(wav_path):
        return True
        
    return False


def clean_manifest_file(manifest_path: Path, purged_filenames: Set[str], dry_run: bool = False):
    """Removes purged filenames from dataset_manifest.jsonl or master_manifest_*.jsonl."""
    if not manifest_path.exists():
        return
    try:
        cleaned_lines = []
        removed_count = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                    fn = rec.get("filename") or ""
                    sid = rec.get("sample_id") or ""
                    if fn in purged_filenames or any(p in fn for p in KNOWN_SYNTH_VID_PREFIXES.union(KNOWN_SYNTH_AUD_PREFIXES)):
                        removed_count += 1
                        continue
                    cleaned_lines.append(line_str)
                except Exception:
                    cleaned_lines.append(line_str)
                    
        if removed_count > 0 and not dry_run:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write("\n".join(cleaned_lines) + "\n" if cleaned_lines else "")
            print(f"    📄 [Manifest Cleaned] {manifest_path.name}: Removed {removed_count} synthetic entries.")
    except Exception as e:
        print(f"    ⚠️ Error cleaning manifest {manifest_path.name}: {e}")


def purge_synthetic_media(extra_mount: str = None, dry_run: bool = False, target_date: str = None) -> Dict:
    """
    Executes deep scan across Google Drive and local paths, purging all synthetic
    videos and audio seeds, and updating manifest files.
    """
    print("\n" + "=" * 80)
    print("🧹 [Option C Drive Purge Engine] Purging Synthetic Videos & Audio Seeds from Google Drive")
    print(f"   Mode: {'[DRY RUN - INSPECT ONLY]' if dry_run else '[PERMANENT DELETE & FLUSH]'}")
    if target_date:
        print(f"   Target Date Filter: {target_date}")
    print("=" * 80)

    paths_to_scan: List[Path] = []
    if extra_mount:
        p_extra = Path(extra_mount)
        if p_extra.exists() and p_extra not in paths_to_scan:
            paths_to_scan.append(p_extra)
            
    for p in CANDIDATE_SEARCH_PATHS:
        if p.exists() and p not in paths_to_scan:
            paths_to_scan.append(p)

    if not paths_to_scan:
        print("⚠️ No valid Google Drive mount or local storage directory found.")
        return {"purged_videos": 0, "purged_audios": 0}

    stats = {
        "purged_videos": 0,
        "purged_audios": 0,
        "bytes_freed": 0,
        "scanned_dirs": 0,
        "purged_filenames": set()
    }

    manifest_files_to_clean: Set[Path] = set()

    for base_p in paths_to_scan:
        print(f"\n📂 [Scanning Base Path] {base_p}")
        stats["scanned_dirs"] += 1
        
        # Traverse directory tree
        for root, dirs, files in os.walk(str(base_p), topdown=False):
            r_path = Path(root)
            
            # Apply date filter if specified
            if target_date and target_date not in str(r_path):
                continue
                
            for fname in files:
                f_path = r_path / fname
                fn_lower = fname.lower()
                
                # Check for manifest files to clean later
                if fname.endswith(".jsonl") and ("manifest" in fname):
                    manifest_files_to_clean.add(f_path)
                    
                # 1. Check for Synthetic Talking Head Videos
                if fn_lower.endswith(".mp4"):
                    if is_synthetic_talking_head_video(f_path):
                        sz = f_path.stat().st_size
                        stats["purged_videos"] += 1
                        stats["bytes_freed"] += sz
                        stats["purged_filenames"].add(fname)
                        print(f"  🗑️ [Purged Synthetic AI Video] {fname} ({round(sz/1024, 1)} KB) in {r_path.name}")
                        if not dry_run:
                            try:
                                f_path.unlink()
                            except Exception as e:
                                print(f"     ⚠️ Delete error: {e}")
                                
                # 2. Check for Synthetic / Seed Audios
                elif fn_lower.endswith(".wav"):
                    if is_synthetic_or_seed_audio(f_path):
                        sz = f_path.stat().st_size
                        stats["purged_audios"] += 1
                        stats["bytes_freed"] += sz
                        stats["purged_filenames"].add(fname)
                        print(f"  🗑️ [Purged Synthetic/Seed Audio] {fname} ({round(sz/1024, 1)} KB) in {r_path.name}")
                        if not dry_run:
                            try:
                                f_path.unlink()
                            except Exception as e:
                                print(f"     ⚠️ Delete error: {e}")

    # Clean manifests
    print("\n📑 [Cleaning Manifest Files] Removing purged media records...")
    for mf in manifest_files_to_clean:
        clean_manifest_file(mf, stats["purged_filenames"], dry_run=dry_run)

    # Force Linux kernel / FUSE buffer sync on Colab
    if not dry_run and hasattr(os, "sync"):
        try:
            print("⏳ [FUSE Sync] Flushing disk writeback buffer to Google Drive...")
            os.sync()
            time.sleep(2)
            os.sync()
        except Exception:
            pass

    freed_mb = round(stats["bytes_freed"] / (1024 * 1024), 2)
    print("\n" + "=" * 80)
    print("✨ [PURGE COMPLETE] Summary of Google Drive & Local Storage Cleanup:")
    print(f"   - Synthetic AI Talking-Head Videos Removed: {stats['purged_videos']}")
    print(f"   - Artificial / Seed Audio WAVs Removed:     {stats['purged_audios']}")
    print(f"   - Total Storage Freed:                     {freed_mb} MB")
    print(f"   - Locations Scanned:                       {stats['scanned_dirs']}")
    print("=" * 80)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Purge Synthetic Video & Audio Seeds from Google Drive")
    parser.add_argument("--drive-mount", type=str, default=None, help="Explicit Google Drive mount path")
    parser.add_argument("--date", type=str, default=None, help="Target date filter (e.g. 2026-10-26)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect files without deleting")
    args = parser.parse_args()

    purge_synthetic_media(
        extra_mount=args.drive_mount,
        dry_run=args.dry_run,
        target_date=args.date
    )


if __name__ == "__main__":
    main()
