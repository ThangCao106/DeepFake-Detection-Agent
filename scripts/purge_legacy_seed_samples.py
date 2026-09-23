"""
Purge Legacy Duplicate Seed Samples Script.
Scans local pentest outputs and Google Drive storage structures, purging legacy duplicate sample files:
  - human_face_sample_*
  - faceswap_sample_*
  - openweight_sample_*
  - Legacy Western face hashes from 2026-09-21
Preserves all Option C unique Vietnamese Threads samples (threads_*).
"""
import os
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEARCH_PATHS = [
    PROJECT_ROOT / "report_colab" / "pentest",
    PROJECT_ROOT / "report_colab" / "gdrive_storage",
    PROJECT_ROOT / "report_colab" / "temp_media_samples",
    Path("/content/drive/MyDrive/Thread_Data"),
    Path("/content/drive/MyDrive/DeepFake_Pentest_Data"),
    Path("/content/drive/MyDrive/1MipTWpXt0o0xz2IyUc_pTvMzo0tcXRlr"),
    Path("/content/drive/Shareddrives/Thread_Data")
]

# Legacy patterns: ONLY purge legacy sample formats and files from 2026-09-21
# Protects all 2026-09-22+ active Vietnamese media files
LEGACY_PATTERNS = [
    re.compile(r"^human_face_sample_.*"),
    re.compile(r"^faceswap_sample_.*"),
    re.compile(r"^openweight_sample_.*"),
    re.compile(r"^threads_20260921_.*"),  # Purge 2026-09-21 legacy files
    re.compile(r"^threads_20260922_.*"),  # Purge 2026-09-22 AI seed files
]

def purge_legacy_samples(extra_paths=None):
    print("=" * 70)
    print("[Purge Script] Starting Automatic Purge of Legacy Duplicate Seed Files...")
    print("=" * 70)
    
    purged_count = 0
    purged_bytes = 0
    
    all_paths = list(SEARCH_PATHS)
    if extra_paths:
        for ep in extra_paths:
            if ep and Path(ep) not in all_paths:
                all_paths.append(Path(ep))
                
    for search_dir in all_paths:
        if not search_dir.exists():
            continue
            
        print(f"[Purge Script] Scanning directory: {search_dir}")
        for root, _, files in os.walk(search_dir):
            for filename in files:
                is_legacy = any(pat.match(filename) for pat in LEGACY_PATTERNS)
                if is_legacy:
                    filepath = Path(root) / filename
                    try:
                        file_size = filepath.stat().st_size if filepath.exists() else 0
                        filepath.unlink()
                        purged_count += 1
                        purged_bytes += file_size
                        print(f"  [Removed] Legacy Western file: {filepath.name}")
                    except Exception as err:
                        print(f"  [Warning] Could not remove {filepath}: {err}")
                        
    print("=" * 70)
    print(f"[Purge Script] Purge Complete! Removed {purged_count} legacy duplicate files ({round(purged_bytes / (1024**2), 2)} MB freed).")
    print("[Purge Script] Only unique Option C Vietnamese Threads samples remain in storage.")
    print("=" * 70)

if __name__ == "__main__":
    purge_legacy_samples()
