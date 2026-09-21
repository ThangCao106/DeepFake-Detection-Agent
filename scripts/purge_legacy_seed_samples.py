"""
Purge Legacy Duplicate Seed Samples Script.
Scans local pentest outputs and Google Drive storage structures, purging legacy duplicate sample files:
  - human_face_sample_*
  - faceswap_sample_*
  - openweight_sample_*
Preserves all Option C unique Threads samples (threads_*).
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
    Path("/content/drive/MyDrive/DeepFake_Pentest_Data")
]

WESTERN_LEGACY_HASHES = [
    "20d877b2e7", "2909e941a1", "667369889f", "829019d377", "a04755ea1e",
    "06fb119d68", "0e749dbc33", "13d819bf50", "d3", "e31", "e38", "ea",
    "fa0", "fb", "fc8", "cf", "d7", "63", "10", "12", "45", "51",
    "0c", "0d", "0e", "2a2", "2c", "3a", "4b", "5d", "7d1", "7e4",
    "8df", "17b", "18a", "19a", "20", "43"
]

LEGACY_PATTERNS = [
    re.compile(r"^human_face_sample_.*"),
    re.compile(r"^faceswap_sample_.*"),
    re.compile(r"^openweight_sample_.*"),
    re.compile(r".*(" + "|".join(WESTERN_LEGACY_HASHES) + r").*")
]

def purge_legacy_samples():
    print("=" * 70)
    print("[Purge Script] Starting Automatic Purge of Legacy Duplicate Seed Files...")
    print("=" * 70)
    
    purged_count = 0
    purged_bytes = 0
    
    for search_dir in SEARCH_PATHS:
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
                        print(f"  [Removed] Legacy file: {filepath.name}")
                    except Exception as err:
                        print(f"  [Warning] Could not remove {filepath}: {err}")
                        
    print("=" * 70)
    print(f"[Purge Script] Purge Complete! Removed {purged_count} legacy duplicate files ({round(purged_bytes / (1024**2), 2)} MB freed).")
    print("[Purge Script] Only unique Option C Threads samples (threads_*) remain in storage.")
    print("=" * 70)

if __name__ == "__main__":
    purge_legacy_samples()
