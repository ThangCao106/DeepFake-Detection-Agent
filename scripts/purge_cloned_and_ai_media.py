"""
Automated Purge & Cleanup Utility.
Removes legacy cloned / augmented face images, AI-generated duplicates,
and resets deduplication registries both locally and on Google Drive.
Prepares a 100% clean environment for Option A + C unique human identities.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def purge_cloned_media(include_drive: bool = True, date_str: str = None):
    print("=" * 75)
    print("🧹 [PURGE] Cleaning legacy cloned and AI-manipulated duplicate media...")
    print("=" * 75)

    purged_files = 0
    purged_bytes = 0

    # 1. Clean Local Pentest Directories
    pentest_dir = PROJECT_ROOT / "report_colab" / "pentest"
    target_local_dirs = [
        pentest_dir / "01_raw_originals",
        pentest_dir / "02_manipulated_faceswap",
        pentest_dir / "03_manipulated_openweight"
    ]
    if date_str:
        target_local_dirs.append(pentest_dir / date_str)
    else:
        # All date subfolders like 2026-10-*
        for d in pentest_dir.glob("2026-*"):
            if d.is_dir():
                target_local_dirs.append(d)

    for ldir in target_local_dirs:
        if ldir.exists():
            for f in ldir.glob("*.*"):
                if f.is_file() and not f.name.endswith(".md"):
                    try:
                        sz = f.stat().st_size
                        f.unlink()
                        purged_files += 1
                        purged_bytes += sz
                    except Exception:
                        pass
            print(f"  🗑️ Cleared local folder: {ldir.relative_to(PROJECT_ROOT)}")

    # 2. Reset Local Dedup Registry
    registry_file = pentest_dir / "perceptual_dhash_registry.json"
    if registry_file.exists():
        try:
            registry_file.unlink()
            print("  🔄 Reset local perceptual_dhash_registry.json")
        except Exception:
            pass

    # 3. Clean Google Drive Directories if Mounted or Available
    if include_drive:
        try:
            from colab_sync_to_gdrive import find_or_create_colab_drive_path
            drive_root = find_or_create_colab_drive_path()
        except Exception:
            drive_root = None

        if not drive_root:
            # Check standard Colab mount paths
            cand_paths = [
                Path("/content/drive/MyDrive/1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"),
                Path("/content/drive/MyDrive/Thread_Data"),
                Path("/content/drive/MyDrive/1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6"),
                Path("/content/drive/MyDrive/1W6R3QYzTI6GJAD2lByKoVaQnXfURcLO0"),
            ]
            for p in cand_paths:
                if p.exists() and p.is_dir():
                    drive_root = p
                    break

        if drive_root and drive_root.exists():
            print(f"\n📂 Google Drive detected at: {drive_root}")
            drive_targets = []
            
            # Target date folders
            if date_str:
                drive_targets.append(drive_root / date_str)
            else:
                for d in drive_root.glob("2026-*"):
                    if d.is_dir():
                        drive_targets.append(d)
                # Also root subfolders
                for sname in ["01_raw_originals", "02_manipulated_faceswap", "03_manipulated_openweight"]:
                    sub_p = drive_root / sname
                    if sub_p.exists():
                        drive_targets.append(sub_p)

            for ddir in drive_targets:
                if ddir.exists():
                    for f in ddir.glob("*.*"):
                        if f.is_file():
                            try:
                                sz = f.stat().st_size
                                f.unlink()
                                purged_files += 1
                                purged_bytes += sz
                            except Exception:
                                pass
                    # If subdirectories exist inside date folder (01_raw_originals, etc.)
                    for sub in ["01_raw_originals", "02_manipulated_faceswap", "03_manipulated_openweight"]:
                        sdir = ddir / sub
                        if sdir.exists():
                            for f in sdir.glob("*.*"):
                                if f.is_file():
                                    try:
                                        sz = f.stat().st_size
                                        f.unlink()
                                        purged_files += 1
                                        purged_bytes += sz
                                    except Exception:
                                        pass
                    print(f"  🗑️ Cleared Google Drive folder: {ddir.name}")

            # Reset Drive Registry & old manifests in purged targets
            drive_reg = drive_root / "registry" / "perceptual_dhash_registry.json"
            if drive_reg.exists():
                try:
                    drive_reg.unlink()
                    print("  🔄 Reset Google Drive perceptual_dhash_registry.json")
                except Exception:
                    pass

            for mf in drive_root.glob("**/dataset_manifest.jsonl"):
                try:
                    mf.unlink()
                    purged_files += 1
                except Exception:
                    pass
        else:
            print("  ℹ️ Google Drive mount not detected (run on Colab to purge Drive directly).")

    # 4. Clean Colab Local SSD Training Cache (/content/train_cache)
    colab_train_cache = Path("/content/train_cache")
    if colab_train_cache.exists():
        try:
            for f in colab_train_cache.glob("**/*.*"):
                if f.is_file():
                    try:
                        sz = f.stat().st_size
                        f.unlink()
                        purged_files += 1
                        purged_bytes += sz
                    except Exception:
                        pass
            print(f"  🗑️ Cleared Colab local SSD cache: {colab_train_cache}")
        except Exception as e:
            print(f"  Notice cleaning /content/train_cache: {e}")

    mb_freed = purged_bytes / (1024 * 1024)
    print("=" * 75)
    print(f"🎉 [PURGE COMPLETE] Cleared {purged_files} legacy cloned/duplicate files ({mb_freed:.2f} MB freed).")
    print("   Ready for Option A + C unique human identity harvesting!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Purge legacy cloned and duplicate face media")
    parser.add_argument("--date", type=str, default=None, help="Specific date to purge (e.g. 2026-10-14)")
    parser.add_argument("--skip-drive", action="store_true", help="Skip cleaning Google Drive")
    args = parser.parse_args()

    purge_cloned_media(include_drive=not args.skip_drive, date_str=args.date)
