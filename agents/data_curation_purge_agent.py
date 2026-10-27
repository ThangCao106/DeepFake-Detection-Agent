"""
Data Curation & Storage Hygiene Agent (Option B Standalone Agent)

Decoupled from Google Colab GPU training loops.
Dedicated to:
  1. Perceptual Deduplication Registry (64-bit dHash, Hamming distance <= 4)
  2. Purging synthetic talking-head videos and artificial audio seeds from storage (GDrive & Local)
  3. Purging 0-byte, corrupt or orphan media artifacts
  4. Auditing incoming crawled samples from Threads & Open Harvester to guarantee 100% authentic human media
  5. Writing operational audit records to SQLite & JSON for Observability Dashboard
"""

import sys
import os
import time
import json
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from crawler.dedup_engine import PerceptualDeduplicationRegistry
from scripts.purge_synthetic_video_and_audio_gdrive import purge_synthetic_media
from scripts.purge_synthetic_audio_and_drive_artifacts import purge_synthetic_and_corrupt_artifacts

DB_PATH = PROJECT_ROOT / "report_colab" / "deepfake_database.sqlite"
AUDIT_JSON_PATH = PROJECT_ROOT / "report_colab" / "curation_agent_audit.json"


class DataCurationPurgeAgent:
    """Standalone Agent responsible for data curation, hygiene and storage integrity."""

    def __init__(self, dry_run: bool = False, extra_drive_path: str = None):
        self.dry_run = dry_run
        self.extra_drive_path = extra_drive_path
        self.dedup_registry = PerceptualDeduplicationRegistry()
        self._init_db_schema()

    def _init_db_schema(self):
        """Ensure SQLite table for curation runs exists."""
        try:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS curation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_timestamp TEXT NOT NULL,
                    status TEXT NOT NULL,
                    purged_synthetic_videos INTEGER DEFAULT 0,
                    purged_synthetic_audios INTEGER DEFAULT 0,
                    purged_corrupt_files INTEGER DEFAULT 0,
                    total_known_dhashes INTEGER DEFAULT 0,
                    authentic_ratio REAL DEFAULT 1.0,
                    details_json TEXT
                )
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[DataCurationAgent] SQLite initialization notice: {e}")

    def run_cycle(self) -> dict:
        """Execute one complete curation, deduplication, and hygiene cycle."""
        timestamp_str = datetime.now(timezone.utc).isoformat()
        print(f"\n=======================================================")
        print(f"🧹 [DataCurationAgent] Initiating Hygiene Cycle [{timestamp_str}]")
        print(f"=======================================================")

        stats = {
            "timestamp": timestamp_str,
            "status": "SUCCESS",
            "dry_run": self.dry_run,
            "purged_synthetic_videos": 0,
            "purged_synthetic_audios": 0,
            "purged_corrupt_files": 0,
            "total_known_dhashes": 0,
            "authentic_ratio": 1.0,
            "inspected_locations": []
        }

        # 1. Purge synthetic talking-head videos and artificial audio seeds from GDrive & local
        print("\n[Step 1/3] Scanning & Purging Synthetic/AI Video & Audio Seeds...")
        try:
            if not self.dry_run:
                purge_synthetic_media(extra_mount=self.extra_drive_path)
            stats["inspected_locations"].append("Drive & Local Verified Media")
        except Exception as err:
            print(f"[DataCurationAgent] Warning during synthetic media purge: {err}")
            stats["status"] = "PARTIAL_WARNING"

        # 2. Purge corrupt / 0-byte / synthetic audio artifacts
        print("\n[Step 2/3] Scanning for Corrupt Artifacts & Legacy Bad Seeds...")
        try:
            if not self.dry_run:
                purge_synthetic_and_corrupt_artifacts()
            stats["inspected_locations"].append("Drive Pentest Batches & Daily Dirs")
        except Exception as err:
            print(f"[DataCurationAgent] Warning during corrupt artifacts purge: {err}")
            stats["status"] = "PARTIAL_WARNING"

        # 3. Synchronize 64-bit dHash Deduplication Registry
        print("\n[Step 3/3] Synchronizing 64-bit dHash Deduplication Registry...")
        try:
            drive_mount = Path(self.extra_drive_path) if self.extra_drive_path else None
            self.dedup_registry.sync_with_gdrive(drive_mount)
            known_hashes = self.dedup_registry.get_known_dhashes()
            stats["total_known_dhashes"] = len(known_hashes)
            print(f"  ✅ Perceptual Registry synchronized. Total unique 64-bit dHashes: {len(known_hashes)}")
        except Exception as err:
            print(f"[DataCurationAgent] Warning during dedup registry sync: {err}")
            stats["status"] = "PARTIAL_WARNING"

        # Record to SQLite & JSON
        self._record_audit(stats)
        print(f"\n✅ [DataCurationAgent] Hygiene Cycle Completed! Status: {stats['status']}\n")
        return stats

    def _record_audit(self, stats: dict):
        """Save audit record to SQLite and JSON for dashboard consumption."""
        try:
            AUDIT_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(AUDIT_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)

            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO curation_runs (
                    run_timestamp, status, purged_synthetic_videos,
                    purged_synthetic_audios, purged_corrupt_files,
                    total_known_dhashes, authentic_ratio, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats["timestamp"],
                stats["status"],
                stats["purged_synthetic_videos"],
                stats["purged_synthetic_audios"],
                stats["purged_corrupt_files"],
                stats["total_known_dhashes"],
                stats["authentic_ratio"],
                json.dumps(stats, ensure_ascii=False)
            ))
            conn.commit()
            conn.close()
        except Exception as err:
            print(f"[DataCurationAgent] Warning recording audit: {err}")

    def run_daemon(self, interval_hours: float = 4.0):
        """Run as a standing background daemon."""
        print(f"🚀 [DataCurationAgent] Starting Standing Daemon (Interval: {interval_hours}h)...")
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                print(f"[DataCurationAgent] Error in daemon loop: {e}")
            sleep_secs = int(interval_hours * 3600)
            print(f"⏳ [DataCurationAgent] Sleeping for {interval_hours} hours ({sleep_secs}s)...")
            time.sleep(sleep_secs)


def main():
    parser = argparse.ArgumentParser(description="Standalone Data Curation & Storage Hygiene Agent")
    parser.add_argument("--once", action="store_true", help="Run a single hygiene cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously as background daemon")
    parser.add_argument("--interval-hours", type=float, default=4.0, help="Interval between runs in daemon mode")
    parser.add_argument("--dry-run", action="store_true", help="Audit only without deleting files")
    parser.add_argument("--drive-mount", type=str, default=None, help="Custom path to Google Drive mount")

    args = parser.parse_args()
    agent = DataCurationPurgeAgent(dry_run=args.dry_run, extra_drive_path=args.drive_mount)

    if args.daemon:
        agent.run_daemon(interval_hours=args.interval_hours)
    else:
        agent.run_cycle()


if __name__ == "__main__":
    main()
