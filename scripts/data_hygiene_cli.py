# -*- coding: utf-8 -*-
"""
Unified Data Hygiene & Storage Purge CLI (consolidating legacy purge_*.py scripts).
Provides:
  - purge_synthetic_media(...)
  - purge_synthetic_and_corrupt_artifacts(...)
  - purge_legacy_samples(...)
  - prune_pentest_cert_bloat(...)
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

KNOWN_SYNTHETIC_VIDEO_SHAS = {
    "deb185bbf3112c596140ed948b1ad057f7a1d11e7a556c670daba0cd7a976e0d",
    "af4d7ba78b279693e170bd3a70f7ca68932ac1e871ffe0326dc4bc681f350f15",
    "c228cb0770a48bf2714664a496c103dcecebaad4edc42466c4c9e19c31199639",
    "066e4cfd3e73ef19a6b2c26ca45bccf95efa23462898603ed1786e0cf9d87c86",
    "4879688e7ecd8a412bc41c6d7ff74bfa84dbdea255bc7396c80579adbd2676e5",
}
KNOWN_SYNTH_VID_PREFIXES = {"deb185bb", "af4d7ba7", "c228cb07", "066e4cfd", "4879688e"}

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
    "fe99f3dc", "5813b73b", "742b7b9b", "5e094e3f", "7561f8ce",
}

LEGACY_FILENAME_PATTERNS = [
    re.compile(r"^human_face_sample_.*"),
    re.compile(r"^faceswap_sample_.*"),
    re.compile(r"^openweight_sample_.*"),
    re.compile(r"^threads_20260921_.*"),
    re.compile(r"^threads_20260922_.*"),
]

SEARCH_CANDIDATE_PATHS = [
    Path("/content/drive/MyDrive/Thread_Data"),
    Path("/content/drive/MyDrive/1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"),
    Path("/content/drive/MyDrive/1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6"),
    Path("/content/drive/MyDrive/1W6R3QYzTI6GJAD2lByKoVaQnXfURcLO0"),
    Path("/content/drive/MyDrive/DeepFake_Pentest_Data"),
    Path("/content/drive/MyDrive/1MipTWpXt0o0xz2IyUc_pTvMzo0tcXRlr"),
    Path("/content/drive/Shareddrives/Thread_Data"),
    PROJECT_ROOT / "report_colab" / "gdrive_storage",
    PROJECT_ROOT / "report_colab" / "pentest",
    PROJECT_ROOT / "report_colab" / "temp_media_samples",
]


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def is_synthetic_sine_audio(wav_path: Path) -> bool:
    """Detect pure synthetic sine-wave tones previously injected as audio placeholders."""
    try:
        import numpy as np
        with wave.open(str(wav_path), "rb") as wf:
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            if framerate != 16000 or nframes < 16000:
                return False
            frames = wf.readframes(min(32000, nframes))
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        t = np.linspace(0, len(data) / framerate, len(data), endpoint=False)
        expected = 0.5 * np.sin(2 * np.pi * 140 * t) + 0.3 * np.sin(2 * np.pi * 520 * t) + 0.15 * np.sin(2 * np.pi * 1600 * t)
        return float(np.mean(np.abs(data - expected))) < 0.05
    except Exception:
        return False


def is_synthetic_talking_head_video(mp4_path: Path) -> bool:
    fname = mp4_path.name.lower()
    if (
        fname.startswith("real_video_vn_")
        or fname.startswith("real_vid_")
        or "vn_id_official" in fname
        or re.match(r"^threads_\d{8}_vid_\d+\.mp4$", fname)
        or re.match(r"^faceswap_threads_\d{8}_vid_\d+\.mp4$", fname)
        or re.match(r"^openweight_threads_\d{8}_vid_\d+\.mp4$", fname)
    ):
        return True
    if any(pfx in fname for pfx in KNOWN_SYNTH_VID_PREFIXES):
        return True
    try:
        f_sha = compute_sha256(mp4_path)
        return f_sha in KNOWN_SYNTHETIC_VIDEO_SHAS or f_sha[:8] in KNOWN_SYNTH_VID_PREFIXES
    except Exception:
        return False


def is_synthetic_or_seed_audio(wav_path: Path) -> bool:
    fname = wav_path.name.lower()
    if fname.startswith("vn_speech_seed_") or fname.startswith("real_aud_") or re.match(r"^threads_\d{8}_aud_\d+\.wav$", fname):
        return True
    if any(pfx in fname for pfx in KNOWN_SYNTH_AUD_PREFIXES):
        return True
    try:
        f_sha = compute_sha256(wav_path)
        if f_sha in KNOWN_SYNTHETIC_AUDIO_SHAS or f_sha[:8] in KNOWN_SYNTH_AUD_PREFIXES:
            return True
    except Exception:
        pass
    return is_synthetic_sine_audio(wav_path)


def _clean_manifest_file(manifest_path: Path, purged_filenames: Set[str], dry_run: bool = False) -> None:
    if not manifest_path.exists():
        return
    try:
        cleaned_lines: List[str] = []
        removed = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    rec = json.loads(s)
                    fn = rec.get("filename") or ""
                    if fn in purged_filenames or any(p in fn for p in KNOWN_SYNTH_VID_PREFIXES.union(KNOWN_SYNTH_AUD_PREFIXES)):
                        removed += 1
                        continue
                except Exception:
                    pass
                cleaned_lines.append(s)
        if removed > 0 and not dry_run:
            with open(manifest_path, "w", encoding="utf-8") as f:
                f.write("\n".join(cleaned_lines) + "\n" if cleaned_lines else "")
    except Exception:
        pass


def _gather_paths(extra_paths: Optional[List[Any]] = None) -> List[Path]:
    paths: List[Path] = []
    if extra_paths:
        for ep in extra_paths:
            if ep and Path(ep).exists() and Path(ep) not in paths:
                paths.append(Path(ep))
    for p in SEARCH_CANDIDATE_PATHS:
        if p.exists() and p not in paths:
            paths.append(p)
    return paths


def purge_synthetic_media(extra_mount: Optional[str] = None, dry_run: bool = False, target_date: Optional[str] = None) -> Dict[str, Any]:
    paths_to_scan = _gather_paths([extra_mount] if extra_mount else None)
    stats: Dict[str, Any] = {"purged_videos": 0, "purged_audios": 0, "bytes_freed": 0, "scanned_dirs": len(paths_to_scan)}
    purged_names: Set[str] = set()
    manifests: Set[Path] = set()

    for base_p in paths_to_scan:
        for root, _, files in os.walk(str(base_p), topdown=False):
            r_path = Path(root)
            if target_date and target_date not in str(r_path):
                continue
            for fname in files:
                f_path = r_path / fname
                fn_l = fname.lower()
                if fname.endswith(".jsonl") and "manifest" in fname:
                    manifests.add(f_path)
                elif fn_l.endswith(".mp4") and is_synthetic_talking_head_video(f_path):
                    sz = f_path.stat().st_size
                    stats["purged_videos"] += 1
                    stats["bytes_freed"] += sz
                    purged_names.add(fname)
                    if not dry_run:
                        f_path.unlink(missing_ok=True)
                elif fn_l.endswith(".wav") and is_synthetic_or_seed_audio(f_path):
                    sz = f_path.stat().st_size
                    stats["purged_audios"] += 1
                    stats["bytes_freed"] += sz
                    purged_names.add(fname)
                    if not dry_run:
                        f_path.unlink(missing_ok=True)
    for mf in manifests:
        _clean_manifest_file(mf, purged_names, dry_run=dry_run)
    return stats


def purge_synthetic_and_corrupt_artifacts(extra_paths: Optional[List[Any]] = None, dry_run: bool = False) -> Dict[str, Any]:
    paths_to_scan = _gather_paths(extra_paths)
    stats: Dict[str, Any] = {
        "purged_synthetic_audios": 0,
        "purged_legacy_files": 0,
        "purged_corrupt_batches": 0,
        "bytes_freed": 0,
        "scanned_dirs": len(paths_to_scan),
    }
    for base_p in paths_to_scan:
        for root, _, files in os.walk(str(base_p), topdown=False):
            r_path = Path(root)
            for fname in files:
                f_path = r_path / fname
                is_legacy = any(pat.match(fname) for pat in LEGACY_FILENAME_PATTERNS)
                is_synth_audio = fname.lower().endswith(".wav") and is_synthetic_sine_audio(f_path)
                if is_legacy or is_synth_audio:
                    try:
                        sz = f_path.stat().st_size
                        if not dry_run:
                            f_path.unlink(missing_ok=True)
                        stats["bytes_freed"] += sz
                        stats["purged_synthetic_audios" if is_synth_audio else "purged_legacy_files"] += 1
                    except Exception:
                        pass
        for root, _, _ in os.walk(str(base_p), topdown=False):
            r_path = Path(root)
            if r_path.name.startswith("batch_"):
                raw_dir = r_path / "01_raw_originals"
                if raw_dir.exists():
                    imgs = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.jpeg")) + list(raw_dir.glob("*.png"))
                    if len(imgs) == 0:
                        sz_batch = sum(f.stat().st_size for f in r_path.rglob("*") if f.is_file())
                        if not dry_run:
                            shutil.rmtree(str(r_path), ignore_errors=True)
                        stats["purged_corrupt_batches"] += 1
                        stats["bytes_freed"] += sz_batch
    return stats


def purge_legacy_samples(extra_paths: Optional[List[Any]] = None, dry_run: bool = False) -> int:
    stats = purge_synthetic_and_corrupt_artifacts(extra_paths=extra_paths, dry_run=dry_run)
    return int(stats["purged_legacy_files"])


def prune_pentest_cert_bloat() -> int:
    from benchmark.deepfake_memory import prune_automated_pentest_certificates
    return prune_automated_pentest_certificates()


def main():
    parser = argparse.ArgumentParser(description="Unified Data Hygiene & Storage Purge CLI")
    parser.add_argument("action", choices=["all", "synthetic-media", "corrupt-artifacts", "legacy-seeds", "prune-db-certs"], default="all", nargs="?")
    parser.add_argument("--drive-mount", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    out: Dict[str, Any] = {}
    if args.action in ("all", "synthetic-media"):
        out["synthetic_media"] = purge_synthetic_media(extra_mount=args.drive_mount, dry_run=args.dry_run, target_date=args.date)
    if args.action in ("all", "corrupt-artifacts", "legacy-seeds"):
        out["corrupt_artifacts"] = purge_synthetic_and_corrupt_artifacts(extra_paths=[args.drive_mount] if args.drive_mount else None, dry_run=args.dry_run)
    if args.action in ("all", "prune-db-certs") and not args.dry_run:
        out["pruned_db_pentest_certs"] = prune_pentest_cert_bloat()
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
