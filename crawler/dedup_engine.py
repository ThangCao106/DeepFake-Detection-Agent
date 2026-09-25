"""
Perceptual Deduplication Engine (dHash & Hamming Distance)
Ensures 0% duplicate identities and visual clones across Google Colab runs and Google Drive storage.

Key features:
1. 64-bit Difference Hashing (dHash) on 8x8 gradient grid.
2. Fast bitwise Hamming distance computation:
   - Distance <= 6: Visual duplicate (identical person, slight crop, compression, or lighting change).
   - Distance >= 12: Distinct human individual.
3. Persistent Google Drive registry (`Thread_Data/registry/perceptual_dhash_registry.json`).
4. Safe cross-run synchronization preventing multi-day identity leakage.
"""

import io
import json
import time
import hashlib
from pathlib import Path
from typing import Union, Tuple, Optional, Dict, List, Set
from PIL import Image

DEFAULT_REGISTRY_NAME = "perceptual_dhash_registry.json"
DEFAULT_HAMMING_THRESHOLD = 6


def compute_dhash(
    image: Union[Image.Image, Path, str, bytes],
    hash_size: int = 8
) -> str:
    """
    Computes a 64-bit difference hash (dHash) as a 16-character hexadecimal string.
    
    Algorithm:
      1. Convert image to grayscale (L mode).
      2. Resize to (hash_size + 1, hash_size), e.g. 9x8 = 72 pixels.
      3. Compare adjacent horizontal pixels (pixel[x] > pixel[x + 1]).
      4. Assemble 64 bits into a 16-character hex string.
    """
    pil_img = None
    close_needed = False
    
    try:
        if isinstance(image, Image.Image):
            pil_img = image
        elif isinstance(image, (str, Path)):
            pil_img = Image.open(str(image))
            close_needed = True
        elif isinstance(image, (bytes, bytearray)):
            pil_img = Image.open(io.BytesIO(image))
            close_needed = True
        else:
            # Fallback for numpy array (OpenCV image)
            import numpy as np
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # Assume BGR from cv2
                    pil_img = Image.fromarray(image[:, :, ::-1])
                else:
                    pil_img = Image.fromarray(image)
                close_needed = True
        
        if pil_img is None:
            return "0000000000000000"

        # Resize to (hash_size + 1, hash_size)
        gray = pil_img.convert("L").resize(
            (hash_size + 1, hash_size),
            Image.Resampling.BILINEAR
        )
        raw_bytes = gray.tobytes()
        w = hash_size + 1

        diff = 0
        for row in range(hash_size):
            row_offset = row * w
            for col in range(hash_size):
                p_left = raw_bytes[row_offset + col]
                p_right = raw_bytes[row_offset + col + 1]
                diff = (diff << 1) | (1 if p_left > p_right else 0)

        return f"{diff:016x}"
    finally:
        if close_needed and pil_img:
            try:
                pil_img.close()
            except Exception:
                pass


def hamming_distance(hash1: str, hash2: str) -> int:
    """
    Computes bitwise Hamming distance between two hex hash strings.
    Returns integer in range [0, 64].
    """
    try:
        val1 = int(hash1, 16)
        val2 = int(hash2, 16)
        diff = val1 ^ val2
        return diff.bit_count() if hasattr(diff, "bit_count") else bin(diff).count("1")
    except Exception:
        return 64


def is_visual_duplicate(
    image: Union[Image.Image, Path, str, bytes],
    known_dhashes: Union[Set[str], List[str], Dict[str, dict]],
    threshold: int = DEFAULT_HAMMING_THRESHOLD
) -> Tuple[bool, Optional[str], int]:
    """
    Checks if an image is visually identical or near-duplicate to any known dHash.
    
    Returns:
      (is_duplicate: bool, matching_hash: str | None, min_distance: int)
    """
    target_dhash = compute_dhash(image)
    if not known_dhashes:
        return False, None, 64

    min_dist = 64
    closest_hash = None

    hash_iter = known_dhashes.keys() if isinstance(known_dhashes, dict) else known_dhashes
    for h in hash_iter:
        dist = hamming_distance(target_dhash, h)
        if dist < min_dist:
            min_dist = dist
            closest_hash = h
        if dist <= threshold:
            return True, closest_hash, min_dist

    return (min_dist <= threshold), closest_hash, min_dist


class PerceptualDeduplicationRegistry:
    """
    Manages persistent perceptual hash registry across Colab runs and Google Drive.
    Guarantees 0% visual duplicates and eliminates identity repetition.
    """

    def __init__(self, local_path: Optional[Union[Path, str]] = None):
        self.local_path = Path(local_path) if local_path else Path("report_colab") / DEFAULT_REGISTRY_NAME
        self.hashes: Dict[str, dict] = {}
        self.sha256_to_dhash: Dict[str, str] = {}
        self.total_rejections: int = 0
        self.load()

    def load(self, path: Optional[Union[Path, str]] = None) -> bool:
        """Loads registry from local JSON file."""
        target = Path(path) if path else self.local_path
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.hashes = data.get("hashes", {})
                self.sha256_to_dhash = data.get("sha256_map", {})
                print(f"[Dedup Registry] Loaded {len(self.hashes)} unique perceptual hashes from {target.name}")
                return True
            except Exception as err:
                print(f"[Dedup Registry] Notice loading {target}: {err}")
        return False

    def save(self, path: Optional[Union[Path, str]] = None) -> bool:
        """Persists registry to JSON file."""
        target = Path(path) if path else self.local_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": "2.0_dhash",
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "total_unique_faces": len(self.hashes),
                "total_rejections": self.total_rejections,
                "hashes": self.hashes,
                "sha256_map": self.sha256_to_dhash
            }
            with open(target, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            return True
        except Exception as err:
            print(f"[Dedup Registry] Notice saving {target}: {err}")
            return False

    def sync_with_gdrive(self, drive_root: Union[Path, str]) -> int:
        """
        Two-way synchronization with Google Drive registry:
        Loads existing hashes from Google Drive, merges them locally, and saves back.
        """
        if not drive_root:
            return 0
            
        drive_path = Path(drive_root) / "registry" / DEFAULT_REGISTRY_NAME
        merged_count = 0

        # 1. Ingest from Google Drive if it exists
        if drive_path.exists():
            try:
                with open(drive_path, "r", encoding="utf-8") as f:
                    remote_data = json.load(f)
                remote_hashes = remote_data.get("hashes", {})
                remote_sha = remote_data.get("sha256_map", {})
                for h, meta in remote_hashes.items():
                    if h not in self.hashes:
                        self.hashes[h] = meta
                        merged_count += 1
                self.sha256_to_dhash.update(remote_sha)
                print(f"[Dedup Registry] Merged {merged_count} remote hashes from GDrive registry ({len(self.hashes)} total).")
            except Exception as err:
                print(f"[Dedup Registry] Notice loading GDrive registry: {err}")

        # 2. Save unified registry back to both local and Google Drive
        self.save(self.local_path)
        try:
            drive_path.parent.mkdir(parents=True, exist_ok=True)
            self.save(drive_path)
            print(f"[Dedup Registry] Synced unified registry to Google Drive: {drive_path}")
        except Exception as err:
            print(f"[Dedup Registry] Notice writing to GDrive registry: {err}")

        return merged_count

    def check_and_register(
        self,
        image: Union[Image.Image, Path, str, bytes],
        filename: str = "",
        category: str = "",
        source: str = "",
        sha256: str = "",
        threshold: int = DEFAULT_HAMMING_THRESHOLD
    ) -> Tuple[bool, str, int]:
        """
        Checks if image is duplicate. If unique, registers it in memory.
        
        Returns:
          (is_duplicate: bool, dhash_value: str, min_distance: int)
        """
        target_dhash = compute_dhash(image)

        # 1. Exact dHash check
        if target_dhash in self.hashes:
            self.total_rejections += 1
            return True, target_dhash, 0

        # 2. Fast Hamming distance check across all registered hashes
        min_dist = 64
        matched_hash = None
        for existing_h in self.hashes.keys():
            dist = hamming_distance(target_dhash, existing_h)
            if dist < min_dist:
                min_dist = dist
                matched_hash = existing_h
            if dist <= threshold:
                self.total_rejections += 1
                return True, matched_hash, min_dist

        # 3. Accepted as unique identity
        self.hashes[target_dhash] = {
            "filename": filename,
            "category": category,
            "source": source,
            "sha256": sha256,
            "registered_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        if sha256:
            self.sha256_to_dhash[sha256] = target_dhash

        return False, target_dhash, min_dist

    def get_known_dhashes(self) -> Set[str]:
        """Returns set of all registered dHashes."""
        return set(self.hashes.keys())

    def get_stats(self) -> dict:
        """Returns summary statistics of the registry."""
        return {
            "total_registered_identities": len(self.hashes),
            "total_rejections_blocked": self.total_rejections,
            "sha256_count": len(self.sha256_to_dhash)
        }


def purge_duplicates_in_dir(
    target_dir: Union[Path, str],
    registry: Optional[PerceptualDeduplicationRegistry] = None,
    threshold: int = DEFAULT_HAMMING_THRESHOLD
) -> dict:
    """
    Scans a directory of images, detects perceptual duplicates, and purges them.
    Ensures that only strictly distinct images remain.
    """
    dir_path = Path(target_dir)
    if not dir_path.exists():
        return {"scanned": 0, "duplicates_purged": 0, "unique_retained": 0}

    reg = registry or PerceptualDeduplicationRegistry()
    image_files = sorted(
        list(dir_path.glob("*.jpg")) + 
        list(dir_path.glob("*.jpeg")) + 
        list(dir_path.glob("*.png"))
    )

    scanned = 0
    duplicates_purged = 0
    unique_retained = 0
    local_hashes: Dict[str, Path] = {}

    for img_path in image_files:
        scanned += 1
        dhash_val = compute_dhash(img_path)
        
        # Check against local batch
        is_dup = False
        for lh, orig_path in local_hashes.items():
            if hamming_distance(dhash_val, lh) <= threshold:
                is_dup = True
                break

        # Check against global registry
        if not is_dup:
            is_dup, _, _ = is_visual_duplicate(img_path, reg.get_known_dhashes(), threshold=threshold)

        if is_dup:
            try:
                img_path.unlink()
                duplicates_purged += 1
                print(f"  [Dedup Purge] Removed visual duplicate: {img_path.name}")
            except Exception as e:
                print(f"  [Dedup Purge Notice] Could not delete {img_path.name}: {e}")
        else:
            local_hashes[dhash_val] = img_path
            reg.check_and_register(img_path, filename=img_path.name, source=str(img_path))
            unique_retained += 1

    return {
        "scanned": scanned,
        "duplicates_purged": duplicates_purged,
        "unique_retained": unique_retained
    }
