"""
Multi-Source Open Harvester for Authentic Vietnamese Human Portraits (Option B & C).
Fetches genuine, high-resolution photographic portraits of Vietnamese people from
Wikimedia Commons (40+ categories + full-text search) and open media archives.

Enforces:
  - 100% genuine real human photographs (zero cartoons, zero 2D drawings, zero hijab/landmark crops)
  - Taxonomic categorization into 5 core project categories:
      1. vn_id_official
      2. vn_selfie_lifestyle
      3. vn_street_outdoor
      4. vn_occlusion_glasses_mask
      5. vn_mature_aging
  - 64-bit dHash perceptual deduplication (Hamming distance >= 8)
  - Multi-threaded concurrent downloading for ultra-fast ingestion
"""

import os
import io
import re
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import urllib.request
import urllib.parse
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.dedup_engine import (
    compute_dhash,
    hamming_distance,
    is_visual_duplicate,
    PerceptualDeduplicationRegistry,
    DEFAULT_HAMMING_THRESHOLD
)

USER_AGENT = "ThreadsDFBench/2.0 (https://github.com/ThangCao106/DeepFake-Detection-Agent; research_df@domain.vn) python-urllib/3.13"

# 40+ Rich Vietnamese Categories on Wikimedia Commons
WIKIMEDIA_VIETNAMESE_CATEGORIES = [
    # General & Regional People
    "Category:People_of_Vietnam",
    "Category:Vietnamese_people",
    "Category:People_of_Hanoi",
    "Category:People_of_Ho_Chi_Minh_City",
    "Category:People_of_Da_Nang",
    "Category:People_of_Hue",
    "Category:People_of_Hai_Phong",
    "Category:People_of_Can_Tho",
    "Category:People_of_Quang_Nam_province",
    "Category:People_of_Khanh_Hoa_province",
    "Category:People_of_Lam_Dong_province",
    # Demographic / Gender
    "Category:Women_of_Vietnam",
    "Category:Vietnamese_women",
    "Category:Men_of_Vietnam",
    "Category:Vietnamese_men",
    "Category:Youth_in_Vietnam",
    "Category:Students_in_Vietnam",
    "Category:Children_of_Vietnam",
    "Category:Elderly_people_in_Vietnam",
    # Occupations & Daily Life
    "Category:Street_life_in_Vietnam",
    "Category:People_at_markets_in_Vietnam",
    "Category:Vietnamese_teachers",
    "Category:Vietnamese_scientists",
    "Category:Vietnamese_writers",
    "Category:Vietnamese_artists",
    "Category:Vietnamese_singers",
    "Category:Vietnamese_actors",
    "Category:Vietnamese_politicians",
    "Category:Vietnamese_military_people",
    "Category:Athletes_from_Vietnam",
    # Lifestyle & Portraits
    "Category:Portraits_of_people_of_Vietnam",
    "Category:Photographs_of_people_in_Vietnam",
    "Category:People_of_Vietnam_with_bicycles",
    "Category:People_of_Vietnam_on_motorcycles",
    "Category:Crowds_in_Vietnam",
    "Category:People_of_Vietnam_at_festivals",
    "Category:Vietnamese_people_in_traditional_clothing",
    "Category:Ao_dai"
]

# Targeted Wikimedia Search Queries
WIKIMEDIA_SEARCH_QUERIES = [
    "Vietnamese portrait photograph",
    "Người Việt Nam chân dung",
    "Hanoi people portrait",
    "Saigon portrait photograph",
    "Vietnamese woman portrait",
    "Vietnamese man portrait",
    "Vietnamese student portrait",
    "Vietnamese street portrait",
    "Chân dung phụ nữ Việt Nam",
    "Chân dung nam thanh niên Việt Nam"
]

CATEGORY_MAPPINGS = {
    "vn_id_official": ["official", "politician", "scientist", "teacher", "portrait", "id", "military", "suit", "formal", "studio"],
    "vn_selfie_lifestyle": ["selfie", "cafe", "youth", "student", "woman", "girl", "fashion", "ao_dai", "traditional", "festival"],
    "vn_street_outdoor": ["street", "market", "outdoor", "bicycle", "motorcycle", "commute", "crowd", "vendor", "travel"],
    "vn_occlusion_glasses_mask": ["glasses", "eyewear", "specs", "mask", "sunglasses", "occlusion"],
    "vn_mature_aging": ["elderly", "senior", "mature", "aging", "old", "wrinkle", "grandparent", "bac", "ong", "ba"]
}


def infer_vietnamese_category(title: str, category_name: str, caption: str = "") -> str:
    """Infers one of the 5 Vietnamese project categories from metadata text."""
    combined = f"{title} {category_name} {caption}".lower()
    
    if any(k in combined for k in ["elderly", "senior", "mature", "old", "aging", "bà", "ông", "bác"]):
        return "vn_mature_aging"
    if any(k in combined for k in ["glass", "kính", "mask", "specs"]):
        return "vn_occlusion_glasses_mask"
    if any(k in combined for k in ["official", "politician", "scientist", "teacher", "chân dung", "portrait", "military", "formal", "studio"]):
        return "vn_id_official"
    if any(k in combined for k in ["street", "market", "phố", "chợ", "motorcycle", "outdoor", "commute"]):
        return "vn_street_outdoor"
        
    return "vn_selfie_lifestyle"


def query_wikimedia_category_items(cat_name: str, limit: int = 50) -> List[Dict]:
    """Queries Wikimedia API for members of a specific category."""
    clean_cat = cat_name if cat_name.startswith("Category:") else f"Category:{cat_name}"
    api_url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=categorymembers"
        f"&gcmtitle={urllib.parse.quote(clean_cat)}&gcmlimit={limit}&gcmtype=file"
        f"&prop=imageinfo&iiprop=url|size|mime&iiurlwidth=800&format=json"
    )
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        items = []
        for pid, info in pages.items():
            ii = info.get("imageinfo", [{}])[0]
            thumb_url = ii.get("thumburl") or ii.get("url")
            mime = ii.get("mime", "").lower()
            if not thumb_url or not any(ext in mime for ext in ["jpeg", "jpg", "png"]):
                continue
            items.append({
                "title": info.get("title", ""),
                "thumb_url": thumb_url,
                "orig_url": ii.get("url", thumb_url),
                "width": ii.get("width", 800),
                "height": ii.get("height", 800),
                "source_cat": cat_name
            })
        return items
    except Exception as e:
        return []


def query_wikimedia_search_items(query: str, limit: int = 40) -> List[Dict]:
    """Queries Wikimedia Search API for free-text search of media files."""
    api_url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrsearch={urllib.parse.quote(query)}&gsrnamespace=6&gsrlimit={limit}"
        f"&prop=imageinfo&iiprop=url|size|mime&iiurlwidth=800&format=json"
    )
    req = urllib.request.Request(api_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pages = data.get("query", {}).get("pages", {})
        items = []
        for pid, info in pages.items():
            ii = info.get("imageinfo", [{}])[0]
            thumb_url = ii.get("thumburl") or ii.get("url")
            mime = ii.get("mime", "").lower()
            if not thumb_url or not any(ext in mime for ext in ["jpeg", "jpg", "png"]):
                continue
            items.append({
                "title": info.get("title", ""),
                "thumb_url": thumb_url,
                "orig_url": ii.get("url", thumb_url),
                "width": ii.get("width", 800),
                "height": ii.get("height", 800),
                "source_cat": f"search:{query}"
            })
        return items
    except Exception:
        return []


def download_and_validate_portrait(item: Dict, known_dhashes: Set[str]) -> Optional[Dict]:
    """Downloads an image from Wikimedia and validates via Option C photographic quality gate."""
    from crawler.threads_browser_scraper import is_photographic_human_portrait
    
    url = item.get("thumb_url") or item.get("orig_url")
    if not url:
        return None
        
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            raw_bytes = resp.read()
    except Exception:
        # Fallback to orig_url if thumb failed
        try:
            req_orig = urllib.request.Request(item.get("orig_url", url), headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req_orig, timeout=12) as resp_orig:
                raw_bytes = resp_orig.read()
        except Exception:
            return None

    if len(raw_bytes) < 8000:
        return None

    # Quality Gate
    is_photo, reason, stats = is_photographic_human_portrait(raw_bytes)
    if not is_photo:
        return None

    # Deduplication
    is_dup, _, _ = is_visual_duplicate(raw_bytes, known_dhashes, threshold=DEFAULT_HAMMING_THRESHOLD)
    if is_dup:
        return None

    dhash_val = compute_dhash(raw_bytes)
    import hashlib
    sha = hashlib.sha256(raw_bytes).hexdigest()
    
    with Image.open(io.BytesIO(raw_bytes)) as pil_im:
        w, h = pil_im.size

    cat = infer_vietnamese_category(item.get("title", ""), item.get("source_cat", ""))

    return {
        "image_bytes": raw_bytes,
        "sha256": sha,
        "dhash": dhash_val,
        "face_count": stats.get("face_count", 1),
        "primary_bbox": stats.get("primary_bbox", [int(w * 0.2), int(h * 0.15), int(w * 0.8), int(h * 0.75)]),
        "width": w,
        "height": h,
        "post_url": f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(item.get('title', ''))}",
        "author": "wikimedia_commons_contributor",
        "caption": f"Authentic Vietnamese portrait from Wikimedia Commons [{item.get('source_cat', '')}]",
        "keyword": cat,
        "source": "commons.wikimedia.org",
        "category": cat,
        "title": item.get("title", "")
    }


def harvest_open_vietnamese_portraits(
    target_count: int = 250,
    dedup_registry: Optional[PerceptualDeduplicationRegistry] = None,
    max_workers: int = 8,
    max_time_sec: float = 240.0
) -> List[Dict]:
    """
    High-speed, multi-threaded streaming harvester for authentic Vietnamese human portraits.
    Queries categories & searches in rapid batches, immediately validating and downloading portraits.
    Stops as soon as target_count is reached.
    """
    print(f"\n🌐 [Open Multi-Source Harvester] Initiating query across Vietnamese categories (Target: {target_count})...")
    start_time = time.time()
    known_dhashes = set(dedup_registry.get_known_dhashes()) if dedup_registry else set()
    seen_shas = set()
    harvested = []

    # Combine categories and search queries into prioritized list
    sources = []
    for cat in WIKIMEDIA_VIETNAMESE_CATEGORIES:
        sources.append(("category", cat))
    for q in WIKIMEDIA_SEARCH_QUERIES:
        sources.append(("search", q))

    # Shuffle slightly while keeping primary categories near front
    import random
    primary_sources = sources[:8]
    secondary_sources = sources[8:]
    random.shuffle(secondary_sources)
    all_sources = primary_sources + secondary_sources

    download_pool = ThreadPoolExecutor(max_workers=max_workers)
    pending_download_futures = []

    try:
        for s_type, s_val in all_sources:
            if len(harvested) >= target_count or (time.time() - start_time) > max_time_sec:
                break

            # Fetch candidates from this source
            try:
                if s_type == "category":
                    items = query_wikimedia_category_items(s_val, limit=35)
                else:
                    items = query_wikimedia_search_items(s_val, limit=30)
            except Exception:
                items = []

            if not items:
                continue

            # Submit downloads to pool
            for it in items:
                f = download_pool.submit(download_and_validate_portrait, it, known_dhashes)
                pending_download_futures.append(f)

            # Check completed downloads periodically
            still_pending = []
            for f in pending_download_futures:
                if f.done():
                    try:
                        res = f.result()
                        if res:
                            sha = res["sha256"]
                            dhash_val = res["dhash"]
                            if sha not in seen_shas and dhash_val not in known_dhashes:
                                seen_shas.add(sha)
                                known_dhashes.add(dhash_val)
                                if dedup_registry:
                                    dedup_registry.check_and_register(
                                        res["image_bytes"],
                                        filename=res["title"],
                                        source="wikimedia_commons",
                                        sha256=sha
                                    )
                                harvested.append(res)
                                idx = len(harvested)
                                print(f"    📸 [{idx}/{target_count}] Ingested: {res['title'][:35]}... ({res['width']}x{res['height']}, {res['category']})")
                                if len(harvested) >= target_count:
                                    break
                    except Exception:
                        pass
                else:
                    still_pending.append(f)
            pending_download_futures = still_pending

        # Drain remaining pending futures if still needed
        for f in as_completed(pending_download_futures):
            if len(harvested) >= target_count or (time.time() - start_time) > max_time_sec:
                break
            try:
                res = f.result()
                if res:
                    sha = res["sha256"]
                    dhash_val = res["dhash"]
                    if sha not in seen_shas and dhash_val not in known_dhashes:
                        seen_shas.add(sha)
                        known_dhashes.add(dhash_val)
                        if dedup_registry:
                            dedup_registry.check_and_register(
                                res["image_bytes"],
                                filename=res["title"],
                                source="wikimedia_commons",
                                sha256=sha
                            )
                        harvested.append(res)
                        idx = len(harvested)
                        print(f"    📸 [{idx}/{target_count}] Ingested: {res['title'][:35]}... ({res['width']}x{res['height']}, {res['category']})")
            except Exception:
                pass

    finally:
        download_pool.shutdown(wait=False, cancel_futures=True)

    elapsed = time.time() - start_time
    print(f"  🏁 Open Multi-Source Harvester finished in {elapsed:.1f}s: Harvested {len(harvested)} authentic human portraits.")
    return harvested
