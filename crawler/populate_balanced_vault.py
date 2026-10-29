import os
import sys
import io
import time
import urllib.request
import urllib.parse
from pathlib import Path
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from crawler.open_harvester import query_wikimedia_search_items, download_and_validate_portrait
from crawler.dedup_engine import PerceptualDeduplicationRegistry
from crawler.media_generator import VIETNAMESE_PORTRAIT_POOLS

VAULT_DIR = PROJECT_ROOT / "crawler" / "vietnamese_face_vault"
SEEDS_DIR = PROJECT_ROOT / "crawler" / "verified_threads_seeds"
REGISTRY_PATH = PROJECT_ROOT / "crawler" / "dedup_registry.json"

CATEGORIES = [
    "vn_id_official",
    "vn_selfie_lifestyle",
    "vn_street_outdoor",
    "vn_occlusion_glasses_mask",
    "vn_mature_aging"
]

print("=== BẮT ĐẦU NẠP BỔ SUNG CÂN BẰNG 5 BỘ DỮ LIỆU IN-THE-WILD VIỆT NAM ===")

for cat in CATEGORIES:
    (VAULT_DIR / cat).mkdir(parents=True, exist_ok=True)

registry = PerceptualDeduplicationRegistry(REGISTRY_PATH)

# 1. Nạp từ Curated Unsplash CDN Pools
print("\n[Bước 1] Nạp từ Curated High-Res Pools (Unsplash CDN)...")
for cat, pool in VIETNAMESE_PORTRAIT_POOLS.items():
    cat_dir = VAULT_DIR / cat
    current_count = len(list(cat_dir.glob("*.jpg")))
    print(f" -> Danh mục {cat}: Hiện có {current_count} ảnh")
    
    for idx, photo_id in enumerate(pool):
        dest_filename = f"curated_vn_{cat}_{photo_id}.jpg"
        dest_path = cat_dir / dest_filename
        if dest_path.exists():
            continue
            
        url = f"https://images.unsplash.com/photo-{photo_id}?w=512&h=512&fit=crop&crop=faces"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw_bytes = resp.read()
            if len(raw_bytes) > 10000:
                with open(dest_path, "wb") as f:
                    f.write(raw_bytes)
                # Copy sang seeds
                seed_dest = SEEDS_DIR / f"seed_curated_vn_{cat}_{photo_id[:10]}.jpg"
                if not seed_dest.exists():
                    with open(seed_dest, "wb") as sf:
                        sf.write(raw_bytes)
                print(f"    + Đã lưu ảnh curated: {dest_filename}")
        except Exception as e:
            print(f"    ! Lỗi tải curated {photo_id}: {e}")

# 2. Nạp thêm từ Wikimedia Commons bằng các truy vấn chuyên biệt
print("\n[Bước 2] Nạp từ Wikimedia Commons bằng từ khóa mục tiêu...")
TARGET_QUERIES = {
    "vn_id_official": [
        "Vietnamese official portrait",
        "Vietnamese politician portrait",
        "Vietnamese professor portrait"
    ],
    "vn_street_outdoor": [
        "Vietnamese street vendor portrait",
        "People of Vietnam outdoor street",
        "People in Hanoi street"
    ],
    "vn_occlusion_glasses_mask": [
        "Vietnamese person glasses",
        "Vietnamese wearing glasses",
        "Vietnamese wearing mask"
    ],
    "vn_mature_aging": [
        "Elderly Vietnamese person portrait",
        "Old Vietnamese person",
        "Senior citizen Vietnam"
    ]
}

for cat, q_list in TARGET_QUERIES.items():
    cat_dir = VAULT_DIR / cat
    existing = len(list(cat_dir.glob("*.jpg")))
    print(f"\n* Tìm kiếm cho {cat} (Hiện có: {existing} ảnh)...")
    if existing >= 15:
        print(f"  Đã đủ số lượng tối thiểu cho {cat}.")
        continue
        
    for q in q_list:
        items = query_wikimedia_search_items(q, limit=8)
        print(f"  - Truy vấn '{q}': Tìm thấy {len(items)} kết quả")
        for item in items:
            known_h = registry.get_known_dhashes()
            res = download_and_validate_portrait(item, known_h)
            if res and res.get("image_bytes"):
                is_dup, _, _ = registry.check_and_register(
                    res["image_bytes"],
                    filename=item.get("title", ""),
                    category=cat,
                    source="commons.wikimedia.org",
                    sha256=res["sha256"]
                )
                if is_dup:
                    continue
                sha_short = res["sha256"][:10]
                dest_fn = f"wiki_vn_{cat}_{sha_short}.jpg"
                dest_p = cat_dir / dest_fn
                if not dest_p.exists():
                    with open(dest_p, "wb") as f:
                        f.write(res["image_bytes"])
                    seed_dest = SEEDS_DIR / f"seed_wiki_vn_{cat}_{sha_short}.jpg"
                    if not seed_dest.exists():
                        with open(seed_dest, "wb") as sf:
                            sf.write(res["image_bytes"])
                    print(f"    + Lưu thành công: {dest_fn}")

registry.save()

print("\n=== HOÀN TẤT NẠP BỘ DỮ LIỆU IN-THE-WILD VIỆT NAM ===")
