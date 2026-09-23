"""
Centralized Configuration Module for Threads Human Media Crawler.
Supports:
  - Option A Core: Flat comma-separated THREADS_KEYWORDS in .env
  - Option C Hybrid: Automated Profiles (quick_inspection, standard_daily, full_pentest, custom)
  - Hierarchical Cascading: CLI Arguments > .env Overrides > Profile Defaults > Safe Fallbacks
"""
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_dotenv_safely(env_path: Path = None):
    """Loads environment variables from .env file without mandatory third-party dependency."""
    if env_path is None:
        env_path = PROJECT_ROOT / ".env"
    
    if not env_path.exists():
        return
        
    try:
        from dotenv import load_dotenv
        load_dotenv(str(env_path))
    except ImportError:
        # Standalone parser fallback if python-dotenv is not installed
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

# Initialize environment from .env
load_dotenv_safely()

# Standard Curated Vietnamese Search Keywords for Meta Threads
DEFAULT_VIETNAMESE_KEYWORDS = [
    "hcm",
    "saigon",
    "vietnam",
    "thesinhvien",
    "selfievietnam",
    "hanoi",
    "nguoivietnam",
    "chandungvietnam",
    "sinhvien",
    "dslrvietnam",
    "duongphovietnam",
    "chandung"
]

CRAWLER_PROFILES = {
    "quick_inspection": {
        "num_images": 25,
        "num_videos": 2,
        "num_audios": 2,
        "num_samples": 25,
        "keywords": ["hcm", "saigon", "vietnam", "thesinhvien", "selfievietnam", "hanoi", "sinhvien"]
    },
    "standard_daily": {
        "num_images": 100,
        "num_videos": 10,
        "num_audios": 10,
        "num_samples": 25,
        "keywords": DEFAULT_VIETNAMESE_KEYWORDS
    },
    "full_pentest": {
        "num_images": 10000,
        "num_videos": 500,
        "num_audios": 500,
        "num_samples": 50,
        "keywords": DEFAULT_VIETNAMESE_KEYWORDS + [
            "hochiminh", "kinhcan", "khautrang", "chupanh", "dulichvietnam", "gioitre"
        ]
    }
}

def parse_keywords_list(raw_input: str) -> list[str]:
    """Cleans and standardizes raw keywords string into list of clean topic strings."""
    if not raw_input or not str(raw_input).strip():
        return []
        
    cleaned = []
    for item in str(raw_input).split(","):
        kw = item.strip().lstrip("#").lower()
        if kw and kw not in cleaned:
            cleaned.append(kw)
    return cleaned

def get_crawler_config() -> dict:
    """
    Resolves active configuration by merging Profile Defaults and .env Overrides.
    Priority:
      1. Explicit environment variables (.env / Colab env):
         - THREADS_KEYWORDS / THREADS_HASHTAGS / THREADS_TAGS
         - THREADS_MAX_IMAGES / CRAWL_NUM_IMAGES
         - THREADS_MAX_VIDEOS / CRAWL_NUM_VIDEOS
         - THREADS_MAX_AUDIOS / CRAWL_NUM_AUDIOS
         - SAMPLE_INSPECTION_COUNT
      2. Profile preset (CRAWL_PROFILE: quick_inspection | standard_daily | full_pentest | custom)
      3. Safe defaults
    """
    profile_name = os.environ.get("CRAWL_PROFILE", "quick_inspection").strip().lower()
    base_profile = CRAWLER_PROFILES.get(profile_name, CRAWLER_PROFILES["quick_inspection"])
    
    # 1. Keywords resolution
    env_kw_raw = (
        os.environ.get("THREADS_KEYWORDS")
        or os.environ.get("THREADS_HASHTAGS")
        or os.environ.get("THREADS_TAGS")
        or ""
    )
    user_keywords = parse_keywords_list(env_kw_raw)
    active_keywords = user_keywords if user_keywords else list(base_profile["keywords"])
    
    # Exclude keywords known to produce non-human retail products (e.g. cccd phone cases)
    blacklisted = {"cccd", "case", "oplung", "cardholder"}
    active_keywords = [k for k in active_keywords if k not in blacklisted]
    if not active_keywords:
        active_keywords = list(DEFAULT_VIETNAMESE_KEYWORDS)

    # 2. Quota resolution
    def _int_env(*keys, default_val):
        for k in keys:
            v = os.environ.get(k)
            if v and str(v).strip().isdigit():
                return int(v.strip())
        return default_val

    num_images = _int_env("THREADS_MAX_IMAGES", "CRAWL_NUM_IMAGES", default_val=base_profile["num_images"])
    num_videos = _int_env("THREADS_MAX_VIDEOS", "CRAWL_NUM_VIDEOS", default_val=base_profile["num_videos"])
    num_audios = _int_env("THREADS_MAX_AUDIOS", "CRAWL_NUM_AUDIOS", default_val=base_profile["num_audios"])
    num_samples = _int_env("SAMPLE_INSPECTION_COUNT", default_val=base_profile["num_samples"])

    return {
        "profile": profile_name,
        "keywords": active_keywords,
        "num_images": num_images,
        "num_videos": num_videos,
        "num_audios": num_audios,
        "num_samples": num_samples,
    }
