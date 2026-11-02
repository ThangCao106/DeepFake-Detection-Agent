"""
Dynamic Trending Keyword & Hashtag Discovery Engine for Vietnam Media Crawlers.
Discovers and queues high-yield, authentic Vietnamese topics across:
  - Meta Threads (threads.net)
  - Instagram (instagram.com)
  - Facebook (facebook.com)
  - Wikimedia Commons & Open Web

Key Capabilities:
  1. Core Curated Topics: viral, thinhhanh, mauanh, caulong, tennis, running, gym, pickleball, etc.
  2. Live Google Trends Vietnam (RSS geo=VN): Real-time daily hot topics in Vietnam.
  3. In-flight Hashtag Extractor: Dynamically mines hashtags from post captions during crawling.
"""

import re
import sys
import json
import time
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
from typing import List, Set, Dict, Optional

# Core Curated Seed Topics (Sports, Fitness, Visual, Lifestyle, Youth)
CORE_CURATED_KEYWORDS = [
    # User-requested topics
    "viral",
    "thinhhanh",
    "mauanh",
    "caulong",
    "tennis",
    "running",
    "gym",
    # Sports & Fitness
    "pickleball",
    "chaybo",
    "marathon",
    "fitness",
    "yoga",
    "pilates",
    "badminton",
    "thethao",
    "gymvietnam",
    "runningvietnam",
    # Lifestyle & Visual Trends
    "checkin",
    "ootd",
    "dulich",
    "chupanh",
    "selfie",
    "saigonlife",
    "hanoilife",
    "dananglife",
    "streetstyle",
    "vibe",
    "aodai",
    "chandung",
    "portraitvn",
    "travelvietnam",
    # Demographic & Youth
    "xuhuong",
    "trend",
    "genz",
    "sinhvien",
    "congso",
    "nguoivietnam",
    "vietnamesegirl",
    "vietnameseboy",
    "vietnam"
]

# Curated Instagram Explore & Profile Targets
EXPANDED_INSTAGRAM_TARGETS = [
    # Sport & Fitness Tags
    "https://www.instagram.com/explore/tags/caulong/",
    "https://www.instagram.com/explore/tags/chaybo/",
    "https://www.instagram.com/explore/tags/runningvietnam/",
    "https://www.instagram.com/explore/tags/gymvietnam/",
    "https://www.instagram.com/explore/tags/tennisvietnam/",
    "https://www.instagram.com/explore/tags/pickleballvietnam/",
    # Lifestyle & Viral Tags
    "https://www.instagram.com/explore/tags/viral/",
    "https://www.instagram.com/explore/tags/thinhhanh/",
    "https://www.instagram.com/explore/tags/mauanh/",
    "https://www.instagram.com/explore/tags/ootdvietnam/",
    "https://www.instagram.com/explore/tags/saigonstreetstyle/",
    "https://www.instagram.com/explore/tags/checkinvietnam/",
    "https://www.instagram.com/explore/tags/selfievietnam/",
    "https://www.instagram.com/explore/tags/saigon/",
    "https://www.instagram.com/explore/tags/hanoi/",
    "https://www.instagram.com/explore/tags/danang/",
    # Top Public Creators / Photography Accounts
    "https://www.instagram.com/humansofvietnam/",
    "https://www.instagram.com/humansofsaigon/",
    "https://www.instagram.com/humansofhanoi.official/",
    "https://www.instagram.com/vietnamtravelers/",
    "https://www.instagram.com/vietnam.portraits/",
    "https://www.instagram.com/chuyen.cua.hanoi/",
    "https://www.instagram.com/streetphotography.vietnam/",
    "https://www.instagram.com/saigonese.lifestyle/",
    "https://www.instagram.com/vietnam_streetphotography/"
]

# Curated Facebook Public Mobile Targets (m.facebook.com - lightweight and fast)
EXPANDED_FACEBOOK_TARGETS = [
    # Hashtags
    "https://m.facebook.com/hashtag/caulong",
    "https://m.facebook.com/hashtag/chaybo",
    "https://m.facebook.com/hashtag/gymvietnam",
    "https://m.facebook.com/hashtag/mauanh",
    "https://m.facebook.com/hashtag/thinhhanh",
    "https://m.facebook.com/hashtag/viral",
    "https://m.facebook.com/hashtag/tennis",
    "https://m.facebook.com/hashtag/pickleball",
    "https://m.facebook.com/hashtag/checkinvietnam",
    "https://m.facebook.com/hashtag/streetphotographyvietnam",
    # Public Pages Photo Albums
    "https://m.facebook.com/humansofhanoi.page/photos",
    "https://m.facebook.com/humansofsaigonvn/photos",
    "https://m.facebook.com/chuyencuahanoi.official/photos",
    "https://m.facebook.com/VietnamTravelersOfficial/photos",
    "https://m.facebook.com/streetphotographyvietnam/photos",
    "https://m.facebook.com/nguoisaigon.page/photos"
]


def strip_accents_slug(text: str) -> str:
    """Converts Vietnamese string into clean ascii alphanumeric slug for hashtags."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"[^a-zA-Z0-9]+", "", text.lower())
    return text


def fetch_google_trends_vietnam(max_items: int = 15, timeout: int = 8) -> List[str]:
    """
    Fetches real-time trending search queries from Google Trends Vietnam RSS.
    Returns sanitized keywords suitable for hashtags and search queries.
    """
    rss_url = "https://trends.google.com/trending/rss?geo=VN"
    trending_keywords: List[str] = []
    
    try:
        req = urllib.request.Request(
            rss_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            xml_bytes = resp.read()
            
        root = ET.fromstring(xml_bytes)
        for item in root.findall(".//item")[:max_items]:
            title_node = item.find("title")
            if title_node is not None and title_node.text:
                raw_title = title_node.text.strip()
                # 1. Clean slug for hashtag
                slug = strip_accents_slug(raw_title)
                if len(slug) >= 3 and slug not in trending_keywords:
                    trending_keywords.append(slug)
                    
        print(f"🔥 [Google Trends VN] Discovered {len(trending_keywords)} real-time trending topics in Vietnam.")
    except Exception as e:
        print(f"⚠️ [Google Trends VN Notice] Could not fetch live trends: {e}. Using curated seeds.")
        
    return trending_keywords


def extract_hashtags_from_text(text: str) -> List[str]:
    """Extracts valid Vietnamese and English hashtags from a block of text."""
    if not text:
        return []
    matches = re.findall(r"#([a-zA-Z0-9_À-ỹ]+)", text)
    cleaned = []
    for m in matches:
        t = strip_accents_slug(m)
        if 3 <= len(t) <= 30 and not t.isdigit():
            cleaned.append(t)
    return cleaned


class TrendingKeywordEngine:
    """
    Dynamic keyword engine that manages seeds, real-time Google trends,
    and in-flight crawled hashtags for Threads, Instagram, and Facebook.
    """
    def __init__(self, seed_keywords: Optional[List[str]] = None):
        self.seen_keywords: Set[str] = set()
        self.queue: List[str] = []
        
        # 1. Initialize with core curated topics
        base_seeds = seed_keywords or CORE_CURATED_KEYWORDS
        for kw in base_seeds:
            clean_kw = strip_accents_slug(kw)
            if clean_kw and clean_kw not in self.seen_keywords:
                self.seen_keywords.add(clean_kw)
                self.queue.append(clean_kw)
                
        # 2. Try fetching live Google Trends
        try:
            live_trends = fetch_google_trends_vietnam(max_items=12)
            for tk in live_trends:
                if tk not in self.seen_keywords:
                    self.seen_keywords.add(tk)
                    self.queue.append(tk)
        except Exception:
            pass

    def add_discovered_tags(self, tags: List[str]):
        """Dynamically appends new trending tags mined from visited posts."""
        added = 0
        for tag in tags:
            clean_t = strip_accents_slug(tag)
            if 3 <= len(clean_t) <= 25 and not clean_t.isdigit() and clean_t not in self.seen_keywords:
                self.seen_keywords.add(clean_t)
                self.queue.append(clean_t)
                added += 1
        if added > 0:
            print(f"  ✨ [Trend Discovery] Enqueued {added} newly discovered hashtags (Queue size: {len(self.queue)})")

    def get_all_keywords(self) -> List[str]:
        """Returns ordered list of all registered keywords."""
        return list(self.queue)

    def get_tag_urls_for_threads(self, limit: int = 25) -> List[str]:
        """Generates list of target tag URLs on threads.net."""
        return [f"https://www.threads.net/tag/{kw}" for kw in self.queue[:limit]]

    def get_search_urls_for_threads(self, limit: int = 15) -> List[str]:
        """Generates search query URLs on threads.net."""
        return [f"https://www.threads.net/search?q={kw}" for kw in self.queue[:limit]]

    def get_instagram_tag_urls(self, limit: int = 20) -> List[str]:
        """Generates Instagram explore tag URLs."""
        return [f"https://www.instagram.com/explore/tags/{kw}/" for kw in self.queue[:limit]]

    def get_facebook_hashtag_urls(self, limit: int = 20) -> List[str]:
        """Generates Facebook mobile hashtag URLs."""
        return [f"https://m.facebook.com/hashtag/{kw}" for kw in self.queue[:limit]]
