# -*- coding: utf-8 -*-
"""
Social Media & Web Stream URL Downloader Module
PhD Research & In-The-Wild Digital Forensic Deepfake Detection Agent
Supports automated link ingestion from TikTok, Instagram Reels, YouTube Shorts,
Facebook Watch, Threads, Twitter/X, and direct media URLs (image/video/audio).
"""

import os
import re
import time
import uuid
import logging
import urllib.parse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import requests

try:
    import yt_dlp
    HAS_YT_DLP = True
except ImportError:
    HAS_YT_DLP = False

logger = logging.getLogger("url_downloader")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

MAX_FILESIZE_BYTES = 100 * 1024 * 1024  # 100 MB Limit for in-the-wild media
SOCKET_TIMEOUT_SECS = 30
USER_AGENT_DESKTOP = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

VIDEO_EXTS = {".mp4", ".webm", ".mkv", ".mov", ".avi", ".flv", ".m4v"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".aac", ".m4a", ".wma"}


def detect_platform(url: str) -> str:
    """Identify the source platform of the incoming link."""
    u = url.lower()
    if "tiktok.com" in u:
        return "TikTok"
    elif "instagram.com" in u:
        return "Instagram"
    elif "youtube.com" in u or "youtu.be" in u:
        return "YouTube"
    elif "facebook.com" in u or "fb.watch" in u:
        return "Facebook"
    elif "threads.net" in u:
        return "Threads"
    elif "twitter.com" in u or "x.com" in u:
        return "X/Twitter"
    elif "reddit.com" in u or "v.redd.it" in u:
        return "Reddit"
    return "WebMedia"


def get_media_type_from_filename(filename: str) -> str:
    """Classify media type into image, video, or audio based on filename."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in VIDEO_EXTS:
        return "video"
    elif ext in IMAGE_EXTS:
        return "image"
    elif ext in AUDIO_EXTS:
        return "audio"
    return "image"


def is_direct_media_url(url: str) -> bool:
    """Check if URL points directly to a known media file extension."""
    parsed = urllib.parse.urlparse(url)
    clean_path = parsed.path.lower()
    all_exts = VIDEO_EXTS | IMAGE_EXTS | AUDIO_EXTS
    return any(clean_path.endswith(ext) for ext in all_exts)


def download_direct_media(url: str, dest_dir: str) -> Dict[str, Any]:
    """
    Download direct media URL via HTTP streaming with strict size & timeout guards.
    """
    os.makedirs(dest_dir, exist_ok=True)
    platform = detect_platform(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_id = uuid.uuid4().hex[:6]

    headers = {
        "User-Agent": USER_AGENT_DESKTOP,
        "Accept": "*/*",
        "Referer": url
    }

    try:
        # Pre-flight HEAD or GET stream
        with requests.get(url, headers=headers, stream=True, timeout=SOCKET_TIMEOUT_SECS, allow_redirects=True) as resp:
            resp.raise_for_status()

            # Check content length
            content_length = resp.headers.get("content-length")
            if content_length and int(content_length) > MAX_FILESIZE_BYTES:
                return {
                    "success": False,
                    "error": f"Tệp vượt quá giới hạn dung lượng cho phép ({int(content_length)/(1024*1024):.1f}MB > {MAX_FILESIZE_BYTES/(1024*1024):.0f}MB)."
                }

            # Determine extension
            content_type = resp.headers.get("content-type", "").lower()
            ext = ""
            parsed_path = urllib.parse.urlparse(resp.url).path
            for candidate in (VIDEO_EXTS | IMAGE_EXTS | AUDIO_EXTS):
                if parsed_path.lower().endswith(candidate):
                    ext = candidate
                    break

            if not ext:
                if "video/mp4" in content_type: ext = ".mp4"
                elif "video/webm" in content_type: ext = ".webm"
                elif "video/" in content_type: ext = ".mp4"
                elif "image/png" in content_type: ext = ".png"
                elif "image/jpeg" in content_type or "image/jpg" in content_type: ext = ".jpg"
                elif "image/webp" in content_type: ext = ".webp"
                elif "audio/mpeg" in content_type or "audio/mp3" in content_type: ext = ".mp3"
                elif "audio/wav" in content_type or "audio/x-wav" in content_type: ext = ".wav"
                else: ext = ".jpg"  # default fallback

            filename = f"social_{timestamp}_{platform.lower()}_{rand_id}{ext}"
            filepath = os.path.join(dest_dir, filename)

            downloaded = 0
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if chunk:
                        downloaded += len(chunk)
                        if downloaded > MAX_FILESIZE_BYTES:
                            f.close()
                            if os.path.exists(filepath):
                                os.remove(filepath)
                            return {
                                "success": False,
                                "error": f"Dung lượng vượt quá {MAX_FILESIZE_BYTES/(1024*1024):.0f}MB trong khi tải."
                            }
                        f.write(chunk)

            media_type = get_media_type_from_filename(filename)
            return {
                "success": True,
                "filename": filename,
                "filepath": os.path.abspath(filepath),
                "media_type": media_type,
                "source_url": url,
                "platform": platform,
                "title": os.path.basename(parsed_path) or f"Media from {platform}",
                "author": platform,
                "duration": None,
                "thumbnail_url": None,
                "filesize_bytes": downloaded
            }

    except Exception as e:
        logger.error(f"Direct media download failed: {e}")
        return {
            "success": False,
            "error": f"Lỗi khi tải tệp trực tiếp: {str(e)}"
        }


def download_with_ytdlp(url: str, dest_dir: str) -> Dict[str, Any]:
    """
    Download video or audio using yt-dlp across TikTok, Reels, Shorts, Threads, etc.
    """
    if not HAS_YT_DLP:
        return {
            "success": False,
            "error": "yt-dlp chưa được cài đặt trên hệ thống."
        }

    os.makedirs(dest_dir, exist_ok=True)
    platform = detect_platform(url)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rand_id = uuid.uuid4().hex[:6]
    out_tmpl = os.path.join(dest_dir, f"social_{timestamp}_{platform.lower()}_{rand_id}_%(id)s.%(ext)s")

    ydl_opts = {
        "outtmpl": out_tmpl,
        # Prefer pre-merged mp4 or best single mp4 file (no mandatory ffmpeg merge needed)
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "max_filesize": MAX_FILESIZE_BYTES,
        "socket_timeout": SOCKET_TIMEOUT_SECS,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "user_agent": USER_AGENT_DESKTOP,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Extracting media via yt-dlp from: {url}")
            info = ydl.extract_info(url, download=True)
            if not info:
                return {
                    "success": False,
                    "error": "Không thể trích xuất luồng media từ đường dẫn được cung cấp."
                }

            # If it's a playlist/multi-item, get first entry
            if "entries" in info and info["entries"]:
                info = info["entries"][0]

            expected_filename = ydl.prepare_filename(info)
            actual_filepath = expected_filename

            # If expected_filename not on disk (e.g. extension changed to mp4), search directory
            if not os.path.exists(actual_filepath):
                base_part = f"social_{timestamp}_{platform.lower()}_{rand_id}"
                matches = [f for f in os.listdir(dest_dir) if f.startswith(base_part)]
                if matches:
                    actual_filepath = os.path.join(dest_dir, matches[0])
                else:
                    return {
                        "success": False,
                        "error": "Tệp đã được tải nhưng không tìm thấy trên đĩa lưu trữ."
                    }

            filename = os.path.basename(actual_filepath)
            media_type = get_media_type_from_filename(filename)
            filesize = os.path.getsize(actual_filepath) if os.path.exists(actual_filepath) else 0

            return {
                "success": True,
                "filename": filename,
                "filepath": os.path.abspath(actual_filepath),
                "media_type": media_type,
                "source_url": url,
                "platform": platform,
                "title": info.get("title") or f"{platform} Video",
                "author": info.get("uploader") or info.get("channel") or info.get("creator") or platform,
                "duration": info.get("duration"),
                "thumbnail_url": info.get("thumbnail"),
                "filesize_bytes": filesize
            }

    except Exception as e:
        logger.warning(f"yt-dlp extraction failed for {url}: {e}")
        # Try fallback to direct streaming download if feasible
        fallback_res = download_direct_media(url, dest_dir)
        if fallback_res.get("success"):
            return fallback_res

        clean_err = str(e)
        if "Unsupported URL" in clean_err:
            clean_err = "Nền tảng hoặc định dạng liên kết chưa được hỗ trợ tự động bóc tách luồng."
        return {
            "success": False,
            "error": f"Lỗi bóc tách phương tiện: {clean_err}"
        }


def download_media_from_url(url: str, dest_dir: str) -> Dict[str, Any]:
    """
    Main entry point for unified URL ingestion.
    Supports TikTok, Instagram Reels, YouTube Shorts, Threads, FB, and direct URLs.
    """
    if not url or not isinstance(url, str):
        return {"success": False, "error": "URL không hợp lệ hoặc để trống."}

    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return {"success": False, "error": "URL phải bắt đầu bằng http:// hoặc https://"}

    # Route 1: If direct media file, download immediately via streaming
    if is_direct_media_url(url):
        logger.info(f"Handling direct media URL: {url}")
        res = download_direct_media(url, dest_dir)
        if res.get("success"):
            return res

    # Route 2: Social media / complex page via yt-dlp
    return download_with_ytdlp(url, dest_dir)

