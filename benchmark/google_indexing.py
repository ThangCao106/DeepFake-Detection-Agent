"""
Google Search Indexing & Sitemap Ping Engine for Deepfake Forensic Certificates.

Provides automated Googlebot notification via:
1. Google Sitemap Ping API (No authentication required)
2. Google Indexing API v3 (Using Google Cloud Service Account JSON key if configured)
"""

import os
import json
import logging
import urllib.request
import urllib.parse
from datetime import datetime

logger = logging.getLogger(__name__)

DEFAULT_SITEMAP_URL = "https://detection.dtpsoft.vn/sitemap.xml"


def ping_google_sitemap(sitemap_url: str = DEFAULT_SITEMAP_URL) -> dict:
    """
    Google deprecated sitemap pings in late 2023. Skip blocking HTTP call unless explicitly enabled.
    """
    if os.environ.get("ENABLE_SITEMAP_PING", "0") != "1":
        return {"status": "SKIPPED", "message": "Google sitemap ping is deprecated"}

    encoded_url = urllib.parse.quote(sitemap_url, safe="")
    ping_endpoint = f"https://www.google.com/ping?sitemap={encoded_url}"

    try:
        req = urllib.request.Request(
            ping_endpoint,
            headers={"User-Agent": "DeepfakeAgentForensicPublisher/2.0"}
        )
        with urllib.request.urlopen(req, timeout=8) as response:
            status_code = response.getcode()
            logger.info(f"[google_indexing] Google Sitemap Ping successful (HTTP {status_code}) for '{sitemap_url}'.")
            return {
                "status": "SUCCESS",
                "http_code": status_code,
                "ping_url": ping_endpoint,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    except Exception as e:
        logger.warning(f"[google_indexing] Google Sitemap Ping warning: {e}")
        return {
            "status": "WARNING",
            "error": str(e),
            "ping_url": ping_endpoint,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }


def publish_google_indexing_url(cert_url: str, action: str = "URL_UPDATED") -> dict:
    """
    Publish instant indexing notification to Google Indexing API v3.
    Requires Service Account credentials (GOOGLE_APPLICATION_CREDENTIALS or google_service_account.json).
    """
    sa_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.path.join(
        os.path.dirname(__file__), "..", "google_service_account.json"
    )

    if not os.path.exists(sa_path):
        # Fall back to standard sitemap ping if Google Cloud Service Account is not configured
        logger.info(f"[google_indexing] Service Account key not found at '{sa_path}'. Falling back to Google Sitemap Ping.")
        return ping_google_sitemap()

    try:
        # If google-auth / google-api-python-client is available, execute API request
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        scopes = ["https://www.googleapis.com/auth/indexing"]
        credentials = service_account.Credentials.from_service_account_file(sa_path, scopes=scopes)
        indexing_service = build("indexing", "v3", credentials=credentials)

        body = {
            "url": cert_url,
            "type": action
        }
        response = indexing_service.urlNotifications().publish(body=body).execute()
        logger.info(f"[google_indexing] Google Indexing API success for '{cert_url}': {response}")
        return {
            "status": "SUCCESS",
            "api_response": response,
            "url": cert_url
        }
    except Exception as e:
        logger.warning(f"[google_indexing] Google Indexing API notification failed: {e}. Falling back to Sitemap Ping.")
        return ping_google_sitemap()


if __name__ == "__main__":
    res = ping_google_sitemap()
    print("Ping Result:", res)
