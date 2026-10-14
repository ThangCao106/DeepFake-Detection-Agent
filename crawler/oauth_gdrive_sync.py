"""
OAuth2 Google Drive Authentication & Sync Engine.
Supports Google OAuth 2.0 Playground Refresh Tokens and User Credentials.
Enables uploading directly under the user's personal Google Drive quota (bypassing 0 MB Service Account quota limits).
"""

import os
import sys
import json
import time
from pathlib import Path
import requests

def exchange_refresh_token_for_access_token(refresh_token: str) -> str:
    """
    Exchanges OAuth2 refresh_token for an active access_token.
    Supports Google OAuth Playground client, custom desktop client, and standard OAuth token endpoint.
    """
    token_url = "https://oauth2.googleapis.com/token"
    clean_token = refresh_token.strip()
    
    # Remove surrounding quotes if user pasted with quotes
    if (clean_token.startswith('"') and clean_token.endswith('"')) or (clean_token.startswith("'") and clean_token.endswith("'")):
        clean_token = clean_token[1:-1].strip()

    # If the user pasted a raw JSON blob instead of just the token string:
    if clean_token.startswith("{"):
        try:
            tjson = json.loads(clean_token)
            clean_token = tjson.get("refresh_token") or tjson.get("access_token") or clean_token
        except Exception:
            pass

    # Check if already a direct access token
    if clean_token.startswith("ya29."):
        print("[OAuth Engine] Input token detected as direct Google Access Token.")
        return clean_token

    # Candidate OAuth client IDs
    DEFAULT_CLIENT_ID = "891972860412-4chuehnfujg29c9gp78mrs4e49r7ejkn.apps.googleusercontent.com"
    DEFAULT_CLIENT_SECRET = "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"

    custom_client_id = os.environ.get("GDRIVE_CLIENT_ID") or DEFAULT_CLIENT_ID
    custom_client_secret = os.environ.get("GDRIVE_CLIENT_SECRET") or DEFAULT_CLIENT_SECRET

    candidates = [
        {"client_id": custom_client_id, "client_secret": custom_client_secret},
        {"client_id": DEFAULT_CLIENT_ID, "client_secret": DEFAULT_CLIENT_SECRET},
        {"client_id": "407408718192.apps.googleusercontent.com", "client_secret": None},
        {"client_id": None, "client_secret": None}
    ]

    last_error = None
    for cfg in candidates:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": clean_token,
        }
        if cfg["client_id"]:
            data["client_id"] = cfg["client_id"]
        if cfg["client_secret"]:
            data["client_secret"] = cfg["client_secret"]

        try:
            resp = requests.post(token_url, data=data, timeout=15)
            if resp.status_code == 200:
                body = resp.json()
                acc_token = body.get("access_token")
                if acc_token:
                    cid_label = cfg['client_id'] or 'public_token_endpoint'
                    print(f"[OAuth Engine] [TOKEN SUCCESS] Successfully refreshed Google Drive Access Token via {cid_label}!")
                    return acc_token
            else:
                last_error = f"HTTP {resp.status_code}: {resp.text}"
                print(f"[OAuth Engine] Candidate {cfg['client_id']} failed ({resp.status_code}): {resp.text}")
        except Exception as e:
            last_error = str(e)
            print(f"[OAuth Engine] Candidate {cfg['client_id']} request exception: {e}")

    raise RuntimeError(f"Failed to refresh Google Drive token from refresh_token. Last response: {last_error}")


def get_oauth_drive_service(refresh_token: str):
    """Initializes Google Drive API v3 client using user's OAuth access token."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    access_token = exchange_refresh_token_for_access_token(refresh_token)
    creds = Credentials(token=access_token)
    service = build('drive', 'v3', credentials=creds)
    return service
