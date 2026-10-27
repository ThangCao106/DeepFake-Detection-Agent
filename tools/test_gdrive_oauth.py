"""
Fast Diagnostic & Verification Script for Google Drive OAuth2 Sync (Option B).
Tests OAuth2 token exchange and uploads a test file to the target Google Drive folder.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_test():
    target_folder_id = os.environ.get("GDRIVE_TARGET_FOLDER_ID") or "1s6G-cscNpEMBbUlvJfb4i24s9_7UVjet"
    refresh_token = os.environ.get("GDRIVE_REFRESH_TOKEN")

    print("=" * 75)
    print("🔍 [TEST] Google Drive OAuth2 Verification (Option B)")
    print(f"   Target Folder ID: {target_folder_id}")
    print(f"   URL: https://drive.google.com/drive/folders/{target_folder_id}")
    print("=" * 75)

    if not refresh_token:
        print("❌ ERROR: GDRIVE_REFRESH_TOKEN environment variable is NOT set!")
        print("   Please ensure GDRIVE_REFRESH_TOKEN is added to GitHub Repository Secrets.")
        sys.exit(1)

    print(f"🔑 GDRIVE_REFRESH_TOKEN detected ({len(refresh_token)} chars, prefix: {refresh_token[:6]}...)")

    try:
        from crawler.oauth_gdrive_sync import get_oauth_drive_service
        service = get_oauth_drive_service(refresh_token)
        print("✅ Google Drive API v3 Service successfully initialized!")
    except Exception as e:
        print(f"❌ Failed to initialize Google Drive service: {e}")
        sys.exit(1)

    # Verify folder access
    try:
        folder_info = service.files().get(
            fileId=target_folder_id, 
            fields="id, name, mimeType, owners, capabilities",
            supportsAllDrives=True
        ).execute()
        folder_name = folder_info.get("name", "Unknown")
        print(f"📁 Target Folder verified: '{folder_name}' (ID: {target_folder_id})")
        can_add = folder_info.get("capabilities", {}).get("canAddChildren", False)
        print(f"   Permission to add files: {can_add}")
    except Exception as e:
        print(f"⚠️ Warning querying folder info: {e} (continuing upload test...)")

    # Create temporary test file
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    test_content = (
        f"🎉 Google Drive OAuth2 Sync Verification Successful!\n"
        f"Timestamp: {now_str}\n"
        f"Target Folder: {target_folder_id}\n"
        f"Method: Option B (Personal Quota - Google OAuth2 Playground)\n"
    )
    test_file = PROJECT_ROOT / "test_oauth_sync_success.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)

    print(f"\n📤 Uploading test file '{test_file.name}' to Google Drive...")
    try:
        from googleapiclient.http import MediaFileUpload
        file_metadata = {
            'name': f"test_oauth_sync_success_{int(time.time())}.txt",
            'parents': [target_folder_id]
        }
        media = MediaFileUpload(str(test_file), mimetype='text/plain', resumable=True)
        uploaded = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink',
            supportsAllDrives=True
        ).execute()
        
        file_id = uploaded.get('id')
        link = uploaded.get('webViewLink')
        print("=" * 75)
        print(f"🎉 SUCCESS! File uploaded to Google Drive!")
        print(f"   File Name: {uploaded.get('name')}")
        print(f"   File ID:   {file_id}")
        if link:
            print(f"   View Link: {link}")
        print(f"   Check your folder: https://drive.google.com/drive/folders/{target_folder_id}")
        print("=" * 75)
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
    finally:
        if test_file.exists():
            test_file.unlink()

if __name__ == "__main__":
    run_test()
