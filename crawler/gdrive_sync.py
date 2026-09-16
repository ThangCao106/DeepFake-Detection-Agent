"""
Google Drive Sync Agent for DeepFake Detection Agent
Manages daily structured folder sync:
  GDrive/DeepFake_Pentest_Data/YYYY-MM-DD/
    ├── 01_raw_originals/
    ├── 02_manipulated_faceswap/
    └── 03_manipulated_openweight/
"""
import os
import sys
import json
import time
from pathlib import Path

# Local storage fallback directory if GDrive credentials are not configured
DEFAULT_LOCAL_STORAGE = os.path.join("report_colab", "gdrive_storage")

class GDriveSyncAgent:
    def __init__(self, credentials_json=None, base_folder_name="DeepFake_Pentest_Data"):
        self.base_folder_name = base_folder_name
        self.credentials_json = credentials_json or os.environ.get("GDRIVE_CREDENTIALS_JSON")
        self.service = None
        self.use_mock_storage = True
        
        self._initialize_service()

    def _initialize_service(self):
        """Initializes Google Drive API service if credentials exist, else uses local structured storage."""
        if self.credentials_json:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build

                creds_dict = json.loads(self.credentials_json) if isinstance(self.credentials_json, str) and self.credentials_json.startswith("{") else None
                if creds_dict:
                    creds = service_account.Credentials.from_service_account_info(
                        creds_dict, scopes=['https://www.googleapis.com/auth/drive']
                    )
                    self.service = build('drive', 'v3', credentials=creds)
                    self.use_mock_storage = False
                    print("[GDrive Sync] Connected to Google Drive API via Service Account.")
                    return
            except Exception as e:
                print(f"[GDrive Sync] Warning: Google Drive API init failed ({e}). Falling back to local storage structure.")
        
        print(f"[GDrive Sync] Active Mode: Structured Storage at {DEFAULT_LOCAL_STORAGE}")

    def get_or_create_daily_folder(self, date_str=None):
        """Creates or gets daily folder structure YYYY-MM-DD."""
        if not date_str:
            date_str = time.strftime("%Y-%m-%d")
        
        subfolders = ["01_raw_originals", "02_manipulated_faceswap", "03_manipulated_openweight"]
        paths = {}

        if self.use_mock_storage:
            base_dir = os.path.join(DEFAULT_LOCAL_STORAGE, date_str)
            for sub in subfolders:
                full_p = os.path.join(base_dir, sub)
                os.makedirs(full_p, exist_ok=True)
                paths[sub] = full_p
            return paths

        # Google Drive API Implementation
        try:
            # Check or create base folder
            base_id = self._find_or_create_drive_folder(self.base_folder_name, parent_id=None)
            daily_id = self._find_or_create_drive_folder(date_str, parent_id=base_id)
            
            for sub in subfolders:
                sub_id = self._find_or_create_drive_folder(sub, parent_id=daily_id)
                paths[sub] = sub_id
            return paths
        except Exception as e:
            print(f"[GDrive Sync] API Error in folder creation: {e}. Falling back to local storage.")
            base_dir = os.path.join(DEFAULT_LOCAL_STORAGE, date_str)
            for sub in subfolders:
                full_p = os.path.join(base_dir, sub)
                os.makedirs(full_p, exist_ok=True)
                paths[sub] = full_p
            return paths

    def _find_or_create_drive_folder(self, folder_name, parent_id=None):
        """Helper to query or create folder on Google Drive via API v3."""
        if not self.service:
            return None
        
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
            
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        
        if files:
            return files[0]['id']
        
        # Create folder
        metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            metadata['parents'] = [parent_id]
            
        folder = self.service.files().create(body=metadata, fields='id').execute()
        return folder.get('id')

    def upload_file(self, local_path, date_str=None, target_subfolder="01_raw_originals"):
        """Uploads a local media file to the specified Google Drive daily subfolder."""
        paths = self.get_or_create_daily_folder(date_str)
        filename = os.path.basename(local_path)

        if self.use_mock_storage:
            dest_dir = paths.get(target_subfolder, os.path.join(DEFAULT_LOCAL_STORAGE, date_str, target_subfolder))
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(local_path) and local_path != dest_path:
                with open(local_path, "rb") as sf, open(dest_path, "wb") as df:
                    df.write(sf.read())
            return dest_path

        # Upload via Google Drive API
        try:
            from googleapiclient.http import MediaFileUpload
            folder_id = paths.get(target_subfolder)
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            media = MediaFileUpload(local_path, resumable=True)
            uploaded_file = self.service.files().create(
                body=file_metadata, media_body=media, fields='id'
            ).execute()
            print(f"[GDrive Sync] Uploaded {filename} to GDrive folder {target_subfolder} (ID: {uploaded_file.get('id')})")
            return uploaded_file.get('id')
        except Exception as e:
            print(f"[GDrive Sync] Failed to upload {filename} via API: {e}")
            return None

    def sync_batch_registry(self, registry_data, date_str=None, target_subfolder="01_raw_originals"):
        """Syncs a metadata registry file (JSON) to Google Drive."""
        paths = self.get_or_create_daily_folder(date_str)
        reg_filename = f"registry_{target_subfolder}_{int(time.time())}.json"

        if self.use_mock_storage:
            dest_dir = paths.get(target_subfolder, os.path.join(DEFAULT_LOCAL_STORAGE, date_str, target_subfolder))
            dest_path = os.path.join(dest_dir, reg_filename)
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            print(f"[GDrive Sync] Synced {len(registry_data.get('images', []))} images & {len(registry_data.get('videos', []))} videos to {dest_path}")
            return dest_path

        # Upload JSON registry string via API
        try:
            temp_path = os.path.join("report_colab", reg_filename)
            os.makedirs("report_colab", exist_ok=True)
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            res = self.upload_file(temp_path, date_str, target_subfolder)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return res
        except Exception as e:
            print(f"[GDrive Sync] Failed registry sync: {e}")
            return None

if __name__ == "__main__":
    agent = GDriveSyncAgent()
    paths = agent.get_or_create_daily_folder()
    print("[GDrive Sync Test] Daily paths created/verified:", paths)
