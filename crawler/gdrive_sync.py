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
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Local storage fallback directory if GDrive credentials are not configured
DEFAULT_LOCAL_STORAGE = str(PROJECT_ROOT / "report_colab" / "gdrive_storage")
DEFAULT_TEMP_STAGING = str(PROJECT_ROOT / "report_colab" / "temp_media_samples")

class DiskGuardAgent:
    """
    Monitors VPS disk space and manages automatic cleanup of temporary media files.
    Prevents VPS disk full errors by purging temporary scratch files after GDrive sync.
    Never deletes files from persistent storage.
    """
    def __init__(self, target_dir=DEFAULT_TEMP_STAGING, min_free_gb=5.0, max_cache_gb=3.0):
        self.target_dir = target_dir
        self.min_free_gb = min_free_gb
        self.max_cache_gb = max_cache_gb

    def get_disk_usage_info(self):
        """Returns disk total, used, free space (in GB) and percentage free."""
        try:
            total, used, free = shutil.disk_usage("/")
            total_gb = round(total / (1024**3), 2)
            used_gb = round(used / (1024**3), 2)
            free_gb = round(free / (1024**3), 2)
            pct_free = round((free / total) * 100, 1)
            return {"total_gb": total_gb, "used_gb": used_gb, "free_gb": free_gb, "pct_free": pct_free}
        except Exception:
            return {"total_gb": 100.0, "used_gb": 50.0, "free_gb": 50.0, "pct_free": 50.0}

    def get_cache_size_gb(self):
        """Calculates total size of local cache storage in GB."""
        total_size = 0
        if os.path.exists(self.target_dir):
            for dirpath, _, filenames in os.walk(self.target_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
        return round(total_size / (1024**3), 3)

    def check_and_clean_disk(self, force=False):
        """
        Checks disk space and triggers cleanup if free space < 20% or cache size > max_cache_gb.
        Purges binary media files (*.jpg, *.mp4, *.webp, *.mov, *.png) while preserving JSON registries.
        """
        info = self.get_disk_usage_info()
        cache_gb = self.get_cache_size_gb()
        
        needs_cleanup = force or (info["pct_free"] < 20.0) or (info["free_gb"] < self.min_free_gb) or (cache_gb > self.max_cache_gb)
        
        if needs_cleanup:
            print(f"[Disk Guard] Clean Triggered: VPS Free Disk={info['free_gb']}GB ({info['pct_free']}%), Local Cache={cache_gb}GB. Executing Purge...")
            purged_bytes, purged_files = self.purge_binary_media()
            print(f"[Disk Guard] Purge Complete. Freed {round(purged_bytes / (1024**2), 2)}MB ({purged_files} binary files removed). JSON registries preserved.")
            return True
        else:
            print(f"[Disk Guard] VPS Disk Status OK: Free={info['free_gb']}GB ({info['pct_free']}%), Local Cache={cache_gb}GB.")
            return False

    def purge_binary_media(self, max_age_hours=24):
        """Purges binary media files while keeping metadata JSON files intact."""
        purged_bytes = 0
        purged_files = 0
        now = time.time()
        binary_exts = {".jpg", ".jpeg", ".png", ".webp", ".mp4", ".mov", ".avi", ".mkv"}

        if os.path.exists(self.target_dir):
            for dirpath, _, filenames in os.walk(self.target_dir):
                for f in filenames:
                    ext = os.path.splitext(f)[1].lower()
                    if ext in binary_exts:
                        fp = os.path.join(dirpath, f)
                        try:
                            file_size = os.path.getsize(fp)
                            os.remove(fp)
                            purged_bytes += file_size
                            purged_files += 1
                        except Exception as e:
                            print(f"[Disk Guard] Notice removing {fp}: {e}")

        return purged_bytes, purged_files


# Default target shared folder on Google Drive (User shared workspace)
DEFAULT_TARGET_FOLDER_ID = "1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6"

class GDriveSyncAgent:
    def __init__(self, credentials_json=None, base_folder_name="Thread_Data"):
        self.base_folder_name = base_folder_name
        self.credentials_json = credentials_json or os.environ.get("GDRIVE_CREDENTIALS_JSON")
        self.target_folder_id = (
            os.environ.get("GDRIVE_TARGET_FOLDER_ID") 
            or os.environ.get("GDRIVE_FOLDER_ID") 
            or DEFAULT_TARGET_FOLDER_ID
        )
        self.service = None
        self.use_mock_storage = True
        self.sync_mode = "MOCK_LOCAL_STORAGE"
        self.colab_mount_path = None
        self.disk_guard = DiskGuardAgent()
        
        self._initialize_service()

    def _initialize_service(self):
        """Initializes Google Drive connection via Colab Direct Mount or API v3, else uses local structured storage."""
        # 1. Check if running on Google Colab with drive.mount('/content/drive')
        colab_mydrive = Path("/content/drive/MyDrive")
        if colab_mydrive.exists():
            candidate_folders = [
                self.target_folder_id,
                "1M9s_pCn7nd9yaqXmJlf062uzrYm7ASa6",
                "1W6R3QYzTI6GJAD2lByKoVaQnXfURcLO0",
                "Thread_Data",
                "DeepFake_Pentest_Data",
                "Deepfake Detection",
                self.base_folder_name
            ]
            found_folder = None
            for cname in candidate_folders:
                cand = colab_mydrive / cname
                if cand.exists() and cand.is_dir():
                    found_folder = cand
                    break
            if not found_folder:
                found_folder = colab_mydrive / "Thread_Data"
                found_folder.mkdir(parents=True, exist_ok=True)
                
            self.colab_mount_path = str(found_folder)
            self.use_mock_storage = False
            self.sync_mode = "COLAB_DIRECT_MOUNT"
            print(f"[GDrive Sync] [AUTH SUCCESS] Connected directly to mounted Google Drive: {self.colab_mount_path}")
            return

        # 2. Check if Google Drive Service Account Credentials JSON is provided
        if self.credentials_json:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build

                creds_dict = None
                # Support both raw JSON string and path to credentials JSON file
                if os.path.isfile(str(self.credentials_json)):
                    with open(self.credentials_json, "r", encoding="utf-8") as cf:
                        creds_dict = json.load(cf)
                elif isinstance(self.credentials_json, str) and self.credentials_json.strip().startswith("{"):
                    creds_dict = json.loads(self.credentials_json)
                elif isinstance(self.credentials_json, dict):
                    creds_dict = self.credentials_json

                if creds_dict:
                    creds = service_account.Credentials.from_service_account_info(
                        creds_dict, scopes=['https://www.googleapis.com/auth/drive']
                    )
                    self.service = build('drive', 'v3', credentials=creds)
                    self.use_mock_storage = False
                    self.sync_mode = "API_SERVICE_ACCOUNT"
                    bot_email = creds_dict.get("client_email", "service_account_bot")
                    print(f"[GDrive Sync] [AUTH SUCCESS] Connected to Google Drive API via Service Account: {bot_email}")
                    if self.target_folder_id:
                        print(f"[GDrive Sync] [FOLDER] Target Shared Drive Folder ID: {self.target_folder_id}")
                        print(f"[GDrive Sync] [URL] https://drive.google.com/drive/folders/{self.target_folder_id}")
                    else:
                        print("[GDrive Sync] [NOTICE] GDRIVE_TARGET_FOLDER_ID not set. Data will be written to Service Account root drive.")
                    return
            except Exception as e:
                print(f"[GDrive Sync] [AUTH ERROR] Google Drive API init failed ({e}). Falling back to local storage structure.")
        else:
            print("[GDrive Sync] [NOTICE] No GDRIVE_CREDENTIALS_JSON found. Falling back to local storage structure.")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print("::warning title=Google Drive Sync Notice::GDRIVE_CREDENTIALS_JSON is not configured in GitHub Secrets! Files are saved to the runner's ephemeral storage only and will NOT appear on Google Drive. To upload files to Google Drive, add your Service Account JSON to Repository Secrets > GDRIVE_CREDENTIALS_JSON.")
        
        self.sync_mode = "MOCK_LOCAL_STORAGE"
        print(f"[GDrive Sync] Active Mode: Structured Local Storage at {DEFAULT_LOCAL_STORAGE}")

    def get_or_create_daily_folder(self, date_str=None):
        """Creates or gets daily folder structure YYYY-MM-DD."""
        if not date_str:
            date_str = time.strftime("%Y-%m-%d")
        
        subfolders = ["01_raw_originals", "02_manipulated_faceswap", "03_manipulated_openweight"]
        paths = {}

        # Mode 1: Google Colab Direct Mount
        if self.sync_mode == "COLAB_DIRECT_MOUNT" and self.colab_mount_path:
            base_dir = os.path.join(self.colab_mount_path, date_str)
            for sub in subfolders:
                full_p = os.path.join(base_dir, sub)
                os.makedirs(full_p, exist_ok=True)
                paths[sub] = full_p
            return paths

        # Mode 2: Local Mock Storage
        if self.use_mock_storage:
            base_dir = os.path.join(DEFAULT_LOCAL_STORAGE, date_str)
            for sub in subfolders:
                full_p = os.path.join(base_dir, sub)
                os.makedirs(full_p, exist_ok=True)
                paths[sub] = full_p
            return paths

        # Mode 3: Google Drive API v3
        try:
            # Check or create base folder inside target_folder_id if specified
            base_id = self._find_or_create_drive_folder(self.base_folder_name, parent_id=self.target_folder_id)
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
        """Helper to query or create folder on Google Drive via API v3 with Shared Drive support."""
        if not self.service:
            return None
        
        query = f"name = '{folder_name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
            
        results = self.service.files().list(
            q=query, 
            fields="files(id, name)",
            supportsAllDrives=True,
            includeItemsFromAllDrives=True
        ).execute()
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
            
        folder = self.service.files().create(
            body=metadata, 
            fields='id',
            supportsAllDrives=True
        ).execute()
        return folder.get('id')

    def upload_file(self, local_path, date_str=None, target_subfolder="01_raw_originals"):
        """Uploads a local media file to specified GDrive daily subfolder and cleans local file if uploaded."""
        paths = self.get_or_create_daily_folder(date_str)
        filename = os.path.basename(local_path)

        # Mode 1: Colab Direct Mount (Direct copy into mounted Google Drive)
        if self.sync_mode == "COLAB_DIRECT_MOUNT" and self.colab_mount_path:
            dest_dir = paths.get(target_subfolder, os.path.join(self.colab_mount_path, date_str, target_subfolder))
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(local_path) and local_path != dest_path:
                shutil.copy2(local_path, dest_path)
            return dest_path

        # Mode 2: Local Mock Storage
        if self.use_mock_storage:
            dest_dir = paths.get(target_subfolder, os.path.join(DEFAULT_LOCAL_STORAGE, date_str, target_subfolder))
            dest_path = os.path.join(dest_dir, filename)
            if os.path.exists(local_path) and local_path != dest_path:
                with open(local_path, "rb") as sf, open(dest_path, "wb") as df:
                    df.write(sf.read())
            self.disk_guard.check_and_clean_disk()
            return dest_path

        # Mode 3: Upload via Google Drive API
        try:
            from googleapiclient.http import MediaFileUpload
            folder_id = paths.get(target_subfolder)
            file_metadata = {
                'name': filename,
                'parents': [folder_id]
            }
            media = MediaFileUpload(local_path, resumable=True)
            uploaded_file = self.service.files().create(
                body=file_metadata, 
                media_body=media, 
                fields='id',
                supportsAllDrives=True
            ).execute()
            print(f"[GDrive Sync] Uploaded {filename} to GDrive folder {target_subfolder} (ID: {uploaded_file.get('id')})")
            
            # Post-sync cleanup: Remove binary local file after successful GDrive upload
            if os.path.exists(local_path) and not local_path.endswith(".json"):
                os.remove(local_path)
                print(f"[Disk Guard] Post-Sync Purge: Deleted local file {filename} after GDrive sync.")
                
            return uploaded_file.get('id')
        except Exception as e:
            print(f"[GDrive Sync] Failed to upload {filename} via API: {e}")
            return None


    def sync_batch_registry(self, registry_data, date_str=None, target_subfolder="01_raw_originals"):
        """Syncs a metadata registry file (JSON) to Google Drive and runs Disk Guard checks."""
        paths = self.get_or_create_daily_folder(date_str)
        reg_filename = f"registry_{target_subfolder}_{int(time.time())}.json"

        # Check disk space status
        self.disk_guard.check_and_clean_disk()

        # Mode 1: Colab Direct Mount
        if self.sync_mode == "COLAB_DIRECT_MOUNT" and self.colab_mount_path:
            dest_dir = paths.get(target_subfolder, os.path.join(self.colab_mount_path, date_str, target_subfolder))
            dest_path = os.path.join(dest_dir, reg_filename)
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
            print(f"[GDrive Sync] [COLAB DIRECT MOUNT] Synced {len(registry_data.get('images', []))} images & {len(registry_data.get('videos', []))} videos to Google Drive: {dest_path}")
            return dest_path

        # Mode 2: Local Mock Storage
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

    def get_or_create_auth_folder(self):
        """Creates or gets the persistent 'auth/' folder on Google Drive or local storage."""
        if self.use_mock_storage:
            auth_dir = os.path.join(DEFAULT_LOCAL_STORAGE, "auth")
            os.makedirs(auth_dir, exist_ok=True)
            return auth_dir

        try:
            base_id = self._find_or_create_drive_folder(self.base_folder_name, parent_id=self.target_folder_id)
            auth_folder_id = self._find_or_create_drive_folder("auth", parent_id=base_id)
            return auth_folder_id
        except Exception as e:
            print(f"[GDrive Sync] Warning getting auth folder: {e}. Using local storage fallback.")
            auth_dir = os.path.join(DEFAULT_LOCAL_STORAGE, "auth")
            os.makedirs(auth_dir, exist_ok=True)
            return auth_dir

    def save_remote_auth_state(self, auth_data: dict, auth_key="threads_auth"):
        """Saves or updates rolling authentication session JSON to Google Drive auth/ folder."""
        auth_folder = self.get_or_create_auth_folder()
        filename = f"{auth_key}.json"

        if self.use_mock_storage or not self.service:
            dest_path = os.path.join(auth_folder if isinstance(auth_folder, str) else os.path.join(DEFAULT_LOCAL_STORAGE, "auth"), filename)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "w", encoding="utf-8") as f:
                json.dump(auth_data, f, indent=2, ensure_ascii=False)
            print(f"[GDrive Sync] [ROLLING-AUTH] Saved local auth cache to {dest_path}")
            return dest_path

        try:
            import io
            from googleapiclient.http import MediaIoBaseUpload
            
            # Check if file exists in auth folder
            query = f"name = '{filename}' and '{auth_folder}' in parents and trashed = false"
            results = self.service.files().list(
                q=query, 
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            existing_files = results.get('files', [])

            content_bytes = json.dumps(auth_data, indent=2, ensure_ascii=False).encode('utf-8')
            media = MediaIoBaseUpload(io.BytesIO(content_bytes), mimetype='application/json', resumable=False)

            if existing_files:
                file_id = existing_files[0]['id']
                updated = self.service.files().update(
                    fileId=file_id, 
                    media_body=media,
                    supportsAllDrives=True
                ).execute()
                print(f"[GDrive Sync] [ROLLING-AUTH] Updated existing {filename} in GDrive auth/ (ID: {file_id})")
                return file_id
            else:
                file_metadata = {'name': filename, 'parents': [auth_folder]}
                created = self.service.files().create(
                    body=file_metadata, 
                    media_body=media, 
                    fields='id',
                    supportsAllDrives=True
                ).execute()
                print(f"[GDrive Sync] [ROLLING-AUTH] Created {filename} in GDrive auth/ (ID: {created.get('id')})")
                return created.get('id')
        except Exception as e:
            print(f"[GDrive Sync] [WARNING] Failed to save rolling auth state to GDrive: {e}")
            return None

    def load_remote_auth_state(self, auth_key="threads_auth"):
        """Loads persistent authentication session JSON from Google Drive auth/ folder."""
        filename = f"{auth_key}.json"

        # Check local storage fallback first
        local_path = os.path.join(DEFAULT_LOCAL_STORAGE, "auth", filename)
        if os.path.isfile(local_path):
            try:
                with open(local_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        if self.use_mock_storage or not self.service:
            return None

        try:
            import io
            from googleapiclient.http import MediaIoBaseDownload
            
            auth_folder = self.get_or_create_auth_folder()
            query = f"name = '{filename}' and '{auth_folder}' in parents and trashed = false"
            results = self.service.files().list(
                q=query, 
                fields="files(id, name)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True
            ).execute()
            files = results.get('files', [])

            if files:
                file_id = files[0]['id']
                request = self.service.files().get_media(fileId=file_id, supportsAllDrives=True)

                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                fh.seek(0)
                data = json.loads(fh.read().decode('utf-8'))
                print(f"[GDrive Sync] [ROLLING-AUTH] Successfully loaded {filename} from GDrive auth/ folder.")
                return data
        except Exception as e:
            print(f"[GDrive Sync] [NOTICE] Could not load rolling auth state from GDrive ({e}).")

        return None

if __name__ == "__main__":
    agent = GDriveSyncAgent()
    paths = agent.get_or_create_daily_folder()
    print("[GDrive Sync Test] Daily paths created/verified:", paths)
    agent.disk_guard.check_and_clean_disk(force=True)

