# 🔐 Hướng Dẫn Xác Thực & Đăng Nhập Tự Động: Meta Threads & Google Drive (Option B4 + B2)

Tài liệu này cung cấp hướng dẫn toàn diện về cơ chế xác thực tự động 100% cho **Job Crawler & Pentest** (`threads_crawler.py`, `pentest_runner.py`, và GitHub Actions `threads_pentest_job.yml`).

---

## 📌 Tổng Quan Kiến Trúc Tự Động (Option B4 + Option B2)

```
                     ┌─────────────────────────────────────────────────────────┐
                     │            KHÔNG CẦN COPY COOKIE MỖI NGÀY               │
                     └─────────────────────────────────────────────────────────┘
                                                  │
                    ┌─────────────────────────────┴─────────────────────────────┐
                    ▼                                                           ▼
      [Option B4: Meta Threads Guest]                             [Option B2: GDrive Rolling Cache]
  - Tự động sinh Ephemeral Guest Session                      - Tự động tạo thư mục 'auth/' trên GDrive
  - Handshake với App ID: 238260118697367                     - Nạp và tự cập nhật 'threads_session.json'
  - Sinh device id: ig_did, mid, csrftoken                    - Giữ ấm (keep-alive) phiên đăng nhập 4h/lần
  - Không cần tài khoản, không lo bị ban                      - Hoạt động liền mạch xuyên suốt các run
```

---

## 🧵 PHẦN 1: Meta Threads - Cơ Chế Tự Động Không Cần Tài Khoản (Option B4)

### 🌟 Bạn Không Cần Làm Gì Cả! (Zero-Config)
Hệ thống crawler đã được tích hợp engine **Option B4 (Ephemeral Guest Session)**:
1. Mỗi khi job pentest kích hoạt (4 tiếng/lần trên GitHub Actions hoặc VPS), crawler sẽ tự động đóng vai một thiết bị ứng dụng di động hợp lệ với định danh `X-IG-App-ID: 238260118697367`.
2. Hệ thống tự sinh bộ cookie khách hợp lệ (`ig_did`, `mid`, `csrftoken`) và gửi handshake tới máy chủ của Meta Threads để nhận cookie phiên từ máy chủ edge.
3. Toàn bộ quá trình diễn ra ngầm trong **0.3 giây**, hoàn toàn không cần tài khoản cá nhân, không lo bị khóa nick, và quan trọng nhất: **bạn không bao giờ phải copy session thủ công mỗi ngày**.

---

## 📁 PHẦN 2: Google Drive - Rolling Session Cache & Đồng Bộ Dữ Liệu (Option B2)

### 📍 Bước 1: Chuẩn Bị Thư Mục Google Drive Cá Nhân
1. Mở [Google Drive](https://drive.google.com/) của bạn.
2. Tạo một thư mục (ví dụ: `Deepfake Detection Data`).
3. Mở thư mục đó và lấy **Folder ID** từ thanh địa chỉ trình duyệt:
   ```text
   https://drive.google.com/drive/folders/1aBcD_eFgHiJkLmNoPqRsTuVwXyZ12345
                                          ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                                              Đây chính là GDRIVE_TARGET_FOLDER_ID
   ```
4. Chia sẻ (Share) thư mục này cho email Service Account của bạn (ví dụ: `deepfake-gdrive-bot@...iam.gserviceaccount.com`) với quyền **Người chỉnh sửa (Editor)**.

### 📍 Bước 2: Cấu Trúc Thư Mục Tự Động Sinh Trên Drive
Khi chạy, `GDriveSyncAgent` sẽ tự động tổ chức cấu trúc dữ liệu chuẩn mực:
```text
GDrive/DeepFake_Pentest_Data/
  ├── auth/                                    <-- [Option B2] Rolling session cache
  │    └── threads_session.json               <-- Tự động lưu & làm mới session
  └── YYYY-MM-DD/
       ├── 01_raw_originals/                  <-- Dữ liệu Threads crawl được
       ├── 02_manipulated_faceswap/           <-- Dữ liệu FaceSwap test vector
       └── 03_manipulated_openweight/         <-- Dữ liệu Flux.1/Wan2.1 test vector
```

---

## ⚙️ PHẦN 3: Thiết Lập Biến Môi Trường (Chỉ Cần Cài 1 Lần Duy Nhất)

Bạn chỉ cần thiết lập **2 biến môi trường Google Drive** trên GitHub Secrets (hoặc file `.env` trên VPS). Không cần cấu hình Threads nếu dùng Option B4!

### 🔹 Cấu Hình Trên GitHub Secrets:
1. Vào GitHub Repo -> **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**.
2. Thêm 2 secret bắt buộc sau:

| Tên Secret | Bắt buộc? | Mô tả giá trị |
| :--- | :---: | :--- |
| **`GDRIVE_CREDENTIALS_JSON`** | ✅ **Bắt buộc** | Toàn bộ nội dung file JSON Service Account tải từ Google Cloud Console. |
| **`GDRIVE_TARGET_FOLDER_ID`** | ✅ **Bắt buộc** | Chuỗi ID thư mục Google Drive bạn đã chia sẻ ở Phần 2. |
| **`THREADS_SESSION_COOKIES`** | ⚪ *Tùy chọn* | **Bỏ trống** để hệ thống tự động chạy **Option B4 Guest Mode**. Chỉ điền nếu bạn muốn ưu tiên dùng nick cá nhân. |

---

## 🔄 PHẦN 4: Nâng Cao - Dùng Nick Cá Nhân 1 Lần Duy Nhất (Option B2 Persistent)

Nếu bạn muốn bot crawl với tư cách tài khoản cá nhân thay vì Guest, bạn **chỉ cần làm 1 lần duy nhất**:
1. Đăng nhập Threads trên máy tính cá nhân.
2. Dùng extension Cookie-Editor trích xuất JSON hoặc chuỗi `sessionid=...; ds_user_id=...`.
3. Lưu nội dung đó vào file `threads_session.json` với cấu trúc:
   ```json
   {
     "auth_mode": "USER_SESSION_COOKIE",
     "cookies": "sessionid=your_actual_sessionid; ds_user_id=your_id; csrftoken=...",
     "user_id": "your_id"
   }
   ```
4. Thả file này vào thư mục `auth/` trên Google Drive (hoặc nạp vào secret `THREADS_SESSION_COOKIES`).
5. Kể từ thời điểm đó: Cứ mỗi 4 tiếng bot chạy, bot sẽ đọc session này, dùng để crawl, và **tự động ghi đè session mới nhất ngược lại Google Drive**. Session được giữ ấm liên tục và không bị hết hạn!

---

## 🧪 PHẦN 5: Kiểm Tra Nhanh (Pre-Flight Verification)

Chạy thử trên máy local để kiểm tra cơ chế Option B4 + B2:

### 1. Kiểm tra Guest Session tự động (Option B4):
```powershell
py -3 -c "from crawler.threads_crawler import create_threads_guest_session; s, d, h = create_threads_guest_session(); print('Guest Handshake:', h); print('Device ID:', d.get('ig_did'))"
```

### 2. Kiểm tra Rolling Session trên Google Drive (Option B2):
```powershell
py -3 -c "from crawler.gdrive_sync import GDriveSyncAgent; a = GDriveSyncAgent(); print('Rolling session loaded:', a.load_remote_auth_state('threads_session'))"
```

### 3. Chạy thử một vòng pentest hoàn chỉnh:
```powershell
py -3 pentest_runner.py --target general
```
*Kết quả kỳ vọng:*
```text
[Threads Crawler] [AUTH] Activating Ephemeral Guest Session (Option B4: Zero-Config)...
[Threads Crawler] [AUTH] Meta Threads Guest Session Ready: ID=... (Status: GUEST_HANDSHAKE_ONLINE)
[GDrive Sync] [ROLLING-AUTH] Saved local auth cache to ...\threads_session.json
[Pentest Runner] Pentest Jobs Complete! 1 Reports saved...
```
