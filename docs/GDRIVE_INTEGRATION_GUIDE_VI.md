# 📖 Hướng Dẫn Tích Hợp Google Drive Cho Dự Án DeepFake Detection (Option A & Option C)

Tài liệu này hướng dẫn chi tiết từng bước để kết nối dự án **DeepFake Detection Agent** với thư mục / project workspace **Deepfake Detection** trên Google Drive.

---

## 🛠️ PHẦN 1: Cấu Hình Option A (Google Cloud Service Account)
> **Mục đích**: Tự động hóa 100% ngầm 24/7 trên **GitHub Actions** và **VPS Server** mà không cần đăng nhập lại.

### 📍 Bước 1: Tạo Google Cloud Project & Kích hoạt Google Drive API
1. Truy cập [Google Cloud Console](https://console.cloud.google.com/).
2. Nhấp vào menu chọn Project ở góc trên bên trái -> Chọn **New Project** (Dự án mới).
3. Đặt tên project: `Deepfake-Detection-Drive` -> Bấm **Create**.
4. Vào thanh tìm kiếm trên cùng, gõ `Google Drive API` -> Chọn **Google Drive API**.
5. Nhấn nút **Enable** (Bật).

### 📍 Bước 2: Tạo Service Account & Tải File JSON Key
1. Vào menu bên trái -> Chọn **APIs & Services** -> **Credentials** (Thông tin xác thực).
2. Nhấp nút **+ CREATE CREDENTIALS** -> Chọn **Service Account**.
3. Nhập thông tin:
   - **Service account name**: `deepfake-gdrive-bot`
   - **Service account ID**: `deepfake-gdrive-bot` (tự động tạo)
4. Nhấn **CREATE AND CONTINUE** -> Nhấn **DONE**.
5. Tại danh sách Service Accounts vừa tạo, nhấp vào email của service account (dạng `deepfake-gdrive-bot@xxxx.iam.gserviceaccount.com`).
6. Chuyển sang tab **KEYS** -> Nhấp **ADD KEY** -> Chọn **Create new key**.
7. Chọn định dạng **JSON** -> Nhấn **CREATE**.
8. File JSON chứa khóa riêng tư sẽ tự động tải về máy tính của bạn (ví dụ: `deepfake-detection-drive-xxxx.json`).

### 📍 Bước 3: Chia Sẻ Thư Mục Google Drive Cho Service Account
1. Mở file JSON key vừa tải bằng Notepad/VS Code, tìm và sao chép giá trị trong trường `"client_email"` (ví dụ: `deepfake-gdrive-bot@deepfake-detection-drive-xxxx.iam.gserviceaccount.com`).
2. Truy cập [Google Drive](https://drive.google.com/), tìm thư mục project **Deepfake Detection** (hoặc tạo thư mục mới tên `Deepfake Detection`).
3. Nhấp chuột phải vào thư mục -> Chọn **Chia sẻ (Share)**.
4. Dán địa chỉ email `client_email` vừa sao chép vào ô dán email.
5. Đặt quyền truy cập là **Người chỉnh sửa (Editor)**.
6. Bỏ tích ô *"Thông báo cho mọi người"* -> Nhấn **Chia sẻ (Share)**.

### 📍 Bước 4: Thêm Credentials Vào GitHub Secrets & VPS

#### 🔹 Cấu hình trên GitHub Actions:
1. Truy cập Repository GitHub của dự án (`DeepFake-Detection-Agent`).
2. Vào **Settings** -> **Secrets and variables** -> **Actions**.
3. Nhấn nút **New repository secret**.
4. **Name**: `GDRIVE_CREDENTIALS_JSON`
5. **Secret**: Mở file JSON key bằng Notepad, copy TOÀN BỘ nội dung trong file (từ `{` đến `}`) dán vào đây.
6. Nhấn **Add secret**.

#### 🔹 Cấu hình trên VPS (.env):
Mở file `.env` trong thư mục gốc của dự án trên VPS và dán:
```env
GDRIVE_CREDENTIALS_JSON='{"type": "service_account", "project_id": "deepfake-detection-drive", ...}'
```

---

## 💻 PHẦN 2: Cấu Hình Option C (Google Colab GPU Direct Mount)
> **Mục đích**: Dành cho môi trường **Google Colab GPU Pro** khi chạy kịch bản sinh deepfake nặng (Flux.1-dev, Wan2.1, InsightFace).

### 📍 Bước 1: Tạo Lối Tắt (Shortcut) Thư Mục `Deepfake Detection`
1. Mở [Google Drive](https://drive.google.com/) cá nhân của bạn.
2. Tìm thư mục project `Deepfake Detection`.
3. Nhấp chuột phải vào thư mục -> Chọn **Thêm lối tắt vào Drive (Add shortcut to Drive)**.
4. Chọn vị trí lưu lối tắt tại **Drive của tôi (My Drive)** -> Nhấn **Thêm lối tắt**.

### 📍 Bước 2: Chạy Lệnh Mount Drive Trên Google Colab
Trong Notebook `colab_gpu_pipeline.ipynb` trên Colab, thêm ô lệnh đầu tiên:
```python
from google.colab import drive
drive.mount('/content/drive')
```
*Nhấn nút cấp quyền kết nối tài khoản Google Drive khi hộp thoại xuất hiện.*

### 📍 Bước 3: Khởi Chạy Pipeline
Chạy lệnh thực thi trên Colab:
```bash
!python colab_gpu_runner.py
```
Dữ liệu sẽ tự động ghi thẳng vào đường dẫn:
`/content/drive/MyDrive/Deepfake Detection/DeepFake_Pentest_Data/YYYY-MM-DD/`
- `01_raw_originals/`
- `02_manipulated_faceswap/`
- `03_manipulated_openweight/`

---

## 🚀 PHẦN 3: Cấu Hình Option C Cho Colab Pro Scheduled Runner (Khắc Phục Lỗi GitHub Private)
> **Mục đích**: Chạy **Scheduled Run (Headless 4h/lần)** trên Colab Pro mà không bị lỗi xác thực private GitHub (`fatal: could not read Username`).

### 📍 Bước 1: Tạo gói zip mã nguồn trên máy cục bộ
Chạy script đóng gói có sẵn trong thư mục dự án:
```bash
python package_for_gdrive.py
```
Script sẽ tạo ra file gọn nhẹ `DeepFake-Detection-Agent.zip` (~11 MB, đã tự động loại trừ virtualenv, git cache và checkpoint nặng).

### 📍 Bước 2: Tải file Zip lên Google Drive
Tải file `DeepFake-Detection-Agent.zip` vào thư mục Google Drive tại đường dẫn:
```
Drive của tôi / Thread_Data / DeepFake-Detection-Agent.zip
```
*(Hoặc giải nén trực tiếp thành thư mục `MyDrive/Thread_Data/DeepFake-Detection-Agent/`)*

### 📍 Bước 3: Chạy Scheduled Runner trên Colab Pro
Trong notebook `Colab_Pro_Scheduled_Runner.ipynb`:
- **Bước 1**: Mount Google Drive (`drive.mount('/content/drive')`).
- **Bước 2**: Hệ thống sẽ tự động phát hiện `Thread_Data/DeepFake-Detection-Agent.zip` hoặc thư mục đã giải nén trên Drive, copy/extract sang `/content/DeepFake-Detection-Agent` với tốc độ cực nhanh (chưa tới 3 giây).
- **Bước 3 & 4**: Cài đặt dependencies và thực thi pipeline `colab_sync_to_gdrive.py` hoàn toàn độc lập, 100% không yêu cầu GitHub Token!
