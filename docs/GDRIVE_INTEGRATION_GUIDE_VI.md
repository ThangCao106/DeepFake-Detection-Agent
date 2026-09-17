# 📖 Hướng Dẫn Tích Hợp Google Drive & GitHub Token Cho Dự Án DeepFake Detection

Tài liệu này hướng dẫn chi tiết từng bước để kết nối dự án **DeepFake Detection Agent** với thư mục Google Drive và cấu hình GitHub Token cho Google Colab Pro Scheduled Runner.

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

## 🚀 PHẦN 3: Cấu Hình Option B2 Cho Colab Pro Scheduled Runner (GitHub Personal Access Token)
> **Mục đích**: Tự động hóa kéo mã nguồn 100% khi chạy **Scheduled Run (Headless 4h/lần)** trên Colab Pro mà vẫn giữ Private Repository an toàn tuyệt đối, không cần copy file zip thủ công lên Google Drive.

### 📍 Bước 1: Tạo GitHub Personal Access Token (Mất 30 giây)
1. Truy cập trực tiếp: [GitHub Token Settings](https://github.com/settings/tokens) (hoặc vào GitHub cá nhân -> **Settings** -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)**).
2. Nhấn nút **Generate new token** -> Chọn **Generate new token (classic)**.
3. Điền các trường:
   - **Note**: `colab_scheduled_runner`
   - **Expiration**: Chọn thời hạn (ví dụ `90 days` hoặc `No expiration` nếu dùng lâu dài).
   - **Select scopes**: Tích chọn duy nhất ô **`repo`** (Full control of private repositories).
4. Kéo xuống dưới cùng, nhấn nút màu xanh **Generate token**.
5. Sao chép chuỗi mã token vừa tạo (bắt đầu bằng ký tự `ghp_...`). *(Lưu ý: GitHub chỉ hiển thị token này một lần).*

### 📍 Bước 2: Lưu Token Vào Colab Secrets (Bảo Mật An Toàn)
1. Mở file Notebook **`Colab_Pro_Scheduled_Runner.ipynb`** trên Google Colab.
2. Nhìn vào thanh công cụ màu xám ở lề bên trái màn hình Colab, nhấp vào biểu tượng chiếc chìa khóa **🔑 (Secrets)**.
3. Nhấn **+ Add new secret**:
   - **Name**: Nhập chính xác `GH_TOKEN`
   - **Value**: Dán chuỗi token `ghp_...` vừa sao chép ở Bước 1.
4. Bật công tắc **Notebook access** (Cho phép sổ tay truy cập) sang màu xanh.

### 📍 Bước 3: Khởi Chạy Scheduled Run
- Mỗi khi Scheduled Run kích hoạt (mỗi 4 tiếng):
  - Colab sẽ tự động đọc `GH_TOKEN` từ Secrets.
  - Tự động kéo commit mới nhất từ nhánh `main` (`git clone` hoặc `git pull`).
  - Chạy `colab_sync_to_gdrive.py` mượt mà 100%, không bao giờ báo lỗi `fatal: could not read Username`!
