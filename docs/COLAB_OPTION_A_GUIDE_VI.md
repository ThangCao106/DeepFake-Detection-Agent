# 🚀 Hướng Dẫn Sử Dụng Option A: Google Colab Direct Mount & GDrive Sync

> **Không cần Google Cloud Console! Không cần Service Account JSON hay API Key!**  
> Dữ liệu được đọc/ghi trực tiếp vào Google Drive thông qua cơ chế `drive.mount('/content/drive')`.

---

## 📌 Cách Hoạt Động Của Option A

Khi bạn mở Google Colab và chạy `drive.mount('/content/drive')`, Google Colab sẽ cấp quyền cho notebook truy cập toàn quyền vào Google Drive cá nhân của bạn như một ổ đĩa thư mục bình thường (`/content/drive/MyDrive/`).

Script `colab_sync_to_gdrive.py` sẽ:
1. Tự động tìm thư mục `Deepfake Detection` hoặc `DeepFake_Pentest_Data` trên Google Drive của bạn.
2. Tự động sinh số lượng ảnh kiểm định theo mong muốn (ví dụ 20 ảnh/xô dữ liệu).
3. Đẩy thẳng toàn bộ ảnh `.jpg` vào 3 thư mục:
   - `GDrive/DeepFake_Pentest_Data/YYYY-MM-DD/01_raw_originals/`
   - `GDrive/DeepFake_Pentest_Data/YYYY-MM-DD/02_manipulated_faceswap/`
   - `GDrive/DeepFake_Pentest_Data/YYYY-MM-DD/03_manipulated_openweight/`

---

## 🛠️ Các Bước Thực Hiện Trên Google Colab

### Bước 1: Mở Notebook Trên Colab & Gắn Ổ Đĩa Google Drive
Tạo một Notebook mới trên [Google Colab](https://colab.research.google.com/) hoặc mở `Colab_MultiModality_AutoTrainer.ipynb`, thêm ô mã đầu tiên:

```python
from google.colab import drive
drive.mount('/content/drive')
```
*Nhấn nút "Connect to Google Drive" khi cửa sổ popup xuất hiện.*

### Bước 2: Clone / Cập Nhật Repository
Chạy ô mã sau để tải mã nguồn mới nhất:

```bash
!git clone https://github.com/ThangCao106/DeepFake-Detection-Agent.git /content/DeepFake-Detection-Agent || (cd /content/DeepFake-Detection-Agent && git pull origin main)
%cd /content/DeepFake-Detection-Agent
!pip install -q Pillow httpx
```

### Bước 3: Chạy Đồng Bộ & Tùy Chỉnh Số Lượng Ảnh, Video, Audio

Bạn có thể chỉnh sửa số lượng ảnh, video, audio và số lượng ảnh kiểm định (`--samples`) tùy ý:

```bash
# Ví dụ: Sinh 20 ảnh kiểm định trực quan mỗi bucket và crawl 1000 ảnh, 1000 video, 1000 audio:
!python colab_sync_to_gdrive.py --images 1000 --videos 1000 --audios 1000 --samples 20
```

*Nếu bạn muốn chạy thử nhanh với số lượng nhỏ:*
```bash
!python colab_sync_to_gdrive.py --images 100 --videos 100 --audios 100 --samples 10
```

Toàn bộ ảnh `.jpg` sẽ xuất hiện ngay lập tức trong thư mục Google Drive của bạn!

---

## ⚙️ Tùy Chỉnh Số Lượng Trên GitHub Actions (Nếu Muốn Chạy Tự Động)

Nếu bạn chạy qua GitHub Actions:
1. Vào tab **Actions** $\rightarrow$ Chọn **Scheduled Model Pentest & OOD Evaluation Job**.
2. Nhấn nút **Run workflow** (bên phải).
3. Trên giao diện xuất hiện các ô nhập số lượng tùy chỉnh:
   - **Số lượng ảnh cần crawl**: Mặc định `1000` (có thể sửa thành `500`, `2000`...)
   - **Số lượng video cần crawl**: Mặc định `1000`
   - **Số lượng audio cần crawl**: Mặc định `1000`
   - **Số lượng ảnh kiểm định sinh ra (.jpg)**: Mặc định `20`
   - **Mô hình mục tiêu**: `all`, `general`, hoặc `human`.
4. Nhấn **Run workflow**.
