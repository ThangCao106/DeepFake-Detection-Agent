# 📊 BÁO CÁO TOÀN DIỆN HUẤN LUYỆN 2 MÔ HÌNH (DUAL-STREAM BENCHMARK)
**Thời gian tạo**: `2026-09-21 16:46:38Z` | **Chu kỳ Crawl & Train**: `Mỗi 4 Giờ` | **Môi trường**: `Google Colab GPU`

---

## 1. Cấu Trúc Phân Bổ Dataset Giữa 2 Mô Hình (Dataset Allocation)

Để tối ưu hóa khả năng khái quát hóa và ngăn ngừa hiện tượng thiên vị đặc trưng (feature entanglement), hệ thống phân tách nghiêm ngặt dữ liệu huấn luyện thành 2 tập độc lập:

### 🖼️ Mô Hình 1: Tổng Quát (General Vision & Signal Model)
*Tập trung vào phát hiện các vết tích toán học vi sai, tần số DWT Haar, phổ FFT của các mô hình sinh AI tổng quát.*
* **Dataset Cơ Sở**:
  1. `NTIRE-RobustAIGenDetection`: ~120,000 ảnh (MSU 2024 benchmark, diffusion & GAN đa dạng).
  2. `CIFAKE`: 120,000 ảnh (60,000 ảnh CIFAR-10 thật vs 60,000 ảnh Stable Diffusion tổng hợp).
  3. `DF40`: 40,000 mẫu video/ảnh (40 thuật toán giả mạo tiên tiến nhất 2026).
  4. `DFFD (Diverse Fake Face Dataset)`: 298,000 ảnh (58k thật, 240k giả mạo GAN).
* **Dữ liệu Huấn Luyện Tăng Cường (Continual Active Learning)**:
  - Khai phá mẫu biên khó (Hard Negatives) từ Meta Threads: Entropy dự đoán $H(P) \ge 0.70$.
* **Quy mô tập Model 1**: **>578,000+ mẫu dữ liệu tổng quát**.

### 🧬 Mô Hình 2: Sinh Trắc Học Con Người (Human Biometrics Model)
*Tập trung chuyên sâu vào giải phẫu khuôn mặt, sinh trắc học da, mạch đập mao mạch (rPPG), giọng nói thanh quản và tính đồng bộ môi - âm thanh.*
* **4 Dataset Human Chuyên Biệt Bổ Sung**:
  1. **`MultiFakeVerse`**: **>845,000 ảnh** (Person-centric VLM manipulation - biến đổi ngữ cảnh, hành vi, tương tác người-vật tinh vi).
  2. **`HiDF (A Human-Indistinguishable Deepfake Dataset)`**: **~62,000 ảnh & 8,000 video** (Mẫu deepfake độ phân giải cực cao thách thức mắt thường).
  3. **`140k Real and Fake Faces Dataset`**: **140,000 ảnh** (70,000 chân dung thật Flickr Nvidia vs 70,000 mặt giả StyleGAN/StyleGAN2).
  4. **`DeepDetect-2025 / 2026`**: Tập dữ liệu đối kháng đối chứng ảnh chân dung Unsplash thật kết hợp khuôn mặt Midjourney v6, StyleGAN3, Stable Diffusion 3 / Flux.1.
* **Dữ liệu In-The-Wild Meta Threads (Quota mỗi chu kỳ 4h)**:
  - **10,000 ảnh chân dung người thật & can thiệp**.
  - **500 video khuôn mặt & chuyển động môi**.
  - **500 clip âm thanh giọng nói**.
  - **5 Phân Lớp Sinh Trắc Học Người Việt**:
    1. `vn_id_official`: Ảnh thẻ, CCCD gắn chip, chụp nghiêm túc chuẩn sinh trắc học.
    2. `vn_selfie_lifestyle`: Ảnh selfie điện thoại, ánh sáng tự nhiên, góc nghiêng.
    3. `vn_street_outdoor`: Ảnh đường phố Việt Nam, ánh sáng phức tạp, xe cộ, bóng râm.
    4. `vn_occlusion_glasses_mask`: Khuôn mặt che khuất một phần (đeo kính râm, khẩu trang y tế).
    5. `vn_mature_aging`: Người trung niên và cao tuổi với nếp nhăn tự nhiên.
* **Quy mô tập Model 2**: **>1,105,000+ mẫu sinh trắc học người**.

---

## 2. Kết Quả Huấn Luyện & Chỉ Số Đánh Giá (Evaluation Metrics)

| Tiêu Chí Đánh Giá | Mô Hình 1 (ViT General Vision) | Mô Hình 2 (Human Biometric Multi-Modal) |
|:---|:---:|:---:|
| **Kiến Trúc Backbone** | Vision Transformer (ViT-B/16) | Fusion: ViT DWT + rPPG + WavLM + LipSync |
| **Hàm Mất Mát** | Focal Loss ($\gamma=2.0, \alpha=0.25$) | Multi-Modal Cross-Entropy + Contrastive |
| **Độ Chính Xác Tổng Thể (Acc)** | **N/A%** | **N/A%** |
| **Độ Chính Xác Sinh Trắc Mặt (WAVIE + rPPG)** | — | **N/A%** |
| **Độ Chính Xác Sinh Trắc Giọng (WavLM Layer 12)** | — | **N/A%** |
| **Độ Chính Xác Khớp Môi (DF26 LipSync)** | — | **N/A%** |
| **Area Under ROC (AUC)** | **N/A** | **N/A** |
| **Equal Error Rate (EER)** | **1.12%** | **N/A%** |
| **Expected Calibration Error (ECE)** | **0.0145** | **N/A** |

---

## 3. Hiệu Chuẩn Phân Phối Xác Suất (Uncertainty & Calibration)

Nhằm triệt tiêu cảnh báo giả (False Positive) đối với người Việt do điều kiện camera chất lượng thấp, ánh sáng gắt hoặc đeo kính/khẩu trang:
* **Nhiệt độ hiệu chuẩn Platt Scaling**:
  $$P_{calibrated}(y=1|x) = \sigma\left(\frac{z}{T^*}\right) \quad \text{với} \quad T^* = 1.6178$$
* **Ngưỡng quyết định tối ưu Youden's J**:
  $$\tau^* = 0.8118$$
* **Hiệu quả**: ECE giảm từ 0.082 xuống **< 0.016**, độ tin cậy của mô hình phản ánh chính xác xác suất thực nghiệm của mẫu thử.

---

## 4. Danh Mục Lưu Trữ & Đồng Bộ Trên Google Drive

Toàn bộ dữ liệu huấn luyện, siêu dữ liệu và trọng số mô hình đã được đồng bộ tự động vào Google Drive:
* 📁 **Thư mục Dữ Liệu Ngày**: `Thread_Data/2026-09-21/`
  - `01_raw_originals/`: Ảnh & video nguyên bản chưa qua can thiệp.
  - `02_manipulated_faceswap/`: Mẫu hoán đổi khuôn mặt (Face-Swap).
  - `03_manipulated_openweight/`: Mẫu tạo bởi mã nguồn mở (FLUX.1 / SD3).
  - `dataset_manifest.jsonl`: Danh mục chi tiết từng file kèm nhãn 5 phân lớp người Việt và chỉ số $E_{high}, Z_{peak}, H(P)$.
* 💾 **Thư mục Checkpoints**: `Thread_Data/checkpoints/`
  - `model_continual_latest.pt`: Trọng số Mô hình 1 (General Vision).
  - `weights_human.pt`: Trọng số Mô hình 2 (Human Biometrics).
  - `human_training_meta.json`: Siêu dữ liệu huấn luyện chi tiết của Mô hình 2.
* 📄 **Báo Cáo**: `Thread_Data/reports/training_report_dual_stream.md`

---
*Báo cáo được khởi tạo tự động bởi Threads-DFBench Dual-Stream Autonomous Daemon.*
