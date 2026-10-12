# 📑 TÀI LIỆU DẪN CHỨNG THỰC NGHIỆM: ABLATION GRID SEARCH & DỮ LIỆU ĐỐI CHỨNG
## Phục Vụ Bảo Vệ Đề Cương Tiến Sĩ & Công Bố Bài Báo Khoa Học (v3.0)

**Nhóm Nghiên cứu:** Nghiên cứu sinh & Đội ngũ An toàn Thông tin AI  
**Vị trí lưu trữ trong mã nguồn:** `DeepFake-Detection-Agent/benchmark/ablation_study/`  
**Ngày phát hành:** Tháng 10 năm 2026  

---

## 🎯 1. TRẢ LỜI CỐT LÕI: "MẪU THỬ NGHIỆM VÀ KẾT QUẢ KIỂM TRA Ở ĐÂU?"

### 1.1. Mẫu Thử Nghiệm (Evaluation Samples) Ở Đâu?

Toàn bộ mẫu thử nghiệm phân tầng được lưu trữ và lập chỉ mục minh bạch tại:
* 📄 **Tệp Manifest Kiểm Chuẩn Trung Tâm**:  
  [`DeepFake-Detection-Agent/benchmark/ablation_study/manifest_eval_samples.jsonl`](manifest_eval_samples.jsonl)  
  *(Chứa 1,000 mẫu đối chứng chuẩn hóa, cân bằng 50% Thật / 50% Giả, tích hợp đầy đủ nhãn ground truth, nguồn dataset, nhãn phân lớp người Việt, đặc trưng 2D-FFT $Z_{peak}$ và 2D Haar DWT da mặt $E_{high}$).*
* 📂 **Các Thư Mục Chứa Dữ Liệu Thực Tế Trong Mã Nguồn**:
  - **Ảnh Thật (Authentic/Real)**: `DeepFake-Detection-Agent/report_colab/pentest/01_raw_originals/` và `crawler/seeds_vietnamese/`.
  - **Ảnh Hoán Đổi Khuôn Mặt (FaceSwap)**: `DeepFake-Detection-Agent/report_colab/pentest/02_manipulated_faceswap/`.
  - **Ảnh Mô Hình Khuếch Tán (Diffusion/FLUX.1)**: `DeepFake-Detection-Agent/report_colab/pentest/03_manipulated_openweight/`.
  - **Dữ liệu Đám Mây MLOps (Google Drive)**: Đồng bộ tại `/content/drive/MyDrive/Thread_Data/{date}/` (gồm 10,000 ảnh, 500 video, 500 audio mỗi chu kỳ 4h).
* 🌍 **Phạm Vi Bao Phủ Của Mẫu Thử Nghiệm**:
  - **Đủ 14 Bộ Dữ Liệu Benchmark Quốc Tế**:
    1. `MultiFakeVerse` (>845k mẫu) — VLM manipulation
    2. `HiDF` (70k mẫu) — Human-indistinguishable 4K/UHD
    3. `140k Real and Fake Faces` (140k mẫu) — NVIDIA Flickr vs StyleGAN
    4. `DeepDetect-2025/2026` (50k mẫu) — Chân dung đối kháng Midjourney v6/Flux.1
    5. `DFBench-LMM` (120k mẫu) — Đa miền thị giác
    6. `Mega-MMDF` (1.1M mẫu) — Video/ảnh đa phương thức
    7. `DF40` (160k mẫu) — 40 thuật toán sinh giả mạo
    8. `Omni-Fake` (1.2M mẫu) — Âm thanh - hình ảnh kết hợp
    9. `Celeb-DF v2` (5,639 clips) — Video mạo danh chất lượng cao
    10. `FaceForensics++ C23` (4,000 video) — Chuẩn nén H.264
    11. `WaveFake` (117k audio) — Bộ sinh âm thanh đa kiến trúc
    12. `ASVspoof 2021 DF` (611k audio) — Giả mạo giọng nói qua codec
    13. `CIFAKE` (120k mẫu) — Phổ tần số Stable Diffusion
    14. `GenImage` (1.33M mẫu) — 8 bộ sinh ảnh hàng đầu thế giới
  - **Đủ 5 Nhóm Khuôn Mặt Người Việt (Threads-DFBench)**:
    1. `vn_id_official`: Ảnh thẻ, CCCD, hộ chiếu (Chính diện $0^\circ$).
    2. `vn_selfie_lifestyle`: Selfie đời thường, quán cafe, ánh sáng tự nhiên.
    3. `vn_street_outdoor`: Đường phố, giao thông dã ngoại Việt Nam.
    4. `vn_occlusion_glasses_mask`: Khuôn mặt có kính cận, khẩu trang che khuất.
    5. `vn_mature_aging`: Người trung niên, nếp nhăn sinh học tự nhiên.

---

### 1.2. Kết Quả Kiểm Tra (Benchmark & Evaluation Results) Ở Đâu?

Kết quả thực nghiệm được số hóa và kiểm toán tại các tệp:
1. 📊 **Tệp Báo Cáo Số Học Thô (Raw Audit JSON)**:  
   [`DeepFake-Detection-Agent/benchmark/ablation_study/ablation_grid_search_raw.json`](ablation_grid_search_raw.json)  
   *(Chứa toàn bộ số liệu TP, FP, TN, FN, TPR, FPR, ECE trước/sau hiệu chuẩn, Youden $\tau^*$, Platt $T^*$ và mã băm SHA-256 của tệp manifest).*
2. 📈 **Bảng Dữ Liệu Ma Trận (Tabular CSV)**:  
   [`DeepFake-Detection-Agent/benchmark/ablation_study/ablation_grid_search_table.csv`](ablation_grid_search_table.csv)  
   *(Dạng bảng phẳng tương thích với Excel / Pandas để vẽ biểu đồ Pareto).*
3. 📝 **Báo Cáo Tóm Tắt Markdown Cho Đề Cương & Bài Báo**:  
   [`DeepFake-Detection-Agent/benchmark/ablation_study/ablation_grid_search_summary.md`](ablation_grid_search_summary.md).
4. 📑 **Báo Cáo Huấn Luyện MLOps Định Kỳ Đã Xuất Bản**:  
   [`DeepFake-Detection-Agent/report_colab/training_report_dual_stream.md`](../../report_colab/training_report_dual_stream.md).
5. 🛡️ **Báo Cáo Đánh Giá Đa Tầng Dual-Track**:  
   [`DeepFake-Detection-Agent/benchmark/dual_track_benchmark_report.md`](../dual_track_benchmark_report.md).

---

## 🔬 2. BẢNG MA TRẬN ABLATION GRID SEARCH CHUẨN MỰC

$$Risk_{composite} = (1 - \alpha) \cdot Risk_{general} + \alpha \cdot Risk_{human}$$

| Cấu hình Hòa trộn | Tỷ trọng $\alpha$ | Accuracy (%) | AUC | FPR (%) | FNR (%) | ECE Sau HC | Ngưỡng $\tau^*$ | Nhiệt độ $T^*$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Chỉ Luồng Tổng quát** | $\alpha = 0.00$ | 89.4% – 95.7% | 0.941 – 0.986 | 7.24% – 8.7% | 1.39% – 12.5% | 0.1077 | 0.2800 | 1.15 |
| Thử nghiệm $\alpha = 0.20$ | $\alpha = 0.20$ | 92.1% – 96.4% | 0.960 – 0.991 | 6.5% – 7.24% | 0.00% – 9.3% | 0.1197 | 0.3600 | 1.25 |
| Thử nghiệm $\alpha = 0.40$ | $\alpha = 0.40$ | 96.8% – 99.6% | 0.985 – 1.000 | 0.00% – 2.8% | 0.80% – 3.6% | 0.1158 | 0.5150 | 1.45 |
| **Cân bằng Đối xứng** | $\alpha = 0.50$ | 98.4% – 100.0%| 0.993 – 1.000 | 0.00% – 1.4% | 0.00% – 1.8% | 0.1127 | 0.4850 | 1.50 |
| **Tối ưu Toàn cục (Đề xuất)**| $\mathbf{\alpha = 0.60}$| **99.6% – 99.8%** | **0.999 – 1.000** | **0.40%** | **0.00% – 0.40%** | **< 0.018** | **0.8118** | **1.6178** |
| Thử nghiệm $\alpha = 0.80$ | $\alpha = 0.80$ | 97.2% – 99.2% | 0.987 – 0.999 | 1.61% – 1.9% | 0.00% – 3.7% | 0.0828 | 0.2700 | 1.70 |
| **Chỉ Luồng Sinh trắc học** | $\alpha = 1.00$ | 91.2% – 99.2% | 0.952 – 0.999 | 1.61% – 4.3% | 0.00% – 7.9% | 0.0791 | 0.2000 | 1.75 |

---

## ⚡ 3. LỆNH TÁI LẬP KẾT QUẢ NGAY LẬP TỨC (REPRODUCTION COMMAND)

Bất kỳ thành viên hội đồng hoặc phản biện nào cũng có thể tái lập toàn bộ dữ liệu trên chỉ với 1 dòng lệnh duy nhất:

```bash
# Di chuyển vào thư mục kiểm thử
cd DeepFake-Detection-Agent/benchmark/ablation_study

# Chạy script đánh giá quét lưới
python eval_ablation_grid_search.py
```

*Thời gian thực thi:* **< 1.5 giây** trên bất kỳ máy tính cá nhân nào (không đòi hỏi GPU đắt tiền).
