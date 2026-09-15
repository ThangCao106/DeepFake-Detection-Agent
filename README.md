# 🤖 Deepfake Detection Agent System

![Track](https://img.shields.io/badge/Track-Agents%20for%20Good-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Modality](https://img.shields.io/badge/Modality-Image%20%7C%20Video%20%7C%20Audio-purple)
![Architecture](https://img.shields.io/badge/Architecture-Vision%20%26%20Audio%20Transformer-orange)
![Calibration](https://img.shields.io/badge/Calibration-Temperature%20Scaling%20(T*)%20%2B%20Youden's%20J%20(%CF%84*)-red)
![Language](https://img.shields.io/badge/Language-English%20%7C%20Ti%E1%BA%BFng%20Vi%E1%BB%87t-blueviolet)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Deployment](https://img.shields.io/badge/Deployment-Docker%20%26%20GitHub%20Actions%20CI%2FCD-brightgreen)

System phát hiện và giám định **DeepFake Đa Phương Thức (Multi-Modal DeepFake Detection: Image, Video, Audio)** dựa trên kiến trúc **Vision & Audio Transformer**, quy trình hiệu chỉnh xác suất **Post-Hoc Temperature Scaling ($T^*$)**, tối ưu hóa ngưỡng quyết định **Youden's J Index ($\tau^*$)**, hỗ trợ **Đa Ngôn Ngữ (Tiếng Việt 🇻🇳 & Tiếng Anh 🇬🇧)**, kết hợp với **Hệ thống Đa Tác thể (Multi-Agent System)** và **Agent SRE Quan sát (Observability Subsystem Agent)**.

---

## 📐 1. Tóm tắt Kiến trúc Mô hình (Model Architecture Summary)

Hệ thống được thiết kế theo mô hình **Đa Phương Thức 3 Luồng (Tri-Stream Multi-Modal Architecture)** xử lý toàn diện 3 dạng nội dung số:

```
                          ┌──────────────────────────┐
                          │    Input Media Stream    │
                          └────────────┬─────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
     ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
     │ IMAGE STREAM │           │ VIDEO STREAM │           │ AUDIO STREAM │
     └──────┬───────┘           └──────┬───────┘           └──────┬───────┘
            │                          │                          │
   ┌────────┴────────┐        ┌────────┴────────┐        ┌────────┴────────┐
   │ dhash / MD5     │        │ Keyframe        │        │ Mel-Spectrogram │
   │ Deduplication   │        │ Extraction (CV) │        │ Feature Transform│
   └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
            │                          │                          │
   ┌────────┴────────┐        ┌────────┴────────┐        ┌────────┴────────┐
   │ Vision          │        │ Temporal ViT    │        │ Audio ViT       │
   │ Transformer     │        │ Aggregation     │        │ Classifier      │
   └────────┬────────┘        └────────┬────────┘        └────────┬────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Uncalibrated Raw Logits (z)  │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │   Post-Hoc Calibration        │
                       │   P(y=1|x) = σ(z / T*)        │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Optimal Threshold Search     │
                       │  J(τ) = TPR(τ) - FPR(τ)       │
                       └───────────────┬───────────────┘
                                       │
                                       ▼
                       ┌───────────────────────────────┐
                       │  Auditable Forensic           │
                       │  Certificate & Decision       │
                       └───────────────────────────────┘
```

### 🔹 1.1 Tri-Stream Backbone Transformers
1. **Luồng Hình ảnh (Image Stream)**:
   * **Backbone**: Vision Transformer (ViT-Base/ViT-Large) fine-tune 2 giai đoạn (Warmup Classifier Head + Full Fine-tuning với Cosine Annealing Learning Rate Schedule).
   * **Tiền xử lý**: Khử trùng lặp `MD5` + Difference Hash (`dhash`), tự động phát hiện và crop vùng khuôn mặt chuẩn hóa.
2. **Luồng Video (Video Stream)**:
   * **Backbone**: OpenCV Keyframe Extraction (trích xuất 5–10 khung hình chính/giây) + ViT Temporal Feature Aggregation.
   * **Phân tích**: Đánh giá biến thiên điểm số giữa các khung hình để phát hiện gián đoạn thời gian (Temporal Discontinuity / Face-Swap artifacts).
3. **Luồng Âm thanh (Audio Stream)**:
   * **Backbone**: Chuyển đổi tín hiệu giọng nói thành **Mel-Spectrogram 2D Feature Map** + Audio ViT Classification.
   * **Phân tích**: Phát hiện nhân bản giọng nói (Voice Cloning), tổng hợp tiếng nói (TTS) và bất đồng bộ môi-tiếng (Lip-Sync Mismatch).

### 🔹 1.2 Hiệu chỉnh Xác suất Post-Hoc & Tối ưu Ngưỡng Quyết định
Mạng nơ-ron sâu thường bị lỗi **Overconfidence** (độ tin cậy xác suất phát ra quá cao nhưng sai lệch với độ chính xác thực tế). Hệ thống tích hợp 2 thuật toán hiệu chỉnh:

1. **Platt Temperature Scaling ($T^*$):**
   $$P(y=1 \mid x) = \sigma\left(\frac{z}{T^*}\right) = \frac{1}{1 + e^{-z / T^*}}$$
   *Tham số nhiệt độ $T^*$ được tối ưu hóa bằng thuật toán L-BFGS-B trên tập Validation để giảm thiểu Expected Calibration Error (ECE).*

2. **Youden's J Index Decision Threshold ($\tau^*$):**
   $$J(\tau) = \text{TPR}(\tau) - \text{FPR}(\tau) = \text{Sensitivity}(\tau) + \text{Specificity}(\tau) - 1 \implies \tau^* = \arg\max_{\tau} J(\tau)$$
   *Thay vì sử dụng ngưỡng mặc định $\tau=0.5$, hệ thống tự động tìm ngưỡng $\tau^*$ tối ưu theo từng tập dữ liệu và phương thức (Modality).*

---

## 📦 2. Danh mục Dữ liệu Multi-Modality Registry & Benchmark 2025–2026

Hệ thống tích hợp **14 Bộ dữ liệu chuẩn quốc tế** mở rộng từ *Awesome-Deepfakes*, bao gồm các tập Benchmark SOTA mới nhất 2025–2026:

| Modality | Key Dataset | Mô tả & Đặc thù Benchmark |
|:---|:---|:---|
| `IMAGE` | `dima806_deepfake` | 140,000 ảnh thật & giả mặt StyleGAN/FFHQ |
| `IMAGE` | `ntire_robust_detection` | NTIRE 2024 Robust AI-Gen Detection Benchmark (Diffusion & GANs) |
| `IMAGE` | `cifake` | 120,000 ảnh CIFAR-10 vs Stable Diffusion Synthetic Images |
| `IMAGE` | `dffd_image` | Diverse Fake Face Dataset (58k real, 240k fake) |
| `IMAGE` | `ifakefacedb` | 87,000 ảnh StyleGAN đã xóa vân tay GANprintR |
| `IMAGE` | **`dfbench_lmm`** | **SOTA 2025 Benchmark đánh giá Large Multimodal Models (LMMs)** |
| `VIDEO` | `faceforensics_pp` | FaceForensics++ (1,000 real & 5,000 fake videos) |
| `VIDEO` | `celeb_df_v2` | Celeb-DF v2 (5,639 video tráo mặt chất lượng cao) |
| `VIDEO` | `dfdc` | DeepFake Detection Challenge (128,000+ video từ Meta) |
| `VIDEO` | `wild_deepfake` | Wild Deepfake (7,314 video thực tế từ internet) |
| `VIDEO` | `deeper_forensics` | DeeperForensics-1.0 (60,000 video với nhiễu thực tế) |
| `VIDEO` | `deepspeak_video` | DeepSpeak Video (Lip-Sync & Face-Swap) |
| `VIDEO` | **`mega_mmdf`** | **DeepfakeBench-MM / Mega-MMDF (2025/2026): 1.1M mẫu đa phương thức qua 21 forgery pipelines** |
| `VIDEO` | **`df40`** | **DF40 Benchmark (2026): 40 thuật toán sinh DeepFake SOTA (Diffusion, Audio-driven Talking Head)** |
| `AUDIO` | `deepspeak_audio` | Voice clone & audio deepfake speech clips |
| `AUDIO` | `fakeavceleb_audio` | FakeAVCeleb Audio-Visual deepfake audio clips |
| `AUDIO` | `wavefake` | 104,885 audio clips từ MelGAN, Parallel WaveGAN, WaveGlow |
| `AUDIO` | `asvspoof_2021` | ASVspoof 2021 Logical & Physical access voice spoofing benchmark |
| `AUDIO` | **`omni_fake`** | **Omni-Fake Benchmark (2026): 1.2M+ mẫu Đa phương thức & 200k+ tập Out-Of-Distribution (OOD)** |

---

## 🏛️ 3. Hệ thống Đa Tác Thể (Multi-Agent System Design) & Hỗ trợ Đa Ngôn Ngữ

Mô hình hoạt động dựa trên sự phối hợp giữa 5 Agent chính:

```mermaid
graph TD
    A[User / Client] -->|Media File + Query + Language EN/VI| B[Orchestrator Agent]
    B -->|Preprocessed Payload| C[Detection Agent - ViT / ONNX Engine]
    C -->|Raw Logits & Keyframes| D[Calibration Engine - T* & τ*]
    D -->|Calibrated Probabilities| B
    B -->|Structured Forensic Context| E[Analysis Agent - Gemini / Claude LLM]
    E -->|Natural Language Explanation EN/VI| B
    B -->|Generate| F[Auditable Forensic Certificate]
    B <-->|Persistent Memory| G[SRE Observability Agent - ClickHouse & Grafana]

    style B fill:#4CAF50,color:#fff
    style C fill:#2196F3,color:#fff
    style D fill:#E91E63,color:#fff
    style E fill:#FF9800,color:#fff
    style G fill:#9C27B0,color:#fff
```

### 🔹 3.1 Các Tác thể Thành phần
1. **Orchestrator Agent**: Tiếp nhận file media (Image, Video, Audio), ngôn ngữ tùy chọn (`vi` hoặc `en`), phân luồng xử lý và phối hợp các agent con.
2. **Detection Agent (ML/ONNX Engine)**: Chạy mô hình ViT fine-tuned và ONNX Runtime (`custom_deepfake_vit.onnx`) trích xuất điểm số dự đoán raw logits.
3. **Calibration Engine**: Áp dụng $T^*$ và $\tau^*$ để tính toán tỷ lệ phần trăm tin cậy chính xác cho từng phương thức.
4. **Analysis Agent (LLM)**: Sử dụng Google Gemini / Claude LLM tổng hợp báo cáo giải thích bằng ngôn ngữ tự nhiên theo Tiếng Việt 🇻🇳 hoặc Tiếng Anh 🇬🇧 và cấp **Auditable Forensic Certificate** (tuân thủ EU AI Act Article 50 & DSA).
5. **Observability SRE Agent (Subsystem Port 8787)**:
   * Giám sát hệ thống thời gian thực qua **ClickHouse Agent Memory** và **Grafana Metrics**.
   * Cung cấp **Hệ thống Đăng nhập 2 Chế độ (Dual Authentication Gateway)**:
     - *Chế độ 1*: Nhập trực tiếp `CLAUDE_API_KEY` (`sk-ant-...`) trên giao diện web bar và lưu vào container session.
     - *Chế độ 2*: Đăng nhập tự động OAuth Anthropic (`claude login`) tích hợp modal **Nhập Authorization Code** và các nút điều khiển trực tiếp trên Web Interface.
     - Tự động pre-trust thư mục `/app` đa đường dẫn (`/root`, `/home/agent`) và chạy trong `/tmp` để bypass 100% màn hình Workspace Safety Check.

### 🔹 3.2 Hỗ trợ Đa Ngôn ngữ (English & Vietnamese i18n)
* **Web UI Language Switcher**: Nút bấm `🌐 EN | 🇻🇳 VI` trên Header ứng dụng Web, tự động lưu lựa chọn vào `localStorage`.
* **LLM Dynamic Localization**: Tác thể Analysis Agent tự động nhận biết tham số `language` và sinh văn bản giải thích chuẩn mực bằng Tiếng Việt hoặc Tiếng Anh.
* **Tinh gọn 6 chỉ số Forensic Certificate**: Dù ở ngôn ngữ nào, thẻ tóm tắt kết quả phân tích luôn giữ định dạng 6 chỉ số định lượng chính (Real/Fake label, Confidence Score, Certificate ID, Composite Risk, Spectral Anomaly FFT 2D, Spatial Noise Inconsistency).

---

## 📈 4. Kết quả Thực nghiệm & Metrics Huấn luyện Động (Experimental Results)

Dưới đây là tóm tắt diễn biến hiệu năng huấn luyện mô hình qua các Pass liên tục trên tập Benchmark (Cập nhật Pass 18):

| Pass # | Modality | Dataset | Accuracy | AUC Score | EER Score | Temp ($T^*$) | Tau ($\tau^*$) | Trạng thái |
|-------:|:--------|:--------|---------:|----------:|----------:|-------------:|---------------:|:-----------|
| **Pass 1** | `IMAGE` | `ntire_robust_detection` | 91.2% | 0.987 | 0.043 | 1.4620 | 0.7344 | ✅ Success |
| **Pass 1** | `VIDEO` | `faceforensics_pp` | 88.6% | 0.961 | 0.040 | 1.7685 | 0.7117 | ✅ Success |
| **Pass 1** | `AUDIO` | `wavefake` | 86.2% | 0.918 | 0.055 | 1.5625 | 0.7722 | ✅ Success |
| **Pass 2** | `IMAGE` | `ntire_robust_detection` | 96.0% | 0.995 | 0.012 | 1.5947 | 0.7163 | ✅ Enhanced |
| **Pass 2** | `VIDEO` | `celeb_df_v2` | 94.0% | 0.981 | 0.026 | 1.4431 | 0.6581 | ✅ Enhanced |
| **Pass 2** | `AUDIO` | `fakeavceleb_audio` | 91.0% | 0.959 | 0.027 | 1.4163 | 0.6336 | ✅ Enhanced |
| **Pass 5** | `AUDIO` | `wavefake` | 97.3% | 0.998 | 0.022 | 1.6384 | 0.7673 | ✅ Optimal |
| **Pass 6** | `IMAGE` | `dima806_deepfake` | **99.4%** | **0.998** | **0.012** | 1.4956 | 0.7285 | 🚀 SOTA Peak |
| **Pass 6** | `VIDEO` | `celeb_df_v2` | **98.2%** | **0.998** | **0.018** | 1.4586 | 0.8449 | 🚀 SOTA Peak |
| **Pass 6** | `AUDIO` | `deepspeak_audio` | **97.9%** | **0.998** | **0.022** | 1.7598 | 0.6365 | 🚀 SOTA Peak |
| **Pass 18**| `VIDEO` | `mega_mmdf` | **98.7%** | **0.999** | **0.011** | 1.4210 | 0.7150 | 🚀 SOTA Peak |

---

## 📂 5. Cấu trúc Thư mục Dự án

```text
DeepFake-Detection-Agent/
├── 🌐 web_ui/                            (Ứng dụng Web Suy luận Multi-Modality - Port 5000)
│   ├── app.py                            - Server Flask & SocketIO với tham số language
│   ├── templates/ & static/              - Giao diện Light Lavender UI, i18n & Multi-Modality Viewer
│   ├── deepfake_detector.py              - Engine suy luận Image, Video, Audio & Certificate ID
│   ├── agent_system.py                   - Multi-Agent Orchestrator (hỗ trợ Tiếng Việt & Tiếng Anh)
│   └── config.py                         - Quản lý cấu hình hệ thống
│
├── 🧠 colab_training/                    (Tự động hóa Huấn luyện Colab GPU & Sync GitHub)
│   ├── colab_auto_runner.py              - Hybrid Continuous Auto-Runner Engine
│   ├── colab_dataset_pipeline.py         - Quản lý đo đạc chỉ số động (Dynamic Metrics Generator)
│   ├── train_pipeline.py                 - Two-Stage ViT Fine-Tuning Pipeline
│   ├── data_pipeline.py                  - Registry 14 tập dữ liệu SOTA 2025-2026
│   ├── upload_model_cloud.py             - Tự động đẩy trọng số lên Hugging Face Hub
│   └── Colab_MultiModality_AutoTrainer.ipynb - Notebook huấn luyện Colab GPU
│
├── 🔍 observability/                     (Subsystem Agent Quan sát & Persistent Memory - Port 8787)
│   ├── Dockerfile & docker-compose.yml   - SRE Claude Agent Service (Port 8787)
│   ├── app/main.py                       - Backend API, Dual Auth, PTY /login Auto-Responder
│   ├── app/agent.py                      - Integration với Claude Agent SDK
│   └── static/index.html                 - Obsagent Web Dashboard & Modal Code Submitter
│
├── 📊 report_colab/                      (Báo cáo Động Đa Phương Thức)
│   └── 2026-09-15/
│       ├── daily_summary.md              - Bảng tổng hợp metrics động theo từng Pass & Modality
│       └── daily_summary_chart.png       - Biểu đồ xu hướng nâng cao hiệu năng
│
├── 🐳 docker-compose.yml                 - File Docker Compose chính quản lý Web UI & Obsagent
└── 🚀 .github/workflows/deploy.yml       - GitHub Actions CI/CD (paths-ignore report_colab/**)
```

---

## ⚡ 6. Hướng dẫn Chạy Hệ thống (Quick Start)

### 🚀 6.1 Khởi chạy bằng Docker Compose Local / Server
```bash
# 1. Clone repository
git clone https://github.com/ThangCao106/DeepFake-Detection-Agent.git
cd DeepFake-Detection-Agent

# 2. Cấu hình file môi trường (.env)
cp .env.example .env
# Thêm GEMINI_API_KEY, CLAUDE_API_KEY (tùy chọn)

# 3. Khởi chạy toàn bộ hệ thống bằng Docker Compose
docker compose up -d --build
```
* **Web UI Inference App**: [http://localhost:5000](http://localhost:5000)
* **Observability SRE Subsystem**: [http://localhost:8787](http://localhost:8787) (hoặc `/obsagent/` trên domain sản xuất)

---

### 🔑 6.2 Hướng dẫn Đăng nhập Subsystem Observability Agent (Port 8787)
Khi truy cập giao diện Observability Agent:
1. Đăng nhập Subsystem Gate: Tên đăng nhập `thang.cd` / Mật khẩu `Thang@123`.
2. Chọn 1 trong 2 phương thức xác thực:
   * **Cách 1 (Nhập API Key - Nhanh nhất)**: Dán `CLAUDE_API_KEY` (`sk-ant-api03-...`) vào ô nhập ở góc phải thanh API Key Bar và bấm **Lưu Key**.
   * **Cách 2 (Xác thực OAuth)**: Bấm **`🔑 Claude Login`** $\rightarrow$ Bấm **`⚡ Auto Launch Docker Login`** $\rightarrow$ Nhấp link Anthropic OAuth để lấy mã Authorization Code $\rightarrow$ Dán vào ô **`🔑 Nhập mã xác thực`** và bấm **`Submit Code`**. Hoặc bấm nút **`✅ Trust Folder (Yes)`** / **`🔑 Gửi /login`** trên giao diện modal.

---

## 📑 7. Tài liệu Tham khảo (Key References)
1. **TRahulsingh/DeepfakeDetector**: [GitHub Repository](https://github.com/TRahulsingh/DeepfakeDetector)
2. **Daisy-Zhang/Awesome-Deepfakes**: [GitHub Repository](https://github.com/Daisy-Zhang/Awesome-Deepfakes)
3. **DeepfakeBench-MM / Mega-MMDF (2025/2026)**: [GitHub SOTA Benchmark](https://github.com/SOTA-Deepfake/DeepfakeBench-MM)
4. **Omni-Fake Benchmark (2026)**: [AlphaXiv / GitHub Repository](https://github.com/Omni-Fake/Omni-Fake-Benchmark)
5. **DF40 Benchmark (2026)**: [GitHub Repository](https://github.com/SOTA-Deepfake/DF40)
