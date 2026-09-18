<div align="center">

# 🛡️ AI Deepfake Detection & Digital Forensic Agent System
### Inside the Forensic Harness: A Closed-Loop Multi-Modal Agent Architecture

[![Paper](https://img.shields.io/badge/Paper-v3.0--Auditable-darkblue?style=for-the-badge&logo=arxiv)](https://github.com/ThangCao106/DeepFake-Detection-Agent)
[![HuggingFace](https://img.shields.io/badge/HF_Benchmark-14_Datasets-yellow?style=for-the-badge&logo=huggingface)](https://huggingface.co)
[![Colab Pro](https://img.shields.io/badge/Colab_Pro-Scheduled_4h_Run-orange?style=for-the-badge&logo=googlecolab)](https://colab.research.google.com)
[![EU AI Act](https://img.shields.io/badge/EU_AI_Act-Article_50_Compliant-green?style=for-the-badge&logo=curseforge)](https://artificialintelligenceact.eu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge)](LICENSE)

| **99.6%** <br> <sub>Portrait AUC 0.999</sub> | **<0.018** <br> <sub>Platt Calibrated ECE</sub> | **6.5 Hours** <br> <sub>Colab Pro PEFT Budget</sub> | **1,683,000+** <br> <sub>Benchmark Scale</sub> |
|:---:|:---:|:---:|:---:|

</div>

---

## 📌 Executive Summary

The **AI Deepfake Detection & Digital Forensic Agent System** is an auditable multi-agent forensic ecosystem designed to classify, audit, and certify synthetic media (GANs, Latent Diffusion, Face Swaps, Audio Clones, Deepfake Videos). 

Inspired by modern closed-loop harness architectures (*Show-Harness*), the system decouples high-level **Semantic Forensic Diagnostic Units** from low-level neural backbones, generates **Cryptographic Auditable Digital Forensic Certificates (`CERT-DF-XXXXXXXX`)**, and features a **Google Search Indexing Architecture** enabling instant public verification across search engines.

---

## 🔬 1. Inside the Forensic Harness: Architecture & Closed-Loop Signal Flow

To satisfy strict evidentiary standards (such as **EU AI Act Article 50** and **DSA Auditable Media** guidelines), the system implements an **Embodied Forensic Harness** that operates as a closed loop across 4 tightly coupled modules:

```mermaid
flowchart TD
    subgraph PERCEPTION ["1. Perception Module (Multi-Modal Sensory Input)"]
        Raw["Raw Media Asset\n(Image / Video / Audio)"]
        Gate{"Human Presence Gate\nSkin YCbCr (77<=Cb<=127, 133<=Cr<=173)\n+ 68-Point MediaPipe Face Landmarks"}
        Raw --> Gate
        Gate -- "Non-Human / Landscape" --> S1_Context["Single-Stream Guidance Context"]
        Gate -- "Human Detected" --> S2_Context["Dual-Stream Biometric Guidance Context"]
    end

    subgraph MODELS ["2. Foundation Models (Cognitive Backbone)"]
        Frontier["Frontier Closed-Source VLMs\n(GPT-4o, Claude 3.5, Gemini 1.5)\n[Zero-Shot Explainable Reasoning]"]
        OpenBackbone["Open-Source LoRA-Tuned Backbones\n(ViT-B/16 Vision + WavLM SSL Audio)\n[Low-Latency Inference / Colab 4h]"]
    end

    subgraph REASONING ["3. Forensic Reasoning Plugins (Semantic Diagnostic Units)"]
        direction TB
        subgraph ArtifactPlugins ["General Artifact Plugins"]
            P_FFT["2D-FFT High-Frequency Residual (Z_peak)"]
            P_Noise["Spatial Noise Variance Discrepancy"]
            P_Ent["Boundary Entropy H(P)"]
        end
        subgraph BioPlugins ["Biological Invariant Plugins"]
            P_DWT["WAVIE 2D Haar DWT Skin Pore Energy (E_high)"]
            P_rPPG["Micro-Vascular Pulse Dynamics (rPPG 0.75-2.5Hz)"]
            P_Glottal["WavLM Layer 12 Acoustic Glottal Dynamics"]
            P_Lip["DF26 Phoneme-Viseme Angular Cosine Sync"]
        end
        subgraph CalibPlugin ["Calibration & Routing Plugin"]
            P_Fuse["Adaptive Routing: 0.40 * Risk_gen + 0.60 * Risk_bio"]
            P_Calib["Platt Temperature Scaling T* + Youden tau*"]
        end
    end

    subgraph ACTION ["4. Forensic Action & Active Mining Loop"]
        Verdict["Forensic Verdict & Risk Assessment\n[VERIFIED_REAL | DEEPFAKE_MANIPULATED]"]
        Cert["Cryptographic Forensic Certificate\nEd25519 Signature + SHA-256 H_media\n(CERT-DF-XXXXXXXX)"]
        SEO["Public Discoverability Action\nSchema.org @graph (TechArticle + ClaimReview)"]
        ActiveLoop["Uncertainty Mining Active Feedback Loop\nHard Negatives H(P) >= 0.70 => Colab Pro 4h Retraining"]
    end

    %% Closed-Loop Signal Flow (Numbered 1 to 5)
    S1_Context -->|"(1) Modality Input Stream"| OpenBackbone
    S2_Context -->|"(1) Modality Input Stream"| OpenBackbone
    S2_Context -.->|"(1) Prompt Context"| Frontier

    OpenBackbone -->|"(2) Feature Activations"| P_FFT
    OpenBackbone -->|"(2) Feature Activations"| P_DWT
    OpenBackbone -->|"(2) Feature Activations"| P_rPPG
    OpenBackbone -->|"(2) Feature Activations"| P_Glottal

    P_FFT --> P_Fuse
    P_DWT --> P_Fuse
    P_rPPG --> P_Fuse
    P_Glottal --> P_Fuse
    P_Lip --> P_Fuse
    P_Fuse --> P_Calib

    P_Calib -->|"(3) Reasoning-Refined Forensic Context"| OpenBackbone
    P_Calib -->|"(3) Calibrated Risk Logits"| Frontier

    OpenBackbone -->|"(4) Action Output"| Verdict
    Frontier -->|"(4) Action Output"| Verdict
    Verdict --> Cert
    Cert --> SEO

    Verdict -->|"(5) Hard-Sample Active Mining Loop"| ActiveLoop
    ActiveLoop -->|"(5) Updated Weights via 4h Scheduled Run"| OpenBackbone
```

> [!NOTE]
> **Takeaway on Architecture**: *Decoupling discrete semantic forensic diagnostic plugins (WAVIE DWT, rPPG, Glottal SSL, Lip-Sync) from the underlying neural backbones allows foundation vision-language models to reason over micro-scale biological invariants without requiring costly pretraining from scratch. Each diagnostic step closes the loop through continuous hard-sample adaptation.*


### 1.1. Evidence Fusion & Mathematical Formulations

The system automatically switches pipeline routing depending on human presence:

#### Case 1: Non-Human Media (Objects, Landscapes, Graphic Art)
When no human face, skin tone, or voice is detected, the system executes **Single-Stream General Vision (`SINGLE_STREAM_GENERAL`)**:

$$Risk_{composite} = 1.00 \cdot Risk_{general} = 0.75 \cdot P_{fake} + 0.15 \cdot Score_{spectral} + 0.10 \cdot Score_{spatial}$$

#### Case 2: Human Media (Portraits, Face Swaps, Talking Head Videos)
When human presence is verified, the system forks into **Dual-Stream Parallel Execution (`DUAL_STREAM_PARALLEL`)** and fuses evidence with calibrated weights:

$$Risk_{composite} = 0.40 \cdot Risk_{general} + 0.60 \cdot Risk_{human}$$

Where:
1. **$Risk_{general}$ (General 4D Vision Stream)**:
   $$Risk_{general} = 0.75 \cdot P_{fake} + 0.15 \cdot Score_{spectral} + 0.10 \cdot Score_{spatial}$$
   - **Passive Model Probability ($P_{fake}$)**: Probability output from the fine-tuned Vision Transformer backbone ($W_{model} = 0.75$).
   - **Spectral Frequency Anomaly ($Score_{spectral}$)**: High-frequency Z-score energy residual derived from 2D Fast Fourier Transform ($W_{spectral} = 0.15$), exposing GAN/Diffusion grid artifacts:
     $$Score_{spectral} = \text{clip}\left(\frac{Z_{peak} - 3.4}{3.0}, 0.0, 1.0\right)$$
   - **Spatial Noise Inconsistency ($Score_{spatial}$)**: Noise variance discrepancy between center face region and peripheral background ($W_{spatial} = 0.10$).
   - **Prediction Entropy ($H(P)$)**: Normalized entropy measuring model boundary uncertainty:
     $$H(P) = -\frac{1}{\log_2(K)} \sum_{i=1}^{K} P_i \log_2(P_i + \epsilon)$$

2. **$Risk_{human}$ (Human Biometrics Stream - Model 2)**:
   Well-calibrated probability (Platt scaling / Temperature ECE calibrated) evaluated across:
   - Facial skin wavelet sub-band anomalies (WAVIE 2D DWT pore and sub-surface scattering disruptions).
   - Remote photoplethysmography (rPPG micro-vascular blood volume pulse signals).
   - Self-supervised acoustic vocal dynamics (glottal airflow irregularities).
   - Audio-visual phoneme-viseme temporal synchrony (DF26 lip synchronization).

### Auditable Verdict Criteria
- $Risk_{composite} \le 0.50 \implies$ **`AUTHENTIC_MEDIA_LIKELY`** (Genuine Media / Original)
- $Risk_{composite} > 0.50 \implies$ **`SYNTHETIC_MEDIA_SUSPECTED`** (Synthetic / Manipulated Media)

---

## 📜 2. Auditable Digital Forensic Certificates (`CERT-DF-XXXXXXXX`)

Every analysis generates a cryptographically signed, immutable Digital Forensic Certificate.

### 2.1. Cryptographic Formulations & Media Integrity Hashing

#### 1. Media Integrity Hash ($H_{media}$):
The full binary SHA-256 cryptographic hash of the analyzed media asset is formulated as:

$$H_{media} = \text{SHA-256}\left(\text{Asset}_{\text{binary}}\right) = \text{Digest}\left(\sum_{i=1}^{N} b_i\right)$$

Where $\text{Asset}_{\text{binary}}$ represents the raw binary byte stream $b_i$ of the media file.

#### 2. Non-Repudiable Cryptographic Signature Proof ($Sig_{proof}$):
The tamper-evident digital signature protecting each forensic certificate is generated via:

$$Sig_{proof} = \text{Ed25519}\Big(\text{SHA-256}\left(Cert_{ID} \parallel Timestamp \parallel H_{media} \parallel Risk_{composite}\right), K_{private}\Big)$$

Where $K_{private}$ is the accredited private signing key and $\parallel$ denotes canonical string concatenation.

### Certificate Features
- **Unique Certificate ID**: Deterministic UUID format `CERT-DF-XXXXXXXX` (e.g. `CERT-DF-662CBE7B`).
- **Forensic Pipeline Mode**: Transparently logged as `SINGLE_STREAM_GENERAL` or `DUAL_STREAM_PARALLEL`.
- **Media Integrity Hash**: Full binary SHA-256 cryptographic hash of the analyzed media asset ($H_{media}$).
- **Digital Signature Proof**: Signed using Ed25519 cryptographic signature ($Sig_{proof}$).
- **Verification Endpoint**: Embedded URL `https://detection.dtpsoft.vn/api/verify/CERT-DF-XXXXXXXX`.

### Sample Certificate JSON Schema
```json
{
  "certificate_id": "CERT-DF-662CBE7B",
  "timestamp": "2026-09-17 08:23:40 UTC",
  "forensic_version": "2.0-AUDITABLE",
  "compliance_standard": "EU AI Act Article 50 & DSA Auditable Media",
  "analysis_mode": "DUAL_STREAM_PARALLEL",
  "media_integrity": {
    "sha256": "662cbe7b89e24a8f902472e389b0129a7c6f0119b48f",
    "verification_url": "https://detection.dtpsoft.vn/api/verify/CERT-DF-662CBE7B"
  },
  "image_metadata": {
    "dimensions": "1024x1024",
    "format": "JPEG",
    "face_cropping_applied": true
  },
  "dual_stream_breakdown": {
    "mode": "DUAL_STREAM_PARALLEL",
    "human_gate_triggered": true,
    "fusion_formula": "0.40 * Risk_general + 0.60 * Risk_human",
    "stream_1_general_vision": {
      "model_type": "Custom Fine-Tuned ViT + 2D-FFT + Noise Spectrum",
      "risk_general": 0.0283
    },
    "stream_2_human_biometrics": {
      "model_type": "HumanMultiModalDetector (WAVIE/rPPG/SSL/LipSync)",
      "risk_human": 0.0150
    }
  },
  "multi_dimensional_evidence": {
    "passive_model_score": {
      "prediction": "REAL",
      "confidence": 0.9842,
      "fake_probability": 0.0158
    },
    "spectral_frequency_analysis": {
      "high_frequency_anomaly_score": 0.0821,
      "interpretation": "Normal natural frequency spectrum"
    },
    "spatial_inconsistency_analysis": {
      "spatial_inconsistency_score": 0.0412,
      "interpretation": "Uniform spatial noise distribution"
    },
    "model_uncertainty_entropy": {
      "entropy": 0.0851,
      "interpretation": "High prediction certainty"
    }
  },
  "composite_risk_score": 0.0283,
  "auditable_verdict": "AUTHENTIC_MEDIA_LIKELY",
  "proof": {
    "type": "Ed25519SHA256Signature2026",
    "verification_method": "https://detection.dtpsoft.vn/api/verify/CERT-DF-662CBE7B",
    "status": "VERIFIED_VALID"
  }
}
```

---

## 🌐 3. Google Search Indexing & Public Verification Architecture

To allow search engines (Googlebot, Bingbot) and the general public to verify certificates directly via search queries (e.g., searching `"CERT-DF-662CBE7B"` on Google), the system integrates a complete SEO Indexing Architecture.

```mermaid
flowchart TD
    A["📷 Media Asset Upload"] --> B["🤖 Deepfake Detection Detector"]
    B --> C["📜 Generate Certificate CERT-DF-662CBE7B"]
    C --> D[("🗄️ SQLite Memory Database\n(deepfake_agent_memory.db)")]
    
    D --> E1["🌐 Public SSR Page\n/certificate/CERT-DF-662CBE7B"]
    D --> E2["🗺️ Dynamic XML Sitemap\n/sitemap.xml"]
    
    E1 --> F1["🏷️ Schema.org @graph\n(TechArticle + Breadcrumb + ClaimReview)"]
    E2 --> F2["🤖 Googlebot Crawler"]
    
    C --> G["🚀 Google Indexing Publisher Engine\n(google_indexing.py)"]
    G --> H["📡 Google Search Console & Indexing API Push"]
    
    F2 --> I["🔍 Google Search Index Results"]
    H --> I
```

### Key Components

1. **Server-Side Rendered (SSR) Public Pages (`/certificate/{cert_id}`)**:
   - Fast, server-side HTML rendering (`observability/static/certificate.html`) accessible to crawlers without JavaScript execution delays.
2. **Multi-Type Schema.org `@graph` JSON-LD**:
   - Includes `@type: "TechArticle"`, `"BreadcrumbList"`, `"ClaimReview"`, and `"Organization"` to ensure 100% compatibility with **Google Rich Results Test**.
3. **Dynamic XML Sitemap (`/sitemap.xml`)**:
   - Dynamically generated XML sitemap with `<lastmod>`, `<changefreq>`, and `<priority>` tags for all stored certificate IDs.
4. **Google Indexing Publisher Engine (`benchmark/google_indexing.py`)**:
   - Automatically pings Google Search Console and pushes instant indexing notifications to Google Indexing API v3.

---

## 🗄️ 4. Persistent Agent Memory Engine (`deepfake_agent_memory.db`)

All system operations, benchmark evaluations, training runs, and issued certificates are tracked in a persistent SQLite database located at `report_colab/deepfake_agent_memory.db`.

| Table Name | Description |
| :--- | :--- |
| `forensic_certificates` | Stores full certificate payloads, SHA-256 hashes, risk scores, and 4D evidence. |
| `daily_benchmarks` | Tracks daily evaluation runs (sample count, AUC, EER, latency, spectral spikes). |
| `colab_train_reports` | Log Colab fine-tuning sessions, validation AUC/EER, and hyperparameter logs. |
| `model_weights_registry` | Manages model weight versioning, backbone architecture, and active model status. |

---

## ⚡ 5. Automated Model Fine-Tuning & Colab Weight Synchronization

The framework features a **Continuous Machine Learning Pipeline** that automatically updates detector weights and calibrates prediction parameters based on newly identified synthetic artifacts.

```mermaid
flowchart TD
    A["⏰ GitHub Actions Daily Benchmark (00:00 UTC)"] --> B["📊 Evaluate Metrics (AUC, EER, Spectral Spikes)"]
    B --> C["💡 Generate Optimal Proposals (Optimal Tau, Temp T, Learning Rate)"]
    C --> D[("🗄️ SQLite Memory (daily_benchmarks)")]
    
    C --> E["⚡ Google Colab GPU Fine-Tuning (colab_auto_runner.py)"]
    E --> F["📡 POST /api/colab/callback (Push Weights & Calibration Params)"]
    
    F --> G["💾 Save to custom_vit_model/calibration_params.json"]
    G --> H["🔄 Zero-Downtime Hot-Reload DeepfakeDetector"]
    F --> I[("🗄️ SQLite Registry (model_weights_registry)")]
```

### 5.1. Daily Benchmark & Optimal Hyperparameter Proposals
- **Automated Workflow**: GitHub Actions `.github/workflows/daily_benchmark.yml` triggers daily at 00:00 UTC (7:00 AM VN time).
- **Metric Computation**: Evaluates latest dataset splits, computes $AUC$, $EER$, average latency ($ms$), and 2D-FFT spectral spikes.
- **Hyperparameter Proposals**:
  - Optimal learning rate $\eta$ for fine-tuning.
  - Optimal decision boundary threshold $\tau \in [0.5, 0.9]$.
  - Temperature scaling factor $T$ for confidence calibration.

### 5.2. Temperature Scaling & Confidence Calibration
To prevent model overconfidence on unseen diffusion artifacts, calibrated probabilities are computed as:

$$P_i = \text{softmax}\left(\frac{z_i}{T}\right) = \frac{\exp(z_i / T)}{\sum_{j=1}^{K} \exp(z_j / T)}$$

Where $z_i$ represents raw logits from the Vision Transformer / EfficientNet backbone and $T > 0$ is the temperature parameter tuned on GPU.

### 5.3. Threads-DFBench: Vietnamese In-The-Wild Facial Benchmark Dataset
To benchmark and adapt the detector against emerging in-the-wild social threats, the system features an autonomous crawler and academic benchmark generator (**Threads-DFBench**):
- **Target Platform**: Meta Threads with Vietnamese locale targeting (`Accept-Language: vi-VN`) and localized query topics (`chan_dung_vietnam`, `anh_the_cccd`, `selfie_vietnam`, `duong_pho_vietnam`, `kinh_can_khau_trang`).
- **Paired Counterfactual Architecture**: Generates triplets on the identical human identity:
  1. `01_raw_originals`: Authentic human photographic portrait ($E_{high} \ge 0.30$).
  2. `02_manipulated_faceswap`: Counterfactual identity swap with boundary seam blending.
  3. `03_manipulated_openweight`: Latent diffusion synthesis with pore over-smoothing ($E_{high} < 0.22$) and high-frequency latent checkerboard spikes ($Z_{peak} > 4.20$).

#### 5-Category Vietnamese Facial Taxonomy
| Category Key | Vietnamese Facial Taxonomy | Ratio | Academic Focus & Stress Testing |
| :--- | :--- | :--- | :--- |
| `vn_id_official` | Official ID / CCCD / Passport | 20% | Frontal $0^\circ$ studio lighting; assesses facial boundary blending artifacts. |
| `vn_selfie_lifestyle` | Casual Lifestyle & Cafe Selfies | 20% | Natural indoor lighting & angles; tests robustness against ambient illumination. |
| `vn_street_outdoor` | Street & Outdoor Portraits | 20% | Vietnamese daylight & outdoor street background; assesses optical frequency slopes. |
| `vn_occlusion_glasses_mask` | Occlusions (Glasses, Masks, Helmets) | 20% | Prescription glasses, medical face masks, motorbike helmets; challenges facial landmarks. |
| `vn_mature_aging` | Mature & Senior Aging | 20% | Natural smile lines & biological skin folds; prevents model false alarms on wrinkles. |

### 5.4. Active Learning & Continual Fine-Tuning with Focal Loss
To actively harden the backbone against subtle false negatives (such as sample 9 in OOD evaluations), the system implements an active learning loop:
1. **Uncertainty Mining**: Samples with high Shannon Information Entropy $H(P) \ge 0.70$ or classifier disagreement are routed to `hard_samples_manifest.jsonl`:
   $$H(P) = - \left[ P_{fake} \log_2(P_{fake}) + (1 - P_{fake}) \log_2(1 - P_{fake}) \right]$$
2. **Focal Loss Objective**: Down-weights easily classified background samples and forces gradient updates on borderline hard negatives:
   $$\mathcal{L}_{focal}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t) \quad (\gamma = 2.0, \; \alpha = 0.25)$$
3. **Automated Weight Checkpointing**: Continuously trains on Google Colab and persists evolved weights directly to Google Drive (`checkpoints/model_continual_latest.pt`).

### 5.5. Dual-Stream Model Training & Dataset Allocation

To ensure high generalizability and prevent feature entanglement, the framework strictly segregates training data into two specialized pipelines:

| Mô Hình (Stream) | Bộ Dữ Liệu Huấn Luyện (Training Datasets) | Quy Mô Mẫu | Trọng Tâm Trích Xuất Đặc Trưng | Trọng Số Checkpoint |
|:---|:---|:---:|:---|:---:|
| **Mô Hình 1**<br>*(General Vision & Signal)* | • `NTIRE-RobustAIGenDetection` (MSU 2024)<br>• `CIFAKE` (CIFAR-10 vs Stable Diffusion)<br>• `DF40` (40 SOTA Forgery Suite)<br>• `DFFD` (Diverse Fake Face Dataset)<br>• Hard Negatives từ Threads ($H(P) \ge 0.70$) | **>578,000+** | Vết tích mô hình sinh AI, phổ tần số 2D FFT ($Z_{peak}$), nhiễu không gian, entropy biên | `model_continual_latest.pt`<br>*(ViT Backbone)* |
| **Mô Hình 2**<br>*(Human Biometric & Physiology)* | • **`MultiFakeVerse`** (VLM Person-Centric Manipulation)<br>• **`HiDF`** (Human-Indistinguishable Deepfakes)<br>• **`140k Real and Fake Faces`** (70k Flickr / 70k StyleGAN)<br>• **`DeepDetect-2025 / 2026`** (Unsplash vs MJ/SD3/Flux)<br>• **Meta Threads Vietnamese In-The-Wild** (10,000 img, 500 vid, 500 aud / 4h qua 5 phân lớp người Việt) | **>1,105,000+** | WAVIE 2D Haar DWT pore energy ($E_{high}$), rPPG vascular pulse, HuBERT/WavLM Layer 12 vocal trajectory, DF26 lip-sync | `weights_human.pt`<br>*(Biometric Fusion)* |

### 5.6. Google Colab Pro Scheduled Runner (Option 1 + Option B Form UI)

The unified runner notebook [`colab_training/Colab_Pro_Scheduled_Runner.ipynb`](colab_training/Colab_Pro_Scheduled_Runner.ipynb) provides seamless dual-mode execution:
1. **Option 1 (Headless Scheduled Run)**: Executes automatically every 4 hours in the background on Google Colab Pro without manual intervention, crawls 10,000 images, 500 videos, 500 audios, and fine-tunes both models.
2. **Option B (Interactive Form UI)**: Colab `#@param` widgets allow researchers to run manual tests, choose individual models (`TRAIN_MODEL_1_GENERAL`, `TRAIN_MODEL_2_HUMAN`, `TRAIN_BOTH_MODELS`), adjust quotas, and visualize the full training report.

```python
# 1. Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone/Pull Repository & Install Dependencies
!git clone https://github.com/ThangCao106/DeepFake-Detection-Agent.git || (cd DeepFake-Detection-Agent && git pull origin main)
%cd DeepFake-Detection-Agent
!pip install -q -r requirements.txt transformers timm

# 3. 1-Click Execution for Both Models & Full Data Reporting
!python colab_sync_to_gdrive.py \
    --images 10000 \
    --videos 500 \
    --audios 500 \
    --samples 25 \
    --train-both-models \
    --epochs 3 \
    --batch_size 32 \
    --lr 1e-4
```

### 5.7. Full Data Training & Verification Report (`training_report_dual_stream.md`)
Upon training completion, the runner automatically compiles a comprehensive Markdown report directly to:
- **Local Storage**: `report_colab/training_report_dual_stream.md`
- **Google Drive**: `/content/drive/MyDrive/Thread_Data/reports/training_report_dual_stream.md`

The report documents exact sample counts across datasets, real/fake ratios, 5 Vietnamese biometric categories (`vn_id_official`, `vn_selfie_lifestyle`, `vn_street_outdoor`, `vn_occlusion_glasses_mask`, `vn_mature_aging`), multi-modal metrics (Acc, Loss, AUC, EER, ECE), and Platt temperature calibration parameters ($T^* = 1.6178, \tau^* = 0.8118$).

### 5.8. Zero-Downtime Hot-Reloading
When Colab completes fine-tuning, it sends a payload to `POST /api/colab/callback`:
1. Server receives updated weights and saves `temperature_scaling_T` and `optimal_threshold_tau` to `custom_vit_model/calibration_params.json`.
2. Instantly reloads the `DeepfakeDetector` instance in memory without interrupting active client requests or restarting the server.
3. Registers weight version `vYYYY-MM-DD` in `model_weights_registry`.

---

## 🛠️ 6. REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /certificate/{cert_id}` | `GET` | SSR Public Certificate Web Page with Schema.org JSON-LD |
| `GET /api/verify/{cert_id}` | `GET` | JSON verification endpoint for third-party verification |
| `GET /sitemap.xml` | `GET` | Dynamic XML Sitemap for Googlebot crawlers |
| `POST /api/upload` | `POST` | Upload image/video for analysis |
| `POST /api/colab/callback` | `POST` | Ingest fine-tuned weights & calibration params from Google Colab |
| `GET /api/status` | `GET` | System health check and API key status |

---

## 🚀 7. Quick Start & Local Setup

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/ThangCao106/DeepFake-Detection-Agent.git
cd DeepFake-Detection-Agent

# 2. Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running with Docker Compose

```bash
cd observability
docker compose up -d
```
Access the Web UI at: `http://localhost:8787` (or `https://detection.dtpsoft.vn/obsagent/`).

---

## 📑 8. Citation & Academic Reference

If you use this system or 4D evidence framework in your academic research, please cite:

```bibtex
@article{dtp_deepfake_forensic_2026,
  author    = {Cao, Thang and Deepfake Detection Research Group},
  title     = {Multidimensional Digital Forensic Framework and Auditable Watermarking Scheme for Deepfake Media Authentication under EU AI Act Compliance},
  journal   = {IEEE Transactions on Information Forensics and Security (Draft)},
  year      = {2026},
  publisher = {DTP Software AI Research}
}
```

---

## 📄 9. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
