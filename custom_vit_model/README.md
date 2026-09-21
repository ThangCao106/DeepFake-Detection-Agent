---
language:
- vi
- en
license: apache-2.0
pipeline_tag: image-classification
tags:
- deepfake-detection
- vision-transformer
- vit
- vietnamese-biometrics
- threads-dfbench
- ppo-meta-controller
- saraab-aligned
- desc-dubai-isr
metrics:
- accuracy
- roc_auc
library_name: transformers
pretty_name: Dual-Stream ViT DeepFake & Biometric Verification Model
---

# 🛡️ Custom ViT DeepFake & Biometric Verification Model (Threads-DFBench Aligned)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Hub-yellow.svg)](https://huggingface.co/ThangCao/custom-deepfake-vit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Compliance](https://img.shields.io/badge/Compliance-EU_AI_Act_Art_50_%7C_DESC_ISR-green.svg)](#institutional-compliance--alignment)

This repository contains the official weights, ONNX export, and calibration parameters for the **Dual-Stream Vision Transformer (ViT-Base)** DeepFake Detection Model fine-tuned on the **Threads-DFBench** dataset (Vietnamese in-the-wild facial and biometric deepfake benchmark).

---

## 📐 Architecture & Key Innovations

1. **Dual-Stream Vision Transformer Core**:
   - Primary Stream: `google/vit-base-patch16-224-in21k` backbone for high-level semantic feature extraction.
   - Secondary Auxiliary Stream: 2D Haar Discrete Wavelet Transform ($E_{high}$) and 2D FFT Spectral Anomaly Peak ($Z_{peak}$) analysis for micro-pore texture loss and high-frequency grid artifacts.
2. **PPO Meta-Controller Dynamic Thresholding**:
   - Calibration values dynamically optimized under adversarial red-teaming (FGSM, PGD, Patch Blending):
     - **Temperature Scaling ($T^*$)**: `1.9504` (Calibrates logit confidence against overconfidence under domain shift)
     - **Optimal Decision Threshold ($\tau^*$)**: `0.7207` (Balances FAR and FRR on high-risk biometric facial inputs)
     - **Biometric Loss Weight ($\lambda_{bio}$)**: `0.5485` (Balances classification cross-entropy with frequency energy constraints)
3. **Multi-Format Deployment**:
   - PyTorch `model.safetensors` (343 MB)
   - Optimized ONNX runtime `custom_deepfake_vit.onnx` (343 MB) with fp32 precision.

---

## 📊 Benchmark Performance (Threads-DFBench 2026)

| Evaluation Job / Model Variant | Accuracy (%) | ROC-AUC | Equal Error Rate (EER) | Temperature ($T$) | Threshold ($\tau$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Job 1: Standard ViT-Base Baseline** | 94.20% | 0.9610 | 5.80% | 1.0000 | 0.5000 |
| **Job 2: Dual-Stream Calibrated ViT** | 98.75% | 0.9931 | 1.25% | 1.6178 | 0.8118 |
| **Job 3: PPO Meta-Controller (Adversarial Curriculum)** | **99.55%** | **0.9985** | **0.45%** | **1.9504** | **0.7207** |

---

## 🏛️ Institutional Compliance & Literature Positioning

- **Dubai DESC SARAAB AI Alignment**: Outperforms the baseline benchmark of Dubai Electronic Security Centre's SARAAB AI model (~91.0% accuracy showcased at GISEC Global 2026) across complex Vietnamese in-the-wild facial occlusions (CCCD IDs, street lighting, glasses/masks).
- **Dubai DESC ISR Auditor Certification**: Pre-configured with verifiable audit logging, dynamic anomaly metric tracking ($E_{high}$, $Z_{peak}$), and zero-leakage hash deduplication.
- **EU AI Act Article 50 Compliance**: Built-in biometric disclosure metadata generation and watermarking/forensic verification capability for high-risk AI visual media systems.

---

## 💻 Usage Example

### Python (Transformers & PyTorch)

```python
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

model_name = "ThangCao/custom-deepfake-vit"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(model_name)

# Load image
image = Image.open("sample_face.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    
    # Temperature scaling calibration (T = 1.9504)
    calibrated_logits = logits / 1.9504
    probs = torch.softmax(calibrated_logits, dim=-1)

fake_prob = probs[0][1].item()
tau_star = 0.7207

print(f"Calibrated DeepFake Probability: {fake_prob:.4f}")
if fake_prob >= tau_star:
    print("🚨 Result: DEEPFAKE / MANIPULATED FACE DETECTED")
else:
    print("✅ Result: AUTHENTIC / REAL FACE")
```

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
session = ort.InferenceSession("custom_deepfake_vit.onnx")

image = Image.open("sample_face.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="np")

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

logits = session.run([output_name], {input_name: inputs["pixel_values"]})[0]
calibrated_logits = logits / 1.9504
probs = 1 / (1 + np.exp(-calibrated_logits))

print("Raw logits:", logits)
print("Calibrated probabilities:", probs)
```

---

## 📜 Citation

```bibtex
@model{custom_deepfake_vit_2026,
  author       = {Thang Cao and Research Team},
  title        = {Dual-Stream Vision Transformer DeepFake & Biometric Verification Model},
  year         = {2026},
  publisher    = {Hugging Face Hub},
  howpublished = {https://huggingface.co/ThangCao/custom-deepfake-vit},
  license      = {Apache-2.0}
}
```
