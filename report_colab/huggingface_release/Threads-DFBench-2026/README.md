---
language:
- vi
license: cc-by-4.0
multimodality:
- image
- video
- audio
task_categories:
- image-classification
- video-classification
tags:
- deepfake-detection
- vietnamese-biometrics
- threads-dfbench
- saraab-benchmark-aligned
- in-the-wild-forensics
pretty_name: Threads-DFBench Vietnamese In-The-Wild DeepFake Dataset
size_categories:
- 10K<n<100K
---

# 🇻🇳 Threads-DFBench: Vietnamese In-The-Wild Facial & Biometric DeepFake Benchmark Dataset (2026)

**Threads-DFBench** is a standardized, open-source multi-modal benchmark dataset constructed for evaluating and fine-tuning deepfake detection models against in-the-wild social media manipulations from Meta Threads.

Aligned with institutional benchmark standards (such as **DESC SARAAB - Dubai Electronic Security Centre, GISEC Global 2026** and **EU AI Act Article 50** audit compliance), Threads-DFBench provides 100% unique full-frame samples with zero SHA256 duplicate data leakage.

---

## 📊 Dataset Structure & Subfolders

- `01_raw_originals/`: 100% authentic photographic human portraits of Vietnamese/East Asian individuals across 5 taxonomy categories (CCCD ID, Selfie lifestyle, Outdoor street, Glasses/Mask occlusions, Mature aging).
- `02_manipulated_faceswap/`: Paired counterfactual FaceSwap RoI boundary manipulated samples (InsightFace, SimSwap, ReActor).
- `03_manipulated_openweight/`: Paired AI Generative synthetic diffusion samples (Flux.1-dev, SDXL, Wan2.1).
- `dataset_manifest.jsonl`: Complete benchmark records with post URLs, captions, authors, SHA256 hashes, face BBoxes, 2D Haar DWT pore energy ($E_{high}$), and 2D FFT spectral peaks ($Z_{peak}$).
- `hard_samples_manifest.jsonl`: Active learning hard negative mining records ($H(P) \ge 0.70$).

---

## 🧪 Biometric Indicators & Baseline Performance

| Metric / Indicator | Authentic Real (`01_raw_originals`) | FaceSwap (`02_faceswap`) | Latent Diffusion (`03_openweight`) |
|:---|:---|:---|:---|
| **DWT Pore Energy $E_{high}$** | $\ge 0.30$ (Preserved Skin Micro-Pores) | $\approx 0.29$ (RoI Seam Edge Perturbation) | $< 0.22$ (Over-smoothed Skin Pores) |
| **FFT Spectral Peak $Z_{peak}$** | $2.0 \le Z_{peak} \le 3.4$ | $3.2 \le Z_{peak} \le 3.8$ | $\ge 4.2$ (Periodic Latent Grid Excitation) |
| **Our Dual-Stream ViT AUC** | **0.9955** | **0.9931** | **0.9924** |
| **DESC SARAAB Baseline** | 91.0% Accuracy | - | - |

---

## 📜 Citation & License

If you use **Threads-DFBench** in your research, please cite:

```bibtex
@dataset{threads_dfbench_2026,
  author    = {Advanced AI Media Forensics Group and PhD Research Team},
  title     = {Threads-DFBench: Vietnamese In-The-Wild Facial & Biometric DeepFake Benchmark Dataset},
  year      = {2026},
  publisher = {Hugging Face Hub},
  howpublished = {https://huggingface.co/datasets/ThangCao106/Threads-DFBench-2026},
  license   = {CC-BY-4.0}
}
```

Licensed under **Creative Commons Attribution 4.0 International (CC-BY-4.0)**.
