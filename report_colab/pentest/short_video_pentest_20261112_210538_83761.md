# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261112_210538_83761`)
- **Timestamp (UTC)**: 2026-11-12T21:05:38Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4514 | 2851.68 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4493 | 2645.16 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.455 | 2741.87 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4493 | 2724.48 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2837.08 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2829.58 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2702.4 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2679.16 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2832.65 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2657.98 |
