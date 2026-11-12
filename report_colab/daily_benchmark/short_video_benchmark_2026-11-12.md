# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261112_044126_133bd`)
- **Timestamp (UTC)**: 2026-11-12T04:41:26Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 1
- **Total Clips Evaluated**: 41 | **Accuracy**: 82.93% | **TPR**: 72.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 90.0% | 83.3% | 0.0% | 16.7% | 1.0 | 0.0% | 0.1387 | 8696.14 | 0.0 |
| `instagram_reels` | 10 | 80.0% | 66.7% | 0.0% | 33.3% | 1.0 | 0.0% | 0.18 | 8687.07 | 0.0 |
| `youtube_shorts` | 11 | 72.7% | 57.1% | 0.0% | 42.9% | 1.0 | 0.0% | 0.2105 | 8435.17 | 0.0 |
| `tiktok` | 10 | 90.0% | 83.3% | 0.0% | 16.7% | 1.0 | 0.0% | 0.134 | 7933.39 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 5 | 80.0% | 0.0% | 20.0% | 8770.8 |
| `caption_sticker` | 8 | 100.0% | 0.0% | 0.0% | 9081.15 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 7870.82 |
| `h264_heavy_reencode` | 8 | 75.0% | 0.0% | 50.0% | 8312.67 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 8371.71 |
| `vfr_fps_mismatch` | 8 | 100.0% | 0.0% | 0.0% | 8311.86 |
