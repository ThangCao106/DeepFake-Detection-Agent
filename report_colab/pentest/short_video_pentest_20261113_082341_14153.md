# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261113_082341_14153`)
- **Timestamp (UTC)**: 2026-11-13T08:23:41Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.443 | 2689.0 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4416 | 2555.85 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4476 | 2630.64 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4422 | 2412.85 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2902.85 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2498.74 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2492.01 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2599.9 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2758.98 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2438.87 |
