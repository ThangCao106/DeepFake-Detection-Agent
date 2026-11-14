# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261114_074903_ac5e6`)
- **Timestamp (UTC)**: 2026-11-14T07:49:03Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4393 | 2433.98 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4379 | 2374.95 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4439 | 2377.11 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4387 | 2406.44 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2466.15 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2380.85 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2369.19 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2410.04 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2429.74 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2382.57 |
