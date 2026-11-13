# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261113_204137_630a7`)
- **Timestamp (UTC)**: 2026-11-13T20:41:37Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4393 | 2421.81 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4379 | 2397.75 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4439 | 2451.52 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4387 | 2398.61 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2408.05 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2451.02 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2406.05 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2390.06 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2406.96 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2432.49 |
