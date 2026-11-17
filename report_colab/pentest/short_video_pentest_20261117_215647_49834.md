# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261117_215647_49834`)
- **Timestamp (UTC)**: 2026-11-17T21:56:47Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4414 | 1533.68 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4381 | 1423.25 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.444 | 1416.18 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4369 | 1420.71 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 1419.28 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 1425.36 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 1419.75 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 1514.19 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 1424.55 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 1461.06 |
