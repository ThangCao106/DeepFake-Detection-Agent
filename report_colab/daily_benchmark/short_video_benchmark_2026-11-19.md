# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261119_090235_3e019`)
- **Timestamp (UTC)**: 2026-11-19T09:02:35Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4414 | 2153.95 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4381 | 2274.0 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.444 | 2176.56 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4369 | 2167.23 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2181.38 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2188.61 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2278.22 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2143.91 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2152.94 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2186.8 |
