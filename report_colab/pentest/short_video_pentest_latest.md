# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261116_081331_3a19f`)
- **Timestamp (UTC)**: 2026-11-16T08:13:31Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4393 | 1774.22 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4379 | 1994.16 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4439 | 1807.98 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4387 | 1829.82 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 1827.99 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 1812.84 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 1806.31 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 1962.28 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 1811.47 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 1856.57 |
