# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261116_230347_4c29b`)
- **Timestamp (UTC)**: 2026-11-16T23:03:47Z
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `True` | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4393 | 1788.41 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4379 | 1779.35 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4439 | 1796.91 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4387 | 1787.66 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 1786.78 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 1808.77 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 1777.12 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 1778.3 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 1776.09 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 1794.78 |
