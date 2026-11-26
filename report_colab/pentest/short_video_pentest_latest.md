# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261126_094829_81f68`)
- **Timestamp (UTC)**: 2026-11-26T09:48:29Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Model Loaded**: `dima806/deepfake_vs_real_image_detection` (`using_custom_weights=False`)
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `False` (Skipped Reason: `fallback_model_weights`) | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4673 | 2504.59 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4638 | 2533.2 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4692 | 2518.07 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4618 | 2523.22 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2549.81 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2520.39 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2523.6 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2496.19 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2515.02 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2526.26 |
