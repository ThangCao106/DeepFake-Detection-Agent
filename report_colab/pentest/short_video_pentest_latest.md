# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261125_100434_9cf78`)
- **Timestamp (UTC)**: 2026-11-25T10:04:34Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Model Loaded**: `dima806/deepfake_vs_real_image_detection` (`using_custom_weights=False`)
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `False` (Skipped Reason: `fallback_model_weights`) | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4673 | 2549.63 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4638 | 2500.73 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4692 | 2536.75 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4618 | 2511.13 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 2538.91 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 2524.45 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 2521.09 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 2490.74 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 2522.96 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 2555.59 |
