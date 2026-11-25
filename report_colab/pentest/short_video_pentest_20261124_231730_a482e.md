# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261124_231730_a482e`)
- **Timestamp (UTC)**: 2026-11-24T23:17:30Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Model Loaded**: `dima806/deepfake_vs_real_image_detection` (`using_custom_weights=False`)
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `False` (Skipped Reason: `fallback_model_weights`) | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4673 | 1721.92 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4638 | 1688.2 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4692 | 1696.07 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4618 | 1703.62 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 1768.24 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 1672.26 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 1704.8 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 1682.45 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 1694.82 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 1721.23 |
