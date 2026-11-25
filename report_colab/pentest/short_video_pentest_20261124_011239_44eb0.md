# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261124_011239_44eb0`)
- **Timestamp (UTC)**: 2026-11-24T01:12:39Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Model Loaded**: `dima806/deepfake_vs_real_image_detection` (`using_custom_weights=False`)
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `False` (Skipped Reason: `fallback_model_weights`) | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4673 | 1902.23 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4638 | 1893.56 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4692 | 1911.57 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4618 | 1924.92 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 1910.69 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 1890.19 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 1905.13 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 1934.32 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 1903.83 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 1903.44 |
