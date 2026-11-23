# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261122_203022_30366`)
- **Timestamp (UTC)**: 2026-11-22T20:30:22Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Model Loaded**: `dima806/deepfake_vs_real_image_detection` (`using_custom_weights=False`)
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `True` | **Auto-Feedback**: `False` (Skipped Reason: `fallback_model_weights`) | **Manifest Clips Evaluated**: 0
- **Total Clips Evaluated**: 40 | **Accuracy**: 40.00% | **TPR**: 0.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4673 | 1523.59 | 0.0 |
| `instagram_reels` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4638 | 1696.45 | 0.0 |
| `youtube_shorts` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4692 | 1527.37 | 0.0 |
| `tiktok` | 10 | 40.0% | 0.0% | 0.0% | 100.0% | 1.0 | 0.0% | 0.4618 | 1519.82 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 4 | 0.0% | 0.0% | 100.0% | 1554.79 |
| `caption_sticker` | 8 | 50.0% | 0.0% | 100.0% | 1537.25 |
| `clean_baseline` | 8 | 50.0% | 0.0% | 100.0% | 1486.82 |
| `h264_heavy_reencode` | 8 | 50.0% | 0.0% | 100.0% | 1766.15 |
| `partial_splicing` | 4 | 0.0% | 0.0% | 100.0% | 1485.05 |
| `vfr_fps_mismatch` | 8 | 50.0% | 0.0% | 100.0% | 1523.89 |
