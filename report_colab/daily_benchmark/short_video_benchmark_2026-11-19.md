# Multi-Platform Short-Video Benchmark & Adversarial Pentest Report (`SVP-20261119_112008_7c353`)
- **Timestamp (UTC)**: 2026-11-19T11:20:08Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Model Loaded**: `D:\DATA\TC\Py_Ex\NCS\DeepFake-Detection-Agent\custom_vit_model` (`using_custom_weights=True`)
- **Platforms**: threads, instagram_reels, youtube_shorts, tiktok
- **Download URLs**: `False` | **Auto-Feedback**: `False` (Skipped Reason: `None`) | **Manifest Clips Evaluated**: 1
- **Total Clips Evaluated**: 41 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **Mean Duration Err**: 0.0s

## 1. Per-Platform Metrics
| Platform | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) | Dur Err (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `threads` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0676 | 4072.51 | 0.0 |
| `instagram_reels` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0705 | 4021.23 | 0.0 |
| `youtube_shorts` | 11 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0707 | 4626.74 | 0.0 |
| `tiktok` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0642 | 3943.79 | 0.0 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ai_generative_short` | 5 | 100.0% | 0.0% | 0.0% | 5185.86 |
| `caption_sticker` | 8 | 100.0% | 0.0% | 0.0% | 4623.0 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 3813.1 |
| `h264_heavy_reencode` | 8 | 100.0% | 0.0% | 0.0% | 4040.6 |
| `partial_splicing` | 4 | 100.0% | 0.0% | 0.0% | 3856.51 |
| `vfr_fps_mismatch` | 8 | 100.0% | 0.0% | 0.0% | 3762.54 |
