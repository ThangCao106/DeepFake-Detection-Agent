# Multi-Channel Audio Benchmark & Adversarial Pentest Report (`AUDP-20261119_112130_f7414`)
- **Timestamp (UTC)**: 2026-11-19T11:21:30Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Audio Backbone**: `ENHANCED_FUSIONNET_HYBRID` | **Channels**: zalo_voice, tiktok_audio, threads_voice, phone_8khz
- **Manifest & Cache Evaluated**: 12 | **Auto-Feedback**: `False`
- **Total Clips Evaluated**: 52 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **EER**: 0.0% | **ECE**: 0.0677

## 1. Per-Channel Metrics
| Channel | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zalo_voice` | 16 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.1051 | 399.56 |
| `tiktok_audio` | 16 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0476 | 546.68 |
| `threads_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 414.25 |
| `phone_8khz` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 349.22 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ambient_noise_anti_fp` | 4 | 100.0% | 0.0% | 0.0% | 341.84 |
| `cache_regression` | 10 | 100.0% | 0.0% | 0.0% | 300.56 |
| `clean_baseline` | 10 | 100.0% | 0.0% | 0.0% | 471.62 |
| `hf_dither_bypass` | 4 | 100.0% | 0.0% | 0.0% | 503.98 |
| `jitter_spoof` | 4 | 100.0% | 0.0% | 0.0% | 443.31 |
| `partial_audio_splice` | 4 | 100.0% | 0.0% | 0.0% | 615.87 |
| `pitch_speed_shift` | 8 | 100.0% | 0.0% | 0.0% | 452.86 |
| `telephony_8khz_codec` | 8 | 100.0% | 0.0% | 0.0% | 476.25 |
