# Multi-Channel Audio Benchmark & Adversarial Pentest Report (`AUDP-20261120_213627_f60ab`)
- **Timestamp (UTC)**: 2026-11-20T21:36:27Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Audio Backbone**: `ENHANCED_FUSIONNET_HYBRID` | **Channels**: zalo_voice, tiktok_audio, threads_voice, phone_8khz
- **Manifest & Cache Evaluated**: 0 | **Auto-Feedback**: `False`
- **Total Clips Evaluated**: 40 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **EER**: 0.0% | **ECE**: 0.0542

## 1. Per-Channel Metrics
| Channel | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zalo_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0547 | 125.64 |
| `tiktok_audio` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0541 | 120.37 |
| `threads_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 118.9 |
| `phone_8khz` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 120.7 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ambient_noise_anti_fp` | 4 | 100.0% | 0.0% | 0.0% | 99.85 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 128.31 |
| `hf_dither_bypass` | 4 | 100.0% | 0.0% | 0.0% | 124.33 |
| `jitter_spoof` | 4 | 100.0% | 0.0% | 0.0% | 123.04 |
| `partial_audio_splice` | 4 | 100.0% | 0.0% | 0.0% | 122.21 |
| `pitch_speed_shift` | 8 | 100.0% | 0.0% | 0.0% | 121.78 |
| `telephony_8khz_codec` | 8 | 100.0% | 0.0% | 0.0% | 122.21 |
