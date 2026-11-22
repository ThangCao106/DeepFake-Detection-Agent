# Multi-Channel Audio Benchmark & Adversarial Pentest Report (`AUDP-20261122_085553_bf972`)
- **Timestamp (UTC)**: 2026-11-22T08:55:53Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Audio Backbone**: `ENHANCED_FUSIONNET_HYBRID` | **Channels**: zalo_voice, tiktok_audio, threads_voice, phone_8khz
- **Manifest & Cache Evaluated**: 0 | **Auto-Feedback**: `False`
- **Total Clips Evaluated**: 40 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **EER**: 0.0% | **ECE**: 0.0542

## 1. Per-Channel Metrics
| Channel | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zalo_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0547 | 93.51 |
| `tiktok_audio` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0541 | 89.32 |
| `threads_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 91.15 |
| `phone_8khz` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 133.48 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ambient_noise_anti_fp` | 4 | 100.0% | 0.0% | 0.0% | 102.8 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 96.84 |
| `hf_dither_bypass` | 4 | 100.0% | 0.0% | 0.0% | 91.87 |
| `jitter_spoof` | 4 | 100.0% | 0.0% | 0.0% | 96.21 |
| `partial_audio_splice` | 4 | 100.0% | 0.0% | 0.0% | 101.17 |
| `pitch_speed_shift` | 8 | 100.0% | 0.0% | 0.0% | 103.73 |
| `telephony_8khz_codec` | 8 | 100.0% | 0.0% | 0.0% | 112.73 |
