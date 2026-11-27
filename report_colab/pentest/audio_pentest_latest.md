# Multi-Channel Audio Benchmark & Adversarial Pentest Report (`AUDP-20261127_101249_be4f2`)
- **Timestamp (UTC)**: 2026-11-27T10:12:49Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Audio Backbone**: `ENHANCED_FUSIONNET_HYBRID` | **Channels**: zalo_voice, tiktok_audio, threads_voice, phone_8khz
- **Manifest & Cache Evaluated**: 0 | **Auto-Feedback**: `False`
- **Total Clips Evaluated**: 40 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **EER**: 0.0% | **ECE**: 0.0542

## 1. Per-Channel Metrics
| Channel | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zalo_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0547 | 82.8 |
| `tiktok_audio` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0541 | 80.35 |
| `threads_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 80.17 |
| `phone_8khz` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 80.02 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ambient_noise_anti_fp` | 4 | 100.0% | 0.0% | 0.0% | 65.72 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 84.96 |
| `hf_dither_bypass` | 4 | 100.0% | 0.0% | 0.0% | 82.17 |
| `jitter_spoof` | 4 | 100.0% | 0.0% | 0.0% | 81.86 |
| `partial_audio_splice` | 4 | 100.0% | 0.0% | 0.0% | 81.69 |
| `pitch_speed_shift` | 8 | 100.0% | 0.0% | 0.0% | 81.23 |
| `telephony_8khz_codec` | 8 | 100.0% | 0.0% | 0.0% | 82.24 |
