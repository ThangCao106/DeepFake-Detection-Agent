# Multi-Channel Audio Benchmark & Adversarial Pentest Report (`AUDP-20261126_100651_e67dd`)
- **Timestamp (UTC)**: 2026-11-26T10:06:51Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Audio Backbone**: `ENHANCED_FUSIONNET_HYBRID` | **Channels**: zalo_voice, tiktok_audio, threads_voice, phone_8khz
- **Manifest & Cache Evaluated**: 0 | **Auto-Feedback**: `False`
- **Total Clips Evaluated**: 40 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **EER**: 0.0% | **ECE**: 0.0542

## 1. Per-Channel Metrics
| Channel | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zalo_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0547 | 83.36 |
| `tiktok_audio` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0541 | 80.66 |
| `threads_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 81.6 |
| `phone_8khz` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 80.51 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ambient_noise_anti_fp` | 4 | 100.0% | 0.0% | 0.0% | 66.61 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 85.41 |
| `hf_dither_bypass` | 4 | 100.0% | 0.0% | 0.0% | 82.48 |
| `jitter_spoof` | 4 | 100.0% | 0.0% | 0.0% | 82.87 |
| `partial_audio_splice` | 4 | 100.0% | 0.0% | 0.0% | 82.26 |
| `pitch_speed_shift` | 8 | 100.0% | 0.0% | 0.0% | 82.07 |
| `telephony_8khz_codec` | 8 | 100.0% | 0.0% | 0.0% | 83.08 |
