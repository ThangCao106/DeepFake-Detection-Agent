# Multi-Channel Audio Benchmark & Adversarial Pentest Report (`AUDP-20261121_201937_1d8a2`)
- **Timestamp (UTC)**: 2026-11-21T20:19:37Z | **Mode**: `EMPIRICAL_LIVE_INFERENCE`
- **Audio Backbone**: `ENHANCED_FUSIONNET_HYBRID` | **Channels**: zalo_voice, tiktok_audio, threads_voice, phone_8khz
- **Manifest & Cache Evaluated**: 0 | **Auto-Feedback**: `False`
- **Total Clips Evaluated**: 40 | **Accuracy**: 100.00% | **TPR**: 100.00% | **FPR**: 0.00% | **ROC-AUC**: 1.0 | **EER**: 0.0% | **ECE**: 0.0542

## 1. Per-Channel Metrics
| Channel | Samples | Accuracy | TPR | FPR | FNR | ROC-AUC | EER (%) | ECE | Latency (ms) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `zalo_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0547 | 153.98 |
| `tiktok_audio` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0541 | 129.58 |
| `threads_voice` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 86.01 |
| `phone_8khz` | 10 | 100.0% | 100.0% | 0.0% | 0.0% | 1.0 | 0.0% | 0.0539 | 108.93 |

## 2. Per-Attack Vector Robustness
| Attack Vector | Samples | Accuracy | FPR | FNR | Mean Latency (ms) |
|:---|---:|---:|---:|---:|---:|
| `ambient_noise_anti_fp` | 4 | 100.0% | 0.0% | 0.0% | 90.8 |
| `clean_baseline` | 8 | 100.0% | 0.0% | 0.0% | 120.27 |
| `hf_dither_bypass` | 4 | 100.0% | 0.0% | 0.0% | 127.16 |
| `jitter_spoof` | 4 | 100.0% | 0.0% | 0.0% | 108.16 |
| `partial_audio_splice` | 4 | 100.0% | 0.0% | 0.0% | 132.5 |
| `pitch_speed_shift` | 8 | 100.0% | 0.0% | 0.0% | 114.94 |
| `telephony_8khz_codec` | 8 | 100.0% | 0.0% | 0.0% | 133.61 |
