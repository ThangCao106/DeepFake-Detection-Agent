# Benchmark Anh Nguoi Viet Duoi Nen, Hop Nhat Theo Chat Luong Tin Hieu & Hoan Ket Luan DiCoME

- **Provenance**: `EMPIRICAL_LOCKED_VN_COMPRESSION_QUALITY_DEFERRAL` | Runtime: `711.28s`
- **Zero-Leakage Audit**: train=258 / cal_val=50 / test_locked=62 (overlap = 0/0/0)
- **Quality-Aware Fusion Hyperparams (locked on `cal_val`)**: base={'w_vision': 0.45, 'w_freq': 0.35, 'w_eye': 0.2}, gamma=1.0

## 1. Giao thuc A: Song con Tin hieu sau Nen & So sanh 3 Chien luoc Hop nhat (`NTIRE 2026` / `FreqFusion`)

| Muc nen / Bien the (`test_locked`) | `q_freq` | `q_eye` | ViT AUC | Wavelet/FFT AUC | Corneal AUC | Fixed Fusion AUC | **Quality-Aware AUC** | Quality TPR@FPR<=1% | Quality Realized FPR (95% CI) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `clean_raw` | 0.913 | 0.602 | 0.7263 | 0.5619 | 0.5640 | 0.7284 | **0.7263** | 0.0323 [0.006, 0.162] | 0.0000 [0.000, 0.110] |
| `mild_jpeg_q80` | 0.915 | 0.601 | 0.7357 | 0.5567 | 0.6098 | 0.7315 | **0.7305** | 0.0323 [0.006, 0.162] | 0.0000 [0.000, 0.110] |
| `medium_social_q55_r75` | 0.899 | 0.609 | 0.7128 | 0.5640 | 0.5838 | 0.7086 | **0.7118** | 0.0323 [0.006, 0.162] | 0.0000 [0.000, 0.110] |
| `heavy_social_webp_q32_r50_blur` | 0.852 | 0.582 | 0.6774 | 0.5754 | 0.4724 | 0.6712 | **0.6805** | 0.0645 [0.018, 0.207] | 0.0000 [0.000, 0.110] |
| `unseen_diffusion_blend_ood` | 0.849 | 0.513 | 0.8439 | 0.0364 | 0.6275 | 0.8418 | **0.8304** | 0.0323 [0.006, 0.162] | 0.0000 [0.000, 0.110] |

### Tong hop Tren Toan bo Tap Kiem thu Gop (`N=310`, `clean + compressed + unseen_ood`)

| Mo hinh | ROC-AUC (95% Boot CI) | TPR @ Target FPR 1% (95% CI) | Realized FPR @ Target 1% (95% CI) | Youden TPR / FPR | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `single_vision_vit` | 0.7411 [0.685, 0.794] | 0.0516 [0.026, 0.099] | 0.0000 [0.000, 0.024] | 0.7419 / 0.4452 | 0.0764 |
| `fixed_weight_fusion` | 0.7386 [0.683, 0.792] | 0.0516 [0.026, 0.099] | 0.0000 [0.000, 0.024] | 0.7290 / 0.4387 | 0.0484 |
| `quality_aware_fusion` | 0.7375 [0.682, 0.792] | 0.0387 [0.018, 0.082] | 0.0000 [0.000, 0.024] | 0.7419 / 0.4452 | 0.0585 |

## 2. Giao thuc B: Dieu kien Chup Nguoi Viet & Phan tang Quang hoc Mat (`Light2Lie`)

| Dieu kien chup / Phan tang | N | `q_freq` | `q_eye` | ViT AUC | Corneal AUC | Quality-Aware AUC | Quality BalAcc (`tau_youden`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `vn_id_official` | 18 | 0.807 | 0.794 | 0.8889 | 0.7778 | 0.9259 | 0.6666 |
| `vn_mature_aging` | 24 | 0.913 | 0.578 | 0.7292 | 0.4514 | 0.7083 | 0.7500 |
| `vn_occlusion_glasses_mask` | 18 | 0.916 | 0.690 | 0.8765 | 0.4938 | 0.9012 | 0.5555 |
| `vn_selfie_lifestyle` | 102 | 0.904 | 0.566 | 0.7228 | 0.5183 | 0.722 | 0.6177 |
| `vn_street_outdoor` | 24 | 0.836 | 0.533 | 0.7431 | 0.5417 | 0.75 | 0.5416 |
| `optical:eye_roi_small` | 93 | -- | 0.550 | -- | 0.4893 | 0.7516 | 0.6970 |
| `optical:eye_roi_adequate` | 93 | -- | 0.645 | -- | 0.5884 | 0.6674 | 0.5510 |
| `optical:lighting_extreme` | 56 | -- | 0.406 | -- | 0.4218 | 0.6051 | 0.5231 |
| `optical:lighting_balanced` | 130 | -- | 0.680 | -- | 0.5878 | 0.7418 | 0.6641 |
| `optical:occlusion_high_or_glare` | 93 | -- | 0.454 | -- | 0.4893 | 0.7247 | 0.6458 |
| `optical:occlusion_clear_ocular` | 93 | -- | 0.741 | -- | 0.5423 | 0.6842 | 0.5972 |

## 3. Hoan Ket luan Co Chon loc (`DiCoME` ICML 2026 vs ViT Probability Margin) tren Cung Ty le Tu dong Ket luan

| Coverage (Tu dong) | Baseline ViT-Margin Err (FPR / FNR) | **DiCoME Quality+Conflict Err (FPR / FNR)** | Giam Sai so Tuyet doi |
| :---: | :---: | :---: | :---: |
| 100% (310/310) | 0.3548 (0.5871 / 0.1226) | **0.3516** (0.4452 / 0.2581) | +0.32% |
| 90% (279/310) | 0.3477 (0.5852 / 0.1250) | **0.3297** (0.4266 / 0.2279) | +1.80% |
| 85% (264/310) | 0.3485 (0.6016 / 0.1103) | **0.3333** (0.4254 / 0.2385) | +1.52% |
| 80% (248/310) | 0.3427 (0.5966 / 0.1085) | **0.3306** (0.4375 / 0.2167) | +1.21% |
| 70% (217/310) | 0.3180 (0.5960 / 0.0847) | **0.3318** (0.4766 / 0.1909) | -1.38% |

- **AURC (thap hon la tot hon)**: ViT-Margin Baseline = `0.3428`, DiCoME Proposed = `0.3345`.