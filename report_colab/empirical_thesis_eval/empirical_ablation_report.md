# Báo Cáo Thực Nghiệm Định Lượng & Ablation Độc Lập Trên Tập Kiểm Thử Khóa (`test_locked`)

- **Thời điểm chạy (UTC):** `2026-11-27T10:29:06Z` | **Provenance:** `EMPIRICAL_LOCKED_SPLIT_INFERENCE`
- **Kiểm toán chống rò rỉ (`origin_id`):** Train=258 (125 gốc) | Cal/Val=50 (25 gốc) | Test Locked=62 (31 gốc) | Overlap=`0/0/0`
- **Khóa ngưỡng Calibration (`cal_val`):** Trọng số dung hợp `w_dual=0.20`, `w_bio=0.80` (tối ưu & khóa trước khi mở `test_locked`)

## View: `clean` (N=62)
| Cấu hình Mô hình | Locked $\tau^*$ | TPR (95% CI) | FPR (95% CI) | FNR | Bal. Acc | ECE | ROC-AUC (95% CI) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: SBI Spatial CNN (CVPR 2022 methodology) | 0.4200 | 90.3% [77.4–100.0] | 83.9% [71.0–96.8] | 9.7% | 53.2% | 0.0333 | 0.5869 [0.433–0.723] |
| Ablation 1: Spatial RGB Only (FCB Zeroed) | 0.4300 | 90.3% [80.7–100.0] | 64.5% [48.4–80.7] | 9.7% | 62.9% | 0.0959 | 0.6348 [0.491–0.767] |
| Ablation 2: Spatial + Noise/Spectral FCB (FFT+SRM) | 0.5400 | 54.8% [38.7–71.0] | 29.0% [16.1–45.2] | 45.2% | 62.9% | 0.1391 | 0.6972 [0.564–0.817] |
| Ablation 3: Biometric Physiology (DWT+rPPG+Attn) | 0.4600 | 96.8% [90.3–100.0] | 90.3% [80.7–100.0] | 3.2% | 53.2% | 0.0540 | 0.5900 [0.434–0.726] |
| Full Proposed: Multi-Stream Fusion (w=0.20) + Locked Platt | 0.5300 | 58.1% [41.9–74.2] | 35.5% [19.4–51.6] | 41.9% | 61.3% | 0.0956 | 0.7034 [0.571–0.822] |

## View: `social_reupload` (N=62)
| Cấu hình Mô hình | Locked $\tau^*$ | TPR (95% CI) | FPR (95% CI) | FNR | Bal. Acc | ECE | ROC-AUC (95% CI) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: SBI Spatial CNN (CVPR 2022 methodology) | 0.4200 | 90.3% [77.4–100.0] | 83.9% [71.0–96.8] | 9.7% | 53.2% | 0.0443 | 0.5838 [0.435–0.723] |
| Ablation 1: Spatial RGB Only (FCB Zeroed) | 0.4300 | 90.3% [80.7–100.0] | 64.5% [48.4–80.7] | 9.7% | 62.9% | 0.0946 | 0.6337 [0.492–0.765] |
| Ablation 2: Spatial + Noise/Spectral FCB (FFT+SRM) | 0.5400 | 54.8% [38.7–71.0] | 35.5% [19.4–51.6] | 45.2% | 59.7% | 0.1359 | 0.6941 [0.559–0.814] |
| Ablation 3: Biometric Physiology (DWT+rPPG+Attn) | 0.4600 | 96.8% [90.3–100.0] | 93.5% [83.9–100.0] | 3.2% | 51.6% | 0.0395 | 0.5890 [0.436–0.723] |
| Full Proposed: Multi-Stream Fusion (w=0.20) + Locked Platt | 0.5300 | 61.3% [41.9–77.4] | 38.7% [22.6–54.8] | 38.7% | 61.3% | 0.0992 | 0.6982 [0.565–0.819] |

## View: `unseen_manipulation_ood` (N=62)
| Cấu hình Mô hình | Locked $\tau^*$ | TPR (95% CI) | FPR (95% CI) | FNR | Bal. Acc | ECE | ROC-AUC (95% CI) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: SBI Spatial CNN (CVPR 2022 methodology) | 0.4200 | 90.3% [77.4–100.0] | 83.9% [71.0–96.8] | 9.7% | 53.2% | 0.0248 | 0.5744 [0.417–0.709] |
| Ablation 1: Spatial RGB Only (FCB Zeroed) | 0.4300 | 77.4% [61.3–90.3] | 64.5% [48.4–80.7] | 22.6% | 56.5% | 0.0900 | 0.5848 [0.444–0.721] |
| Ablation 2: Spatial + Noise/Spectral FCB (FFT+SRM) | 0.5400 | 51.6% [32.3–67.7] | 32.3% [16.1–48.4] | 48.4% | 59.7% | 0.1090 | 0.6847 [0.550–0.814] |
| Ablation 3: Biometric Physiology (DWT+rPPG+Attn) | 0.4600 | 96.8% [90.3–100.0] | 90.3% [80.7–100.0] | 3.2% | 53.2% | 0.0417 | 0.6015 [0.442–0.744] |
| Full Proposed: Multi-Stream Fusion (w=0.20) + Locked Platt | 0.5300 | 54.8% [38.7–71.0] | 38.7% [22.6–54.8] | 45.2% | 58.1% | 0.0841 | 0.6868 [0.548–0.818] |

## Phân tích Lỗi theo Điều kiện chụp Người Việt (`full_proposed_fusion_calibrated`)
### Điều kiện chụp — View `clean`
| Capture Condition | Samples (n) | Errors | FP | FN | Error Rate | 95% Wilson CI |
|:---|---:|---:|---:|---:|---:|---:|
| `vn_id_official` | 6 | 2 | 1 | 1 | 33.3% | [9.7%–70.0%] |
| `vn_mature_aging` | 8 | 3 | 1 | 2 | 37.5% | [13.7%–69.4%] |
| `vn_occlusion_glasses_mask` | 6 | 3 | 1 | 2 | 50.0% | [18.8%–81.2%] |
| `vn_selfie_lifestyle` | 34 | 13 | 7 | 6 | 38.2% | [23.9%–55.0%] |
| `vn_street_outdoor` | 8 | 3 | 1 | 2 | 37.5% | [13.7%–69.4%] |

### Điều kiện chụp — View `social_reupload`
| Capture Condition | Samples (n) | Errors | FP | FN | Error Rate | 95% Wilson CI |
|:---|---:|---:|---:|---:|---:|---:|
| `vn_id_official` | 6 | 2 | 1 | 1 | 33.3% | [9.7%–70.0%] |
| `vn_mature_aging` | 8 | 4 | 2 | 2 | 50.0% | [21.5%–78.5%] |
| `vn_occlusion_glasses_mask` | 6 | 3 | 1 | 2 | 50.0% | [18.8%–81.2%] |
| `vn_selfie_lifestyle` | 34 | 13 | 7 | 6 | 38.2% | [23.9%–55.0%] |
| `vn_street_outdoor` | 8 | 2 | 1 | 1 | 25.0% | [7.1%–59.1%] |

### Điều kiện chụp — View `unseen_manipulation_ood`
| Capture Condition | Samples (n) | Errors | FP | FN | Error Rate | 95% Wilson CI |
|:---|---:|---:|---:|---:|---:|---:|
| `vn_id_official` | 6 | 3 | 1 | 2 | 50.0% | [18.8%–81.2%] |
| `vn_mature_aging` | 8 | 4 | 2 | 2 | 50.0% | [21.5%–78.5%] |
| `vn_occlusion_glasses_mask` | 6 | 2 | 1 | 1 | 33.3% | [9.7%–70.0%] |
| `vn_selfie_lifestyle` | 34 | 14 | 7 | 7 | 41.2% | [26.4%–57.8%] |
| `vn_street_outdoor` | 8 | 3 | 1 | 2 | 37.5% | [13.7%–69.4%] |

