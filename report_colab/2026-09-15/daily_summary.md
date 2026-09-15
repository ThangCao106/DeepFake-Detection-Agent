# 📊 Colab Auto-Training Multi-Modality Summary Report (2026-09-15)
**Last Updated:** 2026-09-15 03:29:59
**Total Training Runs Completed:** 4

## 📈 Performance Metric Comparison Table

| Pass # | Modality | Dataset Key | Accuracy | AUC Score | EER Score | Temp (T) | Tau (τ) | Status |
|-------:|:--------|:------------|---------:|----------:|----------:|---------:|--------:|:-------|
| 1 | `IMAGE` | `dima806_deepfake` | 0.924 | 0.979 | 0.030 | 1.6747 | 0.6791 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ntire_robust_detection` | 0.912 | 0.987 | 0.043 | 1.4620 | 0.7344 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `cifake` | 0.921 | 0.971 | 0.034 | 1.2634 | 0.6622 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `dffd_image` | 0.910 | 0.987 | 0.031 | 1.4282 | 0.5968 | ✅ COMPLETED_SUCCESSFULLY |

## 📊 Visual Performance Chart

![Daily Training Metrics Chart](daily_summary_chart.png)

### ASCII Bar Chart (AUC Progression Across Runs)
```text
P1 [IMAGE] dima806_deepfake         [████████████████████████░] AUC: 0.979
P1 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.987
P1 [IMAGE] cifake                   [████████████████████████░] AUC: 0.971
P1 [IMAGE] dffd_image               [████████████████████████░] AUC: 0.987
```

## 🔧 Environment & Hyperparameters

- **Device:** `CPU` (Standard CPU, VRAM: `0.0` GB)
- **Base Model:** `dima806/deepfake_vs_real_image_detection`
- **Optimizer:** `AdamW` @ Learning Rate `2e-05`
- **Batch Size / Epochs:** `32` / `5`
- **HuggingFace Target Repo:** `ThangCao/custom-deepfake-vit`
