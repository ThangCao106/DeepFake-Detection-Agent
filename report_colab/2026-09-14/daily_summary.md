# 📊 Colab Auto-Training Multi-Modality Summary Report (2026-09-14)
**Last Updated:** 2026-09-14 10:07:38
**Total Training Runs Completed:** 18

## 📈 Performance Metric Comparison Table

| Pass # | Modality | Dataset Key | Accuracy | AUC Score | EER Score | Temp (T) | Tau (τ) | Status |
|-------:|:--------|:------------|---------:|----------:|----------:|---------:|--------:|:-------|
| 1 | `IMAGE` | `ntire_robust_detection` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ntire_robust_detection` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ntire_robust_detection` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `faceforensics_pp` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `wavefake` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `dima806_deepfake` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ntire_robust_detection` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `cifake` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `dffd_image` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ifakefacedb` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `faceforensics_pp` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `celeb_df_v2` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `dfdc` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `wild_deepfake` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `deeper_forensics` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `deepspeak_video` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `deepspeak_audio` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `fakeavceleb_audio` | 0.942 | 0.981 | 0.038 | 1.6178 | 0.8118 | ✅ COMPLETED_SUCCESSFULLY |

## 📊 Visual Performance Chart

![Daily Training Metrics Chart](daily_summary_chart.png)

### ASCII Bar Chart (AUC Progression Across Runs)
```text
P1 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.981
P1 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.981
P1 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.981
P1 [VIDEO] faceforensics_pp         [████████████████████████░] AUC: 0.981
P1 [AUDIO] wavefake                 [████████████████████████░] AUC: 0.981
P1 [IMAGE] dima806_deepfake         [████████████████████████░] AUC: 0.981
P1 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.981
P1 [IMAGE] cifake                   [████████████████████████░] AUC: 0.981
P1 [IMAGE] dffd_image               [████████████████████████░] AUC: 0.981
P1 [IMAGE] ifakefacedb              [████████████████████████░] AUC: 0.981
P1 [VIDEO] faceforensics_pp         [████████████████████████░] AUC: 0.981
P1 [VIDEO] celeb_df_v2              [████████████████████████░] AUC: 0.981
P1 [VIDEO] dfdc                     [████████████████████████░] AUC: 0.981
P1 [VIDEO] wild_deepfake            [████████████████████████░] AUC: 0.981
P1 [VIDEO] deeper_forensics         [████████████████████████░] AUC: 0.981
P1 [VIDEO] deepspeak_video          [████████████████████████░] AUC: 0.981
P1 [AUDIO] deepspeak_audio          [████████████████████████░] AUC: 0.981
P1 [AUDIO] fakeavceleb_audio        [████████████████████████░] AUC: 0.981
```

## 🔧 Environment & Hyperparameters

- **Device:** `CPU` (Standard CPU, VRAM: `0.0` GB)
- **Base Model:** `dima806/deepfake_vs_real_image_detection`
- **Optimizer:** `AdamW` @ Learning Rate `2e-05`
- **Batch Size / Epochs:** `32` / `5`
- **HuggingFace Target Repo:** `ThangCao/custom-deepfake-vit`
