# 📊 Colab Auto-Training Multi-Modality Summary Report (2026-09-15)
**Last Updated:** 2026-09-15 03:32:32
**Total Training Runs Completed:** 28

## 📈 Performance Metric Comparison Table

| Pass # | Modality | Dataset Key | Accuracy | AUC Score | EER Score | Temp (T) | Tau (τ) | Status |
|-------:|:--------|:------------|---------:|----------:|----------:|---------:|--------:|:-------|
| 1 | `IMAGE` | `dima806_deepfake` | 0.924 | 0.979 | 0.030 | 1.6747 | 0.6791 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ntire_robust_detection` | 0.912 | 0.987 | 0.043 | 1.4620 | 0.7344 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `cifake` | 0.921 | 0.971 | 0.034 | 1.2634 | 0.6622 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `dffd_image` | 0.910 | 0.987 | 0.031 | 1.4282 | 0.5968 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `ifakefacedb` | 0.936 | 0.976 | 0.037 | 1.5659 | 0.7835 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `IMAGE` | `dfbench_lmm` | 0.931 | 0.969 | 0.035 | 1.5080 | 0.6642 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `faceforensics_pp` | 0.886 | 0.961 | 0.040 | 1.7685 | 0.7117 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `celeb_df_v2` | 0.912 | 0.967 | 0.045 | 1.7363 | 0.7660 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `dfdc` | 0.922 | 0.941 | 0.035 | 1.7218 | 0.6263 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `wild_deepfake` | 0.919 | 0.948 | 0.036 | 1.5511 | 0.7611 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `deeper_forensics` | 0.911 | 0.960 | 0.046 | 1.6932 | 0.7284 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `deepspeak_video` | 0.901 | 0.951 | 0.047 | 1.6180 | 0.8147 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `mega_mmdf` | 0.895 | 0.936 | 0.038 | 1.6498 | 0.8074 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `VIDEO` | `df40` | 0.919 | 0.942 | 0.044 | 1.4810 | 0.6009 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `deepspeak_audio` | 0.880 | 0.968 | 0.040 | 1.3959 | 0.7366 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `fakeavceleb_audio` | 0.856 | 0.973 | 0.054 | 1.5908 | 0.7124 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `wavefake` | 0.862 | 0.918 | 0.055 | 1.5625 | 0.7722 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `asvspoof_2021` | 0.873 | 0.956 | 0.052 | 1.6200 | 0.8634 | ✅ COMPLETED_SUCCESSFULLY |
| 1 | `AUDIO` | `omni_fake` | 0.866 | 0.922 | 0.048 | 1.5848 | 0.8429 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `IMAGE` | `dima806_deepfake` | 0.923 | 0.998 | 0.023 | 1.5083 | 0.6482 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `IMAGE` | `ntire_robust_detection` | 0.960 | 0.995 | 0.012 | 1.5947 | 0.7163 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `IMAGE` | `cifake` | 0.955 | 0.972 | 0.012 | 1.3690 | 0.7850 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `IMAGE` | `dffd_image` | 0.928 | 0.998 | 0.022 | 1.3310 | 0.6252 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `IMAGE` | `ifakefacedb` | 0.934 | 0.980 | 0.013 | 1.3717 | 0.7423 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `IMAGE` | `dfbench_lmm` | 0.923 | 0.989 | 0.015 | 1.3044 | 0.7076 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `VIDEO` | `faceforensics_pp` | 0.899 | 0.958 | 0.019 | 1.4104 | 0.8340 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `VIDEO` | `celeb_df_v2` | 0.940 | 0.981 | 0.026 | 1.4431 | 0.6581 | ✅ COMPLETED_SUCCESSFULLY |
| 2 | `VIDEO` | `dfdc` | 0.943 | 0.995 | 0.024 | 1.5120 | 0.6423 | ✅ COMPLETED_SUCCESSFULLY |

## 📊 Visual Performance Chart

![Daily Training Metrics Chart](daily_summary_chart.png)

### ASCII Bar Chart (AUC Progression Across Runs)
```text
P1 [IMAGE] dima806_deepfake         [████████████████████████░] AUC: 0.979
P1 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.987
P1 [IMAGE] cifake                   [████████████████████████░] AUC: 0.971
P1 [IMAGE] dffd_image               [████████████████████████░] AUC: 0.987
P1 [IMAGE] ifakefacedb              [████████████████████████░] AUC: 0.976
P1 [IMAGE] dfbench_lmm              [████████████████████████░] AUC: 0.969
P1 [VIDEO] faceforensics_pp         [████████████████████████░] AUC: 0.961
P1 [VIDEO] celeb_df_v2              [████████████████████████░] AUC: 0.967
P1 [VIDEO] dfdc                     [███████████████████████░░] AUC: 0.941
P1 [VIDEO] wild_deepfake            [███████████████████████░░] AUC: 0.948
P1 [VIDEO] deeper_forensics         [████████████████████████░] AUC: 0.960
P1 [VIDEO] deepspeak_video          [███████████████████████░░] AUC: 0.951
P1 [VIDEO] mega_mmdf                [███████████████████████░░] AUC: 0.936
P1 [VIDEO] df40                     [███████████████████████░░] AUC: 0.942
P1 [AUDIO] deepspeak_audio          [████████████████████████░] AUC: 0.968
P1 [AUDIO] fakeavceleb_audio        [████████████████████████░] AUC: 0.973
P1 [AUDIO] wavefake                 [██████████████████████░░░] AUC: 0.918
P1 [AUDIO] asvspoof_2021            [███████████████████████░░] AUC: 0.956
P1 [AUDIO] omni_fake                [███████████████████████░░] AUC: 0.922
P2 [IMAGE] dima806_deepfake         [████████████████████████░] AUC: 0.998
P2 [IMAGE] ntire_robust_detection   [████████████████████████░] AUC: 0.995
P2 [IMAGE] cifake                   [████████████████████████░] AUC: 0.972
P2 [IMAGE] dffd_image               [████████████████████████░] AUC: 0.998
P2 [IMAGE] ifakefacedb              [████████████████████████░] AUC: 0.980
P2 [IMAGE] dfbench_lmm              [████████████████████████░] AUC: 0.989
P2 [VIDEO] faceforensics_pp         [███████████████████████░░] AUC: 0.958
P2 [VIDEO] celeb_df_v2              [████████████████████████░] AUC: 0.981
P2 [VIDEO] dfdc                     [████████████████████████░] AUC: 0.995
```

## 🔧 Environment & Hyperparameters

- **Device:** `CPU` (Standard CPU, VRAM: `0.0` GB)
- **Base Model:** `dima806/deepfake_vs_real_image_detection`
- **Optimizer:** `AdamW` @ Learning Rate `2e-05`
- **Batch Size / Epochs:** `32` / `5`
- **HuggingFace Target Repo:** `ThangCao/custom-deepfake-vit`
