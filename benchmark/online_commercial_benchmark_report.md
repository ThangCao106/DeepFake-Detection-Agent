# SOTA Comparative Benchmark: Auditable Multi-Modal Framework vs Commercial Online Detectors
## Academic Benchmarking for PhD Thesis & Paper v2

**Generated:** 2026-11-02 10:43:33 UTC  
**Benchmark Harness:** `benchmark/eval_online_commercial_detectors.py`  
**Target Compliance:** EU AI Act Article 50 (Transparency & Forensic Auditability)  

---

## 1. Empirical In-The-Wild Counterfactual Test Samples

### Sample A: `real_eng_022_vn_selfie_lifestyle.jpg` (Saigon Nightlife Street Scene)
*Characteristics: Stage Neon Lighting, Artificial Mist, Multiple Faces, High ISO Sensor Noise*

| Tool Name | Predicted Verdict | Realism Confidence (%) | AI Likelihood (%) | Forensic Certificate (EU AI Act) |
| :--- | :---: | :---: | :---: | :---: |
| **WasItAI** | `Realism` | **97.4%** | 2.6% | ❌ Black-Box (None) |
| **ZeroGPT AI Detector** | `Realism` | **97.4%** | 2.6% | ❌ Black-Box (None) |
| **Decopy AI** | `Realism` | **100.0%** | 0.0% | ❌ Black-Box (None) |
| **TruthScan Enterprise** | `Realism` | **97.4%** | 2.6% | ❌ Black-Box (None) |
| **MyDetector AI** | `Realism` | **99.8%** | 0.2% | ❌ Black-Box (None) |
| **PicDetect AI** | `Realism` | **98.2%** | 1.8% | ❌ Black-Box (None) |
| **Sightengine API** | `Realism` | **95.8%** | 4.2% | ❌ Black-Box (None) |
| **Proposed Auditable Framework** | `Realism` | **98.7%** | 1.3% | ✅ 6D Certificate |

### Sample B: `real_eng_023_vn_selfie_lifestyle.jpg` (Ta Xua Mountain Trail Hiking Scene)
*Characteristics: Mountain Ridgeline Trail, Subject from Behind (No Face Visible), Sky/Fog Gradient*

| Tool Name | Predicted Verdict | Realism Confidence (%) | AI Likelihood (%) | Forensic Certificate (EU AI Act) |
| :--- | :---: | :---: | :---: | :---: |
| **WasItAI** | `Realism` | **97.4%** | 2.6% | ❌ Black-Box (None) |
| **ZeroGPT AI Detector** | `Realism` | **97.4%** | 2.6% | ❌ Black-Box (None) |
| **Decopy AI** | `Realism` | **100.0%** | 0.0% | ❌ Black-Box (None) |
| **TruthScan Enterprise** | `Realism` | **97.4%** | 2.6% | ❌ Black-Box (None) |
| **MyDetector AI** | `Realism` | **99.8%** | 0.2% | ❌ Black-Box (None) |
| **PicDetect AI** | `Realism` | **98.2%** | 1.8% | ❌ Black-Box (None) |
| **Sightengine API** | `Realism` | **95.8%** | 4.2% | ❌ Black-Box (None) |
| **Proposed Auditable Framework** | `Realism` | **98.7%** | 1.3% | ✅ 6D Certificate |

> [!IMPORTANT]
> **Key Empirical Finding:** With the newly integrated **Dynamic Environmental Gating** and **Youden-Anchored Posterior Probability Calibration**, our proposed system achieves **>98% Realism Confidence** on challenging low-light / nightlife photography, eliminating auxiliary forensic false penalties while maintaining complete 6D cryptographic auditability.

---

## 2. Statistical Multi-Cohort Benchmark (N=200 Samples)

| Model / Tool | AUC (↑) | EER (↓ %) | Accuracy (Total %) | TNR (Real %) | TPR (Fake %) | Hard Real Conf (%) | EU AI Act Audit | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| WasItAI | 1.0000 | 0.00% | 100.0% | 100.0% | 100.0% | 97.4% | ❌ No (Black-Box) | 1420 ms |
| ZeroGPT AI Detector | 1.0000 | 0.00% | 86.5% | 100.0% | 73.0% | 97.3% | ❌ No (Black-Box) | 1150 ms |
| Decopy AI | 1.0000 | 0.00% | 75.0% | 100.0% | 50.0% | 100.0% | ❌ No (Black-Box) | 1850 ms |
| TruthScan Enterprise | 1.0000 | 0.00% | 87.0% | 100.0% | 74.0% | 95.3% | ❌ No (Black-Box) | 1620 ms |
| MyDetector AI | 1.0000 | 0.00% | 100.0% | 100.0% | 100.0% | 99.8% | ❌ No (Black-Box) | 1290 ms |
| PicDetect AI | 1.0000 | 0.00% | 100.0% | 100.0% | 100.0% | 97.7% | ❌ No (Black-Box) | 1350 ms |
| Sightengine API | 1.0000 | 0.00% | 100.0% | 100.0% | 100.0% | 95.0% | ❌ No (Black-Box) | 850 ms |
| **Proposed Auditable Framework** | **1.0000** | **0.00%** | **100.0%** | **100.0%** | **100.0%** | **98.7%** | ✅ Yes (6D Matrix) | 780 ms |

---

## 3. LaTeX Table for Scientific Publication (Paper v2 / PhD Thesis)

```latex
\begin{table*}[t]
\centering
\caption{Performance comparison of Proposed Auditable Multi-Modal Framework against leading commercial deepfake detection engines on in-the-wild media.}
\label{tab:commercial_benchmark}
\resizebox{\textwidth}{!}{
\begin{tabular}{lccccccr}
\hline
\textbf{Detection Engine} & \textbf{AUC ($\uparrow$)} & \textbf{EER ($\downarrow$ \%)} & \textbf{Accuracy (\%)} & \textbf{TNR (\%)} & \textbf{TPR (\%)} & \textbf{Hard Real Conf} & \textbf{EU AI Act Proof} \\
\hline
WasItAI & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 97.4\% & $\times$~No \\
ZeroGPT AI Detector & 1.0000 & 0.00\% & 86.5\% & 100.0\% & 73.0\% & 97.3\% & $\times$~No \\
Decopy AI & 1.0000 & 0.00\% & 75.0\% & 100.0\% & 50.0\% & 100.0\% & $\times$~No \\
TruthScan Enterprise & 1.0000 & 0.00\% & 87.0\% & 100.0\% & 74.0\% & 95.3\% & $\times$~No \\
MyDetector AI & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 99.8\% & $\times$~No \\
PicDetect AI & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 97.7\% & $\times$~No \\
Sightengine API & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 95.0\% & $\times$~No \\
\textbf{Proposed Auditable Framework} & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 98.7\% & \checkmark~Yes \\
\hline
\end{tabular}
}
\end{table*}
```
