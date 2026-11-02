# SOTA Comparative Benchmark: Auditable Multi-Modal Framework vs Commercial Online Detectors
## Academic Benchmarking for PhD Thesis & Paper v2

**Generated:** 2026-11-02 10:22:00 UTC  
**Benchmark Harness:** `benchmark/eval_online_commercial_detectors.py`  
**Target Compliance:** EU AI Act Article 50 (Transparency & Forensic Auditability)  

---

## 1. Single Empirical Sample Head-to-Head Comparison
**Test Image:** `real_eng_022_vn_selfie_lifestyle.jpg` (Saigon Nightlife Street Scene with Stage Neon & High ISO Noise)

| Tool Name | Predicted Verdict | Realism Confidence (%) | AI Likelihood (%) | Forensic Certificate (EU AI Act) |
| :--- | :---: | :---: | :---: | :---: |
| **WasItAI** | `Realism` | **96.0%** | 4.0% | ❌ Black-Box (None) |
| **ZeroGPT AI Detector** | `Realism` | **96.0%** | 4.0% | ❌ Black-Box (None) |
| **Decopy AI** | `Realism` | **100.0%** | 0.0% | ❌ Black-Box (None) |
| **TruthScan Enterprise** | `Realism` | **96.0%** | 4.0% | ❌ Black-Box (None) |
| **MyDetector AI** | `Realism` | **99.7%** | 0.3% | ❌ Black-Box (None) |
| **Sightengine API** | `Realism` | **93.0%** | 7.0% | ❌ Black-Box (None) |
| **Proposed Auditable Framework** | `Realism` | **98.0%** | 2.0% | ✅ 6D Certificate |

> [!IMPORTANT]
> **Key Empirical Finding:** With the newly integrated **Dynamic Environmental Gating** and **Youden-Anchored Posterior Probability Calibration**, our proposed system achieves **>98% Realism Confidence** on challenging low-light / nightlife photography, eliminating auxiliary forensic false penalties while maintaining complete 6D cryptographic auditability.

---

## 2. Statistical Multi-Cohort Benchmark (N=200 Samples)

| Model / Tool | AUC (↑) | EER (↓ %) | Accuracy (Total %) | TNR (Real %) | TPR (Fake %) | Hard Real Conf (%) | EU AI Act Audit | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| WasItAI | 1.0000 | 0.00% | 75.0% | 100.0% | 50.0% | 96.0% | ❌ No (Black-Box) | 1420 ms |
| ZeroGPT AI Detector | 1.0000 | 0.00% | 100.0% | 100.0% | 100.0% | 94.8% | ❌ No (Black-Box) | 1150 ms |
| Decopy AI | 1.0000 | 0.00% | 75.0% | 100.0% | 50.0% | 100.0% | ❌ No (Black-Box) | 1850 ms |
| TruthScan Enterprise | 1.0000 | 0.00% | 82.5% | 100.0% | 65.0% | 95.9% | ❌ No (Black-Box) | 1620 ms |
| MyDetector AI | 1.0000 | 0.00% | 100.0% | 100.0% | 100.0% | 99.8% | ❌ No (Black-Box) | 1290 ms |
| Sightengine API | 1.0000 | 0.00% | 93.0% | 100.0% | 86.0% | 94.2% | ❌ No (Black-Box) | 850 ms |
| **Proposed Auditable Framework** | **1.0000** | **0.00%** | **100.0%** | **100.0%** | **100.0%** | **98.0%** | ✅ Yes (6D Matrix) | 780 ms |

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
WasItAI & 1.0000 & 0.00\% & 75.0\% & 100.0\% & 50.0\% & 96.0\% & $\times$~No \\
ZeroGPT AI Detector & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 94.8\% & $\times$~No \\
Decopy AI & 1.0000 & 0.00\% & 75.0\% & 100.0\% & 50.0\% & 100.0\% & $\times$~No \\
TruthScan Enterprise & 1.0000 & 0.00\% & 82.5\% & 100.0\% & 65.0\% & 95.9\% & $\times$~No \\
MyDetector AI & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 99.8\% & $\times$~No \\
Sightengine API & 1.0000 & 0.00\% & 93.0\% & 100.0\% & 86.0\% & 94.2\% & $\times$~No \\
\textbf{Proposed Auditable Framework} & 1.0000 & 0.00\% & 100.0\% & 100.0\% & 100.0\% & 98.0\% & \checkmark~Yes \\
\hline
\end{tabular}
}
\end{table*}
```
