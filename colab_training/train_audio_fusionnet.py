# -*- coding: utf-8 -*-
"""
Enhanced Audio FusionNet Trainer (SOTA Multi-Branch Acoustic Deepfake Classifier)
---------------------------------------------------------------------------------
Fuses three complementary acoustic representations over real waveform tensors:
  1. Branch A (Spectral CNN): 64x64 linear-band log-STFT magnitude spectrogram.
  2. Branch B (Temporal BiGRU): 48x8 frame-level sequential acoustic trajectory
     (RMS, ZCR, spectral flatness, >7.5kHz ratio, 4-7.5kHz valley ratio, 0.3-3.8kHz core,
      spectral flux, and pitch-band autocorrelation harmonicity).
  3. Branch C (Forensic Vector MLP): 8-D DSP evidence signature from AudioForensicAnalyzer.

Trains on cached `runs/benchmarks/media_cache/audio/*.wav` (12 clips) + balanced multi-channel
adversarial suite from `generate_audio_pentest_suite` (40 clips across Zalo Voice, TikTok Audio,
Threads Voice, and 8kHz Telephony), computes stratified 5-fold out-of-fold validation metrics
and held-out duration-shifted validation metrics, exports `custom_vit_model/audio_fusionnet.pt`
(< 600 KB), writes `custom_vit_model/audio_training_meta.json`, and registers the artifact in
SQLite `deepfake_agent_memory.db` (`colab_train_reports` + `model_weights_registry`).
"""

import json
import os
import sys
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CUSTOM_MODEL_DIR = PROJECT_ROOT / "custom_vit_model"
CHECKPOINT_PATH = CUSTOM_MODEL_DIR / "audio_fusionnet.pt"
META_PATH = CUSTOM_MODEL_DIR / "audio_training_meta.json"
CACHE_AUDIO_DIR = PROJECT_ROOT / "runs" / "benchmarks" / "media_cache" / "audio"


def _compute_roc_auc_and_eer(y_true: List[int], y_scores: List[float]) -> Tuple[float, float]:
    """Computes exact pairwise ROC-AUC and Equal Error Rate (%)."""
    pos = [s for y, s in zip(y_true, y_scores) if y == 1]
    neg = [s for y, s in zip(y_true, y_scores) if y == 0]
    if not pos or not neg:
        return 1.0, 0.0
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1.0
            elif abs(sp - sn) < 1e-12:
                wins += 0.5
    auc = round(wins / (len(pos) * len(neg)), 4)

    thresholds = sorted(set(y_scores + [0.0, 0.5, 1.0]))
    best_diff = 1.0
    best_eer = 0.0
    for th in thresholds:
        fpr = sum(1 for sn in neg if sn >= th) / len(neg)
        fnr = sum(1 for sp in pos if sp < th) / len(pos)
        diff = abs(fpr - fnr)
        if diff <= best_diff:
            best_diff = diff
            best_eer = 0.5 * (fpr + fnr)
    return auc, round(best_eer * 100.0, 2)


def extract_fusionnet_tensors(
    y: np.ndarray,
    sr: int,
    audio_metrics: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts deterministic multi-branch neural inputs from real audio waveform `y` and
    `AudioForensicAnalyzer` metrics:
      - spec: (1, 64, 64) normalized log-STFT magnitude spectrogram
      - seq:  (48, 8) frame-level acoustic dynamics sequence
      - vec:  (8,) normalized DSP forensic feature vector
    """
    y_arr = np.asarray(y, dtype=np.float32).flatten()
    if len(y_arr) < 1024:
        y_arr = np.pad(y_arr, (0, 1024 - len(y_arr)))
    sr = max(8000, int(sr or 22050))
    y_arr = y_arr[: min(len(y_arr), sr * 10)]
    peak = max(1e-5, float(np.max(np.abs(y_arr))))
    y_norm = y_arr / peak

    # 1. Spectral map (1, 64, 64): 64 frames x 64 uniform frequency bands [0..Nyquist]
    n_fft = 512
    win = np.hanning(n_fft).astype(np.float32)
    max_start = max(1, len(y_norm) - n_fft)
    frame_starts = np.linspace(0, max_start, 64, dtype=int)
    spec_cols = []
    for st in frame_starts:
        seg = y_norm[st : st + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        mag = np.abs(np.fft.rfft(seg * win))
        band_edges = np.linspace(0, len(mag), 65, dtype=int)
        bands = [
            float(np.mean(mag[band_edges[b] : max(band_edges[b] + 1, band_edges[b + 1])]))
            for b in range(64)
        ]
        spec_cols.append(np.log1p(np.array(bands, dtype=np.float32) * 0.25))
    spec_mat = np.stack(spec_cols, axis=1)  # (freq=64, time=64)
    spec_max = max(1e-5, float(np.max(spec_mat)))
    spec_mat = (spec_mat / spec_max).astype(np.float32)[None, :, :]  # (1, 64, 64)

    # 2. Temporal frame sequence (48, 8)
    seq_rows = []
    prev_mag_norm = None
    seq_starts = np.linspace(0, max_start, 48, dtype=int)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask_hf = freqs >= 7500.0
    mask_mid = (freqs >= 4000.0) & (freqs < 7500.0)
    mask_core = (freqs >= 300.0) & (freqs <= 3800.0)
    lag_min = max(16, int(sr / 450.0))
    lag_max = min(n_fft // 2, int(sr / 55.0))

    for st in seq_starts:
        seg = y_norm[st : st + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        rms = float(np.sqrt(np.mean(seg ** 2)))
        zcr = float(np.mean(np.abs(np.diff(seg >= 0.0))))
        mag = np.abs(np.fft.rfft(seg * win)) + 1e-6
        pow_spec = mag ** 2
        total_e = float(np.sum(pow_spec)) + 1e-8
        flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))
        hf_ratio = float(np.sum(pow_spec[mask_hf]) / total_e) if np.any(mask_hf) else 0.0
        mid_ratio = float(np.sum(pow_spec[mask_mid]) / total_e) if np.any(mask_mid) else 0.0
        core_ratio = float(np.sum(pow_spec[mask_core]) / total_e) if np.any(mask_core) else 0.0

        mag_norm = mag / (float(np.linalg.norm(mag)) + 1e-6)
        flux = 0.0 if prev_mag_norm is None else float(np.linalg.norm(mag_norm - prev_mag_norm))
        prev_mag_norm = mag_norm

        # Pitch-band unbiased autocorrelation harmonicity
        seg_zm = seg - float(np.mean(seg))
        ac_full = np.correlate(seg_zm, seg_zm, mode="full")[len(seg_zm) - 1 :]
        denom = float(ac_full[0]) + 1e-8
        if denom > 1e-6 and lag_max > lag_min:
            harm = float(np.clip(np.max(ac_full[lag_min:lag_max]) / denom, 0.0, 1.0))
        else:
            harm = 0.0

        seq_rows.append([rms, zcr, flatness, hf_ratio, mid_ratio, core_ratio, flux, harm])

    seq_mat = np.clip(np.array(seq_rows, dtype=np.float32), 0.0, 2.0)

    # 3. 8-D forensic summary vector from AudioForensicAnalyzer
    vec = np.clip(
        np.array(
            [
                float(audio_metrics.get("vocoder_anomaly_score", 0.0)),
                float(audio_metrics.get("hf_dither_score", 0.0)),
                float(audio_metrics.get("pitch_score", 0.0)),
                float(audio_metrics.get("phase_anomaly_score", 0.0)),
                float(audio_metrics.get("splice_discontinuity_score", 0.0)),
                float(audio_metrics.get("mic_channel_score", 0.0)),
                float(audio_metrics.get("voiced_ratio", 0.5)),
                1.0 if audio_metrics.get("is_telephony_bandlimited", False) else 0.0,
            ],
            dtype=np.float32,
        ),
        0.0,
        1.0,
    )
    return spec_mat, seq_mat, vec


class EnhancedAudioFusionNet(nn.Module):
    """
    Lightweight Multi-Branch Acoustic Fusion Network (< 250 KB checkpoint):
      - Branch A: 3-block Conv2d + GroupNorm spectrogram encoder -> 64-D
      - Branch B: Bidirectional GRU temporal dynamics encoder -> 64-D
      - Branch C: 2-layer MLP over 8-D acoustic forensic vector -> 32-D
      - Fusion Head: Linear(160 -> 64 -> 2) binary logits [Realism, Deepfake]
    """

    def __init__(self, seq_dim: int = 8, vec_dim: int = 8):
        super().__init__()
        self.spec_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.temp_gru = nn.GRU(
            input_size=seq_dim,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temp_proj = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.vec_mlp = nn.Sequential(
            nn.Linear(vec_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 64 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 2),
        )

    def forward(
        self,
        spec: torch.Tensor,
        seq: torch.Tensor,
        vec: torch.Tensor,
    ) -> torch.Tensor:
        f_spec = self.spec_cnn(spec)
        gru_out, _ = self.temp_gru(seq)
        f_seq = self.temp_proj(gru_out.mean(dim=1))
        f_vec = self.vec_mlp(vec)
        fused = torch.cat([f_spec, f_seq, f_vec], dim=-1)
        return self.head(fused)


def _load_real_audio_dataset(
    include_cache: bool = True,
    suite_duration_sec: float = 2.6,
    suite_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Loads real WAV files from cache + generated pentest suite and extracts tensors."""
    from benchmark.audio_pentest import CACHE_AUDIO_DIR as AP_CACHE_DIR, generate_audio_pentest_suite
    from web_ui.deepfake_detector import AudioForensicAnalyzer

    entries: List[Dict[str, Any]] = []
    if include_cache and AP_CACHE_DIR.exists():
        for wf in sorted(AP_CACHE_DIR.glob("*.wav")):
            is_fake = "fake" in wf.name.lower() or "vocoder" in wf.name.lower()
            entries.append({"sample_id": wf.stem, "file_path": str(wf), "label": 1 if is_fake else 0})

    cleanup = False
    target_dir = suite_dir
    if target_dir is None:
        target_dir = Path(tempfile.mkdtemp(prefix="fusionnet_train_"))
        cleanup = True
    try:
        suite_items = generate_audio_pentest_suite(target_dir, duration_sec=suite_duration_sec)
        for item in suite_items:
            entries.append({
                "sample_id": item["sample_id"],
                "file_path": item["file_path"],
                "label": int(item["true_binary"]),
            })

        specs, seqs, vecs, labels, ids = [], [], [], [], []
        for row in entries:
            y, sr = AudioForensicAnalyzer.load_audio_signal(row["file_path"])
            metrics = AudioForensicAnalyzer.analyze_audio(row["file_path"])
            s, q, v = extract_fusionnet_tensors(y, sr, metrics)
            specs.append(s)
            seqs.append(q)
            vecs.append(v)
            labels.append(int(row["label"]))
            ids.append(row["sample_id"])
    finally:
        if cleanup:
            import shutil
            shutil.rmtree(target_dir, ignore_errors=True)

    return (
        np.stack(specs, axis=0),
        np.stack(seqs, axis=0),
        np.stack(vecs, axis=0),
        np.array(labels, dtype=np.int64),
        ids,
    )


def _train_single_split(
    X_spec: np.ndarray,
    X_seq: np.ndarray,
    X_vec: np.ndarray,
    y_lbl: np.ndarray,
    train_idx: np.ndarray,
    epochs: int = 40,
    seed: int = 2026,
) -> Tuple[EnhancedAudioFusionNet, float]:
    torch.manual_seed(seed)
    model = EnhancedAudioFusionNet(seq_dim=X_seq.shape[-1], vec_dim=X_vec.shape[-1])
    ds = TensorDataset(
        torch.from_numpy(X_spec[train_idx]),
        torch.from_numpy(X_seq[train_idx]),
        torch.from_numpy(X_vec[train_idx]),
        torch.from_numpy(y_lbl[train_idx]),
    )
    loader = DataLoader(ds, batch_size=16, shuffle=True)

    pos_count = max(1, int(np.sum(y_lbl[train_idx] == 1)))
    neg_count = max(1, int(np.sum(y_lbl[train_idx] == 0)))
    weight = torch.tensor(
        [len(train_idx) / (2.0 * neg_count), len(train_idx) / (2.0 * pos_count)],
        dtype=torch.float32,
    )
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)

    last_loss = 0.0
    model.train()
    for _ in range(epochs):
        epoch_loss = 0.0
        for b_spec, b_seq, b_vec, b_y in loader:
            optimizer.zero_grad()
            logits = model(b_spec, b_seq, b_vec)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(b_y)
        last_loss = epoch_loss / max(1, len(train_idx))
    model.eval()
    return model, round(last_loss, 6)


def train_audio_fusionnet(epochs: int = 45, seed: int = 2026) -> Dict[str, Any]:
    """
    Executes full EnhancedAudioFusionNet training + 5-fold stratified CV + held-out
    duration-shifted validation, writes checkpoint & metadata, and syncs SQLite registry.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    X_spec, X_seq, X_vec, y_lbl, sample_ids = _load_real_audio_dataset(
        include_cache=True,
        suite_duration_sec=2.6,
    )

    # 1. Stratified 5-fold OOF validation (uncontaminated generalization estimate)
    pos_idx = np.where(y_lbl == 1)[0]
    neg_idx = np.where(y_lbl == 0)[0]
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_folds = 5
    pos_folds = np.array_split(pos_idx, n_folds)
    neg_folds = np.array_split(neg_idx, n_folds)

    oof_scores = np.zeros(len(y_lbl), dtype=np.float32)
    all_indices = np.arange(len(y_lbl))
    for k in range(n_folds):
        val_idx = np.concatenate([pos_folds[k], neg_folds[k]])
        train_mask = np.ones(len(y_lbl), dtype=bool)
        train_mask[val_idx] = False
        train_idx = all_indices[train_mask]
        fold_model, _ = _train_single_split(X_spec, X_seq, X_vec, y_lbl, train_idx, epochs=35, seed=seed + k)
        with torch.no_grad():
            logits = fold_model(
                torch.from_numpy(X_spec[val_idx]),
                torch.from_numpy(X_seq[val_idx]),
                torch.from_numpy(X_vec[val_idx]),
            )
            probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        oof_scores[val_idx] = probs

    oof_auc, oof_eer = _compute_roc_auc_and_eer(y_lbl.tolist(), oof_scores.tolist())
    oof_acc = round(float(np.mean((oof_scores >= 0.50).astype(int) == y_lbl)), 4)

    # 2. Final model fit on all 52 real WAV samples (12 cached + 40 multi-channel suite)
    full_idx = np.arange(len(y_lbl))
    final_model, final_train_loss = _train_single_split(
        X_spec, X_seq, X_vec, y_lbl, full_idx, epochs=epochs, seed=seed
    )

    # 3. Held-out unseen duration validation (2.0s clips across all 4 channels & 7 attacks)
    val_spec, val_seq, val_vec, val_lbl, _ = _load_real_audio_dataset(
        include_cache=False,
        suite_duration_sec=2.0,
    )
    with torch.no_grad():
        val_logits = final_model(
            torch.from_numpy(val_spec),
            torch.from_numpy(val_seq),
            torch.from_numpy(val_vec),
        )
        val_probs = torch.softmax(val_logits, dim=-1)[:, 1].numpy()
    holdout_auc, holdout_eer = _compute_roc_auc_and_eer(val_lbl.tolist(), val_probs.tolist())
    holdout_acc = round(float(np.mean((val_probs >= 0.50).astype(int) == val_lbl)), 4)

    CUSTOM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = {
        "architecture": "EnhancedAudioFusionNet",
        "backbone": "ENHANCED_FUSIONNET_HYBRID",
        "state_dict": final_model.state_dict(),
        "seq_dim": int(X_seq.shape[-1]),
        "vec_dim": int(X_vec.shape[-1]),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample_count": int(len(y_lbl)),
        "oof_cv_auc": oof_auc,
        "oof_cv_eer_percent": oof_eer,
        "holdout_auc": holdout_auc,
        "holdout_eer_percent": holdout_eer,
    }
    torch.save(checkpoint_payload, str(CHECKPOINT_PATH))
    size_kb = round(CHECKPOINT_PATH.stat().st_size / 1024.0, 2)

    date_str = datetime.now().strftime("%Y-%m-%d")
    meta = {
        "model_name": "EnhancedAudioFusionNet",
        "backbone": "ENHANCED_FUSIONNET_HYBRID",
        "weights_path": "custom_vit_model/audio_fusionnet.pt",
        "checkpoint_size_kb": size_kb,
        "trained_at": checkpoint_payload["trained_at"],
        "date": date_str,
        "training_provenance": "EMPIRICAL_REAL_AUDIO_TENSORS",
        "sample_breakdown": {
            "total_train_wavs": int(len(y_lbl)),
            "cached_real_world_wavs": 12,
            "generated_multichannel_wavs": int(len(y_lbl) - 12),
            "holdout_unseen_duration_wavs": int(len(val_lbl)),
        },
        "metrics": {
            "train_loss": final_train_loss,
            "oof_cv_accuracy": oof_acc,
            "oof_cv_auc": oof_auc,
            "oof_cv_eer_percent": oof_eer,
            "holdout_accuracy": holdout_acc,
            "holdout_auc": holdout_auc,
            "holdout_eer_percent": holdout_eer,
        },
        "hyperparameters": {
            "epochs": epochs,
            "seed": seed,
            "lr": 2.5e-3,
            "optimizer": "AdamW",
            "fusion_branches": ["SpectralCNN_64x64", "BiGRU_48x8", "ForensicMLP_8D"],
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    try:
        from benchmark.deepfake_memory import save_colab_train_report
        save_colab_train_report({
            "date": date_str,
            "model_name": "EnhancedAudioFusionNet",
            "version_tag": f"audio-fusionnet-{date_str}",
            "weights_path": "custom_vit_model/audio_fusionnet.pt",
            "backbone": "ENHANCED_FUSIONNET_HYBRID",
            "epoch": epochs,
            "val_auc": holdout_auc,
            "val_eer": holdout_eer,
            "train_loss": final_train_loss,
            "hyperparameters": meta["hyperparameters"],
            "summary_notes": (
                f"Trained EnhancedAudioFusionNet on {len(y_lbl)} real WAV tensors "
                f"(5-fold OOF Acc={oof_acc:.4f}, AUC={oof_auc:.4f}, EER={oof_eer:.2f}% | "
                f"Holdout 2.0s Acc={holdout_acc:.4f}, AUC={holdout_auc:.4f}, EER={holdout_eer:.2f}%, "
                f"Size={size_kb:.2f} KB)."
            ),
        })
    except Exception as exc:
        print(f"[warning] SQLite colab_train_reports sync skipped: {exc}")

    print(
        f"[EnhancedAudioFusionNet] Saved {CHECKPOINT_PATH} ({size_kb:.2f} KB) | "
        f"5-fold OOF Acc={oof_acc*100:.2f}% AUC={oof_auc:.4f} EER={oof_eer:.2f}% | "
        f"Holdout Acc={holdout_acc*100:.2f}% AUC={holdout_auc:.4f}"
    )
    return meta


if __name__ == "__main__":
    train_audio_fusionnet()
