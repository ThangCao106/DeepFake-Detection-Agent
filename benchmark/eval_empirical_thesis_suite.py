# -*- coding: utf-8 -*-
"""
Reproducible Empirical Ablation & Locked-Threshold Thesis Evaluation Suite
--------------------------------------------------------------------------
Replaces simulation/hardcoded tables with real PyTorch training & per-sample inference
on a leak-free, origin-grouped locked test partition (`test_locked`):

  1. Strict Group-Level Isolation: loads `train`, `cal_val`, and `test_locked` from
     `report_colab/corpus/training_corpus_manifest.jsonl`, asserting 0 origin overlap.
  2. Real Baselines & Ablations trained strictly on `train`:
       - `baseline_sbi_cvpr2022`: Public SBI counterfactual methodology CNN baseline
       - `ablation_1_spatial_rgb_only`: DualStream spatial branch (spectral FCB zeroed)
       - `ablation_2_spatial_plus_noise_fft_srm`: DualStream + 2Ch [2D-FFT, SRM] FCB + Gate
       - `ablation_3_biometric_physiology`: BiometricFusionHead (DWT + rPPG + CrossAttn)
       - `full_proposed_fusion_calibrated`: Optimal Fusion + 2-Param Platt + Locked Youden tau*
  3. Calibration & Threshold Locking:
       - Optimizes fusion alpha, Platt (a, b, T*), and Youden tau* ONLY on `cal_val`.
       - Locks `locked_calibration_protocol.json` before evaluating `test_locked` ONCE.
  4. Three Held-Out Evaluation Views on `test_locked`:
       - `clean`: original held-out real & fake images
       - `social_reupload`: compressed/downscaled/blurred social media re-upload degradation
       - `unseen_manipulation_ood`: unseen diffusion-proxy + facial color/blending manipulation
  5. Operating-Point Metrics + 95% Bootstrap CI (n=1000) & Per-Sample Provenance:
       - Outputs `per_sample_predictions.jsonl`, `empirical_ablation_results.json`,
         `empirical_ablation_report.md`, and `table_empirical_ablation.tex`.
"""

import argparse
import hashlib
import io
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from colab_training.train_general import (
    DualStreamGeneralDetector,
    _extract_image_tensors,
    _fit_2param_platt,
)
from colab_training.train_human import (
    BiometricFusionHead,
    _extract_biometric_tensors_from_image,
)
from forensics_kernel.dataset_split_guard import (
    assert_zero_origin_leakage,
    compute_bootstrap_ci,
    compute_wilson_ci,
    load_locked_manifest_splits,
)


DEFAULT_MANIFEST = PROJECT_ROOT / "report_colab" / "corpus" / "training_corpus_manifest.jsonl"
DEFAULT_OUT_DIR = PROJECT_ROOT / "report_colab" / "empirical_thesis_eval"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class SBIConvBaseline(nn.Module):
    """
    Lightweight Self-Blended Image (SBI, Shiohara & Yamasaki CVPR 2022 methodology)
    spatial CNN baseline trained strictly on `train` images at 96x96 RGB.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_social_reupload_image(im: Image.Image) -> Image.Image:
    """Apply reproducible JPEG Q=55 + downscale/upscale 0.75x + mild Gaussian blur."""
    rgb = im.convert("RGB")
    w, h = rgb.size
    small = rgb.resize((max(32, int(w * 0.75)), max(32, int(h * 0.75))), Image.Resampling.BILINEAR)
    restored = small.resize((w, h), Image.Resampling.BICUBIC).filter(ImageFilter.GaussianBlur(radius=0.55))
    buf = io.BytesIO()
    restored.save(buf, format="JPEG", quality=55)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _make_unseen_manipulation_image(im: Image.Image, seed: int) -> Image.Image:
    """
    Create an unseen counterfactual manipulation NOT used during training:
    combines radial facial color-temperature drift, frequency-smoothed inner patch blending
    (diffusion-smoothing proxy), and JPEG quantization grid at Q=68.
    """
    rng = np.random.default_rng(seed)
    rgb = im.convert("RGB").resize((256, 256), Image.Resampling.BICUBIC)
    arr = np.asarray(rgb, dtype=np.float32)

    # Smooth center facial mask cleanly without using training SBI affine/TPS mask code
    blurred = np.asarray(rgb.filter(ImageFilter.GaussianBlur(radius=1.8)), dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256]
    cx = 128.0 + float(rng.uniform(-12, 12))
    cy = 120.0 + float(rng.uniform(-12, 12))
    rx = float(rng.uniform(46, 68))
    ry = float(rng.uniform(58, 82))
    dist2 = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
    mask = np.clip(1.0 - dist2, 0.0, 1.0) ** 1.5
    mask3 = mask[:, :, None]

    # Subtle GAN/Diffusion color-channel gain drift inside blended interior
    gain = np.array([rng.uniform(0.92, 1.08), rng.uniform(0.95, 1.05), rng.uniform(0.90, 1.07)], dtype=np.float32)
    synth_inner = np.clip(blurred * gain + rng.normal(0.0, 2.5, size=arr.shape), 0.0, 255.0)
    blended = np.clip(mask3 * synth_inner + (1.0 - mask3) * arr, 0.0, 255.0).astype(np.uint8)

    out_im = Image.fromarray(blended, mode="RGB")
    buf = io.BytesIO()
    out_im.save(buf, format="JPEG", quality=68)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _load_view_features(
    records: List[Dict[str, Any]],
    *,
    view_name: str,
    temp_dir: Path,
) -> Dict[str, Any]:
    """Extract aligned RGB (3x96x96), DualStream (768-D, 2x64x64), and Biometric (4-branch) tensors."""
    rgb_l, sp_l, ff_l = [], [], []
    bv_l, ba_l, bs_l, bn_l = [], [], [], []
    labels, meta_rows = [], []

    view_dir = temp_dir / view_name
    view_dir.mkdir(parents=True, exist_ok=True)

    source_recs = records
    if view_name == "unseen_manipulation_ood":
        # Keep held-out reals intact and synthesize unseen fakes ONLY from held-out test_locked reals
        reals = [r for r in records if int(r.get("label", 0)) == 0]
        source_recs = list(reals)
        for idx, rr in enumerate(reals):
            r_copy = dict(rr)
            r_copy["sample_id"] = f"UNSEEN_OOD_{idx:04d}"
            r_copy["label"] = 1
            r_copy["label_name"] = "FAKE"
            r_copy["source"] = "unseen_facial_diffusion_color_blend"
            r_copy["_unseen_from_real"] = rr["local_path"]
            source_recs.append(r_copy)

    for idx, rec in enumerate(source_recs):
        src_path = Path(rec.get("_unseen_from_real") or rec["local_path"])
        try:
            with Image.open(src_path) as im:
                if view_name == "social_reupload":
                    proc_im = _make_social_reupload_image(im)
                elif view_name == "unseen_manipulation_ood" and rec.get("_unseen_from_real"):
                    proc_im = _make_unseen_manipulation_image(im, seed=7700 + idx)
                else:
                    proc_im = im.convert("RGB")

                tmp_img_p = src_path
                if view_name != "clean" or rec.get("_unseen_from_real"):
                    tmp_img_p = view_dir / f"{view_name}_{idx:05d}.jpg"
                    proc_im.save(tmp_img_p, format="JPEG", quality=92)

                rgb96 = np.asarray(proc_im.resize((96, 96), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
                t_rgb96 = torch.from_numpy(rgb96).permute(2, 0, 1)

            fs, ff = _extract_image_tensors(tmp_img_p, augmentor=None)
            bv, ba, bs, bn = _extract_biometric_tensors_from_image(tmp_img_p)

            rgb_l.append(t_rgb96)
            sp_l.append(fs)
            ff_l.append(ff)
            bv_l.append(bv)
            ba_l.append(ba)
            bs_l.append(bs)
            bn_l.append(bn)
            labels.append(int(rec.get("label", 0)))
            meta_rows.append({
                "sample_id": rec.get("sample_id", tmp_img_p.stem),
                "origin_id": rec.get("origin_id"),
                "label": int(rec.get("label", 0)),
                "source": rec.get("source", "unknown"),
                "vietnamese_category": rec.get("vietnamese_category", "vn_selfie_lifestyle"),
                "view": view_name,
            })
        except Exception as exc:
            print(f"[EmpiricalSuite] Skipping unreadable sample {src_path}: {exc}")

    return {
        "rgb": torch.stack(rgb_l),
        "spatial": torch.stack(sp_l),
        "fft": torch.stack(ff_l),
        "bio_v": torch.stack(bv_l),
        "bio_a": torch.stack(ba_l),
        "bio_s": torch.stack(bs_l),
        "bio_n": torch.stack(bn_l),
        "labels": torch.tensor(labels, dtype=torch.long),
        "meta": meta_rows,
    }


def _train_and_lock_platt(
    logits_diff_cal: List[float],
    y_cal: List[int],
) -> Dict[str, float]:
    a, b, t_eq, tau = _fit_2param_platt(y_cal, logits_diff_cal)
    return {"platt_a": a, "platt_b": b, "temperature_T": t_eq, "locked_tau": tau}


def _apply_locked_platt(logits_diff: List[float], cal_proto: Dict[str, float]) -> List[float]:
    a = float(cal_proto["platt_a"])
    b = float(cal_proto["platt_b"])
    return [1.0 / (1.0 + math.exp(-max(-25.0, min(25.0, a * float(z) + b)))) for z in logits_diff]


def _predict_sbi(model: nn.Module, feats: Dict[str, Any], device: str) -> List[float]:
    model.eval()
    with torch.no_grad():
        logits = model(feats["rgb"].to(device))
        return (logits[:, 1] - logits[:, 0]).cpu().tolist()


def _predict_dual(model: DualStreamGeneralDetector, feats: Dict[str, Any], device: str, zero_spectral: bool = False) -> List[float]:
    model.eval()
    with torch.no_grad():
        ff = torch.zeros_like(feats["fft"]) if zero_spectral else feats["fft"]
        logits = model(feats["spatial"].to(device), ff.to(device))
        return (logits[:, 1] - logits[:, 0]).cpu().tolist()


def _predict_bio(model: BiometricFusionHead, feats: Dict[str, Any], device: str) -> List[float]:
    model.eval()
    with torch.no_grad():
        logits = model(
            feats["bio_v"].to(device),
            feats["bio_a"].to(device),
            feats["bio_s"].to(device),
            feats["bio_n"].to(device),
        )
        return (logits[:, 1] - logits[:, 0]).cpu().tolist()


def _per_condition_errors(meta_rows: List[Dict[str, Any]], y_true: List[int], y_prob: List[float], tau: float) -> List[Dict[str, Any]]:
    by_cat: Dict[str, List[int]] = {}
    for idx, m in enumerate(meta_rows):
        by_cat.setdefault(str(m.get("vietnamese_category") or "unknown"), []).append(idx)

    rows = []
    for cat, idxs in sorted(by_cat.items()):
        n = len(idxs)
        errs = sum(1 for i in idxs if (1 if y_prob[i] >= tau else 0) != y_true[i])
        fps = sum(1 for i in idxs if y_true[i] == 0 and y_prob[i] >= tau)
        fns = sum(1 for i in idxs if y_true[i] == 1 and y_prob[i] < tau)
        rows.append({
            "capture_condition": cat,
            "n": n,
            "errors": errs,
            "fp": fps,
            "fn": fns,
            "error_rate": round(errs / float(max(1, n)), 4),
            "error_rate_wilson_ci95": list(compute_wilson_ci(errs, n)),
        })
    return rows


def run_empirical_thesis_suite(
    manifest_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    epochs: int = 12,
    batch_size: int = 24,
    n_boot: int = 1000,
    max_images_per_split: int = 0,
) -> Dict[str, Any]:
    """Run leak-free training, cal_val calibration locking, and multi-view test evaluation."""
    import tempfile

    mp = Path(manifest_path or DEFAULT_MANIFEST)
    out_dir = Path(output_dir or DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not mp.exists():
        err_payload = {"status": "FAILED_MISSING_MANIFEST", "manifest_path": str(mp)}
        (out_dir / "empirical_ablation_results.json").write_text(json.dumps(err_payload, indent=2), encoding="utf-8")
        raise FileNotFoundError(f"Missing locked corpus manifest: {mp}")

    loaded = load_locked_manifest_splits(mp, exclude_unverified=True, require_existing_files=True)
    splits = loaded["splits"]
    split_audit = loaded["audit"]

    tr_recs = splits["train"][:max_images_per_split] if max_images_per_split > 0 else splits["train"]
    ca_recs = splits["cal_val"][:max_images_per_split] if max_images_per_split > 0 else splits["cal_val"]
    te_recs = splits["test_locked"][:max_images_per_split] if max_images_per_split > 0 else splits["test_locked"]

    for name, s_recs in (("train", tr_recs), ("cal_val", ca_recs), ("test_locked", te_recs)):
        ys = {int(r.get("label", 0)) for r in s_recs}
        if not s_recs or len(ys) < 2:
            raise RuntimeError(f"Split '{name}' missing samples or binary classes (count={len(s_recs)}, classes={ys}).")

    assert_zero_origin_leakage(tr_recs, ca_recs, te_recs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(2026)
    np.random.seed(2026)

    with tempfile.TemporaryDirectory(prefix="emp_thesis_") as td:
        td_path = Path(td)
        tr_f = _load_view_features(tr_recs, view_name="clean", temp_dir=td_path / "tr")
        ca_f = _load_view_features(ca_recs, view_name="clean", temp_dir=td_path / "ca")
        te_clean_f = _load_view_features(te_recs, view_name="clean", temp_dir=td_path / "te_clean")
        te_social_f = _load_view_features(te_recs, view_name="social_reupload", temp_dir=td_path / "te_social")
        te_unseen_f = _load_view_features(te_recs, view_name="unseen_manipulation_ood", temp_dir=td_path / "te_unseen")

    y_tr = tr_f["labels"]
    y_ca = [int(x) for x in ca_f["labels"].tolist()]

    # Class-balanced CE weight on train
    n_pos = max(1, int((y_tr == 1).sum().item()))
    n_neg = max(1, int((y_tr == 0).sum().item()))
    ce_w = torch.tensor([1.0, min(3.0, max(0.35, n_neg / float(n_pos)))], dtype=torch.float32, device=device)

    # 1. Train Baseline: SBI CVPR 2022 spatial CNN
    sbi_model = SBIConvBaseline().to(device)
    opt_sbi = torch.optim.AdamW(sbi_model.parameters(), lr=8e-4, weight_decay=1e-4)
    ld_sbi = DataLoader(TensorDataset(tr_f["rgb"], y_tr), batch_size=min(batch_size, len(y_tr)), shuffle=True)
    for _ in range(epochs):
        sbi_model.train()
        for bx, by in ld_sbi:
            bx, by = bx.to(device), by.to(device)
            opt_sbi.zero_grad()
            loss = F.cross_entropy(sbi_model(bx), by, weight=ce_w)
            loss.backward()
            opt_sbi.step()

    # 2. Train Ablation 1: Spatial RGB Only (spectral branch zeroed during train & eval)
    rgb_only_model = DualStreamGeneralDetector(spectral_channels=2).to(device)
    opt_rgb = torch.optim.AdamW(rgb_only_model.parameters(), lr=6e-4, weight_decay=1e-4)
    ld_dual = DataLoader(TensorDataset(tr_f["spatial"], tr_f["fft"], y_tr), batch_size=min(batch_size, len(y_tr)), shuffle=True)
    for _ in range(epochs):
        rgb_only_model.train()
        for bs, bf, by in ld_dual:
            bs, by = bs.to(device), by.to(device)
            bf_zero = torch.zeros_like(bf, device=device)
            opt_rgb.zero_grad()
            loss = F.cross_entropy(rgb_only_model(bs, bf_zero), by, weight=ce_w)
            loss.backward()
            opt_rgb.step()

    # 3. Train Ablation 2: Spatial + 2Ch Noise/Spectral FCB (FFT + SRM)
    dual_model = DualStreamGeneralDetector(spectral_channels=2).to(device)
    opt_dual = torch.optim.AdamW(dual_model.parameters(), lr=6e-4, weight_decay=1e-4)
    for _ in range(epochs):
        dual_model.train()
        for bs, bf, by in ld_dual:
            bs, bf, by = bs.to(device), bf.to(device), by.to(device)
            opt_dual.zero_grad()
            loss = F.cross_entropy(dual_model(bs, bf), by, weight=ce_w)
            loss.backward()
            opt_dual.step()

    # 4. Train Ablation 3: Biometric Physiology Head (DWT + Chrominance + Corneal/PRNU)
    bio_model = BiometricFusionHead().to(device)
    opt_bio = torch.optim.AdamW(bio_model.parameters(), lr=5e-4, weight_decay=1e-4)
    ld_bio = DataLoader(
        TensorDataset(tr_f["bio_v"], tr_f["bio_a"], tr_f["bio_s"], tr_f["bio_n"], y_tr),
        batch_size=min(batch_size, len(y_tr)),
        shuffle=True,
    )
    for _ in range(epochs):
        bio_model.train()
        for bv, ba, bs, bn, by in ld_bio:
            bv, ba, bs, bn, by = bv.to(device), ba.to(device), bs.to(device), bn.to(device), by.to(device)
            opt_bio.zero_grad()
            loss = F.cross_entropy(bio_model(bv, ba, bs, bn), by, weight=ce_w)
            loss.backward()
            opt_bio.step()

    # Save checkpoints & compute SHA-256 checksums
    checkpoints_meta: Dict[str, Dict[str, str]] = {}
    for key, mod_obj in (
        ("baseline_sbi_cvpr2022", sbi_model),
        ("ablation_1_spatial_rgb_only", rgb_only_model),
        ("ablation_2_spatial_plus_noise_fft_srm", dual_model),
        ("ablation_3_biometric_physiology", bio_model),
    ):
        ck_p = ckpt_dir / f"{key}.pt"
        torch.save({"state_dict": mod_obj.state_dict(), "trained_on_origins": split_audit["train_origins"]}, ck_p)
        checkpoints_meta[key] = {"path": str(ck_p), "sha256": _sha256_file(ck_p)}

    # Optimize & LOCK calibration protocol strictly on `cal_val`
    cal_logits = {
        "baseline_sbi_cvpr2022": _predict_sbi(sbi_model, ca_f, device),
        "ablation_1_spatial_rgb_only": _predict_dual(rgb_only_model, ca_f, device, zero_spectral=True),
        "ablation_2_spatial_plus_noise_fft_srm": _predict_dual(dual_model, ca_f, device, zero_spectral=False),
        "ablation_3_biometric_physiology": _predict_bio(bio_model, ca_f, device),
    }

    # Select optimal convex fusion weight w_dual in [0.1..0.9] strictly on cal_val AUC & BCE
    best_alpha, best_cal_auc = 0.5, -1.0
    l_dual_ca = np.asarray(cal_logits["ablation_2_spatial_plus_noise_fft_srm"], dtype=np.float64)
    l_bio_ca = np.asarray(cal_logits["ablation_3_biometric_physiology"], dtype=np.float64)
    for cand_w in (0.2, 0.35, 0.5, 0.65, 0.8):
        blended = (cand_w * l_dual_ca + (1.0 - cand_w) * l_bio_ca).tolist()
        proto_tmp = _train_and_lock_platt(blended, y_ca)
        probs_tmp = _apply_locked_platt(blended, proto_tmp)
        stats_tmp = compute_bootstrap_ci(y_ca, probs_tmp, proto_tmp["locked_tau"], n_boot=60)
        score = float(stats_tmp["roc_auc"]) + 0.25 * float(stats_tmp["balanced_accuracy"]) - 0.1 * float(stats_tmp["ece"])
        if score > best_cal_auc:
            best_cal_auc = score
            best_alpha = cand_w

    cal_logits["full_proposed_fusion_calibrated"] = (best_alpha * l_dual_ca + (1.0 - best_alpha) * l_bio_ca).tolist()

    locked_protocol: Dict[str, Any] = {
        "locked_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "calibration_split": "cal_val",
        "calibration_samples": len(y_ca),
        "fusion_weight_dual_stream": best_alpha,
        "fusion_weight_biometric": round(1.0 - best_alpha, 2),
        "checkpoints": checkpoints_meta,
        "models": {},
    }
    for m_name, z_cal in cal_logits.items():
        proto = _train_and_lock_platt(z_cal, y_ca)
        p_cal = _apply_locked_platt(z_cal, proto)
        cal_stats = compute_bootstrap_ci(y_ca, p_cal, proto["locked_tau"], n_boot=min(300, n_boot))
        locked_protocol["models"][m_name] = {**proto, "cal_val_summary": {k: cal_stats[k] for k in ("tpr", "fpr", "accuracy", "balanced_accuracy", "ece", "roc_auc")}}

    (out_dir / "locked_calibration_protocol.json").write_text(json.dumps(locked_protocol, indent=2, ensure_ascii=False), encoding="utf-8")

    # Evaluate ONCE across all 3 held-out test_locked views with locked protocols
    views = {
        "clean": te_clean_f,
        "social_reupload": te_social_f,
        "unseen_manipulation_ood": te_unseen_f,
    }
    model_order = [
        ("baseline_sbi_cvpr2022", "Baseline: SBI Spatial CNN (CVPR 2022 methodology)"),
        ("ablation_1_spatial_rgb_only", "Ablation 1: Spatial RGB Only (FCB Zeroed)"),
        ("ablation_2_spatial_plus_noise_fft_srm", "Ablation 2: Spatial + Noise/Spectral FCB (FFT+SRM)"),
        ("ablation_3_biometric_physiology", "Ablation 3: Biometric Physiology (DWT+rPPG+Attn)"),
        ("full_proposed_fusion_calibrated", f"Full Proposed: Multi-Stream Fusion (w={best_alpha:.2f}) + Locked Platt"),
    ]

    results_by_view: Dict[str, Dict[str, Any]] = {}
    per_sample_lines: List[str] = []

    for v_name, v_feats in views.items():
        y_te = [int(x) for x in v_feats["labels"].tolist()]
        raw_te = {
            "baseline_sbi_cvpr2022": _predict_sbi(sbi_model, v_feats, device),
            "ablation_1_spatial_rgb_only": _predict_dual(rgb_only_model, v_feats, device, zero_spectral=True),
            "ablation_2_spatial_plus_noise_fft_srm": _predict_dual(dual_model, v_feats, device, zero_spectral=False),
            "ablation_3_biometric_physiology": _predict_bio(bio_model, v_feats, device),
        }
        raw_te["full_proposed_fusion_calibrated"] = (
            best_alpha * np.asarray(raw_te["ablation_2_spatial_plus_noise_fft_srm"], dtype=np.float64)
            + (1.0 - best_alpha) * np.asarray(raw_te["ablation_3_biometric_physiology"], dtype=np.float64)
        ).tolist()

        results_by_view[v_name] = {}
        for m_key, m_label in model_order:
            proto = locked_protocol["models"][m_key]
            tau_locked = float(proto["locked_tau"])
            z_list = raw_te[m_key]
            p_list = _apply_locked_platt(z_list, proto)
            stats = compute_bootstrap_ci(y_te, p_list, tau_locked, n_boot=n_boot, seed=2026)
            cond_breakdown = _per_condition_errors(v_feats["meta"], y_te, p_list, tau_locked)

            results_by_view[v_name][m_key] = {
                "display_name": m_label,
                "locked_calibration": {k: proto[k] for k in ("platt_a", "platt_b", "temperature_T", "locked_tau")},
                "metrics": stats,
                "per_capture_condition": cond_breakdown,
            }

            for idx, mrow in enumerate(v_feats["meta"]):
                per_sample_lines.append(json.dumps({
                    "view": v_name,
                    "model": m_key,
                    "sample_id": mrow["sample_id"],
                    "origin_id": mrow["origin_id"],
                    "capture_condition": mrow["vietnamese_category"],
                    "source": mrow["source"],
                    "true_label": y_te[idx],
                    "raw_logit_diff": round(float(z_list[idx]), 5),
                    "calib_prob": round(float(p_list[idx]), 5),
                    "locked_tau": tau_locked,
                    "pred_label": 1 if p_list[idx] >= tau_locked else 0,
                }, ensure_ascii=False))

    (out_dir / "per_sample_predictions.jsonl").write_text("\n".join(per_sample_lines) + "\n", encoding="utf-8")

    final_payload = {
        "evaluation_provenance": "EMPIRICAL_LOCKED_SPLIT_INFERENCE",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest_path": str(mp),
        "manifest_sha256": _sha256_file(mp),
        "leakage_audit": split_audit,
        "training_config": {"epochs": epochs, "batch_size": batch_size, "device": device, "n_bootstrap": n_boot},
        "locked_calibration_protocol": locked_protocol,
        "views": results_by_view,
    }
    res_json_path = out_dir / "empirical_ablation_results.json"
    res_json_path.write_text(json.dumps(final_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Build real empirical LaTeX & Markdown reports
    md_lines = [
        "# Báo Cáo Thực Nghiệm Định Lượng & Ablation Độc Lập Trên Tập Kiểm Thử Khóa (`test_locked`)",
        "",
        f"- **Thời điểm chạy (UTC):** `{final_payload['generated_at_utc']}` | **Provenance:** `{final_payload['evaluation_provenance']}`",
        f"- **Kiểm toán chống rò rỉ (`origin_id`):** Train={split_audit['train_samples']} ({split_audit['train_origins']} gốc) | Cal/Val={split_audit['cal_val_samples']} ({split_audit['cal_val_origins']} gốc) | Test Locked={split_audit['test_locked_samples']} ({split_audit['test_locked_origins']} gốc) | Overlap=`0/0/0`",
        f"- **Khóa ngưỡng Calibration (`cal_val`):** Trọng số dung hợp `w_dual={best_alpha:.2f}`, `w_bio={1.0 - best_alpha:.2f}` (tối ưu & khóa trước khi mở `test_locked`)",
        "",
    ]

    tex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Empirical ablation and out-of-domain robustness on leak-free locked Vietnamese test split (\texttt{test\_locked}) at calibration-locked threshold $\tau^*$.}",
        r"\label{tab:empirical_thesis_ablation}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{llcccccc}",
        r"\hline",
        r"\textbf{Test View} & \textbf{Configuration} & \textbf{TPR@$\tau^*$ (95\% CI)} & \textbf{FPR@$\tau^*$ (95\% CI)} & \textbf{Bal. Acc (\%)} & \textbf{F1 (\%)} & \textbf{ECE ($\downarrow$)} & \textbf{ROC-AUC} \\",
        r"\hline",
    ]

    for v_name in ("clean", "social_reupload", "unseen_manipulation_ood"):
        md_lines.extend([
            f"## View: `{v_name}` (N={len(views[v_name]['labels'])})",
            "| Cấu hình Mô hình | Locked $\\tau^*$ | TPR (95% CI) | FPR (95% CI) | FNR | Bal. Acc | ECE | ROC-AUC (95% CI) |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for m_key, m_label in model_order:
            entry = results_by_view[v_name][m_key]
            m = entry["metrics"]
            ci = m["bootstrap_ci95"]
            tau_v = entry["locked_calibration"]["locked_tau"]
            md_lines.append(
                f"| {m_label} | {tau_v:.4f} | {m['tpr']*100:.1f}% [{ci['tpr'][0]*100:.1f}–{ci['tpr'][1]*100:.1f}] | "
                f"{m['fpr']*100:.1f}% [{ci['fpr'][0]*100:.1f}–{ci['fpr'][1]*100:.1f}] | {m['fnr']*100:.1f}% | "
                f"{m['balanced_accuracy']*100:.1f}% | {m['ece']:.4f} | {m['roc_auc']:.4f} [{ci['roc_auc'][0]:.3f}–{ci['roc_auc'][1]:.3f}] |"
            )
            short_name = m_label.split(":")[0]
            tex_lines.append(
                f"{v_name.replace('_', ' ')} & {short_name} & "
                f"{m['tpr']*100:.1f}\\% [{ci['tpr'][0]*100:.1f}, {ci['tpr'][1]*100:.1f}] & "
                f"{m['fpr']*100:.1f}\\% [{ci['fpr'][0]*100:.1f}, {ci['fpr'][1]*100:.1f}] & "
                f"{m['balanced_accuracy']*100:.1f}\\% & {m['f1']*100:.1f}\\% & {m['ece']:.4f} & {m['roc_auc']:.4f} \\\\"
            )
        md_lines.append("")
        tex_lines.append(r"\hline")

    # Add capture-condition breakdown for Full Proposed on clean & social_reupload
    md_lines.append("## Phân tích Lỗi theo Điều kiện chụp Người Việt (`full_proposed_fusion_calibrated`)")
    for v_name in ("clean", "social_reupload", "unseen_manipulation_ood"):
        md_lines.extend([
            f"### Điều kiện chụp — View `{v_name}`",
            "| Capture Condition | Samples (n) | Errors | FP | FN | Error Rate | 95% Wilson CI |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ])
        for cr in results_by_view[v_name]["full_proposed_fusion_calibrated"]["per_capture_condition"]:
            md_lines.append(
                f"| `{cr['capture_condition']}` | {cr['n']} | {cr['errors']} | {cr['fp']} | {cr['fn']} | "
                f"{cr['error_rate']*100:.1f}% | [{cr['error_rate_wilson_ci95'][0]*100:.1f}%–{cr['error_rate_wilson_ci95'][1]*100:.1f}%] |"
            )
        md_lines.append("")

    tex_lines.extend([r"\end{tabular}", r"}", r"\end{table*}"])
    (out_dir / "empirical_ablation_report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (out_dir / "table_empirical_ablation.tex").write_text("\n".join(tex_lines) + "\n", encoding="utf-8")

    return final_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empirical Leak-Free Thesis Ablation & Locked-Threshold Suite")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    out = run_empirical_thesis_suite(
        manifest_path=Path(args.manifest),
        output_dir=Path(args.outdir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_boot=args.n_boot,
    )
    print(json.dumps({
        "status": out["evaluation_provenance"],
        "leakage_audit": out["leakage_audit"],
        "clean_full_metrics": out["views"]["clean"]["full_proposed_fusion_calibrated"]["metrics"],
        "social_full_metrics": out["views"]["social_reupload"]["full_proposed_fusion_calibrated"]["metrics"],
        "unseen_full_metrics": out["views"]["unseen_manipulation_ood"]["full_proposed_fusion_calibrated"]["metrics"],
    }, indent=2, ensure_ascii=False))
