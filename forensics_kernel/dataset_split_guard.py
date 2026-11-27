# -*- coding: utf-8 -*-
"""
Leak-Free Group-Level Dataset Splitting, Provenance Verification & Statistical CI Utilities.
Pure Python stdlib (100% compatible with `python -S` bare stdlib CI).

Eliminates identity/origin-image data leakage between train, calibration/val (cal_val),
and locked held-out test (test_locked) partitions by grouping all generated variants
(SBI counterfactuals, social augmentations, unseen manipulations) under their canonical
`origin_id` BEFORE partitioning.
"""

import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


UNVERIFIED_VERIFICATION_BASES = frozenset({"UNVERIFIED_SOCIAL", "UNVERIFIED"})


def extract_origin_id(rec_or_path: Any) -> str:
    """
    Resolve canonical origin identity/image key so a source image and ALL synthetic variants
    (SBI counterfactuals, augmentations, crops, unseen fakes) share the exact same `origin_id`.
    """
    if isinstance(rec_or_path, dict):
        explicit = str(
            rec_or_path.get("origin_id")
            or rec_or_path.get("identity_group_id")
            or ""
        ).strip()
        if explicit:
            return explicit
        raw_name = (
            rec_or_path.get("origin_filename")
            or rec_or_path.get("filename")
            or rec_or_path.get("local_path")
            or rec_or_path.get("filepath")
            or rec_or_path.get("path")
            or rec_or_path.get("sample_id")
            or ""
        )
    else:
        raw_name = str(rec_or_path or "")

    stem = Path(str(raw_name)).stem.strip()
    if not stem:
        return "unknown_origin"

    # Strip generated prefixes/suffixes repeatedly to recover source portrait stem
    prev = None
    cur = stem
    while cur != prev:
        prev = cur
        cur = re.sub(r"^(?:sbi|unseen|cf|aug)_(?:val_)?\d*_*", "", cur, flags=re.IGNORECASE)
        cur = re.sub(r"_(?:sbi|unseen|aug|crop|jpeg|reup|blur|mask)[_-]?\d*$", "", cur, flags=re.IGNORECASE)
    return cur or stem


def normalize_split_name(raw_split: Any, default: str = "train") -> str:
    s = str(raw_split or default).strip().lower()
    if s in ("train", "tr"):
        return "train"
    if s in ("cal_val", "val", "val_ood", "calibration", "dev"):
        return "cal_val"
    if s in ("test_locked", "test", "holdout", "held_out", "locked_test"):
        return "test_locked"
    return default


def assign_leak_free_group_splits(
    records: Sequence[Dict[str, Any]],
    train_ratio: float = 0.70,
    cal_ratio: float = 0.15,
    seed: int = 2026,
) -> List[Dict[str, Any]]:
    """
    Group records by `origin_id`, stratify origin groups by (label, category), and assign
    each origin_id atomically to `train`, `cal_val`, or `test_locked`.
    """
    rng = random.Random(int(seed))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        r_copy = dict(rec)
        oid = extract_origin_id(r_copy)
        r_copy["origin_id"] = oid
        groups.setdefault(oid, []).append(r_copy)

    # Stratify groups by primary label + capture category
    strata: Dict[Tuple[int, str], List[str]] = {}
    for oid, grp in sorted(groups.items()):
        lbl = int(grp[0].get("label", 0))
        cat = str(grp[0].get("vietnamese_category") or grp[0].get("cohort") or "default")
        strata.setdefault((lbl, cat), []).append(oid)

    oid_to_split: Dict[str, str] = {}
    for _, oids in sorted(strata.items()):
        oids_shuf = list(oids)
        rng.shuffle(oids_shuf)
        n = len(oids_shuf)
        if n == 1:
            oid_to_split[oids_shuf[0]] = "train"
            continue
        if n == 2:
            oid_to_split[oids_shuf[0]] = "train"
            oid_to_split[oids_shuf[1]] = "cal_val"
            continue

        n_train = max(1, int(math.floor(n * float(train_ratio))))
        n_cal = max(1, int(math.floor(n * float(cal_ratio))))
        n_test = n - n_train - n_cal
        if n_test < 1:
            if n_train > 1:
                n_train -= 1
            elif n_cal > 1:
                n_cal -= 1
            n_test = n - n_train - n_cal

        for idx, oid in enumerate(oids_shuf):
            if idx < n_train:
                oid_to_split[oid] = "train"
            elif idx < n_train + n_cal:
                oid_to_split[oid] = "cal_val"
            else:
                oid_to_split[oid] = "test_locked"

    out: List[Dict[str, Any]] = []
    for rec in records:
        r_copy = dict(rec)
        oid = extract_origin_id(r_copy)
        r_copy["origin_id"] = oid
        r_copy["split"] = oid_to_split.get(oid, "train")
        out.append(r_copy)
    return out


def assert_zero_origin_leakage(
    train_recs: Sequence[Any],
    cal_recs: Sequence[Any],
    test_recs: Optional[Sequence[Any]] = None,
) -> Dict[str, int]:
    """
    Strictly assert pairwise disjointness of `origin_id` sets across splits.
    Raises ValueError immediately if any origin identity or source image leaks across splits.
    """
    train_oids = {extract_origin_id(r) for r in train_recs}
    cal_oids = {extract_origin_id(r) for r in cal_recs}
    test_oids = {extract_origin_id(r) for r in (test_recs or [])}

    ov_tc = sorted(train_oids & cal_oids)
    ov_tt = sorted(train_oids & test_oids)
    ov_ct = sorted(cal_oids & test_oids)

    if ov_tc or ov_tt or ov_ct:
        raise ValueError(
            "DATA LEAKAGE DETECTED across origin_id splits! "
            f"train∩cal={len(ov_tc)} ({ov_tc[:3]}), "
            f"train∩test={len(ov_tt)} ({ov_tt[:3]}), "
            f"cal∩test={len(ov_ct)} ({ov_ct[:3]})"
        )

    return {
        "train_samples": len(train_recs),
        "cal_val_samples": len(cal_recs),
        "test_locked_samples": len(test_recs or []),
        "train_origins": len(train_oids),
        "cal_val_origins": len(cal_oids),
        "test_locked_origins": len(test_oids),
        "train_cal_overlap": 0,
        "train_test_overlap": 0,
        "cal_test_overlap": 0,
    }


def load_locked_manifest_splits(
    manifest_path: Path,
    *,
    exclude_unverified: bool = True,
    require_existing_files: bool = True,
) -> Dict[str, Any]:
    """
    Load manifest (.jsonl/.json), filter unverified social posts if requested, partition into
    `train`, `cal_val`, `test_locked`, and verify 0 origin leakage.
    """
    mp = Path(manifest_path)
    if not mp.exists():
        raise FileNotFoundError(f"Manifest not found: {mp}")

    raw_rows: List[Dict[str, Any]] = []
    if mp.suffix.lower() == ".jsonl":
        for line in mp.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                raw_rows.append(json.loads(s))
    else:
        data = json.loads(mp.read_text(encoding="utf-8"))
        if isinstance(data, list):
            raw_rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            raw_rows = [r for r in (data.get("samples") or data.get("records") or []) if isinstance(r, dict)]

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "cal_val": [], "test_locked": []}
    skipped_unverified = 0
    skipped_missing = 0

    for rec in raw_rows:
        basis = str(rec.get("label_verification_basis") or "").strip().upper()
        if exclude_unverified and basis in UNVERIFIED_VERIFICATION_BASES:
            skipped_unverified += 1
            continue
        lp = rec.get("local_path") or rec.get("filepath") or rec.get("path")
        if require_existing_files and (not lp or not os.path.exists(str(lp)) or os.path.getsize(str(lp)) <= 128):
            skipped_missing += 1
            continue
        r_copy = dict(rec)
        r_copy["origin_id"] = extract_origin_id(r_copy)
        sp = normalize_split_name(r_copy.get("split"), default="train")
        r_copy["split"] = sp
        splits[sp].append(r_copy)

    audit = assert_zero_origin_leakage(splits["train"], splits["cal_val"], splits["test_locked"])
    audit["skipped_unverified"] = skipped_unverified
    audit["skipped_missing"] = skipped_missing
    return {"splits": splits, "audit": audit}


def compute_wilson_ci(successes: int, total: int, z: float = 1.959964) -> Tuple[float, float]:
    """95% Wilson score confidence interval for binomial proportions in [0.0, 1.0]."""
    if total <= 0:
        return (0.0, 0.0)
    p = max(0.0, min(1.0, float(successes) / float(total)))
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def _roc_auc_binary(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    pos = [float(p) for y, p in zip(y_true, y_prob) if int(y) == 1]
    neg = [float(p) for y, p in zip(y_true, y_prob) if int(y) == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / float(len(pos) * len(neg))


def _ece_10bin(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    n = len(y_true)
    if n == 0:
        return 0.0
    ece = 0.0
    for b in range(10):
        lo = b / 10.0
        hi = (b + 1) / 10.0
        idxs = [i for i, p in enumerate(y_prob) if (lo <= p < hi) or (b == 9 and p == 1.0)]
        if idxs:
            conf = sum(float(y_prob[i]) for i in idxs) / len(idxs)
            acc = sum(int(y_true[i]) for i in idxs) / len(idxs)
            ece += (len(idxs) / float(n)) * abs(acc - conf)
    return ece


def _operating_metrics_point(y_true: Sequence[int], y_prob: Sequence[float], tau: float) -> Dict[str, float]:
    tp = tn = fp = fn = 0
    brier_sum = 0.0
    for y_raw, p_raw in zip(y_true, y_prob):
        y = int(y_raw)
        p = float(p_raw)
        pred = 1 if p >= tau else 0
        brier_sum += (p - y) ** 2
        if pred == 1 and y == 1:
            tp += 1
        elif pred == 0 and y == 0:
            tn += 1
        elif pred == 1 and y == 0:
            fp += 1
        else:
            fn += 1

    n = max(1, len(y_true))
    tpr = tp / float(max(1, tp + fn))
    tnr = tn / float(max(1, tn + fp))
    fpr = fp / float(max(1, fp + tn))
    fnr = fn / float(max(1, tp + fn))
    prec = tp / float(max(1, tp + fp))
    f1 = (2.0 * prec * tpr / (prec + tpr)) if (prec + tpr) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    acc = (tp + tn) / float(n)
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": tpr,
        "tnr": tnr,
        "fpr": fpr,
        "fnr": fnr,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "brier": brier_sum / float(n),
        "ece": _ece_10bin(y_true, y_prob),
        "roc_auc": _roc_auc_binary(y_true, y_prob),
    }


def compute_bootstrap_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    tau: float,
    *,
    n_boot: int = 1000,
    seed: int = 2026,
) -> Dict[str, Any]:
    """
    Compute point metrics at locked operating threshold `tau` plus stratified 95% Bootstrap CIs
    and binomial Wilson CIs for TPR/FPR/FNR.
    """
    n = len(y_true)
    if n == 0:
        return {"sample_count": 0, "locked_tau": tau}

    point = _operating_metrics_point(y_true, y_prob, tau)
    pos_idx = [i for i, y in enumerate(y_true) if int(y) == 1]
    neg_idx = [i for i, y in enumerate(y_true) if int(y) == 0]

    metric_keys = ("tpr", "fpr", "fnr", "accuracy", "balanced_accuracy", "f1", "ece", "brier", "roc_auc")
    boot_vals: Dict[str, List[float]] = {k: [] for k in metric_keys}
    rng = random.Random(int(seed))

    for _ in range(max(50, int(n_boot))):
        # Stratified bootstrap preserves class ratio for reliable ROC-AUC & balanced metrics
        sampled: List[int] = []
        if pos_idx:
            sampled.extend(rng.choice(pos_idx) for _ in range(len(pos_idx)))
        if neg_idx:
            sampled.extend(rng.choice(neg_idx) for _ in range(len(neg_idx)))
        yb = [int(y_true[i]) for i in sampled]
        pb = [float(y_prob[i]) for i in sampled]
        m = _operating_metrics_point(yb, pb, tau)
        for k in metric_keys:
            boot_vals[k].append(m[k])

    ci95: Dict[str, List[float]] = {}
    for k in metric_keys:
        vals = sorted(boot_vals[k])
        lo = vals[max(0, int(0.025 * len(vals)))]
        hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
        ci95[k] = [round(lo, 4), round(hi, 4)]

    tp, tn, fp, fn = int(point["tp"]), int(point["tn"]), int(point["fp"]), int(point["fn"])
    return {
        "sample_count": n,
        "locked_tau": round(float(tau), 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tpr": round(point["tpr"], 4),
        "fpr": round(point["fpr"], 4),
        "fnr": round(point["fnr"], 4),
        "accuracy": round(point["accuracy"], 4),
        "balanced_accuracy": round(point["balanced_accuracy"], 4),
        "f1": round(point["f1"], 4),
        "ece": round(point["ece"], 4),
        "brier": round(point["brier"], 4),
        "roc_auc": round(point["roc_auc"], 4),
        "bootstrap_ci95": ci95,
        "wilson_ci95": {
            "tpr": list(compute_wilson_ci(tp, tp + fn)),
            "fpr": list(compute_wilson_ci(fp, fp + tn)),
            "fnr": list(compute_wilson_ci(fn, tp + fn)),
            "accuracy": list(compute_wilson_ci(tp + tn, n)),
        },
    }
