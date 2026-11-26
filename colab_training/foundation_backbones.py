# -*- coding: utf-8 -*-
"""
Foundation Vision Backbones & Effort Orthogonal Subspace Adaptation (ICML 2025).
Supports:
  - "vit": dima806/deepfake_vs_real_image_detection
  - "clip": openai/clip-vit-large-patch14 (fallback openai/clip-vit-base-patch16)
  - "siglip2": prithivMLmods/deepfake-detector-model-v1 (fallback google/siglip2-base-patch16-224)
  - "dinov2": facebook/dinov2-base

Effort (Orthogonal Subspace Decomposition):
  Decomposes pretrained W_0 = U_r S_r V_r^T + W_residual via SVD, freezes the principal
  semantic basis (U_r, V_r), and optimizes residual low-rank updates constrained orthogonal
  to the semantic subspace so generative forgery cues generalize to unseen Diffusion / Flow-Matching models.
"""

import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

SUPPORTED_BACKBONES: Dict[str, List[str]] = {
    "vit": [
        "dima806/deepfake_vs_real_image_detection",
        "prithivMLmods/Deep-Fake-Detector-v2-Model",
    ],
    "clip": [
        "openai/clip-vit-large-patch14",
        "openai/clip-vit-base-patch16",
    ],
    "siglip2": [
        "prithivMLmods/deepfake-detector-model-v1",
        "google/siglip2-base-patch16-224",
    ],
    "dinov2": [
        "facebook/dinov2-base",
    ],
}


class OrthogonalSubspaceLinear(nn.Module):
    """
    Effort-inspired Orthogonal Residual Subspace Layer (ICML 2025).
    Freezes the top-k principal SVD subspace of W_0 (preserving general semantics)
    and learns low-rank residual projections orthogonal to the frozen principal subspace.
    """

    def __init__(self, base_layer: nn.Linear, residual_rank: int = 16, svd_freeze_ratio: float = 0.75):
        super().__init__()
        self.base_layer = base_layer
        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        with torch.no_grad():
            w = self.base_layer.weight.data.float()
            min_dim = min(out_features, in_features)
            max_rank = max(1, min_dim - 1)
            r_sem = max(1, min(max_rank, int(round(min_dim * svd_freeze_ratio))))
            r_res = max(1, min(residual_rank, min_dim - r_sem if min_dim > r_sem else min_dim))

            # Full or reduced SVD on CPU/current device
            u, s, vh = torch.linalg.svd(w, full_matrices=False)
            u_sem = u[:, :r_sem]
            v_sem = vh[:r_sem, :]
            w_sem = (u_sem * s[:r_sem].unsqueeze(0)) @ v_sem

            # Residual component initialized orthogonally from trailing singular vectors
            if u.size(1) > r_sem:
                u_tail = u[:, r_sem : r_sem + r_res]
                s_tail = s[r_sem : r_sem + r_res]
                v_tail = vh[r_sem : r_sem + r_res, :]
                if u_tail.size(1) < r_res:
                    pad_c = r_res - u_tail.size(1)
                    u_tail = torch.cat([u_tail, torch.zeros(out_features, pad_c, device=w.device)], dim=1)
                    s_tail = torch.cat([s_tail, torch.zeros(pad_c, device=w.device)], dim=0)
                    v_tail = torch.cat([v_tail, torch.zeros(pad_c, in_features, device=w.device)], dim=0)
            else:
                u_tail = torch.zeros(out_features, r_res, device=w.device)
                s_tail = torch.zeros(r_res, device=w.device)
                v_tail = torch.zeros(r_res, in_features, device=w.device)

        self.register_buffer("w_semantic", w_sem.to(base_layer.weight.dtype))
        self.register_buffer("u_semantic", u_sem.to(base_layer.weight.dtype))
        self.register_buffer("v_semantic", v_sem.to(base_layer.weight.dtype))

        self.res_U = nn.Parameter((u_tail * torch.sqrt(s_tail.clamp(min=1e-6)).unsqueeze(0)).to(base_layer.weight.dtype))
        self.res_V = nn.Parameter((torch.sqrt(s_tail.clamp(min=1e-6)).unsqueeze(1) * v_tail).to(base_layer.weight.dtype))
        self.residual_rank = r_res

    def effective_delta_weight(self) -> torch.Tensor:
        return self.res_U @ self.res_V

    def merged_weight(self) -> torch.Tensor:
        return self.w_semantic + self.effective_delta_weight()

    def orthogonality_loss(self) -> torch.Tensor:
        """Enforces U_sem^T @ res_U == 0 and res_V @ V_sem^T == 0."""
        u_proj = self.u_semantic.to(self.res_U.dtype).T @ self.res_U
        v_proj = self.res_V @ self.v_semantic.to(self.res_V.dtype).T
        return (u_proj ** 2).mean() + (v_proj ** 2).mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.base_layer.bias.to(x.dtype) if self.base_layer.bias is not None else None
        sem_out = F.linear(x, self.w_semantic.to(x.dtype), bias)
        res_out = F.linear(F.linear(x, self.res_V.to(x.dtype)), self.res_U.to(x.dtype))
        return sem_out + res_out


def apply_effort_orthogonal_subspace(
    model: nn.Module,
    rank: int = 16,
    target_keywords: Optional[List[str]] = None,
) -> nn.Module:
    """
    Injects Effort OrthogonalSubspaceLinear layers across Transformer attention & MLP weights.
    """
    if target_keywords is None:
        target_keywords = ["query", "key", "value", "q_proj", "k_proj", "v_proj", "out_proj", "dense"]

    for p in model.parameters():
        p.requires_grad = False

    injected = 0
    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Linear):
                full_name = f"{name}.{child_name}".lower() if name else child_name.lower()
                if full_name.endswith(".classifier") or full_name == "classifier" or full_name.endswith(".head"):
                    for p in child_module.parameters():
                        p.requires_grad = True
                elif any(kw in full_name for kw in target_keywords):
                    setattr(module, child_name, OrthogonalSubspaceLinear(child_module, residual_rank=rank))
                    injected += 1

    print(f"[Effort Subspace] Injected {injected} OrthogonalSubspaceLinear layers (rank={rank}).")
    return model


def compute_effort_orthogonality_loss(model: nn.Module) -> torch.Tensor:
    terms = [m.orthogonality_loss() for m in model.modules() if isinstance(m, OrthogonalSubspaceLinear)]
    if not terms:
        dev = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
        return torch.tensor(0.0, device=dev)
    return torch.stack(terms).mean()


class FoundationDeepfakeClassifier(nn.Module):
    """
    Unified Vision Foundation Classifier wrapping CLIP, SigLIP-2, DINOv2, or ViT encoders.
    Supports `return_embeddings=True` for Supervised Contrastive Loss & Dual-Stream feature extraction.
    """

    def __init__(self, encoder: nn.Module, hidden_dim: int, num_labels: int = 2, head: Optional[nn.Module] = None):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.classifier = head if head is not None else nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, num_labels),
        )
        self.config = getattr(encoder, "config", None)
        if self.config is not None:
            self.config.id2label = {0: "Real", 1: "Fake"}
            self.config.label2id = {"Real": 0, "Fake": 1}

    def extract_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        out = self.encoder(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state[:, 0, :]
        if isinstance(out, (tuple, list)):
            first = out[0]
            return first[:, 0, :] if first.ndim == 3 else first
        return out

    def forward(self, pixel_values: torch.Tensor, return_embeddings: bool = False):
        emb = self.extract_embedding(pixel_values)
        logits = self.classifier(emb)
        if return_embeddings:
            return logits, emb
        return logits


def load_foundation_backbone_resilient(
    backbone_family: str = "vit",
    explicit_model_id: Optional[str] = None,
    device: str = "cpu",
):
    """
    Resiliently loads a Hugging Face Vision Foundation Model (`vit`, `clip`, `siglip2`, `dinov2`)
    with `token=False` anonymous fallback so expired `HF_TOKEN` never breaks training.
    """
    from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

    fam = (backbone_family or "vit").strip().lower()
    candidates = []
    if explicit_model_id:
        candidates.append(explicit_model_id)
    for c in SUPPORTED_BACKBONES.get(fam, SUPPORTED_BACKBONES["vit"]):
        if c not in candidates:
            candidates.append(c)

    env_tok = (os.environ.get("HF_TOKEN") or "").strip() or None
    last_err = None

    for repo_id in candidates:
        attempts = [env_tok, False] if (env_tok and not os.path.exists(repo_id)) else [False]
        for tok in attempts:
            try:
                processor = AutoImageProcessor.from_pretrained(repo_id, token=tok)
                if fam in ("vit", "siglip2") and ("deepfake" in repo_id.lower() or os.path.exists(repo_id)):
                    model = AutoModelForImageClassification.from_pretrained(
                        repo_id, num_labels=2, ignore_mismatched_sizes=True, token=tok
                    ).to(device)
                    return processor, model, True, repo_id

                # Load pure vision encoder (CLIP / SigLIP2 / DINOv2)
                base_mod = AutoModel.from_pretrained(repo_id, token=tok)
                if hasattr(base_mod, "vision_model"):
                    encoder = base_mod.vision_model
                    hidden_dim = int(getattr(encoder.config, "hidden_size", 768))
                else:
                    encoder = base_mod
                    hidden_dim = int(getattr(encoder.config, "hidden_size", 768))
                model = FoundationDeepfakeClassifier(encoder=encoder, hidden_dim=hidden_dim, num_labels=2).to(device)
                return processor, model, False, repo_id
            except Exception as err:
                last_err = err

    raise RuntimeError(f"Unable to load foundation backbone family='{fam}' ({candidates}): {last_err}")
