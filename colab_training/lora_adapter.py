"""
LoRA / DoRA / LoRA+ / Orthogonal Dual-Adapter / Diagonal EWC Module for Vision & Audio Transformers.
Supports both native PyTorch (zero external dependencies) and Hugging Face PEFT.

References:
  - Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
  - Liu et al., "DoRA: Weight-Decomposed Low-Rank Adaptation", ICML 2024.
  - Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models", ICML 2024.
  - Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (EWC), PNAS 2017.
"""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import peft
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class NativeLoRALinear(nn.Module):
    """
    Native PyTorch LoRA Layer wrapping an existing nn.Linear layer.
    Computes: output = W_orig(x) + (alpha / r) * (B(A(x)))
    Zero initial perturbation since B is initialized to 0.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(max(1, r))

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def effective_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        return self.base_layer.weight.data + self.effective_delta_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.base_layer(x)
        dropped = self.dropout(x)
        lora_out = F.linear(F.linear(dropped, self.lora_A), self.lora_B) * self.scaling
        return orig_out + lora_out


class NativeDoRALinear(nn.Module):
    """
    Native PyTorch Weight-Decomposed Low-Rank Adaptation (DoRA, ICML 2024).
    Decomposes adapted weight matrix W' into magnitude vector `m` and directional unit matrix:
      W_dora = m * (W_0 + (alpha/r) * B @ A) / ||W_0 + (alpha/r) * B @ A||_row
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(max(1, r))

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        with torch.no_grad():
            row_norms = self.base_layer.weight.data.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        self.magnitude = nn.Parameter(row_norms.clone())
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def effective_delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    def merged_weight(self) -> torch.Tensor:
        w_eff = self.base_layer.weight.to(self.lora_A.dtype) + self.effective_delta_weight()
        denom = w_eff.norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        return self.magnitude * (w_eff / denom)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = self.base_layer.weight.to(x.dtype) + self.effective_delta_weight().to(x.dtype)
        denom = w_eff.detach().norm(p=2, dim=1, keepdim=True).clamp(min=1e-6)
        mag = self.magnitude.to(x.dtype)
        if not self.training or isinstance(self.dropout, nn.Identity):
            w_dora = mag * (w_eff / denom)
            bias = self.base_layer.bias.to(x.dtype) if self.base_layer.bias is not None else None
            return F.linear(x, w_dora, bias)
        # Apply dropout along the low-rank directional branch while scaling direction norm by magnitude
        dir_weight = (self.base_layer.weight.to(x.dtype) / denom) * mag
        base_scaled = F.linear(x, dir_weight, self.base_layer.bias.to(x.dtype) if self.base_layer.bias is not None else None)
        dropped = self.dropout(x)
        delta_out = F.linear(F.linear(dropped, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling
        delta_scaled = delta_out * (mag.squeeze(1) / denom.squeeze(1))
        return base_scaled + delta_scaled


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Returns (total_params, trainable_params, trainable_ratio_percent)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    return total, trainable, ratio


DEFAULT_VIT_TARGET_KEYWORDS = [
    "query",
    "key",
    "value",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "attention.attention",
    "attention.output.dense",
    "intermediate.dense",
    "fc1",
    "fc2",
]


def apply_lora_to_vit(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_module_keywords: Optional[List[str]] = None,
    prefer_peft: bool = True,
    peft_method: str = "dora",
) -> nn.Module:
    """
    Applies DoRA or LoRA to the attention & MLP projection layers of a Transformer model.
    peft_method: "dora" (default, ICML 2024) | "lora"
    """
    method = (peft_method or "dora").strip().lower()
    use_dora = method == "dora"
    if target_module_keywords is None:
        target_module_keywords = list(DEFAULT_VIT_TARGET_KEYWORDS)

    # 1. Try Hugging Face PEFT if available and preferred
    if PEFT_AVAILABLE and prefer_peft and hasattr(model, "config"):
        try:
            target_mods = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    lname = name.lower()
                    if lname.endswith(".classifier") or lname == "classifier" or lname.endswith(".head"):
                        continue
                    for kw in target_module_keywords:
                        if kw in lname:
                            leaf_name = name.split(".")[-1]
                            if leaf_name not in target_mods:
                                target_mods.append(leaf_name)

            if not target_mods:
                target_mods = ["query", "value"]

            cfg_kwargs = dict(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_mods,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=["classifier"] if hasattr(model, "classifier") else None,
            )
            if use_dora:
                cfg_kwargs["use_dora"] = True
            lora_config = LoraConfig(**cfg_kwargs)
            peft_model = get_peft_model(model, lora_config)
            print(f"[{'DoRA' if use_dora else 'LoRA'} Adapter] Successfully initialized via HuggingFace PEFT:")
            peft_model.print_trainable_parameters()
            return peft_model
        except Exception as err:
            print(f"[LoRA/DoRA Adapter] PEFT initialization notice: {err}. Falling back to native PyTorch wrapper.")

    # 2. Native PyTorch DoRA / LoRA Injection Fallback
    wrapper_cls = NativeDoRALinear if use_dora else NativeLoRALinear
    label = "DoRA" if use_dora else "LoRA"
    print(f"[{label} Adapter] Injecting Native {wrapper_cls.__name__} (r={r}, alpha={lora_alpha}, dropout={lora_dropout})...")

    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Linear):
                full_name = f"{name}.{child_name}".lower() if name else child_name.lower()
                if full_name.endswith(".classifier") or full_name == "classifier" or full_name.endswith(".head"):
                    for p in child_module.parameters():
                        p.requires_grad = True
                elif any(kw in full_name for kw in target_module_keywords):
                    wrapped = wrapper_cls(child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    setattr(module, child_name, wrapped)
                    replaced_count += 1

    total, trainable, ratio = count_parameters(model)
    print(f"[{label} Adapter] Injected {replaced_count} layers.")
    print(f"[{label} Adapter] Trainable Params: {trainable:,} / {total:,} ({ratio:.2f}%) | VRAM footprint minimized.")
    return model


def build_loraplus_param_groups(
    model: nn.Module,
    lr: float = 1e-4,
    loraplus_lr_ratio: float = 16.0,
    weight_decay: float = 0.01,
) -> List[Dict]:
    """
    LoRA+ (Hayou et al., ICML 2024) differential learning rates:
      - Group A (lora_A & standard weights): base `lr`
      - Group B (lora_B, magnitude, classifier): `lr * loraplus_lr_ratio`
    """
    group_a: List[nn.Parameter] = []
    group_b: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if "lora_b" in lname or "magnitude" in lname or "classifier" in lname or ".head." in lname:
            group_b.append(param)
        else:
            group_a.append(param)

    groups: List[Dict] = []
    if group_a:
        groups.append({"params": group_a, "lr": float(lr), "weight_decay": float(weight_decay)})
    if group_b:
        groups.append({
            "params": group_b,
            "lr": float(lr) * float(max(1.0, loraplus_lr_ratio)),
            "weight_decay": float(weight_decay),
        })
    return groups


def compute_orthogonal_adapter_loss(
    model_a: nn.Module,
    model_b: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Enforces subspace orthogonality || (DeltaW_1)^T @ DeltaW_2 ||_F^2 across LoRA/DoRA layers
    so General-domain and Human-biometric adapters avoid representation interference.
    When `model_b` is None, penalizes intra-model similarity between adjacent adapter layers.
    """
    deltas_a = [
        m.effective_delta_weight()
        for m in model_a.modules()
        if hasattr(m, "effective_delta_weight")
    ]
    if not deltas_a:
        dev = next(model_a.parameters()).device if any(True for _ in model_a.parameters()) else "cpu"
        return torch.tensor(0.0, device=dev)

    if model_b is not None:
        deltas_b = [
            m.effective_delta_weight()
            for m in model_b.modules()
            if hasattr(m, "effective_delta_weight")
        ]
        terms = []
        for da, db in zip(deltas_a, deltas_b):
            if da.shape == db.shape:
                gram = da.T @ db
                terms.append((gram ** 2).mean())
        return torch.stack(terms).mean() if terms else deltas_a[0].sum() * 0.0

    # Single-model mode: enforce orthonormal low-rank projections within each adapter
    terms = []
    for m in model_a.modules():
        if hasattr(m, "lora_A"):
            a = m.lora_A
            gram = a @ a.T
            eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
            terms.append(((gram - eye) ** 2).mean())
    return torch.stack(terms).mean() if terms else deltas_a[0].sum() * 0.0


def merge_native_adapters(model: nn.Module) -> nn.Module:
    """
    Folds NativeLoRALinear / NativeDoRALinear or PEFT LoRA weights cleanly back into standard
    `nn.Linear` layers so `save_pretrained(..., safe_serialization=True)` produces standard `model.safetensors`.
    """
    if hasattr(model, "merge_and_unload"):
        try:
            return model.merge_and_unload()
        except Exception:
            pass

    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, (NativeLoRALinear, NativeDoRALinear)):
                merged_w = child_module.merged_weight().detach().clone()
                orig = child_module.base_layer
                new_linear = nn.Linear(
                    orig.in_features,
                    orig.out_features,
                    bias=(orig.bias is not None),
                    device=merged_w.device,
                    dtype=orig.weight.dtype,
                )
                with torch.no_grad():
                    new_linear.weight.copy_(merged_w.to(orig.weight.dtype))
                    if orig.bias is not None and new_linear.bias is not None:
                        new_linear.bias.copy_(orig.bias.detach())
                setattr(module, child_name, new_linear)
    return model


class DiagonalEWC:
    """
    Diagonal Elastic Weight Consolidation (EWC) to prevent Catastrophic Forgetting
    across repeated continual fine-tuning cycles:
      L_EWC = (lambda / 2) * sum_k F_k * (theta_k - theta_k*)^2
    """

    def __init__(
        self,
        fisher_diag: Optional[Dict[str, torch.Tensor]] = None,
        anchor_params: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.fisher_diag: Dict[str, torch.Tensor] = fisher_diag or {}
        self.anchor_params: Dict[str, torch.Tensor] = anchor_params or {}

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        dataloader=None,
        device: str = "cpu",
        max_batches: int = 8,
    ) -> "DiagonalEWC":
        inst = cls()
        if dataloader is not None:
            inst.estimate_fisher(model, dataloader, device=device, max_batches=max_batches)
        else:
            for n, p in model.named_parameters():
                if p.requires_grad:
                    inst.anchor_params[n] = p.detach().cpu().clone()
                    inst.fisher_diag[n] = torch.ones_like(inst.anchor_params[n])
        return inst

    def estimate_fisher(
        self,
        model: nn.Module,
        dataloader,
        device: str = "cpu",
        max_batches: int = 8,
    ) -> None:
        was_training = model.training
        model.eval()
        fisher_acc: Dict[str, torch.Tensor] = {
            n: torch.zeros_like(p.detach().cpu())
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        anchor: Dict[str, torch.Tensor] = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        n_steps = 0
        for b_idx, batch in enumerate(dataloader):
            if b_idx >= max_batches:
                break
            pixel_values = batch[0].to(device)
            labels = batch[1].to(device)
            model.zero_grad()
            out = model(pixel_values)
            logits = out.logits if hasattr(out, "logits") else out
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs.gather(1, labels.view(-1, 1)).squeeze(1).mean()
            (-selected).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None and n in fisher_acc:
                    fisher_acc[n] += (p.grad.detach().cpu() ** 2)
            n_steps += 1
        if n_steps > 0:
            for n in fisher_acc:
                fisher_acc[n] /= float(n_steps)
        self.fisher_diag = fisher_acc
        self.anchor_params = anchor
        model.train(was_training)

    def penalty(self, model: nn.Module, lambda_ewc: float = 50.0) -> torch.Tensor:
        if not self.fisher_diag:
            dev = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
            return torch.tensor(0.0, device=dev)
        total = None
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.fisher_diag and n in self.anchor_params:
                f = self.fisher_diag[n].to(p.device, dtype=p.dtype)
                a = self.anchor_params[n].to(p.device, dtype=p.dtype)
                term = (f * ((p - a) ** 2)).sum()
                total = term if total is None else (total + term)
        if total is None:
            dev = next(model.parameters()).device if any(True for _ in model.parameters()) else "cpu"
            return torch.tensor(0.0, device=dev)
        return 0.5 * float(lambda_ewc) * total

    def save(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"fisher_diag": self.fisher_diag, "anchor_params": self.anchor_params}, str(p))
        return str(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DiagonalEWC":
        payload = torch.load(str(path), map_location="cpu")
        return cls(
            fisher_diag=payload.get("fisher_diag", {}),
            anchor_params=payload.get("anchor_params", {}),
        )


def save_lora_weights(model: nn.Module, save_path: Union[str, Path]) -> str:
    """Saves ONLY trainable LoRA/DoRA parameters and classification head (~2-15 MB)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if PEFT_AVAILABLE and hasattr(model, "save_pretrained") and hasattr(model, "peft_config"):
        model.save_pretrained(str(save_path))
        return str(save_path)

    lora_state = {
        name: param.data.cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(lora_state, str(save_path))
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"[LoRA/DoRA Adapter] Saved compact checkpoint: {save_path.name} ({file_size_mb:.2f} MB)")
    return str(save_path)


def load_lora_weights(model: nn.Module, weights_path: Union[str, Path]) -> nn.Module:
    """Loads LoRA/DoRA weights into an existing base model."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at: {weights_path}")

    if PEFT_AVAILABLE and weights_path.is_dir() and (weights_path / "adapter_config.json").exists():
        from peft import PeftModel
        return PeftModel.from_pretrained(model, str(weights_path))

    state = torch.load(str(weights_path), map_location="cpu")
    model.load_state_dict(state, strict=False)
    print(f"[LoRA/DoRA Adapter] Successfully loaded weights from {weights_path.name}.")
    return model
