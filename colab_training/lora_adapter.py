"""
LoRA (Low-Rank Adaptation / PEFT) Module for Vision Transformers (ViT).
Supports both native PyTorch (zero external dependencies) and Hugging Face PEFT.

Reference:
  Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022.
  PhD Thesis Section 3.4: Parameter-Efficient Fine-Tuning for In-the-Wild Social Deepfakes.
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
    Computes: output = W_orig(x) + (alpha / r) * (x @ A^T @ B^T)
    Zero initial perturbation since B is initialized to 0.
    """
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = float(lora_alpha) / float(r)

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank decomposition matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize: Kaiming normal for A, Zeros for B -> ΔW = B @ A = 0 at start
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_out = self.base_layer(x)
        # Low rank adaptation
        dropped = self.dropout(x)
        lora_out = F.linear(F.linear(dropped, self.lora_A), self.lora_B) * self.scaling
        return orig_out + lora_out


def count_parameters(model: nn.Module) -> Tuple[int, int, float]:
    """Returns (total_params, trainable_params, trainable_ratio_percent)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ratio = (trainable / total * 100.0) if total > 0 else 0.0
    return total, trainable, ratio


def apply_lora_to_vit(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_module_keywords: Optional[List[str]] = None,
    prefer_peft: bool = True
) -> nn.Module:
    """
    Applies LoRA to the attention projection layers of a Vision Transformer model.
    Targets query, key, value, and dense projection layers by default.
    """
    if target_module_keywords is None:
        target_module_keywords = ["query", "key", "value", "q_proj", "v_proj", "attention.attention"]

    # 1. Try Hugging Face PEFT if available and preferred
    if PEFT_AVAILABLE and prefer_peft and hasattr(model, "config"):
        try:
            target_mods = []
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    for kw in target_module_keywords:
                        if kw in name.lower():
                            leaf_name = name.split(".")[-1]
                            if leaf_name not in target_mods:
                                target_mods.append(leaf_name)

            if not target_mods:
                target_mods = ["query", "value"]

            lora_config = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                target_modules=target_mods,
                lora_dropout=lora_dropout,
                bias="none",
                modules_to_save=["classifier"] if hasattr(model, "classifier") else None
            )
            peft_model = get_peft_model(model, lora_config)
            print("[LoRA Adapter] Successfully initialized via HuggingFace PEFT:")
            peft_model.print_trainable_parameters()
            return peft_model
        except Exception as err:
            print(f"[LoRA Adapter] PEFT initialization notice: {err}. Falling back to native PyTorch LoRA.")

    # 2. Native PyTorch LoRA Injection Fallback
    print(f"[LoRA Adapter] Injecting Native PyTorch LoRALinear (r={r}, alpha={lora_alpha}, dropout={lora_dropout})...")
    
    # Freeze all model parameters first
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0
    # Walk modules and replace target linear projections
    for name, module in list(model.named_modules()):
        for child_name, child_module in list(module.named_children()):
            if isinstance(child_module, nn.Linear):
                full_name = f"{name}.{child_name}".lower()
                is_target = any(kw in full_name for kw in target_module_keywords)
                # Keep classification head trainable
                if full_name.endswith(".classifier") or full_name == "classifier":
                    for p in child_module.parameters():
                        p.requires_grad = True
                elif is_target:
                    lora_layer = NativeLoRALinear(child_module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
                    setattr(module, child_name, lora_layer)
                    replaced_count += 1

    total, trainable, ratio = count_parameters(model)
    print(f"[LoRA Adapter] Injected {replaced_count} LoRA layers into ViT.")
    print(f"[LoRA Adapter] Trainable Params: {trainable:,} / {total:,} ({ratio:.2f}%) | VRAM footprint minimized.")
    return model


def save_lora_weights(model: nn.Module, save_path: Union[str, Path]) -> str:
    """
    Saves ONLY the trainable LoRA parameters and classification head (~5 - 15 MB).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if PEFT_AVAILABLE and hasattr(model, "save_pretrained"):
        model.save_pretrained(str(save_path))
        return str(save_path)

    # Native state dict of trainable parameters only
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    torch.save(lora_state, str(save_path))
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"[LoRA Adapter] Saved compact LoRA checkpoint: {save_path.name} ({file_size_mb:.2f} MB)")
    return str(save_path)


def load_lora_weights(model: nn.Module, weights_path: Union[str, Path]) -> nn.Module:
    """Loads LoRA weights into an existing base model."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"LoRA weights not found at: {weights_path}")

    if PEFT_AVAILABLE and weights_path.is_dir() and (weights_path / "adapter_config.json").exists():
        from peft import PeftModel
        return PeftModel.from_pretrained(model, str(weights_path))

    # Native PyTorch load
    state = torch.load(str(weights_path), map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[LoRA Adapter] Successfully loaded LoRA weights from {weights_path.name}.")
    return model
