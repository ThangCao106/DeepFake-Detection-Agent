"""
Probability Calibration & Threshold Optimization Module.
1. Temperature Scaling (Platt Scaling via Logit Optimization)
2. Youden's J Index Threshold Optimization (TPR - FPR)
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from train_pipeline import DeepfakeDataset
from transformers import AutoImageProcessor, AutoModelForImageClassification


class TemperatureScaler(nn.Module):
    """
    Temperature scaling optimizer for post-hoc probability calibration.
    Scales logits by 1/T to align output confidence with true accuracy.
    """
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, val_loader, model, device="cpu"):
        model.eval()
        logits_list, labels_list = [], []

        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                outputs = model(pixel_values=pixel_values)
                logits_list.append(outputs.logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list, dim=0).to(device)
        labels = torch.cat(labels_list, dim=0).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        calibrated_temp = float(self.temperature.item())
        print(f"[calibrate] Optimal Temperature Scaling T*: {calibrated_temp:.4f}")
        return calibrated_temp, logits, labels


def optimize_threshold(logits, labels, temperature=1.0):
    """
    Optimize classification threshold tau* using Youden's J Index (TPR - FPR).
    """
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)[:, 1].cpu().numpy()
    targets = labels.cpu().numpy()

    thresholds = np.linspace(0.01, 0.99, 100)
    best_j = -1.0
    best_tau = 0.5

    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        tp = np.sum((preds == 1) & (targets == 1))
        tn = np.sum((preds == 0) & (targets == 0))
        fp = np.sum((preds == 1) & (targets == 0))
        fn = np.sum((preds == 0) & (targets == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j_index = tpr - fpr

        if j_index > best_j:
            best_j = j_index
            best_tau = tau

    print(f"[calibrate] Optimal Decision Threshold tau*: {best_tau:.4f} (Youden's J: {best_j:.4f})")
    return round(float(best_tau), 4)


def calibrate_and_optimize(model_dir="custom_vit_model", manifest_file="dataset_pipeline/source_split/test_manifest.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(model_dir)
    model = AutoModelForImageClassification.from_pretrained(model_dir).to(device)

    dataset = DeepfakeDataset(manifest_file, processor)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    scaler = TemperatureScaler()
    temp, logits, labels = scaler.calibrate(val_loader, model, device=device)
    tau = optimize_threshold(logits, labels, temperature=temp)

    params = {
        "temperature_scaling_T": temp,
        "optimal_threshold_tau": tau,
        "calibration_status": "CALIBRATED"
    }

    param_file = os.path.join(model_dir, "calibration_params.json")
    with open(param_file, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    print(f"[calibrate] Saved Calibration Parameters to {param_file}")
    return params


if __name__ == "__main__":
    calibrate_and_optimize()
