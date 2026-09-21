"""
PPO Meta-Controller for Autonomous Deepfake Hyper-Parameter Optimization.
Evaluates 3 Threads Benchmark Jobs (01_raw_originals, 02_manipulated_faceswap, 03_manipulated_openweight)
and predicts optimal hyperparameters:
  - Decision Threshold tau* in [0.50, 0.95]
  - Softmax Temperature T* in [1.0, 3.0]
  - Dual-Stream Fusion Weight lambda_bio in [0.10, 0.90]
  - Hard Negative Mining Threshold eta in [0.60, 0.90]
  - Trigger LoRA Adaptation Signal (Boolean)
"""
import os
import json
import time
import math
import random
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class PPONetwork(nn.Module if HAS_TORCH else object):
    def __init__(self, state_dim=6, action_dim=4):
        if HAS_TORCH:
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Sigmoid()
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

    def forward(self, state):
        if HAS_TORCH:
            action_probs = self.actor(state)
            state_value = self.critic(state)
            return action_probs, state_value
        return None, None

class PPOMetaController:
    """
    PPO RL Controller managing dynamic hyperparameters & LoRA triggers for Threads-DFBench.
    """
    def __init__(self, state_dim=6, action_dim=4, lr=0.001, gamma=0.99, clip_eps=0.2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.trajectories = []
        
        if HAS_TORCH:
            self.model = PPONetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            self.model = None

    def construct_state_vector(self, metrics: dict) -> np.ndarray:
        """
        Builds State Vector S_t in R^6:
          0: auc (0.0 - 1.0)
          1: ece (0.0 - 1.0)
          2: fpr_1pct (0.0 - 1.0)
          3: mean_entropy (0.0 - 1.0)
          4: hard_sample_ratio (0.0 - 1.0)
          5: dwt_zpeak_gap (0.0 - 5.0 scaled)
        """
        auc = float(metrics.get("auc", 0.90))
        ece = float(metrics.get("ece", 0.05))
        fpr = float(metrics.get("fpr_percent", 2.0)) / 100.0
        entropy = float(metrics.get("mean_entropy", 0.50))
        hard_ratio = float(metrics.get("hard_sample_ratio", 0.20))
        gap = float(metrics.get("dwt_zpeak_gap", 1.50)) / 5.0

        state = np.array([auc, ece, fpr, entropy, hard_ratio, gap], dtype=np.float32)
        return state

    def predict_action(self, metrics: dict) -> dict:
        """
        Predicts action A_t given environment state from 3 Threads Benchmark Jobs.
        """
        state_np = self.construct_state_vector(metrics)
        
        if HAS_TORCH and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_np).unsqueeze(0)
                raw_actions, _ = self.model(state_tensor)
                raw = raw_actions.numpy()[0]
        else:
            # Fallback heuristic mapping if torch is not active
            raw = np.array([
                0.60 + 0.35 * (1.0 - state_np[1]), # tau
                0.30 + 0.60 * state_np[1],         # T*
                0.40 + 0.40 * state_np[3],         # lambda_bio
                0.70 + 0.20 * state_np[4]          # eta
            ], dtype=np.float32)

        # Map raw sigmoid outputs [0, 1] to physical bounds
        optimal_tau = round(float(0.50 + 0.45 * raw[0]), 4)      # [0.50, 0.95]
        optimal_temp = round(float(1.0 + 2.0 * raw[1]), 4)        # [1.00, 3.00]
        lambda_bio = round(float(0.10 + 0.80 * raw[2]), 4)        # [0.10, 0.90]
        hard_threshold_eta = round(float(0.60 + 0.30 * raw[3]), 4)# [0.60, 0.90]
        
        # Trigger LoRA if ECE > 0.05 or hard_sample_ratio > 0.35
        trigger_lora = bool(metrics.get("ece", 0.05) > 0.05 or metrics.get("hard_sample_ratio", 0.0) > 0.35)

        action_dict = {
            "optimal_tau": optimal_tau,
            "optimal_temp_t": optimal_temp,
            "lambda_bio": lambda_bio,
            "hard_threshold_eta": hard_threshold_eta,
            "trigger_lora": trigger_lora,
            "state_vector": state_np.tolist(),
            "predicted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        return action_dict

    def compute_reward(self, metrics: dict) -> float:
        """
        Computes PPO Reward R_t:
          R_t = AUC - 2.0 * ECE - 1.5 * FPR_1pct
        """
        auc = float(metrics.get("auc", 0.90))
        ece = float(metrics.get("ece", 0.05))
        fpr = float(metrics.get("fpr_percent", 2.0)) / 100.0
        
        reward = auc - (2.0 * ece) - (1.5 * fpr)
        return round(float(reward), 4)

    def record_step(self, metrics: dict, action_dict: dict):
        """
        Stores trajectory tuple (State, Action, Reward) in controller memory.
        """
        reward = self.compute_reward(metrics)
        trajectory_step = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "state": action_dict.get("state_vector", []),
            "action": action_dict,
            "reward": reward
        }
        self.trajectories.append(trajectory_step)
        print(f"[PPO Meta-Controller] Recorded Trajectory Step #{len(self.trajectories)} | Reward={reward:.4f} (Tau={action_dict['optimal_tau']}, Temp={action_dict['optimal_temp_t']}, Lambda={action_dict['lambda_bio']})")
        return trajectory_step

if __name__ == "__main__":
    controller = PPOMetaController()
    test_metrics = {"auc": 0.925, "ece": 0.018, "fpr_percent": 1.2, "mean_entropy": 0.55, "hard_sample_ratio": 0.28, "dwt_zpeak_gap": 1.8}
    act = controller.predict_action(test_metrics)
    print("Test PPO Action Prediction:", json.dumps(act, indent=2))
