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
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.distributions import Normal
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
    def __init__(self, state_dim=6, action_dim=4, lr=0.001, gamma=0.99, clip_eps=0.2, checkpoint_path: Optional[str] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.trajectories = []
        self.step_count = 0
        self.last_reward = 0.0
        self.baseline_reward = 0.85

        default_ckpt = Path(__file__).resolve().parent.parent / "custom_vit_model" / "ppo_meta_controller.pt"
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else default_ckpt

        if HAS_TORCH:
            torch.manual_seed(2026)
            self.model = PPONetwork(state_dim, action_dim)
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
            if self.checkpoint_path.exists():
                try:
                    ckpt = torch.load(str(self.checkpoint_path), map_location="cpu", weights_only=False)
                    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                        self.model.load_state_dict(ckpt["model_state_dict"])
                        if "optimizer_state_dict" in ckpt:
                            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        self.step_count = int(ckpt.get("step_count", 0))
                        self.last_reward = float(ckpt.get("last_reward", 0.0))
                        self.baseline_reward = float(ckpt.get("baseline_reward", 0.85))
                except Exception as exc:
                    print(f"[PPO Meta-Controller] Checkpoint reload skipped ({exc})")
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
            "raw_action": [round(float(x), 6) for x in raw],
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

    def _reconstruct_raw_action(self, action_dict: dict) -> np.ndarray:
        """Recovers [0, 1]^4 policy action vector from action_dict when raw_action is absent."""
        if "raw_action" in action_dict and len(action_dict["raw_action"]) == self.action_dim:
            return np.clip(np.array(action_dict["raw_action"], dtype=np.float32), 1e-3, 1.0 - 1e-3)
        tau_r = (float(action_dict.get("optimal_tau", 0.72)) - 0.50) / 0.45
        temp_r = (float(action_dict.get("optimal_temp_t", 1.50)) - 1.0) / 2.0
        lam_r = (float(action_dict.get("lambda_bio", 0.50)) - 0.10) / 0.80
        eta_r = (float(action_dict.get("hard_threshold_eta", 0.75)) - 0.60) / 0.30
        return np.clip(np.array([tau_r, temp_r, lam_r, eta_r], dtype=np.float32), 1e-3, 1.0 - 1e-3)

    def update_policy(self, state_np: np.ndarray, reward: float, raw_action: Optional[np.ndarray] = None) -> dict:
        """
        Runs a clipped PPO Actor-Critic policy gradient update (4 epochs) on the observed
        (S_t, A_t, R_t) tuple and persists the updated checkpoint to disk.
        """
        if not HAS_TORCH or self.model is None:
            return {"policy_loss": 0.0, "checkpoint_saved": False}

        state_arr = np.asarray(state_np, dtype=np.float32).reshape(1, self.state_dim)
        state_t = torch.from_numpy(state_arr)
        target_return = torch.tensor([[float(reward)]], dtype=torch.float32)

        if raw_action is None:
            with torch.no_grad():
                mean_init, _ = self.model(state_t)
                action_t = mean_init.detach().clamp(1e-3, 1.0 - 1e-3)
        else:
            raw_arr = np.asarray(raw_action, dtype=np.float32).reshape(1, self.action_dim)
            action_t = torch.from_numpy(raw_arr).clamp(1e-3, 1.0 - 1e-3)

        self.model.eval()
        with torch.no_grad():
            old_mean, old_val = self.model(state_t)
            old_std = torch.full_like(old_mean, 0.08)
            old_dist = Normal(old_mean, old_std)
            old_log_prob = old_dist.log_prob(action_t).sum(dim=-1, keepdim=True)
            advantage = target_return - old_val
            if float(advantage.abs().item()) < 1e-4:
                advantage = target_return - torch.tensor([[float(self.baseline_reward)]], dtype=torch.float32)

        self.model.train()
        last_loss = 0.0
        for _ in range(4):
            self.optimizer.zero_grad()
            mean, val = self.model(state_t)
            std = torch.full_like(mean, 0.08)
            dist = Normal(mean, std)
            log_prob = dist.log_prob(action_t).sum(dim=-1, keepdim=True)
            ratio = torch.exp((log_prob - old_log_prob).clamp(-10.0, 10.0))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantage
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(val, target_return)
            entropy_bonus = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_bonus
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            last_loss = float(loss.item())

        self.model.eval()
        self.step_count += 1
        self.last_reward = float(reward)
        self.baseline_reward = float(0.9 * self.baseline_reward + 0.1 * float(reward))

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step_count": int(self.step_count),
                "last_reward": float(self.last_reward),
                "baseline_reward": float(self.baseline_reward),
                "state_dim": int(self.state_dim),
                "action_dim": int(self.action_dim),
            },
            str(self.checkpoint_path),
        )
        return {
            "policy_loss": round(last_loss, 6),
            "checkpoint_saved": self.checkpoint_path.exists(),
            "checkpoint_path": str(self.checkpoint_path),
        }

    def record_step(self, metrics: dict, action_dict: dict):
        """
        Stores trajectory tuple (State, Action, Reward), runs a PPO policy gradient update,
        and persists the updated Actor-Critic checkpoint.
        """
        reward = self.compute_reward(metrics)
        state_vec = action_dict.get("state_vector", [])
        if isinstance(state_vec, (list, tuple, np.ndarray)) and len(state_vec) == self.state_dim:
            state_np = np.array(state_vec, dtype=np.float32)
        else:
            state_np = self.construct_state_vector(metrics)
        raw_action = self._reconstruct_raw_action(action_dict)
        update_info = self.update_policy(state_np, reward, raw_action=raw_action)

        trajectory_step = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "state": state_np.tolist(),
            "action": action_dict,
            "reward": reward,
            "policy_loss": update_info.get("policy_loss", 0.0),
            "checkpoint_saved": update_info.get("checkpoint_saved", False),
            "checkpoint_path": update_info.get("checkpoint_path"),
            "step_count": int(self.step_count),
        }
        self.trajectories.append(trajectory_step)
        print(
            f"[PPO Meta-Controller] Step #{self.step_count} (Session #{len(self.trajectories)}) | "
            f"Reward={reward:.4f} | Loss={trajectory_step['policy_loss']:.6f} | "
            f"Saved={trajectory_step['checkpoint_saved']} "
            f"(Tau={action_dict['optimal_tau']}, Temp={action_dict['optimal_temp_t']}, Lambda={action_dict['lambda_bio']})"
        )
        return trajectory_step

if __name__ == "__main__":
    controller = PPOMetaController()
    test_metrics = {"auc": 0.925, "ece": 0.018, "fpr_percent": 1.2, "mean_entropy": 0.55, "hard_sample_ratio": 0.28, "dwt_zpeak_gap": 1.8}
    act = controller.predict_action(test_metrics)
    print("Test PPO Action Prediction:", json.dumps(act, indent=2))
