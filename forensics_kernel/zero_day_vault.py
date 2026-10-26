"""
Zero-Day Adversarial Active Learning Vault.
Acts as the closed-loop bridge between:
  1. Live Interactive Investigation (dfagent): captures ambiguous/hard samples.
  2. Autonomous Batch Pipeline: feeds hard samples to Colab GPU trainer.
"""

import os
import time
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VAULT_DIR = PROJECT_ROOT / "colab_datasets" / "zero_day_vault"
QUEUE_FILE = VAULT_DIR / "active_learning_queue.json"


class ZeroDayActiveLearningVault:
    """
    Manages the active learning sample queue.
    Ambiguous media files discovered during live investigations are staged here
    to be prioritized during nightly continual training.
    """
    _instance = None

    def __init__(self):
        VAULT_DIR.mkdir(parents=True, exist_ok=True)
        self.queue_file = QUEUE_FILE
        self._ensure_queue()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_queue(self):
        if not self.queue_file.exists():
            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump([], f)

    def enqueue_adversarial_sample(self, file_path: str, modality: str, confidence: float, anomaly_reason: str) -> Dict[str, Any]:
        """
        Enqueues an adversarial outlier or hard sample into the active learning vault.
        """
        try:
            p = Path(file_path)
            if not p.exists():
                return {"status": "error", "message": "Source file not found"}

            target_name = f"zeroday_{int(time.time())}_{p.name}"
            target_path = VAULT_DIR / target_name
            shutil.copy2(str(p), str(target_path))

            record = {
                "sample_id": f"zd_{int(time.time() * 1000)}",
                "filename": target_name,
                "local_path": str(target_path),
                "modality": modality,
                "uncertain_confidence": round(confidence, 4),
                "anomaly_reason": anomaly_reason,
                "timestamp": time.time(),
                "status": "pending_continual_training"
            }

            with open(self.queue_file, "r", encoding="utf-8") as f:
                queue = json.load(f)

            queue.append(record)

            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2, ensure_ascii=False)

            print(f"🚨 [Active Learning Vault] Enqueued hard sample: {target_name} (Confidence: {confidence:.2f}) -> {anomaly_reason}")
            return {"status": "enqueued", "record": record}
        except Exception as e:
            print(f"[Vault Error] Failed to enqueue sample: {e}")
            return {"status": "error", "message": str(e)}

    def get_pending_samples(self) -> List[Dict[str, Any]]:
        """Retrieves all pending adversarial samples awaiting training."""
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                queue = json.load(f)
            return [q for q in queue if q.get("status") == "pending_continual_training"]
        except Exception:
            return []

    def mark_samples_trained(self, sample_ids: List[str]):
        """Marks samples as trained after batch Colab fine-tuning."""
        try:
            with open(self.queue_file, "r", encoding="utf-8") as f:
                queue = json.load(f)

            for q in queue:
                if q.get("sample_id") in sample_ids:
                    q["status"] = "trained"
                    q["trained_at"] = time.time()

            with open(self.queue_file, "w", encoding="utf-8") as f:
                json.dump(queue, f, indent=2, ensure_ascii=False)
            print(f"✅ [Active Learning Vault] Marked {len(sample_ids)} samples as trained.")
        except Exception as e:
            print(f"[Vault Error] Failed to mark trained: {e}")
