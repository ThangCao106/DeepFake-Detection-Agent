"""
Configuration module for the Deepfake Detection Agent System.
Handles API keys, model settings, and agent parameters.
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def _recover_persistent_keys():
    candidates = [
        Path("/app/report_colab/.credentials/forensic_env.json"),
        Path(__file__).resolve().parent.parent / "report_colab" / ".credentials" / "forensic_env.json",
        Path("/home/agent/.credentials/forensic_env.json")
    ]
    for p in candidates:
        try:
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if v and not os.environ.get(k):
                        os.environ[k] = str(v).strip()
        except Exception:
            pass

_recover_persistent_keys()

class Config:
    """Configuration class for the agent system."""
    
    # API Keys
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))
    OBSERVABILITY_AGENT_URL = os.getenv("OBSERVABILITY_AGENT_URL", "http://127.0.0.1:8787")
    
    # Model Settings (Open-source Hugging Face Baselines & Multi-Modal Backbones)
    DEEPFAKE_MODEL_NAME_VIT = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    DEEPFAKE_MODEL_NAME_SIGLIP = "prithivMLmods/deepfake-detector-model-v1"
    DEEPFAKE_MODEL_NAME = os.getenv("DEEPFAKE_MODEL_NAME", DEEPFAKE_MODEL_NAME_VIT)
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.8-flash")
    CLAUDE_MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "claude-sonnet-5.5")
    
    # Logging
    LOG_LEVEL = "INFO"
    ENABLE_TRACING = True
    
    # Session & Memory Settings
    MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "20"))
    
    # Forensic Analysis Parameters
    FORENSIC_VERSION = "2.0-AUDITABLE"
    HIGH_FREQ_ANOMALY_THRESHOLD = 0.45
    SPATIAL_NOISE_DIFF_THRESHOLD = 0.15
    ENABLE_AUDITABLE_CERTIFICATE = True

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found. Please set it in .env file. "
                "Get your API key from: https://aistudio.google.com/app/apikey"
            )
        return True

