"""
Configuration module for the Deepfake Detection Agent System.
Handles API keys, model settings, and agent parameters.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.6-flash")
    CLAUDE_MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-sonnet-20241022")
    
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

