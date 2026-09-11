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
    
    # Model Settings
    DEEPFAKE_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2-Model"
    GEMINI_MODEL_NAME = "models/gemini-2.5-flash"
    
    # Agent Settings
    MAX_CONVERSATION_HISTORY = 10
    TEMPERATURE = 0.7
    
    # Logging
    LOG_LEVEL = "INFO"
    ENABLE_TRACING = True
    
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

