"""
Generators Package for DeepFake Detection Agent
Contains FaceSwap Engine (InsightFace/SimSwap/InstantID), Open-Weight Generative AI Engine (Flux.1-dev, SDXL, Wan2.1),
Commercial Frontier Engine (Google Imagen 3, DALL-E 3, Meta Imagine), and Visual Autoregressive Engine (DeepSeek Janus-Pro, Chameleon).
"""
from generators.faceswap_engine import FaceSwapEngine
from generators.openweight_gen_engine import OpenWeightGenerativeEngine
from generators.commercial_frontier_engine import CommercialFrontierGenerativeEngine
from generators.autoregressive_engine import AutoregressiveGenerativeEngine

__all__ = [
    "FaceSwapEngine",
    "OpenWeightGenerativeEngine",
    "CommercialFrontierGenerativeEngine",
    "AutoregressiveGenerativeEngine"
]
