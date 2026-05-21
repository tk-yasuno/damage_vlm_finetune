"""Visionモジュール"""

from .llama_cpp_vision import LlamaCppVisionAnalyzer, LlamaCppVisionConfig
from .ollama_vision import OllamaVisionAnalyzer

# HuggingFace版（オプショナル）
try:
    from .granite_vision import GraniteVisionAnalyzer, VisionConfig
    __all__ = ['LlamaCppVisionAnalyzer', 'LlamaCppVisionConfig',
               'OllamaVisionAnalyzer',
               'GraniteVisionAnalyzer', 'VisionConfig']
except ImportError:
    __all__ = ['LlamaCppVisionAnalyzer', 'LlamaCppVisionConfig',
               'OllamaVisionAnalyzer']
