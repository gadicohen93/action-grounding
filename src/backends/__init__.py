"""
Model backends for the action-grounding research project.

This module provides a unified interface for interacting with language models,
abstracting away the differences between PyTorch/Transformers and vLLM.
"""

from .base import ModelBackend, GenerationOutput
from .pytorch import PyTorchBackend

__all__ = ["ModelBackend", "GenerationOutput", "PyTorchBackend"]


def get_backend(backend_type: str = "pytorch") -> type[ModelBackend]:
    """
    Get the backend class by name.

    Args:
        backend_type: "pytorch" or "vllm"

    Returns:
        Backend class (not instance)
    """
    backends = {
        "pytorch": PyTorchBackend,
    }

    if backend_type not in backends:
        raise ValueError(
            f"Unknown backend: {backend_type}. Available: {list(backends.keys())}"
        )

    return backends[backend_type]
