"""Model registry for the plasticity experiments.

The notebooks and experiment runners import models from this package so the
architectures from Appendix A.2 are defined in one place and reused across
figures.
"""

from .cnn import CNN
from .mlp import MLP
from .vit import VisionTransformer

__all__ = ["CNN", "MLP", "VisionTransformer"]
