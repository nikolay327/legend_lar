import importlib.metadata
__version__ = importlib.metadata.version("legend_lar")

from .bce_trainer import train_bce
from .conditional_trainer import train_conditional
from .unconditional_trainer import train_unconditional

__all__ = [
    "train_bce",
    "train_conditional",
    "train_unconditional"
]