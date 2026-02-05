import importlib.metadata
__version__ = importlib.metadata.version("legend_lar")

from .trainer import train

__all__ = [
    "train"
]