import importlib.metadata
__version__ = importlib.metadata.version("legend_lar")

from .kfold_ensemble import nre_c
from .utils import create_base_dataset

__all__ = [
    "create_base_dataset",
    "nre_c"
]
