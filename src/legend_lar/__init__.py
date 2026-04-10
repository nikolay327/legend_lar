import importlib.metadata
__version__ = importlib.metadata.version("legend_lar")

from pathlib import Path
__path__.append(str(Path(__file__).resolve().parent / "kfold_ensemble"))
__path__.append(str(Path(__file__).resolve().parent / "utils"))
__path__.append(str(Path(__file__).resolve().parent / "calibration"))

from .kfold_ensemble import nre_c
from .utils import create_base_dataset
from .calibration import nre_c_inference

__all__ = [
    "create_base_dataset",
    "nre_c",
    "nre_c_inference"
]
