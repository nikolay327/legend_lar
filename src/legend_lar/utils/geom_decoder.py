import yaml

import numpy as np
import pandas as pd

import torch

from .configs import NRECConfig

def decode_geom(
    cfg_yaml: str,
    config: NRECConfig
):
    with open(cfg_yaml, "r") as f:
        cfg_yaml = yaml.safe_load(f)
    cfg_yaml = pd.json_normalize(cfg_yaml.values()).sort_values("id", ascending=True, ignore_index=True)

    cfg_yaml["r"] = cfg_yaml["r"].map(lambda x: (x - config.r_shift) / config.r_inv_scale)
    cfg_yaml["phi"] = cfg_yaml["phi"].map(lambda x: x * np.pi / 180)
    cfg_yaml["z"] = cfg_yaml["z"].map(lambda x: (x - config.z_shift) / config.z_inv_scale)

    hpge_detector_coords = cfg_yaml[cfg_yaml["id"] < config.num_hpges][["r", "phi", "z"]].to_numpy()
    lar_detector_coords = cfg_yaml[cfg_yaml["id"] >= config.num_hpges][["r", "phi", "z"]].to_numpy()
    hpge_detector_coords = torch.from_numpy(hpge_detector_coords).float()
    lar_detector_coords = torch.from_numpy(lar_detector_coords).float()

    return lar_detector_coords, hpge_detector_coords
