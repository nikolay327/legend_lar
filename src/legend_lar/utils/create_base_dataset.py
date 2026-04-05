import shutil
import numpy as np
import scipy as sp
from scipy.sparse import load_npz
from lgdo import lh5

from .db import FileDB
from .configs import NRECConfig, _initialize_configs

def create_base_dataset(
    file_db: FileDB,
    data_config: dict
):
    # bg training data
    geds_data_phy = file_db.build_file(
        tier="dataset",
        partition="p16",
        filename="geds_data_phy.lh5"
    )

    geds_data_phy = lh5.read_as("/geds", geds_data_phy, field_mask=["id", "energy", "drift_time", "aoe", "lq"], library="pd")
    geds_data_phy = geds_data_phy[["id", "energy", "drift_time", "aoe", "lq"]].to_numpy().astype(np.float32)
    np.save(
        file_db.build_file(
            tier="training",
            partition="p16",
            filename="geds_data_phy.npy"
        ), geds_data_phy
    )
    geds_data_phy = None

    sipm_data_sparse_phy = file_db.build_file(
        tier="dataset",
        partition="p16",
        filename="sipm_data_sparse_phy.npz"
    )
    shutil.copy(
        sipm_data_sparse_phy,
        file_db.build_file(
            tier="training",
            partition="p16",
            filename="sipm_data_sparse_phy.npz"
        )
    )

    # sg training and calibration data
    rng = np.random.default_rng(seed=data_config["rng_seed_calib_selection"])
    sipm_data_sparse_fp = file_db.build_file(
        tier="dataset",
        partition="p16",
        filename="sipm_data_sparse_phy.npz"
    )
    sipm_data_sparse_fp = load_npz(sipm_data_sparse_fp)
    indices = rng.permutation(sipm_data_sparse_fp.shape[0])

    calibration_indices = indices[: 2 * data_config["num_calib_data"]] # factor 2 because of additional global calibration
    training_indices = indices[2 * data_config["num_calib_data"]: ]
    
    # calibration
    sipm_data_sparse_calibration = sipm_data_sparse_fp[calibration_indices]
    path = file_db.build_file(
        tier="inference",
        partition="p16",
        filename="sipm_data_sparse_rc_ev_ep.npz"
    )
    sp.sparse.save_npz(path, sipm_data_sparse_calibration[: data_config["num_calib_data"]])

    path = file_db.build_file(
        tier="inference",
        partition="p16",
        filename="sipm_data_sparse_glob.npz"
    )
    sp.sparse.save_npz(path, sipm_data_sparse_calibration[data_config["num_calib_data"]:])

    # training
    sipm_data_sparse_fp = sipm_data_sparse_fp[training_indices]
    # include rc from ge trigger
    sipm_data_sparse_ge = file_db.build_file(
        tier="dataset",
        partition="p16",
        filename="sipm_data_sparse_ge.npz"
    )
    sipm_data_sparse_ge = load_npz(sipm_data_sparse_ge)
    sipm_data_sparse_rc = sp.sparse.vstack([sipm_data_sparse_fp, sipm_data_sparse_ge], format="csr")

    path = file_db.build_file(
        tier="training",
        partition="p16",
        filename="sipm_data_sparse_rc.npz"
    )
    sp.sparse.save_npz(path, sipm_data_sparse_rc)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Creating base dataset")
    parser.add_argument("experiment", type=str, help="Name of the experiment")
    parser.add_argument("partition", type=str, help="partition name")
    parser.add_argument("model_name", type=str, help="Name of the model")
    parser.add_argument("version", type=str, help="Model version")
    parser.add_argument("dataflow_dir", type=str, help="Directory of the dataflow")
    parser.add_argument("base_cfg_name", type=str, help="Name of the base json config")

    args = parser.parse_args()

    file_db = FileDB(
        working_dir=args.dataflow_dir,
        experiment=args.experiment
    )

    base_cfg = file_db.build_file(
        tier="base_configs",
        filename=args.base_cfg_name
    )

    model_cfg = file_db.build_file(
        tier="model_config",
        partition=args.partition,
        model_name=args.model_name,
        version=args.version
    )

    model_cfg, data_config = _initialize_configs(
        config_obj=NRECConfig(),
        config_path=model_cfg,
        base_config=base_cfg
    )

    create_base_dataset(
        file_db=file_db,
        data_config=data_config
    )
