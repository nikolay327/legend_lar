import os
import shutil
import numpy as np
import scipy as sp
from scipy.sparse import load_npz
from lgdo import lh5

from .db import FileDB
from .configs import NRECConfig, _initialize_configs

def create_base_dataset(
    partition: str,
    file_db: FileDB,
    data_config: dict,
    include_rc_from_ge_trigger: bool = True
):
    # bg training data
    geds_data_phy = file_db.build_file(
        tier="dataset",
        partition=partition,
        filename="geds_data_phy.lh5"
    )

    geds_data_phy = lh5.read_as("/geds", geds_data_phy, field_mask=["id", "energy", "drift_time", "aoe", "lq"], library="pd")
    geds_data_phy = geds_data_phy[["id", "energy", "drift_time", "aoe", "lq"]].to_numpy().astype(np.float32)
    path = file_db.build_file(
        tier="training",
        partition=partition,
        version="base",
        filename="geds_data_phy.npy"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, geds_data_phy)
    geds_data_phy = None

    sipm_data_sparse_phy = file_db.build_file(
        tier="dataset",
        partition=partition,
        filename="sipm_data_sparse_phy.npz"
    )
    shutil.copy(
        sipm_data_sparse_phy,
        file_db.build_file(
            tier="training",
            partition=partition,
            version="base",
            filename="sipm_data_sparse_phy.npz"
        )
    )

    path = file_db.build_file(
        tier="dataset",
        partition=partition,
        filename="lib_phy.lh5"
    )
    phy_classical_classifier = lh5.read_as("phy/coincident/spms", path, library="np").astype(bool)
    mask = lh5.read_as("phy/spms/energy_sum", path, library="np")
    mask = (mask > 0) & (mask < 500)
    phy_classical_classifier = phy_classical_classifier[mask]
    np.save(
        file_db.build_file(
            tier="training",
            partition=partition,
            version="base",
            filename="classical_classifier_phy.npy"
        ), phy_classical_classifier
    )

    # sg training, coverage, and calibration data
    rng = np.random.default_rng(seed=data_config["rng_seed_calib_selection"])
    sipm_data_sparse_fp = file_db.build_file(
        tier="dataset",
        partition=partition,
        filename="sipm_data_sparse_fp.npz"
    )
    sipm_data_sparse_fp = load_npz(sipm_data_sparse_fp)

    indices = rng.permutation(sipm_data_sparse_fp.shape[0])

    num_rc_coverage_data = data_config["num_rc_coverage_data"]
    num_calib_data = data_config["num_calib_data"]

    n_reserved = num_rc_coverage_data + 2 * num_calib_data
    if sipm_data_sparse_fp.shape[0] < n_reserved:
        raise ValueError(
            "Not enough FP RC data for coverage + calibration: "
            f"have {sipm_data_sparse_fp.shape[0]}, need {n_reserved} "
            f"= num_rc_coverage_data({num_rc_coverage_data}) "
            f"+ 2*num_calib_data({2 * num_calib_data})"
        )

    # First reserve RC coverage/test data.
    coverage_indices = indices[:num_rc_coverage_data]

    # Then reserve calibration data.
    calibration_indices = indices[
        num_rc_coverage_data : num_rc_coverage_data + 2 * num_calib_data
    ]

    # Everything left is used for training.
    training_indices = indices[num_rc_coverage_data + 2 * num_calib_data :]

    path = file_db.build_file(
        tier="dataset",
        partition=partition,
        filename="lib_rc_fp.lh5"
    )
    sipm_fp_classical_classifier = lh5.read_as(
        "evt/coincident/spms", path, library="np"
    ).astype(bool)
    mask = lh5.read_as("evt/spms/energy_sum", path, library="np")
    mask = (mask > 0) & (mask < 500)
    sipm_fp_classical_classifier = sipm_fp_classical_classifier[mask]

    # ------------------------------------------------------------------
    # coverage / test data
    # ------------------------------------------------------------------
    sipm_data_sparse_coverage = sipm_data_sparse_fp[coverage_indices]
    sipm_data_sparse_coverage_classical_classifier = (
        sipm_fp_classical_classifier[coverage_indices]
    )

    path = file_db.build_file(
        tier="coverage_dataset",
        partition=partition,
        version="base",
        filename="sipm_data_sparse_rc_coverage.npz"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sp.sparse.save_npz(path, sipm_data_sparse_coverage)

    np.save(
        file_db.build_file(
            tier="coverage_dataset",
            partition=partition,
            version="base",
            filename="classical_classifier_rc_coverage.npy"
        ),
        sipm_data_sparse_coverage_classical_classifier
    )

    np.save(
        file_db.build_file(
            tier="coverage_dataset",
            partition=partition,
            version="base",
            filename="indices_rc_coverage.npy"
        ),
        coverage_indices
    )

    # ------------------------------------------------------------------
    # calibration data
    # ------------------------------------------------------------------
    sipm_data_sparse_calibration = sipm_data_sparse_fp[calibration_indices]
    sipm_data_sparse_calibration_classical_classifier = (
        sipm_fp_classical_classifier[calibration_indices]
    )

    path = file_db.build_file(
        tier="inference_dataset",
        partition=partition,
        version="base",
        filename="sipm_data_sparse_rc_ev_ep.npz"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sp.sparse.save_npz(
        path,
        sipm_data_sparse_calibration[:num_calib_data]
    )

    np.save(
        file_db.build_file(
            tier="inference_dataset",
            partition=partition,
            version="base",
            filename="classical_classifier_rc_ev_ep.npy"
        ),
        sipm_data_sparse_calibration_classical_classifier[:num_calib_data]
    )

    np.save(
        file_db.build_file(
            tier="inference_dataset",
            partition=partition,
            version="base",
            filename="indices_rc_ev_ep.npy"
        ),
        calibration_indices[:num_calib_data]
    )

    path = file_db.build_file(
        tier="inference_dataset",
        partition=partition,
        version="base",
        filename="sipm_data_sparse_glob.npz"
    )
    sp.sparse.save_npz(
        path,
        sipm_data_sparse_calibration[num_calib_data:]
    )

    np.save(
        file_db.build_file(
            tier="inference_dataset",
            partition=partition,
            version="base",
            filename="classical_classifier_glob.npy"
        ),
        sipm_data_sparse_calibration_classical_classifier[num_calib_data:]
    )

    np.save(
        file_db.build_file(
            tier="inference_dataset",
            partition=partition,
            version="base",
            filename="indices_glob.npy"
        ),
        calibration_indices[num_calib_data:]
    )

    # ------------------------------------------------------------------
    # training data
    # ------------------------------------------------------------------
    sipm_data_sparse_fp = sipm_data_sparse_fp[training_indices]
    sipm_fp_classical_classifier = sipm_fp_classical_classifier[training_indices]

    if include_rc_from_ge_trigger:
        # include rc from ge trigger
        sipm_data_sparse_ge = file_db.build_file(
            tier="dataset",
            partition=partition,
            filename="sipm_data_sparse_ge.npz"
        )
        sipm_data_sparse_ge = load_npz(sipm_data_sparse_ge)
        sipm_data_sparse_rc = sp.sparse.vstack([sipm_data_sparse_fp, sipm_data_sparse_ge], format="csr")
    else:
        sipm_data_sparse_rc = sipm_data_sparse_fp

    path = file_db.build_file(
        tier="training",
        partition=partition,
        version="base",
        filename="sipm_data_sparse_rc.npz"
    )
    sp.sparse.save_npz(path, sipm_data_sparse_rc)

    if include_rc_from_ge_trigger:
        path = file_db.build_file(
            tier="dataset",
            partition=partition,
            filename="lib_rc_ge.lh5"
        )
        sipm_rc_ge_classical_classifier = lh5.read_as("evt/coincident/spms", path, library="np").astype(bool)
        mask = lh5.read_as("evt/spms/energy_sum", path, library="np")
        mask = (mask > 0) & (mask < 500)
        sipm_rc_ge_classical_classifier = sipm_rc_ge_classical_classifier[mask]

        sipm_rc_classical_classifier = np.concatenate([sipm_fp_classical_classifier, sipm_rc_ge_classical_classifier], axis=0)
    else:
        sipm_rc_classical_classifier = sipm_fp_classical_classifier

    np.save(
        file_db.build_file(
            tier="inference_dataset",
            partition=partition,
            version="base",
            filename="classical_classifier_rc.npy"
        ), sipm_rc_classical_classifier
    )
    np.save(
        file_db.build_file(
            tier="inference_dataset",
            partition=partition,
            version="base",
            filename="indices_rc.npy"
        ), training_indices
    )

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
        partition=args.partition,
        file_db=file_db,
        data_config=data_config,
        include_rc_from_ge_trigger=data_config["include_rc_from_ge_trigger"]
    )
