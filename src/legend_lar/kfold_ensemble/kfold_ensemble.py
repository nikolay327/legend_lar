from legend_lar.kfold_ensemble.unconditional import train_unconditional
from legend_lar.kfold_ensemble.conditional import train_conditional

def train_kfold_ensemble(
    experiment: str,
    model_name: str,
    working_dir: str,
    data_dir: str,
    unconditional_base_config: str,
    conditional_base_config: str
):
    train_unconditional(
        experiment=experiment,
        model_name=model_name,
        version="unconditional",
        working_dir=working_dir,
        data_dir=data_dir,
        training_config=unconditional_base_config
    )

    train_conditional(
        experiment=experiment,
        model_name=model_name,
        version="conditional",
        working_dir=working_dir,
        data_dir=data_dir,
        training_config=conditional_base_config
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Training started")
    parser.add_argument("experiment", type=str)
    parser.add_argument("model_name", type=str, help="Name of the model being trained")
    parser.add_argument("working_dir", type=str, help="Top-most dir of the training pipeline")
    parser.add_argument("data_dir", type=str, help="Directory the training data is saved under")
    parser.add_argument("unconditional_base_config", type=str)
    parser.add_argument("conditional_base_config", type=str)
    parser.add_argument("cache_dir", type=str, help="Directory to store torch.inductor and triton cache")
    args = parser.parse_args()

    import os
    from pathlib import Path

    BASE = args.cache_dir
    rank = os.environ["RANK"]
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{BASE}/inductor")
    os.environ.setdefault("TRITON_CACHE_DIR", f"{BASE}/triton/rank_{rank}")
    os.environ.setdefault("NUMBA_CACHE_DIR", f"{BASE}/numba/rank_{rank}")

    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    JOB_SHM_DIR = os.environ["JOB_SHMTMPDIR"] if "JOB_SHMTMPDIR" in os.environ else None

    train_kfold_ensemble(args.experiment, args.model_name, args.working_dir, args.data_dir, args.unconditional_base_config, args.conditional_base_config)
