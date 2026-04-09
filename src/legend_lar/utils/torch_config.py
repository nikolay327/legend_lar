import os
from pathlib import Path
import torch
import torch.multiprocessing as mp

def _init_torch(cache_dir=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._dynamo.config.capture_scalar_outputs = True

    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
    else:
        rank = 0
        world_size = 1

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    device = torch.device("cuda:0") # is always 0 due to gpu-bind

    if cache_dir is not None:
        BASE = cache_dir
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{BASE}/inductor")
        os.environ.setdefault("TRITON_CACHE_DIR", f"{BASE}/triton/rank_{rank}")
        os.environ.setdefault("NUMBA_CACHE_DIR", f"{BASE}/numba/rank_{rank}")

        Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["TRITON_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
        Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    return rank, world_size, device
