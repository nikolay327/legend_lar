import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def _init_torch():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
    torch._dynamo.config.capture_scalar_outputs = True

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1
    
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device, init_method="env://")

    return local_rank, rank, world_size, device
