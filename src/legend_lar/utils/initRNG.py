import torch, math
import torch.nn as nn

class InitRNG:
    def __init__(self, device: str | None = None):
        self.device = device

    @torch.no_grad()
    def reinit_(self, model: nn.Module, seed: int):
        rng = torch.Generator(device=self.device if self.device is not None else "cpu")
        rng.manual_seed(seed)

        handled_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.Embedding,
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
        )
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=rng)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound, generator=rng)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0, generator=rng)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, "weight", None) is not None: nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)

            elif not isinstance(m, handled_modules):
                for name, p in m.named_parameters(recurse=False):
                    if p is None:
                        continue
                    if p.dim() >= 2:
                        nn.init.normal_(p, mean=0.0, std=1.0, generator=rng)
                    else:
                        nn.init.zeros_(p)
