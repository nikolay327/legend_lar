import torch, math
import torch.nn as nn

from legend_lar.model.tokenizer import ParallelContinuousEmbedder

class InitRNG:
    def __init__(self, seed: int, device: str | None = None):
        self.g = torch.Generator(device=device if device is not None else "cpu")
        self.g.manual_seed(seed)

    @torch.no_grad()
    def reinit_(self, model: nn.Module):
        handled_modules = (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d,
            nn.Embedding,
            nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
            ParallelContinuousEmbedder
        )
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, ParallelContinuousEmbedder)):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5), generator=self.g)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound, generator=self.g)

            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=1.0, generator=self.g)

            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if getattr(m, "weight", None) is not None: nn.init.ones_(m.weight)
                if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)

            elif not isinstance(m, handled_modules):
                for name, p in m.named_parameters(recurse=False):
                    if p is None:
                        continue
                    if p.dim() >= 2:
                        nn.init.normal_(p, mean=0.0, std=1.0, generator=self.g)
                    else:
                        nn.init.zeros_(p)
