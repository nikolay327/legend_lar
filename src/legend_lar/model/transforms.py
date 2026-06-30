import torch
import torch.nn as nn


class AsinhTransform(nn.Module):
    """
    Elementwise signed asinh transform.

        y = asinh(x / scale)
        x = scale * sinh(y)

    scale can be scalar or per-feature.
    """

    def __init__(self, scale=1.0, dtype=torch.float32):
        super().__init__()
        scale = torch.as_tensor(scale, dtype=dtype)

        if torch.any(scale <= 0):
            raise ValueError("scale must be positive.")

        self.register_buffer("scale", scale)

    def forward(self, x):
        return torch.asinh(x / self.scale)

    def inverse(self, y):
        return self.scale * torch.sinh(y)

    def log_abs_det_jacobian(self, x):
        """
        dy/dx = 1 / sqrt(x^2 + scale^2)
        """
        return -0.5 * torch.log(x * x + self.scale * self.scale)
