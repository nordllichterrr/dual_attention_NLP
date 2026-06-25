import torch
import torch.nn as nn

from dual_attention.core.tensor import DualTensor


def dual_norm(z: DualTensor) -> torch.Tensor:
    x, y = z.real, z.dual
    return torch.abs(y / 2) + torch.sqrt(x ** 2 + (y / 2) ** 2)


class DualLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, z: DualTensor) -> DualTensor:
        norm = dual_norm(z)
        mean = norm.mean(dim=-1, keepdim=True)
        var = ((norm - mean) ** 2).mean(dim=-1, keepdim=True)

        z_real_norm = (z.real - mean) / torch.sqrt(var + self.eps)
        z_dual_norm = (z.dual - mean) / torch.sqrt(var + self.eps)

        out_real = self.gamma * z_real_norm + self.beta
        out_dual = self.gamma * z_dual_norm + self.beta

        return DualTensor(out_real, out_dual)
