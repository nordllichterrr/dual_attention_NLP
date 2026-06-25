import torch.nn as nn
from dual_attention.core.tensor import DualTensor


class DualAvgPool(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, z: DualTensor) -> DualTensor:
        return DualTensor(
            z.real.mean(dim=self.dim),
            z.dual.mean(dim=self.dim)
        )
