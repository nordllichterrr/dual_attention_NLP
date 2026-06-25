import torch
import torch.nn as nn

from dual_attention.core.tensor import DualTensor


class DualLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W_r = nn.Linear(in_features, out_features, bias=bias)
        self.W_d = nn.Linear(in_features, out_features, bias=False)

        if bias:
            self.b_r = nn.Parameter(torch.zeros(out_features))
            self.b_d = nn.Parameter(torch.zeros(out_features))
        else:
            self.b_r = None
            self.b_d = None

    def forward(self, z: DualTensor) -> DualTensor:
        out_r = self.W_r(z.real)
        if self.b_r is not None:
            out_r = out_r + self.b_r

        out_d = self.W_r(z.dual) + self.W_d(z.real)
        if self.b_d is not None:
            out_d = out_d + self.b_d

        return DualTensor(out_r, out_d)