from dataclasses import dataclass
from typing import Union
import torch
from torch import Tensor


@dataclass
class DualTensor:
    real: Tensor
    dual: Tensor

    def __post_init__(self):
        if self.real.shape != self.dual.shape:
            raise ValueError(f"Shape mismatch: {self.real.shape} vs {self.dual.shape}")

    @property
    def device(self):
        return self.real.device

    def to(self, device: Union[str, torch.device]):
        return DualTensor(self.real.to(device), self.dual.to(device))

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(self.real + other, self.dual)
        return DualTensor(self.real + other.real, self.dual + other.dual)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(self.real + other, self.dual)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(self.real - other, self.dual)
        return DualTensor(self.real - other.real, self.dual - other.dual)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(other - self.real, -self.dual)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(self.real * other, self.dual * other)
        return DualTensor(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(self.real * other, self.dual * other)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return DualTensor(self.real / other, self.dual / other)
        raise TypeError(f"Division by {type(other)} not supported")

    def __matmul__(self, other):
        return DualTensor(
            self.real @ other.real,
            self.real @ other.dual + self.dual @ other.real
        )

    def transpose(self, dim0, dim1):
        return DualTensor(
            self.real.transpose(dim0, dim1),
            self.dual.transpose(dim0, dim1)
        )

    def mean(self, dim=None, keepdim=False):
        return DualTensor(
            self.real.mean(dim=dim, keepdim=keepdim),
            self.dual.mean(dim=dim, keepdim=keepdim)
        )

    def unsqueeze(self, dim):
        return DualTensor(
            self.real.unsqueeze(dim),
            self.dual.unsqueeze(dim)
        )

    def squeeze(self, dim=None):
        return DualTensor(
            self.real.squeeze(dim=dim),
            self.dual.squeeze(dim=dim)
        )

    @staticmethod
    def from_real(x: Tensor):
        return DualTensor(x, torch.zeros_like(x))

    def clone(self):
        return DualTensor(self.real.clone(), self.dual.clone())

    def __repr__(self):
        return f"DualTensor(real={self.real}, dual={self.dual})"
