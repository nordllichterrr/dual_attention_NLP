import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dual_attention.core.tensor import DualTensor


def test_multiplication():
    a = DualTensor(torch.tensor(2.0), torch.tensor(3.0))
    b = DualTensor(torch.tensor(4.0), torch.tensor(5.0))
    c = a * b
    assert torch.allclose(c.real, torch.tensor(8.0))
    assert torch.allclose(c.dual, torch.tensor(22.0))
    print("✓ Multiplication OK")


def test_matmul():
    A = DualTensor(torch.ones(2, 2), torch.zeros(2, 2))
    B = DualTensor(torch.ones(2, 2), torch.ones(2, 2))
    C = A @ B
    assert torch.allclose(C.real, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
    assert torch.allclose(C.dual, torch.tensor([[2.0, 2.0], [2.0, 2.0]]))
    print("✓ Matmul OK")


if __name__ == "__main__":
    test_multiplication()
    test_matmul()
