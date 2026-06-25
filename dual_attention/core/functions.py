import torch

from dual_attention.core.tensor import DualTensor


def dual_softmax(logits: DualTensor):
    # Обычный softmax от действительной части
    s = torch.softmax(logits.real, dim=-1)

    # Якобиан softmax: J = diag(s) - s·s^T
    diag = torch.diag_embed(s)
    outer = s.unsqueeze(-1) @ s.unsqueeze(-2)
    jac = diag - outer

    # Дуальная часть: J * Sb
    s_dual = (jac @ logits.dual.unsqueeze(-1)).squeeze(-1)

    return s, s_dual