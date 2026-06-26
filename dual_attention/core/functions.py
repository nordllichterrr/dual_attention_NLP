import torch
from dual_attention.core.tensor import DualTensor


def dual_softmax(logits: DualTensor):
    s = torch.softmax(logits.real, dim=-1)
    diag = torch.diag_embed(s)
    outer = s.unsqueeze(-1) @ s.unsqueeze(-2)
    jac = diag - outer
    s_dual = (jac @ logits.dual.unsqueeze(-1)).squeeze(-1)
    return s, s_dual
