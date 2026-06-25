import torch
import torch.nn as nn

from dual_attention.core.tensor import DualTensor
from dual_attention.core.functions import dual_softmax
from dual_attention.layers.linear import DualLinear


class DualSelfAttention(nn.Module):
    # Многоголовое self-attention
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim должен делиться на num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = DualLinear(embed_dim, embed_dim)
        self.W_k = DualLinear(embed_dim, embed_dim)
        self.W_v = DualLinear(embed_dim, embed_dim)
        self.W_o = DualLinear(embed_dim, embed_dim)

    def forward(self, x: DualTensor) -> DualTensor:
        B, L, D = x.real.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        # Разбиваем на головы
        q_real = q.real.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        q_dual = q.dual.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_real = k.real.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k_dual = k.dual.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_real = v.real.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v_dual = v.dual.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        q = DualTensor(q_real, q_dual)
        k = DualTensor(k_real, k_dual)
        v = DualTensor(v_real, v_dual)

        # Scaled dot-product attention
        logits = (q @ k.transpose(-2, -1)) * self.scale

        attn_real, attn_dual = dual_softmax(logits)
        attn = DualTensor(attn_real, attn_dual)

        out = attn @ v

        # Объединяем головы
        out_real = out.real.transpose(1, 2).contiguous().view(B, L, D)
        out_dual = out.dual.transpose(1, 2).contiguous().view(B, L, D)
        out = DualTensor(out_real, out_dual)

        out = self.W_o(out)

        return out