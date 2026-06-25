import torch
import torch.nn as nn

from dual_attention.core.tensor import DualTensor
from dual_attention.core.norm import dual_norm
from dual_attention.layers.attention import DualSelfAttention
from dual_attention.layers.pooling import DualAvgPool


class DualClassifierWithNormClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=64, num_heads=4, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = DualSelfAttention(embed_dim, num_heads=num_heads)
        self.pool = DualAvgPool(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.embed(x)
        z = DualTensor.from_real(emb)
        z = self.attn(z)
        z = self.pool(z)
        norm = dual_norm(z)
        norm = self.dropout(norm)
        logits = self.fc(norm)
        return logits
