import torch
import torch.nn as nn
from dual_attention.core.tensor import DualTensor
from dual_attention.layers.attention import DualSelfAttention
from dual_attention.layers.pooling import DualAvgPool


class DualClassifier(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = DualSelfAttention(embed_dim, num_heads=num_heads)
        self.pool = DualAvgPool(dim=1)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.embed(x)
        z = DualTensor.from_real(emb)
        z = self.attn(z)
        z = self.pool(z)
        logits = self.fc(z.real)
        return logits


class RealBaseline(nn.Module):
    def __init__(self, vocab_size, seq_len, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.embed(x)
        out, _ = self.attn(emb, emb, emb)
        pooled = out.mean(dim=1)
        logits = self.fc(pooled)
        return logits
