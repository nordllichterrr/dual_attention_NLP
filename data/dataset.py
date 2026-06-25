import numpy as np
import torch
from torch.utils.data import Dataset


class DualSequenceDataset(Dataset):
    def __init__(self, n_samples, vocab_size=32, seq_len=16, seed=42):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples

        rng = np.random.RandomState(seed)
        half = n_samples // 2

        pos = rng.randint(0, vocab_size, size=(half, seq_len))
        pos[:, 0] = pos[:, -1]

        neg = rng.randint(0, vocab_size, size=(half, seq_len))
        neg[:, 0] = rng.randint(0, vocab_size, size=half)
        for i in range(half):
            last = (neg[i, 0] + rng.randint(1, vocab_size)) % vocab_size
            neg[i, -1] = last

        data = np.vstack([pos, neg])
        labels = np.array([1] * half + [0] * half)

        idx = rng.permutation(n_samples)
        data = data[idx]
        labels = labels[idx]

        self.data = torch.tensor(data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
