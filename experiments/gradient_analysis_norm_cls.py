import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DualSequenceDataset
from dual_attention.models.dual_classifier_norm_cls import DualClassifierWithNormClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def compute_gradient_norms(model):
    grad_norm_a = 0.0
    grad_norm_b = 0.0

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if 'W_r' in name or 'b_r' in name:
            grad_norm_a += param.grad.norm().item() ** 2
        elif 'W_d' in name or 'b_d' in name:
            grad_norm_b += param.grad.norm().item() ** 2

    return np.sqrt(grad_norm_a), np.sqrt(grad_norm_b)


def collect_gradients():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 50
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    SEED = 42
    EMBED_DIM = 64
    NUM_HEADS = 4
    DROPOUT = 0.3

    torch.manual_seed(SEED)

    train_ds = DualSequenceDataset(5000, seed=SEED)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)

    model = DualClassifierWithNormClassifier(32, 16, EMBED_DIM, NUM_HEADS, DROPOUT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    grad_history = []

    logger.info("Collecting gradient norms with Norm in Classifier...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y.long())
            loss.backward()

            grad_a, grad_b = compute_gradient_norms(model)
            grad_history.append((grad_a, grad_b))

            opt.step()

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: ||∇a||={grad_a:.4f}, ||∇b||={grad_b:.4f}")

    grad_history = np.array(grad_history)
    np.save("logs/gradient_history_norm_cls.npy", grad_history)
    logger.info(f"Collected {len(grad_history)} gradient steps")

    Path("reports/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(grad_history[:, 0], label='||∇a||', alpha=0.7, linewidth=1.5)
    plt.plot(grad_history[:, 1], label='||∇b||', alpha=0.7, linewidth=1.5)
    plt.yscale('log')
    plt.xlabel('Training step')
    plt.ylabel('Gradient norm (log scale)')
    plt.legend()
    plt.title('Gradient norms (Norm in Classifier)')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    ratio = grad_history[:, 1] / (grad_history[:, 0] + 1e-8)
    plt.plot(ratio, color='green', linewidth=1.5)
    plt.axhline(y=0.01, color='red', linestyle='--', label='threshold 0.01')
    plt.axhline(y=0.1, color='orange', linestyle='--', label='threshold 0.1')
    plt.xlabel('Training step')
    plt.ylabel('||∇b|| / ||∇a||')
    plt.legend()
    plt.title('Gradient norm ratio (Norm in Classifier)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/gradient_norms_norm_cls.png', dpi=150)
    logger.info("Plot saved to reports/figures/gradient_norms_norm_cls.png")
    plt.close()

    final_ratio = grad_history[-1, 1] / (grad_history[-1, 0] + 1e-8)
    mean_ratio = np.mean(grad_history[:, 1] / (grad_history[:, 0] + 1e-8))

    logger.info("GRADIENT ANALYSIS RESULTS (Norm in Classifier)")
    logger.info(f"Mean ||∇b||/||∇a||: {mean_ratio:.4f}")
    logger.info(f"Final ratio: {final_ratio:.4f}")

    if mean_ratio > 0.01:
        logger.info("b-component receives gradients and can contribute")
    else:
        logger.info("b-component does not receive meaningful gradients")


if __name__ == "__main__":
    collect_gradients()
