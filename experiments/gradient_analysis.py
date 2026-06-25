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
from dual_attention.models.dual_classifier import DualClassifier

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
    SEED = 42
    EMBED_DIM = 64
    NUM_HEADS = 4

    torch.manual_seed(SEED)

    train_ds = DualSequenceDataset(5000, seed=SEED)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)

    model = DualClassifier(32, 16, EMBED_DIM, NUM_HEADS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    grad_history = []

    logger.info("Collecting gradient norms...")

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
    np.save("logs/gradient_history.npy", grad_history)
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
    plt.title('Gradient norms by component')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    ratio = grad_history[:, 1] / (grad_history[:, 0] + 1e-8)
    plt.plot(ratio, color='green', linewidth=1.5)
    plt.axhline(y=0.01, color='red', linestyle='--', label='threshold 0.01')
    plt.axhline(y=0.1, color='orange', linestyle='--', label='threshold 0.1')
    plt.xlabel('Training step')
    plt.ylabel('||∇b|| / ||∇a||')
    plt.legend()
    plt.title('Gradient norm ratio')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/figures/gradient_norms.png', dpi=150)
    logger.info("Plot saved to reports/figures/gradient_norms.png")
    plt.close()

    final_ratio = grad_history[-1, 1] / (grad_history[-1, 0] + 1e-8)
    mean_ratio = np.mean(grad_history[:, 1] / (grad_history[:, 0] + 1e-8))

    logger.info("GRADIENT ANALYSIS RESULTS")
    logger.info(f"Mean ||∇b||/||∇a||: {mean_ratio:.4f}")
    logger.info(f"Final ratio: {final_ratio:.4f}")

    if mean_ratio < 0.01:
        logger.info("HYPOTHESIS CONFIRMED: gradients for b are negligible")
    elif mean_ratio < 0.1:
        logger.info("b-component learns but weakly (gradients 10x smaller than a)")
    else:
        logger.info("b-component learns and may contribute")


if __name__ == "__main__":
    collect_gradients()
