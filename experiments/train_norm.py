import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DualSequenceDataset
from dual_attention.models.dual_classifier_norm import DualClassifierWithNorm
from dual_attention.models.dual_classifier import RealBaseline

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def train_epoch(model, loader, opt, crit, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits, y.long())
        loss.backward()
        opt.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, crit, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y.long())
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return total_loss / len(loader), correct / total


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3
    SEED = 42
    EMBED_DIM = 64
    NUM_HEADS = 4

    torch.manual_seed(SEED)

    train_ds = DualSequenceDataset(10000, seed=SEED)
    valid_ds = DualSequenceDataset(2000, seed=SEED + 1)
    test_ds = DualSequenceDataset(2000, seed=SEED + 2)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    model = DualClassifierWithNorm(32, 16, EMBED_DIM, NUM_HEADS).to(device)
    real_model = RealBaseline(32, 16, EMBED_DIM, NUM_HEADS).to(device)

    logger.info(f"Model with Norm params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    opt_real = torch.optim.Adam(real_model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    logger.info("Training Dual Model with Norm")
    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, opt, crit, device)
        _, val_acc = evaluate(model, valid_loader, crit, device)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "logs/best_dual_with_norm.pt")
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    logger.info("Training Real Baseline")
    best_real_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(real_model, train_loader, opt_real, crit, device)
        _, val_acc = evaluate(real_model, valid_loader, crit, device)
        if val_acc > best_real_acc:
            best_real_acc = val_acc
            torch.save(real_model.state_dict(), "logs/best_real.pt")
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    model.load_state_dict(torch.load("logs/best_dual_with_norm.pt"))
    real_model.load_state_dict(torch.load("logs/best_real.pt"))

    _, test_acc = evaluate(model, test_loader, crit, device)
    _, real_acc = evaluate(real_model, test_loader, crit, device)

    logger.info("FINAL RESULTS")
    logger.info(f"Majority Baseline: 50.00%")
    logger.info(f"Real Baseline: {real_acc * 100:.2f}%")
    logger.info(f"Dual Model with Norm: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
