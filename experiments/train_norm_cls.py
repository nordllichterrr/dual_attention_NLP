import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DualSequenceDataset
from dual_attention.models.dual_classifier_norm_cls import DualClassifierWithNormClassifier
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
    WEIGHT_DECAY = 1e-4
    SEED = 42
    EMBED_DIM = 64
    NUM_HEADS = 4
    DROPOUT = 0.3

    torch.manual_seed(SEED)

    train_ds = DualSequenceDataset(10000, seed=SEED)
    valid_ds = DualSequenceDataset(2000, seed=SEED + 1)
    test_ds = DualSequenceDataset(2000, seed=SEED + 2)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    model = DualClassifierWithNormClassifier(32, 16, EMBED_DIM, NUM_HEADS, DROPOUT).to(device)
    real_model = RealBaseline(32, 16, EMBED_DIM, NUM_HEADS).to(device)

    logger.info(f"Model with Norm in Classifier params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_real = torch.optim.Adam(real_model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    logger.info("Training Dual Model with Norm in Classifier (dropout + weight_decay)")
    best_acc = 0.0
    best_epoch = 0
    patience = 20
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, opt, crit, device)
        _, val_acc = evaluate(model, valid_loader, crit, device)

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), "logs/best_dual_norm_cls.pt")
        else:
            if epoch - best_epoch > patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    logger.info("Training Real Baseline")
    best_real_acc = 0.0
    best_real_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(real_model, train_loader, opt_real, crit, device)
        _, val_acc = evaluate(real_model, valid_loader, crit, device)
        if val_acc > best_real_acc:
            best_real_acc = val_acc
            best_real_epoch = epoch
            torch.save(real_model.state_dict(), "logs/best_real.pt")
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    model.load_state_dict(torch.load("logs/best_dual_norm_cls.pt"))
    real_model.load_state_dict(torch.load("logs/best_real.pt"))

    _, test_acc = evaluate(model, test_loader, crit, device)
    _, real_acc = evaluate(real_model, test_loader, crit, device)

    logger.info("FINAL RESULTS")
    logger.info(f"Majority Baseline: 50.00%")
    logger.info(f"Real Baseline: {real_acc * 100:.2f}%")
    logger.info(f"Dual Model with Norm in Classifier: {test_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
