import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DualSequenceDataset
from dual_attention.models.dual_classifier_norm_cls import DualClassifierWithNormClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


def freeze_dual_weights(model):
    frozen_count = 0
    for name, param in model.named_parameters():
        if 'W_d' in name or 'b_d' in name:
            param.data.zero_()
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"Frozen {frozen_count} b-component parameters")
    return frozen_count


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
    DROPOUT = 0.1
    WEIGHT_DECAY = 1e-5

    torch.manual_seed(SEED)

    train_ds = DualSequenceDataset(10000, seed=SEED)
    valid_ds = DualSequenceDataset(2000, seed=SEED + 1)
    test_ds = DualSequenceDataset(2000, seed=SEED + 2)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    logger.info("Training FULL model with Norm in Classifier (dropout + weight_decay)")
    full_model = DualClassifierWithNormClassifier(32, 16, EMBED_DIM, NUM_HEADS, DROPOUT).to(device)
    opt_full = torch.optim.Adam(full_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    best_full_acc = 0.0
    best_full_epoch = 0
    patience = 20
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(full_model, train_loader, opt_full, crit, device)
        _, val_acc = evaluate(full_model, valid_loader, crit, device)

        if val_acc > best_full_acc:
            best_full_acc = val_acc
            best_full_epoch = epoch
            torch.save(full_model.state_dict(), "logs/full_model_norm_cls.pt")
        else:
            if epoch - best_full_epoch > patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    full_model.load_state_dict(torch.load("logs/full_model_norm_cls.pt"))
    _, full_acc = evaluate(full_model, test_loader, crit, device)
    logger.info(f"Full model test accuracy: {full_acc * 100:.2f}%")

    logger.info("Training ABLATION model (b-component frozen)")
    ablation_model = DualClassifierWithNormClassifier(32, 16, EMBED_DIM, NUM_HEADS, DROPOUT).to(device)
    freeze_dual_weights(ablation_model)

    opt_ablation = torch.optim.Adam(
        [p for p in ablation_model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY
    )

    best_ablation_acc = 0.0
    best_ablation_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(ablation_model, train_loader, opt_ablation, crit, device)
        _, val_acc = evaluate(ablation_model, valid_loader, crit, device)

        if val_acc > best_ablation_acc:
            best_ablation_acc = val_acc
            best_ablation_epoch = epoch
            torch.save(ablation_model.state_dict(), "logs/ablation_model_norm_cls.pt")
        else:
            if epoch - best_ablation_epoch > patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    ablation_model.load_state_dict(torch.load("logs/ablation_model_norm_cls.pt"))
    _, ablation_acc = evaluate(ablation_model, test_loader, crit, device)
    logger.info(f"Ablation model (b=0) test accuracy: {ablation_acc * 100:.2f}%")

    diff = full_acc - ablation_acc
    logger.info("ABLATION RESULTS (Norm in Classifier)")
    logger.info(f"Full model:   {full_acc * 100:.2f}%")
    logger.info(f"Ablation (b=0): {ablation_acc * 100:.2f}%")
    logger.info(f"Difference: {diff * 100:.2f} pp")


if __name__ == "__main__":
    main()
