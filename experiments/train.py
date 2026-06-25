import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import DualSequenceDataset
from dual_attention.models.dual_classifier import DualClassifier, RealBaseline

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

    # Гиперпараметры
    BATCH_SIZE = 64
    EPOCHS = 100
    LR = 1e-3
    SEED = 42
    EMBED_DIM = 64
    NUM_HEADS = 4  

    torch.manual_seed(SEED)

    # Данные
    train_ds = DualSequenceDataset(10000, seed=SEED)
    valid_ds = DualSequenceDataset(2000, seed=SEED + 1)
    test_ds = DualSequenceDataset(2000, seed=SEED + 2)

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, BATCH_SIZE)
    test_loader = DataLoader(test_ds, BATCH_SIZE)

    # Модели
    dual_model = DualClassifier(32, 16, EMBED_DIM, NUM_HEADS).to(device)
    real_model = RealBaseline(32, 16, EMBED_DIM, NUM_HEADS).to(device)

    logger.info(f"Dual model params: {sum(p.numel() for p in dual_model.parameters()):,}")
    logger.info(f"Real model params: {sum(p.numel() for p in real_model.parameters()):,}")

    opt_dual = torch.optim.Adam(dual_model.parameters(), lr=LR)
    opt_real = torch.optim.Adam(real_model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()

    # Обучение Dual модели 
    logger.info("Training Dual Model")
    best_dual_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(dual_model, train_loader, opt_dual, crit, device)
        _, val_acc = evaluate(dual_model, valid_loader, crit, device)
        if val_acc > best_dual_acc:
            best_dual_acc = val_acc
            torch.save(dual_model.state_dict(), "logs/best_dual.pt")
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

    # Обучение Real baseline
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

    # Финальное тестирование 
    dual_model.load_state_dict(torch.load("logs/best_dual.pt"))
    real_model.load_state_dict(torch.load("logs/best_real.pt"))

    _, dual_acc = evaluate(dual_model, test_loader, crit, device)
    _, real_acc = evaluate(real_model, test_loader, crit, device)

    logger.info("FINAL RESULTS")
    logger.info(f"Majority Baseline: 50.00%")
    logger.info(f"Real Baseline: {real_acc * 100:.2f}%")
    logger.info(f"Dual Model: {dual_acc * 100:.2f}%")

    # Диагностический вывод
    if dual_acc > real_acc:
        logger.info("✓ Dual модель превосходит Real baseline")
    elif abs(dual_acc - real_acc) < 0.01:
        logger.info("→ Dual и Real модели показывают одинаковый результат")
    else:
        logger.info("✗ Dual модель уступает Real baseline")


if __name__ == "__main__":
    main()