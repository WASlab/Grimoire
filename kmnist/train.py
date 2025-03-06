import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math
from model import EfficientCNN
from torchvision.datasets import KMNIST
from torchvision.transforms import ToTensor
import os
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cosine_warmup_decay(epoch, total_epochs, lr_base, lr_peak, lr_end):
    p = epoch / (total_epochs - 1)
    if p <= 0.5:
        p2 = p / 0.5
        return lr_base + 0.5 * (lr_peak - lr_base) * (1 - math.cos(math.pi * p2))
    else:
        p2 = (p - 0.5) / 0.5
        return lr_peak + 0.5 * (lr_end - lr_peak) * (1 - math.cos(math.pi * p2))

def train(model, optimizer, criterion, train_loader):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, criterion, val_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def early_stopping(losses, patience=5):
    if len(losses) > patience and losses[-1] > min(losses[:-1]):
        return True
    return False


def main():
    epochs = 15
    lr_base = 1e-4
    lr_peak = 8e-3
    lr_end = 1e-6

    train_data = torch.load(""./kmnist/results_augmentation_search2/kmnist_train_augmented_tensorset.pth"")
    val_data = torch.load("./kmnist/saved_tensorsets/kmnist_val_tensorset.pth")

    train_loader = DataLoader(train_tds, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1024, shuffle=False)

    model = EfficientCNN(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_base, weight_decay=1e-4)

    val_losses = []
    for epoch in range(epochs):
        lr = cosine_warmup_decay(epoch, epochs, lr_base, lr_peak, lr_end)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_loss = train(model, optimizer, criterion, train_loader)
        val_loss = validate(model, criterion, val_loader)
        print(f"Epoch {epoch+1}/{epochs}, LR: {lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        val_losses.append(val_loss)
        if early_stopping(model, val_losses):
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), "efficientcnn.pth")

if __name__ == "__main__":
    main()
