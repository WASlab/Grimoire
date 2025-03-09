import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # Depthwise
            nn.Conv2d(32, 64, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.Conv2d(128, 256, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 512),  # Expand to learn richer interactions
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),      # Regularize the expanded space
            nn.Linear(512, 256),  # Compress to focus on critical features
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 10)    # Final classification
        )


    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x