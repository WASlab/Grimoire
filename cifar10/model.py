import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCNN(nn.Module):
    """
    An "efficient-style" CNN for 3-channel 32x32 inputs (CIFAR10-like).
    Uses depthwise+pointwise convolutions and global average pooling.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32),  # Depthwise
            nn.Conv2d(32, 64, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2)  # 32x32 -> 16x16
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64),  # Depthwise
            nn.Conv2d(64, 128, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)  # 16x16 -> 8x8
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.Conv2d(128, 256, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(256),
            nn.GELU()
            # output shape: (B, 256, 8, 8)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> (B, 256, 1, 1)

        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)   # -> (B, 64, 16, 16)
        x = self.conv_block2(x)   # -> (B, 128, 8, 8)
        x = self.conv_block3(x)   # -> (B, 256, 8, 8)
        x = self.global_pool(x)   # -> (B, 256, 1, 1)
        x = x.view(x.size(0), -1) # -> (B, 256)
        x = self.fc(x)            # -> (B, num_classes)
        return x


class CNN(nn.Module):
    """
    A more "classic-style" CNN for 3-channel 32x32 inputs.
    Uses two convolution+pool blocks, then multiple fully-connected layers.
    The key fix is using 64*8*8 for the first linear layer, matching
    (32x32 -> pool -> 16x16 -> pool -> 8x8).
    """
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2, 2)  # Each pool halves spatial dims

        # After two MaxPool2d(2,2) calls on a 32x32 input:
        #   1st pool: 32x32 -> 16x16
        #   2nd pool: 16x16 -> 8x8
        # So final feature map is (64, 8, 8) = 64*8*8 = 4096
        self.fc1   = nn.Linear(64 * 8 * 8, 1024)
        self.bn3   = nn.BatchNorm1d(1024)
        
        self.dropout = nn.Dropout(0.2)
        self.fc2   = nn.Linear(1024, 512)
        self.bn4   = nn.BatchNorm1d(512)
        self.fc3   = nn.Linear(512, 256)
        self.bn5   = nn.BatchNorm1d(256)
        self.fc4   = nn.Linear(256, 128)
        self.bn6   = nn.BatchNorm1d(128)
        self.fc5   = nn.Linear(128, num_classes)
    
    def forward(self, x):
        # Convolution block 1
        x = self.conv1(x)   # (3->32, 3x3)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.pool(x)    # -> (B, 32, 16, 16)

        # Convolution block 2
        x = self.conv2(x)   # (32->64, 3x3)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.pool(x)    # -> (B, 64, 8, 8)

        # Flatten for fully-connected
        x = x.view(x.size(0), -1)  # -> (B, 64*8*8) = (B, 4096)

        x = self.fc1(x)
        x = self.bn3(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.bn5(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.bn6(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc5(x)     # -> (B, num_classes)
        return x
