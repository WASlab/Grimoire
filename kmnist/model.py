import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(2,2)
        self.fc1   = nn.Linear(64*7*7, 1024)
        self.bn3   = nn.BatchNorm1d(1024)
        self.dropout = nn.Dropout(0.2)
        self.fc2   = nn.Linear(1024,512)
        self.bn4   = nn.BatchNorm1d(512)
        self.fc3   = nn.Linear(512,256)
        self.bn5   = nn.BatchNorm1d(256)
        self.fc4   = nn.Linear(256,128)
        self.bn6   = nn.BatchNorm1d(128)
        self.fc5   = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.gelu(self.bn3(self.fc1(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn4(self.fc2(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn5(self.fc3(x)))
        x = self.dropout(x)
        x = F.gelu(self.bn6(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class ConvAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, heads=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.attn = nn.MultiheadAttention(out_channels, heads, batch_first=True)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = F.gelu(self.bn(self.conv(x)))
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W).transpose(1, 2)
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x = x_attn.transpose(1, 2).view(B, C, H, W)
        x = self.pool(x)
        return x

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
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class ConvSwiGLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1):
        super().__init__()
        self.conv_gate = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)
        self.conv_value = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        gate = F.silu(self.conv_gate(x))
        value = self.conv_value(x)
        return self.bn(gate * value)


class SwiGLUFC(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear_gate = nn.Linear(in_features, out_features)
        self.linear_value = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gate = F.silu(self.linear_gate(x))
        value = self.linear_value(x)
        return self.dropout(self.bn(gate * value))


class SwiGLUEfficientCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            ConvSwiGLU(1, 32),
            ConvSwiGLU(32, 32, groups=32),  # Depthwise
            ConvSwiGLU(32, 64, kernel_size=1, padding=0),  # Pointwise
            nn.MaxPool2d(2)
        )

        self.conv_block2 = nn.Sequential(
            ConvSwiGLU(64, 64),
            ConvSwiGLU(64, 64, groups=64),  # Depthwise
            ConvSwiGLU(64, 128, kernel_size=1, padding=0),  # Pointwise
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            ConvSwiGLU(128, 128),
            ConvSwiGLU(128, 128, groups=128),  # Depthwise
            ConvSwiGLU(128, 256, kernel_size=1, padding=0)  # Pointwise
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            SwiGLUFC(256, 256, dropout=0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x