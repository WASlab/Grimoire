import torch.nn as nn
import torch.nn.functional as F
import torch

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
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.
    reduction: the intermediate FC dim is channels // reduction
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        b, c, _, _ = x.size()
        # Squeeze: global average pool
        y = self.global_pool(x).view(b, c)   # shape: (B, C)
        # Excitation: channel weighting
        y = self.fc(y).view(b, c, 1, 1)      # shape: (B, C, 1, 1)
        # Scale
        return x * y
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




import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Create spatial attention map
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate along channel dimension
        attention = torch.cat([avg_pool, max_pool], dim=1)
        
        # Apply convolution and sigmoid
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        # Apply attention to input feature map
        return x * attention

class EfficientCNNwAttn(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        
        # First convolutional block
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
        
        # Second convolutional block
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
        
        # Spatial attention after second block
        self.spatial_attention = SpatialAttention(kernel_size=5)
        
        # Third convolutional block
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128),  # Depthwise
            nn.Conv2d(128, 256, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(2)  # Added MaxPool to reduce spatial dimensions
        )
        
        # Fourth convolutional block (new)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256),  # Depthwise
            nn.Conv2d(256, 512, kernel_size=1),  # Pointwise
            nn.BatchNorm2d(512),
            nn.GELU()
        )
        
        # Improved pooling - combining global average and max pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Simplified but effective classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 512),  # *2 because we concatenate avg and max pooling
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.4),  # Increased dropout for better regularization
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Apply convolution blocks
        x = self.conv_block1(x)        # -> (B, 64, 16, 16)
        x = self.conv_block2(x)        # -> (B, 128, 8, 8)
        
        # Apply spatial attention
        x = self.spatial_attention(x)  # -> (B, 128, 8, 8) with attention
        
        x = self.conv_block3(x)        # -> (B, 256, 4, 4)
        x = self.conv_block4(x)        # -> (B, 512, 4, 4)
        
        # Apply pooling
        avg_x = self.global_avg_pool(x)  # -> (B, 512, 1, 1)
        max_x = self.global_max_pool(x)  # -> (B, 512, 1, 1)
        
        # Concatenate average and max pooling results
        x = torch.cat([avg_x, max_x], dim=1)  # -> (B, 1024, 1, 1)
        x = x.view(x.size(0), -1)             # -> (B, 1024)
        
        # Apply classifier
        x = self.classifier(x)                # -> (B, num_classes)
        
        return x

if __name__ == '__main__':
    model = EfficientCNN()
    
    # Count total trainable parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total trainable parameters:", total_params)
    model.eval()
    # Test a forward pass with a dummy input (e.g., a 28x28 grayscale image)
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print("Output shape:", output.shape)