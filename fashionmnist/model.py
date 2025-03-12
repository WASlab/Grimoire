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

class EfficientCNN(nn.Module):
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
            nn.Linear(512, num_classes)
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