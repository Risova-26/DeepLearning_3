import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet1DThicknessModel(nn.Module):
    """1D ResNet model for thickness prediction, inspired by Zhou et al. (2021) and mmWave pipeline guide.
    Uses residual Conv1D blocks to capture multi-scale frequency features from S11 data.
    """
    def __init__(self):
        super().__init__()
        # Initial 1D convolution layer (in_channels=2 for magnitude & phase)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)  # normalise feature maps
        # Define residual blocks with increasing channels
        self.layer1 = self._make_layer(16, 16, num_blocks=2, downsample=False)
        self.layer2 = self._make_layer(16, 32, num_blocks=2, downsample=True)
        self.layer3 = self._make_layer(32, 64, num_blocks=2, downsample=True)
        # (Further layers could be added for a deeper network)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # global average pooling over frequency dimension
        self.fc = nn.Linear(64, 2)  # output two thickness values

    def _make_layer(self, in_channels, out_channels, num_blocks, downsample):
        """Create a sequence of residual blocks."""
        layers = []
        for i in range(num_blocks):
            if i == 0 and downsample:
                # first block in this layer with downsampling (stride 2) and channel increase
                layers.append(ResidualBlock(in_channels, out_channels, stride=2))
            else:
                layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2, 200)
        out = F.relu(self.bn1(self.conv1(x)))       # initial conv + BN + ReLU
        out = self.layer1(out)                      # residual blocks at 16 channels
        out = self.layer2(out)                      # residual blocks at 32 channels (downsampled)
        out = self.layer3(out)                      # residual blocks at 64 channels (downsampled)
        out = self.global_pool(out)                 # shape (batch, 64, 1)
        out = out.view(out.size(0), -1)             # flatten to (batch, 64)
        out = self.fc(out)                          # output shape (batch, 2)
        return out

class ResidualBlock(nn.Module):
    """Residual block with two 1D conv layers and an identity skip connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # First conv layer (with possible downsampling)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm1d(out_channels)
        # Second conv layer
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm1d(out_channels)
        # Skip connection layer (if in/out channels differ or stride > 1)
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv to match dimensions of skip connection
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Add identity skip connection
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity  # residual connection adds input feature map
        out = F.relu(out)
        return out

# Example usage:
# model = ResNet1DThicknessModel()
# pred = model(torch.randn(8, 2, 200))
