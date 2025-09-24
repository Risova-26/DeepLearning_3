# modelling/ResNet1D_v2.py
# V2 NEW: Replace BatchNorm1d with GroupNorm (batch-size independent), add light dropout.

import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(ch):  # 8 groups or fallback to 1 group if channels < 8
    g = 8 if ch >= 8 else 1
    return nn.GroupNorm(g, ch)

class ResidualBlockGN(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, drop=0.05):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.gn1   = gn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn2   = gn(out_ch)
        self.drop  = nn.Dropout(drop)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        idt = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.drop(out)
        out = self.gn2(self.conv2(out))
        if self.down is not None: idt = self.down(idt)
        out = F.relu(out + idt)
        return out

class ResNet1DThicknessModelV2(nn.Module):
    """ResNet1D v2: GN + dropout + global avg pool → 2 outputs."""
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            gn(32), nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(ResidualBlockGN(32,32), ResidualBlockGN(32,32))
        self.layer2 = nn.Sequential(ResidualBlockGN(32,64, stride=2), ResidualBlockGN(64,64))
        self.layer3 = nn.Sequential(ResidualBlockGN(64,128, stride=2), ResidualBlockGN(128,128))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(128, 2)

    def forward(self, x):            # x: [B,2,200]
        x = self.stem(x)             # [B,32,200]
        x = self.layer1(x)           # [B,32,200]
        x = self.layer2(x)           # [B,64,100]
        x = self.layer3(x)           # [B,128,50]
        x = self.pool(x).squeeze(-1) # [B,128]
        return self.head(x)          # [B,2]
