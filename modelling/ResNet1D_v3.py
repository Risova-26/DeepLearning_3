# modelling/ResNet1D_v3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def gn(ch, groups=8):
    g = groups if ch >= groups else 1
    return nn.GroupNorm(g, ch)

class SE1d(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(ch, ch//r, 1), nn.ReLU(inplace=True),
            nn.Conv1d(ch//r, ch, 1), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class DilatedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, d=1, drop=0.05, use_se=True):
        super().__init__()
        pad = d
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=d)
        self.gn1   = gn(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=pad, dilation=d)
        self.gn2   = gn(out_ch)
        self.drop  = nn.Dropout(drop)
        self.se    = SE1d(out_ch) if use_se else nn.Identity()
        self.down  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        idt = self.down(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.drop(out)
        out = self.gn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + idt)

class Stream1D(nn.Module):
    """Small stack of dilated residual blocks without downsampling."""
    def __init__(self, in_ch, widths=(32,64,96,128), dilations=(1,2,4,8), drop=0.05):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, widths[0], 5, padding=2), gn(widths[0]), nn.ReLU(inplace=True))
        blocks = []
        ch = widths[0]
        for w, d in zip(widths, dilations):
            blocks.append(DilatedResBlock(ch, w, d=d, drop=drop, use_se=True))
            ch = w
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_ch = ch
    def forward(self, x):                 # [B,C,T]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)      # [B,out_ch]
        return x

class ResNet1DThicknessModelV3(nn.Module):
    """
    Dual-stream (freq + time) 1D CNN with dilations + SE.
    Input: x [B,2,T] with magnitude and unwrapped phase (radians).
    Output: [B,2] (Acetal, Air).
    """
    def __init__(self, T=200):
        super().__init__()
        self.T = T
        # frequency stream expects real/imag channels
        self.freq = Stream1D(in_ch=2, widths=(32,64,96,128), dilations=(1,2,4,8), drop=0.05)
        # time stream on IFFT(real/imag)
        self.time = Stream1D(in_ch=2, widths=(16,32,64), dilations=(1,2,4), drop=0.05)
        fused = self.freq.out_ch + self.time.out_ch
        self.head = nn.Sequential(
            nn.Linear(fused, 128), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(128, 2)
        )

    @staticmethod
    def polar_to_cart(mag, phase):
        # mag, phase: [B,T] -> real/imag [B,2,T]
        real = mag * torch.cos(phase)
        imag = mag * torch.sin(phase)
        return torch.stack([real, imag], dim=1)

    def forward(self, x):                 # x: [B,2,T]  (mag, phase)
        B, C, T = x.shape
        assert C == 2 and T == self.T, f"Expected [B,2,{self.T}], got {x.shape}"
        mag = x[:,0,:]
        ph  = x[:,1,:]

        # complex freq vector
        s_c = torch.polar(mag, ph)       # [B,T] complex
        # freq stream input: real/imag
        freq_ri = torch.stack([s_c.real, s_c.imag], dim=1)           # [B,2,T]
        f_feat = self.freq(freq_ri)                                   # [B,F]

        # time stream input: IFFT(S11)
        t_c = torch.fft.ifft(s_c)                                     # [B,T] complex
        time_ri = torch.stack([t_c.real, t_c.imag], dim=1)            # [B,2,T]
        t_feat = self.time(time_ri)                                   # [B,Tf]

        z = torch.cat([f_feat, t_feat], dim=1)                        # [B,F+Tf]
        return self.head(z)                                           # [B,2]
