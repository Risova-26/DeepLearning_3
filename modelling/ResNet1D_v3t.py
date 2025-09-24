import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock1D(nn.Module):
    def __init__(self, c, k=7, d=1, downsample=False):
        super().__init__()
        pad = ((k - 1) // 2) * d
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv1d(c, c, kernel_size=k, stride=stride, padding=pad, dilation=d, bias=False)
        self.bn1   = nn.BatchNorm1d(c)
        self.conv2 = nn.Conv1d(c, c, kernel_size=k, padding=pad, dilation=d, bias=False)
        self.bn2   = nn.BatchNorm1d(c)
        self.down  = nn.AvgPool1d(kernel_size=2) if downsample else None

    def forward(self, x):
        idt = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            idt = self.down(idt)
        return F.relu(out + idt)

class DualStreamS11Net(nn.Module):
    """
    Input: x of shape (B, C, T), C>=2 expected (channel0 = |S11|, channel1 = unwrapped phase φ)
    Outputs: (B, 2) -> [Acetal_mm, Air_mm]
    """
    def __init__(self, in_chans=2, base=48, tbase=32, dropout=0.10):
        super().__init__()
        self.in_chans = in_chans

        # Frequency stream on [|S11|, φ, d|S11|/df, dφ/df]
        self.freq_in  = nn.Conv1d(4, base, kernel_size=7, padding=3, bias=False)
        self.freq_bn  = nn.BatchNorm1d(base)
        self.freq_trunk = nn.Sequential(
            ResBlock1D(base, k=7, d=1),
            ResBlock1D(base, k=7, d=2),
            ResBlock1D(base, k=7, d=4),
        )

        # Time stream on IFFT envelope of complex S11
        self.time_in  = nn.Conv1d(1, tbase, kernel_size=7, padding=3, bias=False)
        self.time_bn  = nn.BatchNorm1d(tbase)
        self.time_trunk = nn.Sequential(
            ResBlock1D(tbase, k=7, d=1),
            ResBlock1D(tbase, k=7, d=2),
        )

        # Fusion + heads
        fdim = base + tbase
        self.fuse    = nn.Conv1d(fdim, fdim, kernel_size=1, bias=False)
        self.fuse_bn = nn.BatchNorm1d(fdim)
        self.drop    = nn.Dropout(dropout)
        hid = 128
        self.head_acetal = nn.Sequential(nn.Linear(fdim, hid), nn.ReLU(), nn.Linear(hid, 1))
        self.head_air    = nn.Sequential(nn.Linear(fdim, hid), nn.ReLU(), nn.Linear(hid, 1))

    @staticmethod
    def _diff_along_t(x):
        # x: (B,T) -> finite difference with length preserved
        return F.pad(x[:, 1:] - x[:, :-1], (1, 0))

    def forward(self, x):
        # x: (B,C,T) with C>=2: [mag, phase, ...]
        if x.dim() != 3 or x.size(1) < 2:
            raise ValueError(f"Expected (B,C,T) with C>=2, got {tuple(x.shape)}")

        B, C, T = x.shape
        mag = x[:, 0, :]                       # |S11|
        phs = x[:, 1, :]                       # unwrapped phase (radians)

        # Frequency-domain engineered features
        dmag = self._diff_along_t(mag)
        dphs = self._diff_along_t(phs)
        freq_in = torch.stack([mag, phs, dmag, dphs], dim=1)   # (B,4,T)

        f = F.relu(self.freq_bn(self.freq_in(freq_in)))
        f = self.freq_trunk(f)

        # Time-domain envelope via IFFT of complex S11
        real = mag * torch.cos(phs)
        imag = mag * torch.sin(phs)
        s11c = torch.complex(real, imag)                       # (B,T)
        td   = torch.fft.ifft(s11c, dim=-1)                    # complex (B,T)
        env  = torch.abs(td).unsqueeze(1)                      # (B,1,T)
        env  = F.avg_pool1d(env, kernel_size=3, stride=1, padding=1)

        g = F.relu(self.time_bn(self.time_in(env)))
        g = self.time_trunk(g)

        # Fuse and predict
        z   = torch.cat([f, g], dim=1)                         # (B, base+tbase, T)
        z   = self.drop(F.relu(self.fuse_bn(self.fuse(z))))
        gap = z.mean(dim=-1)                                   # (B, base+tbase)

        acetal = self.head_acetal(gap).squeeze(-1)             # (B,)
        air    = self.head_air(gap).squeeze(-1)                # (B,)
        return torch.stack([acetal, air], dim=1)               # (B,2)
