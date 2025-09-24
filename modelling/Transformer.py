import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=4096):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):              # x: (B, T, d_model)
        return self.dropout(x + self.pe[:, :x.size(1)])

class TransformerThicknessModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1, out_dim=2):
        super().__init__()
        # No need to know in_chans ahead of time
        self.input = nn.LazyConv1d(out_channels=d_model, kernel_size=1)  # infers in_channels on first call
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                         dim_feedforward=dim_feedforward,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.pos = PositionalEncoding(d_model, dropout=0.0)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x):
        # Accept (B, T), (B, T, C), or (B, C, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)           # (B, 1, T)
        elif x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)        # (B, C, T) <- (B, T, C)

        x = self.input(x)                # (B, d_model, T)
        x = x.transpose(1, 2)            # (B, T, d_model)
        x = self.pos(x)                  # add positional encoding
        x = self.encoder(x)              # (B, T, d_model)
        x = x.mean(dim=1)                # global average pool over time
        return self.head(x)              # (B, 2)
