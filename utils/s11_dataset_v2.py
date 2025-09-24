# utils/s11_dataset_v2.py
# V2 NEW: same as v1 but with optional, tiny train-time augmentations.

import pandas as pd, numpy as np, torch
from torch.utils.data import Dataset
from utils.normalizer import S11Normalizer
from utils.unwrap_torch import unwrap_phase_deg_torch

def _load_tall_filtered(csv_path, keep_ids=None):
    df = pd.read_csv(csv_path)
    if keep_ids is not None:
        keep_ids = set(map(int, keep_ids))
        df = df[df["ID"].isin(keep_ids)]
    X, Y, ids = [], [], []
    for gid, g in df.groupby("ID", sort=True):
        mag = g["Magnitude(dB)"].to_numpy(np.float32)
        pha = g["Phase(degree)"].to_numpy(np.float32)
        if len(mag) != 200 or len(pha) != 200: continue
        X.append(np.stack([mag, pha], axis=0))      # [2,200]
        a = float(g["Label_1(Acetal)"].iloc[0]); b = float(g["Label_2(Air)"].iloc[0])
        Y.append([a, b]); ids.append(int(gid))
    return np.stack(X, 0), np.array(Y, np.float32), np.array(ids)

class S11DomainDatasetV2(Dataset):
    """
    V2 CHANGE: optional augmentations for training robustness.
    - unwrap phase per sample
    - per-domain normalization (same as v1)
    - optional: tiny gaussian noise + small freq masking (train only)
    """
    def __init__(self, csv_path, scaler_npz, keep_ids=None,
                 train_mode=False, aug_noise_mag=0.0, aug_noise_phase=0.0,
                 aug_freq_mask_frac=0.0, rng_seed=1234):
        self.X, self.Y, self.IDs = _load_tall_filtered(csv_path, keep_ids)
        self.scaler = S11Normalizer(scaler_npz)
        self.train_mode = train_mode
        self.aug_noise_mag = float(aug_noise_mag)
        self.aug_noise_phase = float(aug_noise_phase)
        self.aug_freq_mask_frac = float(aug_freq_mask_frac)
        self.rng = np.random.default_rng(rng_seed)

    def __len__(self): return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i])  # [2,200]
        # unwrap phase before normalization (keep slope)
        ph_u = unwrap_phase_deg_torch(x[1].unsqueeze(0)).squeeze(0)  # [200]
        x = torch.stack([x[0], ph_u], dim=0)                         # [2,200]
        # per-domain normalization
        x = self.scaler.transform(x)

        # V2 CHANGE: tiny train-only augmentations (OFF by default)
        if self.train_mode:
            if self.aug_noise_mag > 0:
                x[0] = x[0] + torch.randn_like(x[0]) * self.aug_noise_mag
            if self.aug_noise_phase > 0:
                x[1] = x[1] + torch.randn_like(x[1]) * self.aug_noise_phase
            if self.aug_freq_mask_frac > 0:
                k = max(1, int(self.aug_freq_mask_frac * x.size(-1)))
                idx = self.rng.choice(x.size(-1), size=k, replace=False)
                x[:, idx] = 0.0  # zero-out a few random frequency bins

        y = torch.from_numpy(self.Y[i])  # [2]
        return x, y
