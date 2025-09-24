# utils/normalizer.py
import numpy as np
import torch

class S11Normalizer:
    """
    Feature-wise normalizer for 2x200 S11 (mag_dB, phase_deg).
    Uses per-(channel,frequency) mu/std saved by fit_normalizer.py
    """
    def __init__(self, npz_path):
        d = np.load(npz_path)
        self.mu  = torch.from_numpy(d["mu"])    # [2,200]
        self.std = torch.from_numpy(d["std"])   # [2,200]
        self.mode = str(d.get("mode","zscore"))

    def to(self, device):
        self.mu  = self.mu.to(device)
        self.std = self.std.to(device)
        return self

    @torch.no_grad()
    def transform(self, x):
        """
        x: float tensor [..., 2, 200]
        returns normalized tensor with same shape
        """
        return (x - self.mu) / (self.std + 1e-8)

    @staticmethod
    def ema_calibrate(pretrain_npz, phys_train_npz, out_npz, alpha=0.2):
        """
        Blend pretrain and physical-train stats: mu' = (1-a)mu + a mu_phys; same for std.
        """
        d0 = np.load(pretrain_npz)
        d1 = np.load(phys_train_npz)
        mu  = (1-alpha) * d0["mu"]  + alpha * d1["mu"]
        std = (1-alpha) * d0["std"] + alpha * d1["std"]
        std = np.where(std < 1e-6, 1e-6, std).astype(np.float32)
        np.savez_compressed(out_npz, mu=mu.astype(np.float32), std=std, mode=d0.get("mode","zscore"))
