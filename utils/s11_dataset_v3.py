# utils/s11_dataset_v3.py
# UPDATED: supports repr from scaler: "magphase" (default/backward compatible) or "reim".
# We keep original logic commented where behavior changed.

import numpy as np, pandas as pd, torch
from torch.utils.data import Dataset

def _unwrap_phase_deg(arr_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(arr_deg)
    unwrapped = np.unwrap(rad, axis=-1)
    return np.rad2deg(unwrapped)

def _db_to_lin(db):
    return (10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)).astype(np.float32)

def _load_feat_scaler(npz_path):
    z = np.load(npz_path)
    mu  = torch.tensor(z["mu"],  dtype=torch.float32)   # (2,200)
    std = torch.tensor(z["std"], dtype=torch.float32)   # (2,200)
    std = torch.where(std == 0, torch.ones_like(std), std)

    # ADDED: read meta with safe defaults for backward compatibility
    repr_mode     = str(z["repr"]) if "repr" in z else "magphase"
    unwrap_flag   = bool(z["unwrap"]) if "unwrap" in z else True
    remove_offset = bool(z["remove_offset"]) if "remove_offset" in z else True
    mode          = str(z["mode"]) if "mode" in z else "zscore"

    return mu, std, repr_mode, unwrap_flag, remove_offset, mode

class S11CSV(Dataset):
    """
    Returns:
        x: Tensor [2,200] built according to scaler's 'repr':
           - "magphase": mag(dB) and phase(deg); phase optionally unwrapped & DC-offset removed.
           - "reim": real and imaginary parts computed from mag(dB) + phase(deg).
        y: Tensor [2] = [Acetal_mm, Air_mm]
    """
    def __init__(self, csv_path, feat_scaler_npz, keep_ids=None, drop_ids=None):
        df = pd.read_csv(csv_path)
        if keep_ids is not None:
            keep = np.loadtxt(keep_ids, dtype=int).tolist() if isinstance(keep_ids, str) else keep_ids
            df = df[df["ID"].isin(keep)]
        if drop_ids is not None:
            drop = np.loadtxt(drop_ids, dtype=int).tolist() if isinstance(drop_ids, str) else drop_ids
            df = df[~df["ID"].isin(drop)]

        groups = df.groupby("ID", sort=True)

        ids, Mag_dB_raw, Pha_deg_raw, Y = [], [], [], []

        for gid, g in groups:
            if len(g) != 200:
                continue
            ids.append(int(gid))
            mag_db = g["Magnitude(dB)"].to_numpy(dtype=float)                 # (200,)
            pha_deg = g["Phase(degree)"].to_numpy(dtype=float)                # (200,)

            # ORIGINAL (always unwrap + remove DC on phase):
            # ph  = _unwrap_phase_deg(pha_deg)
            # ph  = ph - ph[0]
            # Xmag.append(mag_db); Xphase.append(ph)

            # NEW: store raw; we build features per 'repr' in __getitem__
            Mag_dB_raw.append(mag_db)
            Pha_deg_raw.append(pha_deg)

            ac = float(g["Label_1(Acetal)"].iloc[0])
            ar = float(g["Label_2(Air)"].iloc[0])
            Y.append([ac, ar])

        self.ids = ids
        self.Mag_dB_raw = torch.tensor(np.stack(Mag_dB_raw), dtype=torch.float32)  # [N,200]
        self.Pha_deg_raw = torch.tensor(np.stack(Pha_deg_raw), dtype=torch.float32) # [N,200]
        self.Y      = torch.tensor(np.stack(Y),      dtype=torch.float32)           # [N,2]

        # CHANGED: load scaler + metadata
        self.mu, self.std, self.repr_mode, self.unwrap_flag, self.remove_offset, self.mode = _load_feat_scaler(feat_scaler_npz)
        # Sanity
        assert self.mu.shape == (2,200) and self.std.shape == (2,200), "Scaler must be [2,200]"

        # One-time note if repr=reim but unwrap/remove_offset present
        if self.repr_mode == "reim" and (self.unwrap_flag or self.remove_offset):
            print("[WARN][S11CSV] 'unwrap'/'remove_offset' are ignored for repr='reim' (complex features).")

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        mag_db = self.Mag_dB_raw[i]     # [200]
        pha_deg = self.Pha_deg_raw[i]   # [200]

        if self.repr_mode == "magphase":
            pha = pha_deg
            if self.unwrap_flag:
                pha = torch.from_numpy(_unwrap_phase_deg(pha_deg.numpy())).to(dtype=torch.float32)
            # remove DC offset only if requested
            if self.remove_offset:
                pha = pha - pha[0]
            x = torch.stack([mag_db, pha], dim=0)  # [2,200]
        elif self.repr_mode == "reim":
            mag_lin = torch.from_numpy(_db_to_lin(mag_db.numpy()))
            phi = torch.deg2rad(pha_deg)
            real = (mag_lin * torch.cos(phi)).to(dtype=torch.float32)
            imag = (mag_lin * torch.sin(phi)).to(dtype=torch.float32)
            x = torch.stack([real, imag], dim=0)   # [2,200]
        else:
            raise ValueError(f"Unknown repr_mode: {self.repr_mode}")

        x = (x - self.mu) / self.std                  # per-(ch,freq) normalize
        y = self.Y[i]                                 # [Acetal, Air]
        return x, y
