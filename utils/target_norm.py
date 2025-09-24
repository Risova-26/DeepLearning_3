# utils/target_norm.py
# TargetScaler with three label modes:
#   - zscore:      y_n = (y - mu) / std               ↔ linear head
#   - minmax:      y_n = (y - ymin) / (ymax - ymin)   ↔ sigmoid head in [0,1]
#   - minmax_sym:  y_n = 2*(y - ymin)/(ymax - ymin)-1 ↔ tanh head in [-1,1]
# Includes:
#   - encode()/decode()
#   - .to(device) via nn.Module
#   - .state_dict() with all fields (kind + tensors)

import torch
import torch.nn as nn

class TargetScaler(nn.Module):
    def __init__(self, y_train, kind: str = "zscore", pad_frac: float = 0.0, eps: float = 1e-6, device=None):
        """
        y_train : array-like [N,2] from TRAIN SPLIT ONLY
        kind    : 'zscore' | 'minmax' | 'minmax_sym'
        pad_frac: for minmax modes, pad the [ymin,ymax] by this fraction to reduce saturation
        """
        super().__init__()
        self.kind = kind
        self.eps = eps

        if not torch.is_tensor(y_train):
            y = torch.tensor(y_train, dtype=torch.float32)
        else:
            y = y_train.float()
        if device is not None:
            y = y.to(device)

        # Initialize buffers so attributes always exist
        self.register_buffer("mu",   torch.zeros(2))
        self.register_buffer("std",  torch.ones(2))
        self.register_buffer("ymin", torch.zeros(2))
        self.register_buffer("ymax", torch.ones(2))

        if kind == "zscore":
            mu  = y.mean(dim=0)
            std = y.std(dim=0, unbiased=False).clamp_min(eps)
            self.mu.copy_(mu)
            self.std.copy_(std)

        elif kind in ("minmax", "minmax_sym"):
            ymin = y.min(dim=0).values
            ymax = y.max(dim=0).values
            span = (ymax - ymin).clamp_min(eps)
            pad  = pad_frac * span
            ymin = ymin - pad
            ymax = ymax + pad
            # ensure non-zero span
            span = (ymax - ymin).clamp_min(eps)
            self.ymin.copy_(ymin)
            self.ymax.copy_(ymax)

        else:
            raise ValueError(f"Unknown kind: {kind}")

    # ---------- core API ----------

    def encode(self, y):
        """Map mm -> normalized space according to self.kind."""
        if not torch.is_tensor(y):
            y = torch.tensor(y, dtype=torch.float32, device=self.mu.device)
        if self.kind == "zscore":
            return (y - self.mu) / self.std
        elif self.kind == "minmax":
            return (y - self.ymin) / (self.ymax - self.ymin)
        elif self.kind == "minmax_sym":
            return 2.0 * (y - self.ymin) / (self.ymax - self.ymin) - 1.0
        else:
            raise RuntimeError(f"Bad kind: {self.kind}")

    def decode(self, yhat):
        """Map normalized prediction -> mm according to self.kind."""
        if not torch.is_tensor(yhat):
            yhat = torch.tensor(yhat, dtype=torch.float32, device=self.mu.device)
        if self.kind == "zscore":
            return yhat * self.std + self.mu
        elif self.kind == "minmax":
            return self.ymin + yhat * (self.ymax - self.ymin)
        elif self.kind == "minmax_sym":
            return self.ymin + 0.5 * (yhat + 1.0) * (self.ymax - self.ymin)
        else:
            raise RuntimeError(f"Bad kind: {self.kind}")

    # ---------- state I/O ----------

    def state_dict(self, *args, **kwargs):
        # Return a simple dict that eval.py can consume even without nn.Module.load_state_dict
        return {
            "kind": self.kind,
            "mu":   self.mu.detach().cpu(),
            "std":  self.std.detach().cpu(),
            "ymin": self.ymin.detach().cpu(),
            "ymax": self.ymax.detach().cpu(),
        }

    def load_state_dict(self, state, strict: bool = False):
        # Accept dicts with kind + tensors (cpu or cuda); be permissive.
        if "kind" in state:
            self.kind = state["kind"]
        for name in ("mu", "std", "ymin", "ymax"):
            if name in state and state[name] is not None:
                v = state[name]
                if not torch.is_tensor(v):
                    v = torch.tensor(v, dtype=torch.float32)
                # make sure buffer exists, then copy
                getattr(self, name).data.copy_(v.to(self.mu.device))
        return  # mimic nn.Module.load_state_dict signature
