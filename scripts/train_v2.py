# scripts/train_v2.py
# V2 TRAINER: adds
#  - robust loss options (SmoothL1 / weighted per target)
#  - GroupNorm ResNet (resnet1d_v2) support
#  - LR scheduler (ReduceLROnPlateau) + gradient clipping
#  - optional tiny train-time augmentations
#  - label normalization/activation options (zscore / minmax / minmax_sym) + (linear / tanh / sigmoid)
#  - checkpoint saves generic TargetScaler state when available

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split

# --- Make project root importable (Windows-safe) ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Models ---
from modelling.MLP import MLPThicknessModel
from modelling.ResNet1D_v2 import ResNet1DThicknessModelV2   # V2: GroupNorm ResNet
from modelling.Transformer import TransformerThicknessModel

# --- Datasets ---
from utils.s11_dataset import S11DomainDataset               # no-aug (eval/val)
from utils.s11_dataset_v2 import S11DomainDatasetV2          # V2: optional tiny aug for train

# --- Label scaler (must support encode()/decode(); state_dict optional) ---
from utils.target_norm import TargetScaler

# -------------------------------
# Helpers
# -------------------------------

def build_model(kind: str):
    if kind == "mlp":         return MLPThicknessModel()
    if kind == "resnet1d":    return ResNet1DThicknessModelV2()   # V2 default to GN ResNet
    if kind == "transformer": return TransformerThicknessModel()
    raise ValueError(f"unknown model: {kind}")

def batch_collate(batch):
    """Top-level collate (Windows-safe for num_workers>0)."""
    xs, ys = zip(*batch)
    x = torch.stack(xs, 0)  # [B,2,200]
    y = torch.stack(ys, 0)  # [B,2]
    return x, y

def apply_out_act(z, mode: str):
    """Apply output activation before computing loss in label-normalized space."""
    if mode == "linear":  return z
    if mode == "tanh":    return torch.tanh(z)
    if mode == "sigmoid": return torch.sigmoid(z)
    raise ValueError(f"bad out_act: {mode}")

# -------------------------------
# Training / Validation
# -------------------------------

def train_one_epoch(model, loader, device, tnorm, base_loss, w_vec, opt, clip_max, out_act):
    model.train()
    tot = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yb_n = tnorm.encode(yb)              # encode labels to normalized space
        opt.zero_grad()
        yhat_n = model(xb)
        yhat_n = apply_out_act(yhat_n, out_act)  # V2: optional bounded head
        loss_vec = base_loss(yhat_n, yb_n)   # [B,2]
        loss = (loss_vec * w_vec).mean()
        loss.backward()
        if clip_max is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max)  # V2: grad clipping
        opt.step()
        tot += loss.item() * xb.size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, device, tnorm, base_loss, w_vec, out_act):
    model.eval()
    vl_tot = 0.0
    mae_ac, mae_air = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        yb_n = tnorm.encode(yb)
        yhat_n = model(xb)
        yhat_n = apply_out_act(yhat_n, out_act)  # V2: bounded head on val too
        loss_vec = base_loss(yhat_n, yb_n)
        vl_tot += (loss_vec * w_vec).mean().item() * xb.size(0)
        yhat = tnorm.decode(yhat_n)              # back to mm for metrics
        err = (yhat - yb).abs().mean(dim=0)
        mae_ac.append(err[0].item()); mae_air.append(err[1].item())
    vl_loss = vl_tot / len(loader.dataset)
    return vl_loss, float(np.mean(mae_ac)), float(np.mean(mae_air))

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain","finetune"], required=True)
    ap.add_argument("--model", choices=["mlp","resnet1d","transformer"], required=True)

    # DATA: sources & scalers
    ap.add_argument("--math_csv"); ap.add_argument("--cst_csv")
    ap.add_argument("--phy_csv"); ap.add_argument("--phy_train_ids"); ap.add_argument("--phy_val_ids")
    ap.add_argument("--math_scaler"); ap.add_argument("--cst_scaler"); ap.add_argument("--phy_scaler")

    # TRAINING
    ap.add_argument("--init_weights", default=None)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 2))
    ap.add_argument("--outdir", default="runs_v2")

    # V2: LOSS & WEIGHTS
    ap.add_argument("--loss", choices=["mse","smoothl1"], default="smoothl1")  # V2 CHANGE
    ap.add_argument("--loss_weights", type=float, nargs=2, default=[1.0, 1.3],   # [Acetal, Air]
                    help="Per-target weights (default raises pressure on Air).") # V2 CHANGE
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Global grad-norm clip (None to disable).")

    # V2: AUG (train-only; defaults OFF)
    ap.add_argument("--aug_noise_mag", type=float, default=0.0)
    ap.add_argument("--aug_noise_phase", type=float, default=0.0)
    ap.add_argument("--aug_freq_mask_frac", type=float, default=0.0)

    # V2: LABEL NORMALIZATION + OUTPUT ACTIVATION
    ap.add_argument("--y_norm", choices=["zscore","minmax","minmax_sym"], default="zscore",
                    help="Label scaling: zscore or minmax ([0,1]) or minmax_sym ([-1,1]).")
    ap.add_argument("--out_act", choices=["linear","tanh","sigmoid"], default="linear",
                    help="Output activation applied in normalized space before loss.")
    ap.add_argument("--y_pad_frac", type=float, default=0.05,
                    help="Padding fraction for minmax/minmax_sym to avoid saturation.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------
    # Build datasets
    # -----------------------
    if args.stage == "pretrain":
        assert args.math_csv and args.cst_csv and args.math_scaler and args.cst_scaler
        # Train with V2 dataset to optionally allow aug (even if OFF)
        ds_math = S11DomainDatasetV2(args.math_csv, args.math_scaler,
                                     train_mode=True,
                                     aug_noise_mag=args.aug_noise_mag,
                                     aug_noise_phase=args.aug_noise_phase,
                                     aug_freq_mask_frac=args.aug_freq_mask_frac)
        ds_cst  = S11DomainDatasetV2(args.cst_csv,  args.cst_scaler,
                                     train_mode=True,
                                     aug_noise_mag=args.aug_noise_mag,
                                     aug_noise_phase=args.aug_noise_phase,
                                     aug_freq_mask_frac=args.aug_freq_mask_frac)
        full = ConcatDataset([ds_math, ds_cst])
        n = len(full); n_val = max(1, int(0.1*n)); n_train = n - n_val
        train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    else:
        assert args.phy_csv and args.phy_scaler and args.phy_train_ids and args.phy_val_ids
        train_ids = np.loadtxt(args.phy_train_ids, dtype=int).tolist()
        val_ids   = np.loadtxt(args.phy_val_ids,   dtype=int).tolist()
        train_ds = S11DomainDatasetV2(args.phy_csv, args.phy_scaler,
                                      keep_ids=train_ids, train_mode=True,
                                      aug_noise_mag=args.aug_noise_mag,
                                      aug_noise_phase=args.aug_noise_phase,
                                      aug_freq_mask_frac=args.aug_freq_mask_frac)
        val_ds   = S11DomainDataset(args.phy_csv, args.phy_scaler, keep_ids=val_ids)  # no aug for val

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=(args.workers>0), collate_fn=batch_collate)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=False, collate_fn=batch_collate)

    # -----------------------
    # Target scaler (labels)
    # -----------------------
    # V2 CHANGE: compute label stats over FULL train split (not a small sample)
    tmp = DataLoader(train_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=batch_collate)
    ys = []
    for _, yb in tmp:
        ys.append(yb)
    y_train = torch.cat(ys, 0).numpy()  # shape [N,2]

    # Construct TargetScaler; we pass config if supported. If not, it falls back to zscore.
    # EXPECTED: TargetScaler(y_train, kind="zscore|minmax|minmax_sym", pad_frac=float)
    try:
        tnorm = TargetScaler(y_train, kind=args.y_norm, pad_frac=args.y_pad_frac).to(device)
    except TypeError:
        # Backward-compat: older TargetScaler(y_train) only
        tnorm = TargetScaler(y_train).to(device)
        # If the class exposes attributes, set them so encode/decode knows what to do
        try:
            tnorm.kind = args.y_norm
            tnorm.pad_frac = args.y_pad_frac
        except Exception:
            pass

    # -----------------------
    # Model / Optim / Loss
    # -----------------------
    model = build_model(args.model).to(device)
    if args.init_weights:
        ck = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        print(f"[INFO] loaded init weights from {args.init_weights}")

    base_loss = (nn.MSELoss(reduction="none") if args.loss == "mse"
                 else nn.SmoothL1Loss(beta=1.0, reduction="none"))  # V2: robust option
    w_vec = torch.tensor(args.loss_weights, dtype=torch.float32, device=device)  # [2]

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)  # V2: AdamW
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                           patience=10, verbose=True)  # V2

    best_val = float("inf")
    ckpt_path = os.path.join(args.outdir, f"{args.stage}_{args.model}.pt")

    # -----------------------
    # Epoch loop
    # -----------------------
    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, device, tnorm, base_loss, w_vec, opt,
                             clip_max=args.grad_clip, out_act=args.out_act)
        vl, mae_ac, mae_air = validate(model, val_loader, device, tnorm, base_loss, w_vec, out_act=args.out_act)
        scheduler.step(vl)

        print(f"[{epoch:03d}] train={tr:.4f}  val={vl:.4f}  MAE(mm): Acetal={mae_ac:.3f}, Air={mae_air:.3f}")

        if vl < best_val:
            best_val = vl
            save_obj = {"model": model.state_dict()}
            # V2 CHANGE: prefer saving a generic scaler state if available
            if hasattr(tnorm, "state_dict"):
                try:
                    save_obj["tnorm_state"] = tnorm.state_dict()
                except Exception:
                    # fallback to older mu/std fields if present
                    if hasattr(tnorm, "mu") and hasattr(tnorm, "std"):
                        mu = tnorm.mu.detach().cpu() if torch.is_tensor(tnorm.mu) else torch.tensor(tnorm.mu)
                        std = tnorm.std.detach().cpu() if torch.is_tensor(tnorm.std) else torch.tensor(tnorm.std)
                        save_obj["tnorm_mu"] = mu; save_obj["tnorm_std"] = std
            else:
                # ultimate fallback: try mu/std
                if hasattr(tnorm, "mu") and hasattr(tnorm, "std"):
                    mu = tnorm.mu.detach().cpu() if torch.is_tensor(tnorm.mu) else torch.tensor(tnorm.mu)
                    std = tnorm.std.detach().cpu() if torch.is_tensor(tnorm.std) else torch.tensor(tnorm.std)
                    save_obj["tnorm_mu"] = mu; save_obj["tnorm_std"] = std

            torch.save(save_obj, ckpt_path)

    print(f"[OK] saved best to {ckpt_path}")

if __name__ == "__main__":
    main()
