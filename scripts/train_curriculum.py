# scripts/train_curriculum.py
import os, sys, argparse, numpy as np, torch
from torch.utils.data import DataLoader, random_split

# repo import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.s11_dataset_v3 import S11CSV
from utils.target_norm import TargetScaler
from modelling.MLP import MLPThicknessModel
from modelling.ResNet1D_v3 import ResNet1DThicknessModelV3
from modelling.ResNet1D_v3t import DualStreamS11Net as Model
from modelling.Transformer import TransformerThicknessModel

def build_model(name):
    if name == "mlp":         return MLPThicknessModel()
    if name == "resnet1d_v3t": return Model(in_chans=2)
    if name == "transformer": return TransformerThicknessModel()
    raise ValueError(name)

def freeze_stem_if_requested(model, freeze_stem: bool):
    if not freeze_stem: return
    # For ResNet1D_v2 we named the initial stack 'stem'
    if hasattr(model, "stem"):
        for p in model.stem.parameters(): p.requires_grad = False
        print("[INFO] froze 'stem' parameters.")
    else:
        print("[WARN] model has no 'stem' attribute to freeze.")

def train_one_stage(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- dataset(s) --------
    if args.stage == "phys":
        ds_train = S11CSV(args.csv, args.feat_scaler, keep_ids=args.train_ids)
        ds_val   = S11CSV(args.csv, args.feat_scaler, keep_ids=args.val_ids)
    else:
        full = S11CSV(args.csv, args.feat_scaler)
        # make our own val split for Math/CST
        n = len(full); n_val = max(1, int(n * args.val_frac))
        n_train = n - n_val
        ds_train, ds_val = random_split(full, [n_train, n_val],
                                        generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # -------- target normalization (labels) --------
    y_train_np = torch.vstack([y for _, y in DataLoader(ds_train, batch_size=1024)]).numpy()
    tnorm = TargetScaler(y_train_np, kind=args.y_norm, pad_frac=args.y_pad_frac).to(device)

    # -------- model/opt --------
    model = build_model(args.model).to(device)

    if args.init and os.path.isfile(args.init):
        ckpt = torch.load(args.init, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"[INFO] loaded init weights from {args.init}")

    freeze_stem_if_requested(model, args.freeze_stem and args.stage=="cst")

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=False)
    if args.loss == "smoothl1":
        base_loss = torch.nn.SmoothL1Loss(reduction="none", beta=0.1)
    else:
        base_loss = torch.nn.L1Loss(reduction="none")

    def loss_fn(yhat_n, yb_n):
        # yhat_n, yb_n in normalized space
        L = base_loss(yhat_n, yb_n)
        # apply different weights to [Acetal, Air]
        w = torch.tensor([args.w_acetal, args.w_air], device=L.device)[None, :]
        return (L * w).mean()

    # -------- train loop --------
    os.makedirs(args.outdir, exist_ok=True)
    best = (1e9, -1)
    for ep in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            yb_n = tnorm.encode(yb)              # normalize labels
            yhat_n = model(xb)                   # linear head in our models
            loss = loss_fn(yhat_n, yb_n)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # ---- validation (in mm) ----
        model.eval(); val_loss = 0.0
        mae_ac, mae_air = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                yhat_n = model(xb)
                loss = loss_fn(yhat_n, tnorm.encode(yb))
                val_loss += loss.item() * xb.size(0)
                yhat = tnorm.decode(yhat_n)
                err = (yhat - yb).abs()
                mae_ac.append(err[:,0]); mae_air.append(err[:,1])
        val_loss /= len(val_loader.dataset)
        mae_ac = torch.cat(mae_ac).mean().item()
        mae_air = torch.cat(mae_air).mean().item()

        print(f"[{ep:03d}] train={tr_loss:.4f}  val={val_loss:.4f}  MAE(mm): Acetal={mae_ac:.3f}, Air={mae_air:.3f}")
        scheduler.step(val_loss)

        # save best
        if val_loss < best[0]:
            best = (val_loss, ep)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": ep,
                "val_loss": val_loss,
                "tnorm_state": tnorm.state_dict(),
                "tnorm_kind": tnorm.kind,
            }
            torch.save(ckpt, os.path.join(args.outdir, "best.pt"))
            # also dump the last batch MAE in a text for quick glance
            with open(os.path.join(args.outdir, "best_readme.txt"), "w") as f:
                f.write(f"Best epoch {ep}\nMAE(mm) Acetal={mae_ac:.3f}, Air={mae_air:.3f}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["math","cst","phys"])
    ap.add_argument("--model", default="resnet1d_v2", choices=["mlp","resnet1d_v3t","transformer"])
    ap.add_argument("--csv", required=True)
    ap.add_argument("--feat_scaler", required=True)
    ap.add_argument("--train_ids", default=None)
    ap.add_argument("--val_ids",   default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--freeze_stem", type=int, default=0)
    # training settings
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--loss", choices=["l1","smoothl1"], default="smoothl1")
    ap.add_argument("--w_acetal", type=float, default=1.0)
    ap.add_argument("--w_air",    type=float, default=1.3)
    # target norm
    ap.add_argument("--y_norm", choices=["zscore"], default="zscore")
    ap.add_argument("--y_pad_frac", type=float, default=0.0)
    # splits for math/cst
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    # misc
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    train_one_stage(args)

if __name__ == "__main__":
    main()
