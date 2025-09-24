# train.py
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse, os, math, time
import numpy as np, torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from utils.s11_dataset import S11DomainDataset
from utils.target_norm import TargetScaler

# === import your models ===
from modelling.MLP import MLPThicknessModel
from modelling.ResNet1D import ResNet1DThicknessModel
from modelling.Transformer import TransformerThicknessModel

def batch_collate(batch):
    xs, ys = zip(*batch)
    # if you later re-enable domain_onehot, xs[i] will be a tuple (x, onehot)
    if isinstance(xs[0], tuple):
        x = torch.stack([t[0] for t in xs], 0)  # [B, 2, 200]
        d = torch.stack([t[1] for t in xs], 0)  # [B, 2]
        # append domain one-hot as 2 extra constant channels across 200 points
        x = torch.cat([x, d.unsqueeze(-1).repeat(1, 1, 200)], dim=1)  # [B, 4, 200]
    else:
        x = torch.stack(xs, 0)  # [B, 2, 200]
    y = torch.stack(ys, 0)      # [B, 2]
    return x, y

def build_model(kind):
    if kind == "mlp": return MLPThicknessModel()
    if kind == "resnet1d": return ResNet1DThicknessModel()
    if kind == "transformer": return TransformerThicknessModel()
    raise ValueError("model must be mlp|resnet1d|transformer")

def mae_mm(yhat, y): return (yhat - y).abs().mean(dim=0)  # per-target MAE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pretrain","finetune"], required=True)
    ap.add_argument("--model", choices=["mlp","resnet1d","transformer"], required=True)
    ap.add_argument("--math_csv"); ap.add_argument("--cst_csv"); ap.add_argument("--phy_csv")
    ap.add_argument("--math_scaler"); ap.add_argument("--cst_scaler"); ap.add_argument("--phy_scaler")
    ap.add_argument("--phy_train_ids"); ap.add_argument("--phy_val_ids"); ap.add_argument("--phy_test_ids")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--outdir", default="runs")
    ap.add_argument("--workers", type=int, default=(0 if os.name == "nt" else 2),help="DataLoader workers; 0 on Windows is safest")
    ap.add_argument("--init_weights", default=None, help="Path to pretrained checkpoint (.pt) to initialize model")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    # ----- datasets -----
    if args.stage == "pretrain":
        assert args.math_csv and args.cst_csv and args.math_scaler and args.cst_scaler
        ds_math = S11DomainDataset(args.math_csv, args.math_scaler)
        ds_cst  = S11DomainDataset(args.cst_csv,  args.cst_scaler)
        train_ds = ConcatDataset([ds_math, ds_cst])
        # simple split 90/10 for internal val
        n = len(train_ds); n_val = max(1,int(0.1*n))
        n_train = n - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    else:  # finetune
        assert args.phy_csv and args.phy_scaler and args.phy_train_ids and args.phy_val_ids
        phy_train_ids = np.loadtxt(args.phy_train_ids, dtype=int).tolist()
        phy_val_ids   = np.loadtxt(args.phy_val_ids, dtype=int).tolist()
        train_ds = S11DomainDataset(args.phy_csv, args.phy_scaler, keep_ids=phy_train_ids)
        val_ds   = S11DomainDataset(args.phy_csv, args.phy_scaler, keep_ids=phy_val_ids)

    # ----- loaders -----
    def collate(batch):
        xs, ys = zip(*batch)
        if isinstance(xs[0], tuple):  # (x, onehot)
            x = torch.stack([t[0] for t in xs], 0)
            d = torch.stack([t[1] for t in xs], 0)
            x = torch.cat([x, d.unsqueeze(-1).repeat(1,1,200)], dim=1)  # append 2 domain chans
        else:
            x = torch.stack(xs, 0)
        y = torch.stack(ys, 0)
        return x, y

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,num_workers=args.workers, pin_memory=True,persistent_workers=(args.workers > 0),collate_fn=batch_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,num_workers=args.workers, pin_memory=True,persistent_workers=(args.workers > 0),collate_fn=batch_collate)


    # If we appended domain channels, MLP expects 2*200 flattened; quick fix:
    # For MLP only, if input has 4 channels now (mag,phase,dm0,dm1), it will flatten 4*200=800 automatically.

    # ----- target scaler (z-score) -----
    # peek at a few train batches to compute y mean/std
    y_collect = []
    for i,(xb,yb) in enumerate(train_loader):
        y_collect.append(yb)
        if i>=10: break
    y_train = torch.cat(y_collect, 0).numpy()
    tnorm = TargetScaler(y_train).to(device)

    # ----- model/opt -----
    model = build_model(args.model).to(device)
    if args.init_weights:
        ck = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        print(f"[INFO] loaded init weights from {args.init_weights}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.MSELoss()

    best_val = float("inf"); best_path = os.path.join(args.outdir, f"{args.stage}_{args.model}.pt")

    for epoch in range(1, args.epochs+1):
        model.train(); tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb = tnorm.encode(yb)
            optim.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward(); optim.step()
            tr_loss += loss.item()*xb.size(0)
        tr_loss /= len(train_loader.dataset)

        # val
        model.eval(); vl_loss = 0.0; mae_ac=[]; mae_air=[]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                yb_n = tnorm.encode(yb)
                yhat_n = model(xb)
                vl_loss += loss_fn(yhat_n, yb_n).item()*xb.size(0)
                yhat = tnorm.decode(yhat_n)
                m = (yhat - yb).abs().mean(dim=0)
                mae_ac.append(m[0].item()); mae_air.append(m[1].item())
        vl_loss /= len(val_loader.dataset)
        mae_ac = float(np.mean(mae_ac)); mae_air = float(np.mean(mae_air))

        print(f"[{epoch:03d}] train={tr_loss:.4f}  val={vl_loss:.4f}  MAE(mm): Acetal={mae_ac:.3f}, Air={mae_air:.3f}")

        # early stop
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save({"model":model.state_dict(),
                        "tnorm_mu":tnorm.mu.cpu(), "tnorm_std":tnorm.std.cpu()}, best_path)

    print(f"[OK] saved best to {best_path}")

if __name__ == "__main__":
    main()
