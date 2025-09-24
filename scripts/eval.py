# scripts/eval.py
import os, sys, argparse, numpy as np, torch, pandas as pd
from torch.utils.data import DataLoader

# make repo importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.s11_dataset import S11DomainDataset
from utils.target_norm import TargetScaler

from modelling.MLP import MLPThicknessModel
from modelling.ResNet1D import ResNet1DThicknessModel
from modelling.ResNet1D_v2 import ResNet1DThicknessModelV2
from modelling.Transformer import TransformerThicknessModel

def build_model(kind: str):
    if kind == "mlp":            return MLPThicknessModel()
    if kind == "resnet1d":       return ResNet1DThicknessModel()
    if kind == "resnet1d_v2":    return ResNet1DThicknessModelV2()
    if kind == "transformer":    return TransformerThicknessModel()
    raise ValueError(f"unknown model kind: {kind}")

def _as_tensor(x, device):
    return x if torch.is_tensor(x) else torch.tensor(x, dtype=torch.float32, device=device)

def load_tnorm_from_ckpt(ckpt, device):
    state = ckpt.get("tnorm_state", None)
    if state is not None:
        tnorm = TargetScaler(np.zeros((1,2)))
        try: tnorm = tnorm.to(device)
        except Exception: pass
        try:
            tnorm.load_state_dict(state)   # modern path
            return tnorm
        except Exception:
            pass
        if isinstance(state, dict):
            if "kind" in state: setattr(tnorm, "kind", state["kind"])
            for k in ("mu","std","ymin","ymax"):
                if k in state and state[k] is not None:
                    v = state[k]; v = v if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32)
                    setattr(tnorm, k, v.to(device))
        return tnorm

    if "tnorm_mu" in ckpt and "tnorm_std" in ckpt:  # legacy z-score
        tnorm = TargetScaler(np.zeros((1,2)))
        try: tnorm = tnorm.to(device)
        except Exception: pass
        tnorm.mu  = _as_tensor(ckpt["tnorm_mu"],  device)
        tnorm.std = _as_tensor(ckpt["tnorm_std"], device)
        if not hasattr(tnorm, "kind"):
            setattr(tnorm, "kind", "zscore")
        return tnorm

    tnorm = TargetScaler(np.zeros((1,2)))
    try: tnorm = tnorm.to(device)
    except Exception: pass
    return tnorm

def _ensure_tnorm_device(tnorm, device):
    def _move_attr(name):
        if hasattr(tnorm, name) and getattr(tnorm, name) is not None:
            v = getattr(tnorm, name)
            if not torch.is_tensor(v): v = torch.tensor(v, dtype=torch.float32)
            setattr(tnorm, name, v.to(device))
    for name in ("mu","std","ymin","ymax"):
        _move_attr(name)
    try: tnorm.to(device)
    except Exception: pass

# NEW: apply the right head at eval time
def apply_out_act_eval(yhat_logits, tnorm_kind: str, mode: str):
    """
    yhat_logits : raw model outputs
    tnorm_kind  : 'zscore' | 'minmax' | 'minmax_sym'
    mode        : 'auto' | 'linear' | 'sigmoid' | 'tanh' | 'clamp'
    """
    if mode == "linear":  return yhat_logits
    if mode == "sigmoid": return torch.sigmoid(yhat_logits)
    if mode == "tanh":    return torch.tanh(yhat_logits)
    if mode == "clamp":
        if tnorm_kind == "minmax":      return yhat_logits.clamp(0.0, 1.0)
        if tnorm_kind == "minmax_sym":  return yhat_logits.clamp(-1.0, 1.0)
        return yhat_logits

    # mode == "auto"
    if tnorm_kind == "zscore":      return yhat_logits
    if tnorm_kind == "minmax":      return torch.sigmoid(yhat_logits)
    if tnorm_kind == "minmax_sym":  return torch.tanh(yhat_logits)
    return yhat_logits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp","resnet1d","resnet1d_v2","transformer"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--phy_csv", required=True)
    ap.add_argument("--phy_scaler", required=True)
    ap.add_argument("--phy_test_ids", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=0)
    # NEW: control eval-time head (defaults to 'auto' based on scaler kind)
    ap.add_argument("--out_act", choices=["auto","linear","sigmoid","tanh","clamp"], default="auto")
    ap.add_argument("--out_csv", default="preds_test.csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ids = np.loadtxt(args.phy_test_ids, dtype=int).tolist()
    ds = S11DomainDataset(args.phy_csv, args.phy_scaler, keep_ids=test_ids)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    ckpt = torch.load(args.weights, map_location="cpu")
    model = build_model(args.model).to(device)
    model.load_state_dict(ckpt["model"], strict=False)

    tnorm = load_tnorm_from_ckpt(ckpt, device)
    _ensure_tnorm_device(tnorm, device)
    tkind = getattr(tnorm, "kind", "zscore")

    model.eval()
    rows = []
    mae_ac_list, mae_air_list = [], []
    idx_ptr = 0
    with torch.no_grad():
        for xb, yb in dl:
            bsz = xb.size(0)
            ids_batch = ds.IDs[idx_ptr:idx_ptr+bsz].tolist()
            idx_ptr += bsz

            xb = xb.to(device); yb = yb.to(device)
            ylogits = model(xb)                                    # raw outputs
            yhat_n  = apply_out_act_eval(ylogits, tkind, args.out_act)  # NEW
            yhat    = tnorm.decode(yhat_n)                         # back to mm

            err = (yhat - yb).abs()
            mae_ac_list.append(err[:,0]); mae_air_list.append(err[:,1])

            for i in range(bsz):
                rows.append(dict(
                    ID=int(ids_batch[i]),
                    y_ac=float(yb[i,0]), y_air=float(yb[i,1]),
                    yhat_ac=float(yhat[i,0]), yhat_air=float(yhat[i,1])
                ))

    mae_ac = torch.cat(mae_ac_list).mean().item()
    mae_air = torch.cat(mae_air_list).mean().item()
    print(f"TEST MAE (mm): Acetal={mae_ac:.3f}, Air={mae_air:.3f}  [kind={tkind}, out_act={args.out_act}]")

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[OK] wrote {args.out_csv}")

if __name__ == "__main__":
    main()
