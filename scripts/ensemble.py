# scripts/ensemble.py
import pandas as pd
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--mlp", required=True)
ap.add_argument("--resnet", required=True)
ap.add_argument("--trans", required=True)
ap.add_argument("--out_csv", default="preds_ensemble_test.csv")
args = ap.parse_args()

m = pd.read_csv(args.mlp)
r = pd.read_csv(args.resnet)
t = pd.read_csv(args.trans)

# inner-join on ID to guarantee row alignment
df = m.merge(r, on="ID", suffixes=("_mlp","_res"))
df = df.merge(t, on="ID")
# columns now: ID, y_ac_mlp, y_air_mlp, yhat_ac_mlp, yhat_air_mlp, y_ac_res, ..., y_ac, y_air, yhat_ac, yhat_air

# sanity: ground truth should match across files
assert np.allclose(df["y_ac_mlp"], df["y_ac_res"]) and np.allclose(df["y_air_mlp"], df["y_air_res"])
y_ac  = df["y_ac_mlp"].to_numpy()
y_air = df["y_air_mlp"].to_numpy()

# stack predictions
pred_ac = np.vstack([df["yhat_ac_mlp"].to_numpy(),
                     df["yhat_ac_res"].to_numpy(),
                     df["yhat_ac"].to_numpy()])  # transformer's column name from last merge is "yhat_ac"
pred_air = np.vstack([df["yhat_air_mlp"].to_numpy(),
                      df["yhat_air_res"].to_numpy(),
                      df["yhat_air"].to_numpy()])

# simple average
yhat_ac  = pred_ac.mean(axis=0)
yhat_air = pred_air.mean(axis=0)

mae_ac  = np.abs(yhat_ac  - y_ac).mean()
mae_air = np.abs(yhat_air - y_air).mean()
print(f"ENSEMBLE TEST MAE (mm): Acetal={mae_ac:.3f}, Air={mae_air:.3f}")

out = pd.DataFrame(dict(ID=df["ID"], y_ac=y_ac, y_air=y_air, yhat_ac=yhat_ac, yhat_air=yhat_air))
out.to_csv(args.out_csv, index=False)
print(f"[OK] wrote {args.out_csv}")
