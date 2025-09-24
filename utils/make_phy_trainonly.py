# utils/make_phy_trainonly.py
import numpy as np, pandas as pd, os, sys

PHY_CSV = "dataset/Phy_S11.csv"
TRAIN_IDS_TXT = "phy_train.txt"
OUT_CSV = "dataset/Phy_S11_trainOnly.csv"

if not os.path.isfile(PHY_CSV):
    sys.exit(f"Missing {PHY_CSV}")
if not os.path.isfile(TRAIN_IDS_TXT):
    sys.exit(f"Missing {TRAIN_IDS_TXT} (run utils/make_phy_split.py first)")

ids = set(map(int, np.loadtxt(TRAIN_IDS_TXT, dtype=int)))
df = pd.read_csv(PHY_CSV)

df_tr = df[df["ID"].isin(ids)].copy()
if df_tr.empty:
    sys.exit("No rows matched train IDs (check split files).")

os.makedirs("dataset", exist_ok=True)
df_tr.to_csv(OUT_CSV, index=False)

print(f"[OK] Train IDs: {len(ids)} | Rows kept: {len(df_tr)}")
print(f"[OK] Wrote {OUT_CSV}")
