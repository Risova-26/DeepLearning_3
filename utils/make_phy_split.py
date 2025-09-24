#!/usr/bin/env python3
"""
Make train / val / test splits for Physical data (by ID).

- Reads a tall CSV like dataset/Phy_S11.csv
- Groups rows by ID to recover per-sample labels
- STRATIFIES BY (Acetal, Air) so each pair is represented across splits
- Writes: <prefix>_train.txt, <prefix>_val.txt, <prefix>_test.txt
  (each contains one ID per line)

Defaults: 70% train, 15% val, 15% test, seed=42, stratified.
"""

import argparse, os
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/Phy_S11.csv",
                    help="Path to Physical tall CSV (ID,Magnitude,Phase,Label_1,Label_2)")
    ap.add_argument("--train", type=float, default=0.70, help="Train ratio")
    ap.add_argument("--val",   type=float, default=0.15, help="Val ratio")
    ap.add_argument("--test",  type=float, default=None, help="Test ratio (default=1-train-val)")
    ap.add_argument("--seed",  type=int,   default=42,   help="Random seed")
    ap.add_argument("--prefix", default="phy", help="Output prefix for txt files")
    ap.add_argument("--no-stratify", action="store_true",
                    help="If set, do a global random split (no (Acetal,Air) stratification)")
    return ap.parse_args()

def load_id_table(csv_path):
    df = pd.read_csv(csv_path)
    # collapse to one row per ID with labels
    rows = []
    for gid, g in df.groupby("ID", sort=True):
        if len(g) != 200:
            continue  # skip malformed IDs
        a = float(g["Label_1(Acetal)"].iloc[0])
        b = float(g["Label_2(Air)"].iloc[0])
        rows.append((int(gid), a, b))
    meta = pd.DataFrame(rows, columns=["ID","Acetal","Air"])
    return meta

def split_group(ids, r_train, r_val, rng):
    """Split a list of IDs (same (Acetal,Air)) into train/val/test with robust rules for small n."""
    n = len(ids)
    ids = list(ids)
    rng.shuffle(ids)

    if n == 1:
        return ids, [], []              # 1 -> train
    if n == 2:
        return [ids[0]], [], [ids[1]]   # 2 -> 1 train, 1 test
    # n >= 3
    n_tr = int(np.floor(r_train * n))
    n_va = int(np.floor(r_val   * n))
    n_te = n - n_tr - n_va

    # Ensure at least 1 in train and 1 in test when possible
    if n_tr < 1: n_tr = 1
    if n_te < 1:
        if n_tr > 1:
            n_tr -= 1; n_te += 1
        elif n_va > 0:
            n_va -= 1; n_te += 1

    tr = ids[:n_tr]
    va = ids[n_tr:n_tr+n_va]
    te = ids[n_tr+n_va:]
    return tr, va, te

def main():
    args = parse_args()
    r_train = args.train
    r_val   = args.val
    r_test  = 1.0 - r_train - r_val if args.test is None else args.test
    if not (0 < r_train < 1 and 0 <= r_val < 1):
        raise SystemExit("Ratios must be in (0,1).")
    if abs(r_train + r_val + r_test - 1.0) > 1e-6:
        raise SystemExit("train+val+test must sum to 1.0")

    meta = load_id_table(args.csv)
    if meta.empty:
        raise SystemExit(f"No valid 200-pt IDs found in {args.csv}")

    rng = np.random.default_rng(args.seed)

    if args.no_stratify:
        all_ids = meta["ID"].tolist()
        rng.shuffle(all_ids)
        n = len(all_ids)
        n_tr = int(np.floor(r_train*n))
        n_va = int(np.floor(r_val*n))
        tr = all_ids[:n_tr]
        va = all_ids[n_tr:n_tr+n_va]
        te = all_ids[n_tr+n_va:]
    else:
        # stratify by (Acetal, Air)
        tr, va, te = [], [], []
        for (a,b), g in meta.groupby(["Acetal","Air"], sort=True):
            g_ids = g["ID"].tolist()
            gi_tr, gi_va, gi_te = split_group(g_ids, r_train, r_val, rng)
            tr += gi_tr; va += gi_va; te += gi_te

    # final sanity: no overlaps and full coverage of selected IDs
    s_tr, s_va, s_te = set(tr), set(va), set(te)
    overlap = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    if overlap:
        raise SystemExit(f"Internal error: overlapping IDs {sorted(list(overlap))[:10]}")
    total = len(s_tr) + len(s_va) + len(s_te)

    print("=== Physical split summary ===")
    print(f"Source CSV: {args.csv}")
    print(f"Unique IDs found: {len(meta)}")
    print(f"Stratified by (Acetal,Air): {not args.no_stratify}")
    print(f"Ratios (train/val/test): {r_train:.2f}/{r_val:.2f}/{(1-r_train-r_val):.2f}")
    print(f"Split counts: train={len(s_tr)}  val={len(s_va)}  test={len(s_te)}  (total used={total})")

    # write lists
    np.savetxt(f"{args.prefix}_train.txt", sorted(s_tr), fmt="%d")
    np.savetxt(f"{args.prefix}_val.txt",   sorted(s_va), fmt="%d")
    np.savetxt(f"{args.prefix}_test.txt",  sorted(s_te), fmt="%d")
    print(f"[OK] wrote {args.prefix}_train.txt, {args.prefix}_val.txt, {args.prefix}_test.txt")

    # optional: small CSV report per pair
    rep = []
    tag = {i:"train" for i in s_tr} | {i:"val" for i in s_va} | {i:"test" for i in s_te}
    for _, row in meta.iterrows():
        rid = int(row["ID"]); rep.append((rid, row["Acetal"], row["Air"], tag.get(rid, "unused")))
    pd.DataFrame(rep, columns=["ID","Acetal","Air","split"]).to_csv(f"{args.prefix}_split_report.csv", index=False)
    print(f"[OK] wrote {args.prefix}_split_report.csv")

if __name__ == "__main__":
    main()
