# utils/fit_normalizer.py
# UPDATED: adds --repr {magphase,reim}, saves metadata (repr, unwrap, remove_offset)
# Original behavior (mag/phase + unwrap) is preserved by default.

import argparse, os, sys
import numpy as np
import pandas as pd

# ---------- helpers ----------
def unwrap_deg(vec_deg: np.ndarray, remove_offset=True) -> np.ndarray:
    ph = np.deg2rad(vec_deg.astype(np.float64))
    phu = np.unwrap(ph, axis=-1)
    if remove_offset:
        phu = phu - phu[..., :1]
    return np.rad2deg(phu).astype(np.float32)

def db_to_lin(db):
    # |S11| linear magnitude from dB where dB = 20*log10(|S11|)
    return (10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)).astype(np.float32)

def make_features(mag_db, pha_deg, repr_mode, unwrap_phase, remove_offset):
    """Return [2, 200] feature array depending on repr_mode."""
    if repr_mode == "magphase":
        # OLD behavior
        pha = pha_deg.astype(np.float32)
        if unwrap_phase:
            pha = unwrap_deg(pha, remove_offset=remove_offset)
        elif remove_offset:
            # If we don't unwrap but still remove DC, do a simple constant shift
            pha = pha - pha[0]
        x0 = np.stack([mag_db.astype(np.float32), pha.astype(np.float32)], axis=0)  # [2,200]
        return x0
    elif repr_mode == "reim":
        # NEW: build real/imag directly from raw mag(dB) and phase(deg)
        mag_lin = db_to_lin(mag_db)                          # |S11|
        phi = np.deg2rad(pha_deg.astype(np.float32))         # radians
        real = (mag_lin * np.cos(phi)).astype(np.float32)
        imag = (mag_lin * np.sin(phi)).astype(np.float32)
        # NOTE: unwrap/offset are not needed in re/im; ignore if passed.
        return np.stack([real, imag], axis=0)
    else:
        raise ValueError(f"Unknown repr_mode: {repr_mode}")

def load_tall_csv(csv_path, repr_mode="magphase", unwrap_phase=False, remove_offset=True):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    req = {"ID","Magnitude(dB)","Phase(degree)"}
    missing = list(req - set(df.columns))
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    X = []
    n_ids = 0
    for gid, g in df.groupby("ID", sort=True):
        mag = g["Magnitude(dB)"].to_numpy(np.float32)
        pha = g["Phase(degree)"].to_numpy(np.float32)
        if len(mag) != 200 or len(pha) != 200:
            continue
        x0 = make_features(mag, pha, repr_mode, unwrap_phase, remove_offset)
        X.append(x0)                                # [2,200]
        n_ids += 1
    if n_ids == 0:
        raise RuntimeError(f"{csv_path}: no valid 200-pt IDs found")
    X = np.stack(X, axis=0)                         # [N,2,200]
    return X, n_ids

def fit_stats(arr, mode="zscore"):
    if mode == "zscore":
        mu  = arr.mean(axis=0)                      # [2,200]
        std = arr.std(axis=0, ddof=0)
    elif mode == "robust":
        med = np.median(arr, axis=0)
        q25 = np.quantile(arr, 0.25, axis=0)
        q75 = np.quantile(arr, 0.75, axis=0)
        mu, std = med, (q75 - q25) / 1.349
    else:
        raise ValueError("mode must be 'zscore' or 'robust'")
    std = np.where(std < 1e-6, 1e-6, std).astype(np.float32)
    return mu.astype(np.float32), std

def main():
    ap = argparse.ArgumentParser(description="Fit per-(channel,frequency) normalizer for 2x200 S11 data")
    ap.add_argument("--csv", nargs="+", required=True, help="One or more tall CSVs")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--mode", choices=["zscore","robust"], default="zscore")
    # CHANGED: add representation switch
    ap.add_argument("--repr", choices=["magphase","reim"], default="magphase",
                    help="Feature representation to normalize. Default=magphase (backward compatible).")
    # keep legacy flags (only apply to mag/phase)
    ap.add_argument("--unwrap-phase", action="store_true")
    ap.add_argument("--no-offset-remove", action="store_true")
    args = ap.parse_args()

    print("[INFO] settings:",
          f"\n  csvs={args.csv}",
          f"\n  out={args.out}",
          f"\n  mode={args.mode}",
          f"\n  repr={args.repr}",
          f"\n  unwrap_phase={args.unwrap_phase}",
          f"\n  remove_offset={not args.no_offset_remove}", flush=True)

    if args.repr == "reim" and (args.unwrap_phase or args.no_offset_remove):
        print("[WARN] unwrap/offset flags are ignored for repr=reim.", flush=True)

    all_X = []
    total_ids = 0
    for p in args.csv:
        X, n_ids = load_tall_csv(
            p,
            repr_mode=args.repr,
            unwrap_phase=args.unwrap_phase,
            remove_offset=not args.no_offset_remove
        )
        all_X.append(X)
        total_ids += n_ids
        print(f"[OK] loaded {p}: {n_ids} IDs", flush=True)

    X = np.concatenate(all_X, axis=0)               # [N,2,200]
    print(f"[INFO] concatenated: X.shape={X.shape}", flush=True)

    mu, std = fit_stats(X, mode=args.mode)
    print(f"[INFO] mu/std shapes: {mu.shape} / {std.shape}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        mu=mu, std=std,
        mode=args.mode,
        repr=args.repr,                    # ADDED: tell the dataset which representation
        unwrap=args.unwrap_phase,          # ADDED: saved for provenance
        remove_offset=not args.no_offset_remove
    )

    print(f"[OK] wrote {args.out}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(1)
