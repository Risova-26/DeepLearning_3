# scripts/make_report.py
# Make simple, presentation-friendly reports from prediction CSVs produced by scripts/eval.py.
# For each CSV, we compute metrics and export 3 plots per target:
#   1) y (true) vs yhat (pred) with 45° line
#   2) error histogram (yhat - y)
#   3) error vs true (to see bias/heteroscedasticity)
# We also produce a summary CSV across all inputs and a bar chart comparing MAE.

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- metrics ----------
def compute_metrics(y, yhat, tol_list=(0.05, 0.10)):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    err = yhat - y
    abs_err = np.abs(err)
    mae = abs_err.mean()
    rmse = np.sqrt(np.mean(err**2))
    bias = err.mean()
    # R^2 (handle zero variance edge case)
    ss_res = np.sum(err**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    p95 = np.percentile(abs_err, 95)
    within = {f"within_{tol:.3f}mm": float((abs_err <= tol).mean()) for tol in tol_list}
    d = dict(MAE=mae, RMSE=rmse, Bias=bias, R2=r2, P95=p95)
    d.update(within)
    return d

# ---------- plotting helpers ----------
def _ensure_outdir(path):
    os.makedirs(path, exist_ok=True)

def scatter_y_vs_yhat(y, yhat, title, out_png):
    plt.figure(figsize=(6,6))
    plt.scatter(y, yhat, s=8, alpha=0.6)
    lo, hi = float(np.min([y.min(), yhat.min()])), float(np.max([y.max(), yhat.max()]))
    pad = 0.05 * (hi - lo + 1e-9)
    plt.plot([lo-pad, hi+pad], [lo-pad, hi+pad], linestyle='--')  # 45° reference
    plt.xlabel("True (mm)")
    plt.ylabel("Predicted (mm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def hist_error(y, yhat, title, out_png, bins=40):
    err = np.asarray(yhat) - np.asarray(y)
    plt.figure(figsize=(6,4))
    plt.hist(err, bins=bins)
    plt.axvline(0.0, linestyle='--')
    plt.xlabel("Error (Pred - True) [mm]")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def error_vs_true(y, yhat, title, out_png):
    err = np.asarray(yhat) - np.asarray(y)
    plt.figure(figsize=(6,4))
    plt.scatter(y, err, s=8, alpha=0.6)
    plt.axhline(0.0, linestyle='--')
    plt.xlabel("True (mm)")
    plt.ylabel("Error (Pred - True) [mm]")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def bar_compare_mae(summary_df, out_png):
    # summary_df has columns: Dataset, Target, MAE
    piv = summary_df.pivot(index="Dataset", columns="Target", values="MAE")
    plt.figure(figsize=(7,4))
    x = np.arange(len(piv.index))
    w = 0.35
    mae_ac = piv.get("Acetal", pd.Series([np.nan]*len(x), index=piv.index))
    mae_air = piv.get("Air",    pd.Series([np.nan]*len(x), index=piv.index))
    plt.bar(x - w/2, mae_ac.values, width=w, label="Acetal MAE")
    plt.bar(x + w/2, mae_air.values, width=w, label="Air MAE")
    plt.xticks(x, piv.index, rotation=20)
    plt.ylabel("MAE (mm)")
    plt.title("MAE comparison across datasets/models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Build plots + summary from prediction CSVs.")
    ap.add_argument("--pred", action="append", required=True,
                    help="Label=path/to/preds.csv  (repeatable)")
    ap.add_argument("--outdir", default="report_out", help="Where to save plots/summary.")
    ap.add_argument("--tol", type=float, nargs="*", default=[0.05, 0.10],
                    help="Tolerance thresholds (mm) for within-% metrics.")
    args = ap.parse_args()

    _ensure_outdir(args.outdir)

    # Parse inputs
    items = []
    for spec in args.pred:
        if "=" not in spec:
            raise SystemExit(f"Expected Label=path.csv, got: {spec}")
        label, path = spec.split("=", 1)
        if not os.path.isfile(path):
            raise SystemExit(f"CSV not found: {path}")
        items.append((label.strip(), path.strip()))

    rows_summary = []
    for label, csv_path in items:
        df = pd.read_csv(csv_path)
        for target, cols in [
            ("Acetal", ("y_ac","yhat_ac")),
            ("Air",    ("y_air","yhat_air")),
        ]:
            if not all(c in df.columns for c in cols):
                raise SystemExit(f"{csv_path}: missing columns for {target}: {cols}")
            y     = df[cols[0]].values
            yhat  = df[cols[1]].values
            # Metrics
            m = compute_metrics(y, yhat, tol_list=args.tol)
            # Save per-target plots
            base = os.path.join(args.outdir, f"{label}_{target.lower()}")
            scatter_y_vs_yhat(y, yhat, f"{label} — {target}: True vs Pred", base+"_scatter.png")
            hist_error(y, yhat,      f"{label} — {target}: Error Histogram", base+"_errhist.png")
            error_vs_true(y, yhat,   f"{label} — {target}: Error vs True",   base+"_err_vs_true.png")
            # Collect summary
            row = dict(Dataset=label, Target=target)
            row.update(m)
            rows_summary.append(row)

    summary = pd.DataFrame(rows_summary)
    # Friendly rounding
    for c in ["MAE","RMSE","Bias","P95"] + [c for c in summary.columns if c.startswith("within_")]:
        if c in summary.columns:
            summary[c] = summary[c].astype(float)
    summary.to_csv(os.path.join(args.outdir, "summary_metrics.csv"), index=False)

    # Bar comparison (MAE)
    bar_compare_mae(summary, os.path.join(args.outdir, "compare_mae.png"))

    # Console print
    print("\n=== Summary (mm) ===")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(summary.to_string(index=False))
    print(f"\n[OK] Wrote plots + summary to: {args.outdir}")
    print(f"     - {os.path.join(args.outdir, 'summary_metrics.csv')}")
    print(f"     - {os.path.join(args.outdir, 'compare_mae.png')}")
    print("     - Per-dataset plots: *_scatter.png, *_errhist.png, *_err_vs_true.png")

if __name__ == "__main__":
    main()
