"""
ChandraMatch - Results Analysis
=================================
Reads batch_results.csv and produces report-ready plots (saved as PNGs,
not just shown on screen) plus a summary table printed to the terminal.

Run with:
    python analyze_results.py
    python analyze_results.py --csv batch_results.csv --out report_figures
"""

import argparse
import os
import re
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

TIMESTAMP_RE = re.compile(r"(\d{8}T\d{6})")


def parse_timestamp(scene_name):
    """Extracts the acquisition datetime from a scene name like
    ch2_ohr_ncp_20260102T1224107393_d_img_d18 -> datetime(2026,1,2,12,24,10)."""
    m = TIMESTAMP_RE.search(scene_name)
    if not m:
        return None
    # keep date + HHMMSS, drop the extra sub-second digits
    stamp = m.group(1)[:15]  # "20260102T122410"
    return datetime.strptime(stamp, "%Y%m%dT%H%M%S")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="batch_results.csv")
    parser.add_argument("--out", default="report_figures")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df["status"] == "ok"].copy()
    print(f"Loaded {len(df)} successful runs from {args.csv}")

    # --- derive time gap between each pair, in hours ---
    def gap_hours(row):
        t1 = parse_timestamp(row["pair_a"])
        t2 = parse_timestamp(row["pair_b"])
        if t1 is None or t2 is None:
            return None
        return abs((t2 - t1).total_seconds()) / 3600.0

    df["time_gap_hours"] = df.apply(gap_hours, axis=1)

    # --- Plot 1: inlier ratio vs. time gap (the key robustness result) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for det, group in df.groupby("detector"):
        ax.scatter(group["time_gap_hours"], group["inlier_ratio"], label=det, alpha=0.75)
    ax.set_xlabel("Time gap between scenes (hours)")
    ax.set_ylabel("Inlier ratio")
    ax.set_title("Registration quality vs. acquisition time gap")
    ax.set_xscale("symlog")  # gaps span hours to weeks; log-ish scale reads better
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "inlier_ratio_vs_time_gap.png"), dpi=150)
    plt.close(fig)

    # --- Plot 2: ORB vs SIFT, paired comparison per scene-pair ---
    pivot = df.pivot_table(index=["pair_a", "pair_b"], columns="detector",
                            values="inlier_ratio").reset_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(pivot))
    width = 0.35
    if "orb" in pivot.columns:
        ax.bar([i - width / 2 for i in x], pivot["orb"], width, label="ORB")
    if "sift" in pivot.columns:
        ax.bar([i + width / 2 for i in x], pivot["sift"], width, label="SIFT")
    ax.set_xlabel("Scene pair")
    ax.set_ylabel("Inlier ratio")
    ax.set_title("ORB vs SIFT inlier ratio, per pair")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"P{i+1}" for i in x], rotation=0)
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "orb_vs_sift_per_pair.png"), dpi=150)
    plt.close(fig)

    # --- Plot 3: distribution of inlier ratios by detector (boxplot) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    data = [df[df["detector"] == d]["inlier_ratio"].values for d in sorted(df["detector"].unique())]
    ax.boxplot(data, tick_labels=sorted(df["detector"].unique()))
    ax.set_ylabel("Inlier ratio")
    ax.set_title("Inlier ratio distribution by detector")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "inlier_ratio_distribution.png"), dpi=150)
    plt.close(fig)

    # --- Summary stats printed to terminal (paste into your report text) ---
    print("\n=== Summary by detector ===")
    print(df.groupby("detector")["inlier_ratio"].agg(["mean", "median", "std", "count"]).round(4))

    same_day = df[df["time_gap_hours"] < 24]
    far_gap = df[df["time_gap_hours"] >= 24]
    print("\n=== Same-day pairs (<24h gap) ===")
    print(f"  n={len(same_day)}  mean inlier ratio={same_day['inlier_ratio'].mean():.4f}")
    print("=== Larger-gap pairs (>=24h gap) ===")
    print(f"  n={len(far_gap)}  mean inlier ratio={far_gap['inlier_ratio'].mean():.4f}")

    corr = df["time_gap_hours"].corr(df["inlier_ratio"])
    print(f"\nCorrelation (time gap vs inlier ratio): {corr:.3f}")

    print(f"\nFigures saved to {os.path.abspath(args.out)}/")
    print("  - inlier_ratio_vs_time_gap.png")
    print("  - orb_vs_sift_per_pair.png")
    print("  - inlier_ratio_distribution.png")


if __name__ == "__main__":
    main()
