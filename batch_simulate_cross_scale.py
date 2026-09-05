r"""
ChandraMatch - Batch Cross-Scale Simulation
==============================================
Runs simulate_cross_scale.py's ground-truth cross-scale validation
across every real OHRC scene found under a data folder, logging every
result to CSV (same incremental, crash-safe write pattern as
batch_run.py — results are saved after each run, not just at the end).

Usage:
    python batch_simulate_cross_scale.py --scan "C:\...\chandra match\data"
    python batch_simulate_cross_scale.py --scan "C:\...\chandra match\data" --detector both
"""

import argparse
import csv
import glob
import os

from simulate_cross_scale import run_simulation

FIELDNAMES = [
    "scene", "detector", "status",
    "num_features_img1", "num_features_img2", "num_candidate_matches", "num_inlier_matches",
    "inlier_ratio", "reprojection_rmse_px", "processing_time_s",
    "true_scale", "recovered_scale", "scale_error_pct",
    "true_angle_deg", "recovered_angle_deg", "angle_error_deg",
]


def find_all_xml(scan_root):
    """Finds every PDS4 .xml under <product>/data/calibrated/**/ — this
    pattern avoids also matching the .xml labels that sit next to
    geometry CSVs or browse images."""
    pattern = os.path.join(scan_root, "**", "data", "calibrated", "**", "*.xml")
    return sorted(set(glob.glob(pattern, recursive=True)))


def main():
    parser = argparse.ArgumentParser(description="Batch-run the cross-scale simulation over every OHRC scene")
    parser.add_argument("--scan", required=True, help="Root data folder to scan for OHRC products")
    parser.add_argument("--detector", default="orb", choices=["orb", "sift", "both"])
    parser.add_argument("--out", default="cross_scale_results.csv")
    args = parser.parse_args()

    xmls = find_all_xml(args.scan)
    print(f"Found {len(xmls)} PDS4 .xml file(s) under data/calibrated/.\n")
    if not xmls:
        print("Nothing to run.")
        return

    detectors = ["orb", "sift"] if args.detector == "both" else [args.detector]
    total = len(xmls) * len(detectors)
    run_i = 0

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval="")
        writer.writeheader()
        f.flush()

        for xml in xmls:
            scene = os.path.basename(xml)
            for det in detectors:
                run_i += 1
                print(f"[{run_i}/{total}] {scene} ({det})...")
                try:
                    result = run_simulation(xml, detector=det, verbose=False)
                except Exception as e:
                    result = {"status": f"error: {e}"}
                result["scene"] = scene
                result["detector"] = det
                print(f"  -> {result.get('status')}  "
                      f"inlier_ratio={result.get('inlier_ratio', 'n/a')}  "
                      f"scale_err%={result.get('scale_error_pct', 'n/a')}  "
                      f"angle_err_deg={result.get('angle_error_deg', 'n/a')}")
                writer.writerow(result)
                f.flush()  # crash-safe: every row is saved immediately

    print(f"\nDone. Results written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
