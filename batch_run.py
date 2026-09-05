r"""
ChandraMatch - Batch Pair Processing
======================================
Combines find_overlaps.py (geometry-based pair discovery) with
chandramatch_pipeline.py (the registration pipeline): scans a data
folder, finds every overlapping pair automatically, runs the full
pipeline on each one, and logs every result to a CSV.

This is what turns "we ran the pipeline twice by hand" into a proper
batch evaluation — this CSV is the spine of your report's Results chapter.

Usage:
    python batch_run.py --scan "C:\...\chandra match\data"
    python batch_run.py --scan "C:\...\chandra match\data" --detector both
"""

import argparse
import csv
import glob
import os
import time
from itertools import combinations

from find_overlaps import find_geometry_csvs, get_bbox, bboxes_overlap, product_name
from chandramatch_pipeline import (
    load_pds4_image,
    preprocess,
    detect_and_describe,
    match_features,
    estimate_homography,
    evaluate,
)


def find_product_root(geometry_csv_path):
    """Walk up from .../<product>/geometry/... to <product>'s root folder."""
    d = os.path.dirname(geometry_csv_path)
    while d and os.path.basename(d) != "geometry":
        d = os.path.dirname(d)
    return os.path.dirname(d) if d else None


def find_xml_for_product(product_root):
    """Finds the PDS4 .xml label under <product>/data/calibrated/**/*.xml"""
    pattern = os.path.join(product_root, "data", "**", "*.xml")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def run_one_pair(img1, img2, detector, name1, name2):
    """Takes already-loaded images (see main()'s cache) so repeated
    scenes across multiple pairs don't get re-read from disk."""
    t0 = time.time()
    gray1, _ = preprocess(img1)
    gray2, _ = preprocess(img2)
    kp1, desc1 = detect_and_describe(gray1, detector)
    kp2, desc2 = detect_and_describe(gray2, detector)
    matches = match_features(desc1, desc2, detector)
    H, mask = estimate_homography(kp1, kp2, matches)

    if H is None:
        return {
            "pair_a": name1, "pair_b": name2, "detector": detector,
            "status": "failed_no_homography",
        }

    elapsed = time.time() - t0
    metrics = evaluate(kp1, kp2, matches, mask, H, elapsed)
    metrics.update({"pair_a": name1, "pair_b": name2, "detector": detector, "status": "ok"})
    return metrics


def discover_pairs(scan_root):
    csvs = find_geometry_csvs(scan_root)
    print(f"Found {len(csvs)} geometry file(s).")

    bboxes = {}
    for path in csvs:
        try:
            bboxes[path] = get_bbox(path)
        except ValueError as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")

    pairs = [
        (p1, p2) for (p1, b1), (p2, b2) in combinations(bboxes.items(), 2)
        if bboxes_overlap(b1, b2)
    ]
    print(f"Found {len(pairs)} overlapping pair(s).\n")
    return pairs


FIELDNAMES = [
    "pair_a", "pair_b", "detector", "status",
    "num_features_img1", "num_features_img2", "num_candidate_matches",
    "num_inlier_matches", "inlier_ratio", "reprojection_rmse_px", "processing_time_s",
]


def main():
    parser = argparse.ArgumentParser(description="Batch-run ChandraMatch over every overlapping pair")
    parser.add_argument("--scan", required=True, help="Root data folder to scan for products")
    parser.add_argument("--detector", default="orb", choices=["orb", "sift", "both"])
    parser.add_argument("--out", default="batch_results.csv")
    args = parser.parse_args()

    pairs = discover_pairs(args.scan)
    if not pairs:
        print("No overlapping pairs found — nothing to run.")
        return

    detectors = ["orb", "sift"] if args.detector == "both" else [args.detector]
    total_runs = len(pairs) * len(detectors)

    # Cache: load each unique scene from disk at most once per script run,
    # even if it appears in several overlapping pairs. This is what was
    # making a 10-scene run take forever — every pair was re-reading
    # ~1GB files from scratch with no reuse.
    image_cache = {}

    def get_image(xml_path):
        if xml_path not in image_cache:
            print(f"  (loading {os.path.basename(xml_path)} — first time seeing this scene)")
            image_cache[xml_path] = load_pds4_image(xml_path)
        return image_cache[xml_path]

    # Write incrementally + flush after every row, so if this gets
    # interrupted (Ctrl+C, crash, laptop sleeps) you keep every result
    # computed so far instead of losing everything.
    run_i = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval="")
        writer.writeheader()
        f.flush()

        for i, (geo1, geo2) in enumerate(pairs, 1):
            root1, root2 = find_product_root(geo1), find_product_root(geo2)
            xml1, xml2 = find_xml_for_product(root1), find_xml_for_product(root2)
            name1, name2 = product_name(geo1), product_name(geo2)

            if not xml1 or not xml2:
                print(f"[pair {i}/{len(pairs)}] Skipping {name1} x {name2}: no .xml found")
                continue

            try:
                img1 = get_image(xml1)
                img2 = get_image(xml2)
            except Exception as e:
                print(f"[pair {i}/{len(pairs)}] Failed to load images: {e}")
                continue

            for det in detectors:
                run_i += 1
                print(f"[run {run_i}/{total_runs}] {name1} x {name2} ({det})...")
                try:
                    result = run_one_pair(img1, img2, det, name1, name2)
                except Exception as e:
                    result = {"pair_a": name1, "pair_b": name2, "detector": det, "status": f"error: {e}"}
                print(f"  -> {result.get('status')}  inlier_ratio={result.get('inlier_ratio', 'n/a')}")
                writer.writerow(result)
                f.flush()  # <- safe on disk immediately, not just at the end

    print(f"\nDone. Results written incrementally to {args.out}")


if __name__ == "__main__":
    main()
