r"""
ChandraMatch - Cross-Sensor (OHRC <-> TMC-2/IIRS) Registration
==================================================================
Estimates each scene's real ground-sample-distance (GSD, meters per
pixel) from its geometry CSV, using great-circle distance on a lunar
sphere between two widely-separated sample points, then resizes the
higher-resolution image down so both images represent comparable
physical scale before handing them to the normal pipeline.

Usage:
    python cross_sensor.py \
        --geo1 "...\geometry\...\file.csv" --xml1 "...\data\...\file.xml" \
        --geo2 "...\geometry\...\file.csv" --xml2 "...\data\...\file.xml"
"""

import argparse
import csv
import math
import time

import cv2

from find_overlaps import LAT_COL_CANDIDATES, LON_COL_CANDIDATES, pick_column
from chandramatch_pipeline import (
    load_pds4_image,
    preprocess,
    detect_and_describe,
    match_features,
    estimate_homography,
    evaluate,
)

MOON_RADIUS_M = 1_737_400  # mean lunar radius, meters

PIXEL_COL_CANDIDATES = ["Pixel", "Sample", "pixel", "sample"]
LINE_COL_CANDIDATES = ["Scan", "Line", "scan", "line"]


def estimate_gsd(geometry_csv_path):
    """Estimates meters/pixel for a scene from its geometry CSV, using
    the two sampled rows with the largest pixel-index separation."""
    with open(geometry_csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        lat_i = pick_column(header, LAT_COL_CANDIDATES)
        lon_i = pick_column(header, LON_COL_CANDIDATES)
        pix_i = pick_column(header, PIXEL_COL_CANDIDATES)
        line_i = pick_column(header, LINE_COL_CANDIDATES)
        if None in (lat_i, lon_i, pix_i, line_i):
            raise ValueError(
                f"Could not find lat/lon/pixel/line columns in {geometry_csv_path}. "
                f"Header was: {header}"
            )
        rows = []
        for row in reader:
            try:
                rows.append((
                    float(row[lat_i]), float(row[lon_i]),
                    float(row[pix_i]), float(row[line_i]),
                ))
            except (ValueError, IndexError):
                continue

    if len(rows) < 2:
        raise ValueError(f"Not enough valid geometry rows in {geometry_csv_path}")

    def px_dist(a, b):
        return math.hypot(a[2] - b[2], a[3] - b[3])

    sample = rows[::max(1, len(rows) // 50)]
    best_pair, best_d = None, -1
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            d = px_dist(sample[i], sample[j])
            if d > best_d:
                best_d, best_pair = d, (sample[i], sample[j])

    if best_pair is None or best_d == 0:
        raise ValueError(f"Degenerate geometry file: {geometry_csv_path}")

    (lat1, lon1, _, _), (lat2, lon2, _, _) = best_pair
    lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1, math.sqrt(a)))
    ground_dist_m = MOON_RADIUS_M * c
    return ground_dist_m / best_d


def load_scale_normalized_pair(xml1, geo1, xml2, geo2):
    gsd1 = estimate_gsd(geo1)
    gsd2 = estimate_gsd(geo2)
    ratio = max(gsd1, gsd2) / min(gsd1, gsd2)
    print(f"Estimated GSD: image1={gsd1:.3f} m/px, image2={gsd2:.3f} m/px (ratio={ratio:.1f}x)")

    img1 = load_pds4_image(xml1)
    img2 = load_pds4_image(xml2)

    if gsd1 < gsd2:
        scale = gsd1 / gsd2
        h, w = img1.shape[:2]
        img1 = cv2.resize(img1, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        print(f"Resized image1 by {scale:.4f}x to match image2's scale -> {img1.shape[:2]}")
    elif gsd2 < gsd1:
        scale = gsd2 / gsd1
        h, w = img2.shape[:2]
        img2 = cv2.resize(img2, (max(1, int(w * scale)), max(1, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        print(f"Resized image2 by {scale:.4f}x to match image1's scale -> {img2.shape[:2]}")

    return img1, img2


def main():
    parser = argparse.ArgumentParser(description="Cross-sensor registration")
    parser.add_argument("--geo1", required=True)
    parser.add_argument("--xml1", required=True)
    parser.add_argument("--geo2", required=True)
    parser.add_argument("--xml2", required=True)
    parser.add_argument("--detector", default="orb", choices=["orb", "sift"])
    args = parser.parse_args()

    img1, img2 = load_scale_normalized_pair(args.xml1, args.geo1, args.xml2, args.geo2)

    t0 = time.time()
    gray1, _ = preprocess(img1)
    gray2, _ = preprocess(img2)
    kp1, desc1 = detect_and_describe(gray1, args.detector)
    kp2, desc2 = detect_and_describe(gray2, args.detector)
    matches = match_features(desc1, desc2, args.detector)
    H, mask = estimate_homography(kp1, kp2, matches)

    if H is None:
        print("Registration failed: not enough good matches even after scale normalization.")
        return

    metrics = evaluate(kp1, kp2, matches, mask, H, time.time() - t0)
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()