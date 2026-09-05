r"""
ChandraMatch - Geometry-Based Pair Discovery
==============================================
Chandrayaan-2 OHRC/TMC-2 products each ship a "geometry" CSV containing
latitude/longitude samples across the image footprint (per ISSDC's own
product documentation: calibrated geometry products are "CSV ASCII Text
... containing latitude and longitude information at regular interval of
pixels and samples", typically named like
"..._g_grd_gds.csv" and sitting in each product's geometry/ folder).

This script:
  1. Finds every geometry CSV under a root data folder.
  2. Reads each one's lat/lon extent (a rough bounding box for that scene).
  3. Reports which pairs of scenes have overlapping bounding boxes —
     candidates worth actually running through chandramatch_pipeline.py.

This replaces manually guessing pairs from filenames/dates.

STEP 1 — confirm the real column names before trusting the scan (written
against ISSDC's documentation, not a file I've opened myself — same
caution that caught the ARRAY_0 surprise earlier):

    python find_overlaps.py --inspect "path\to\..._g_grd_gds.csv"

If the printed columns don't match Latitude/Longitude, edit
LAT_COL_CANDIDATES / LON_COL_CANDIDATES below to match what you see.

STEP 2 — once confirmed, scan a whole folder of downloaded products:

    python find_overlaps.py --scan "C:\...\chandra match\data"
"""

import argparse
import csv
import glob
import os
from itertools import combinations

# Adjust these if --inspect shows different column header names.
LAT_COL_CANDIDATES = ["Latitude", "Lat", "LATITUDE", "lat"]
LON_COL_CANDIDATES = ["Longitude", "Lon", "Long", "LONGITUDE", "lon"]


def find_geometry_csvs(root):
    pattern = os.path.join(root, "**", "geometry", "**", "*.csv")
    return glob.glob(pattern, recursive=True)


def inspect_csv(path, n=5):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Columns in {os.path.basename(path)}:")
        for i, col in enumerate(header):
            print(f"  [{i}] {col!r}")
        print(f"\nFirst {n} rows:")
        for i, row in enumerate(reader):
            if i >= n:
                break
            print(" ", row)


def pick_column(header, candidates):
    for c in candidates:
        if c in header:
            return header.index(c)
    for i, h in enumerate(header):
        if any(c.lower() in h.lower() for c in candidates):
            return i
    return None


def get_bbox(path):
    """Returns (lat_min, lat_max, lon_min, lon_max) for one geometry CSV."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        lat_i = pick_column(header, LAT_COL_CANDIDATES)
        lon_i = pick_column(header, LON_COL_CANDIDATES)
        if lat_i is None or lon_i is None:
            raise ValueError(
                f"Could not find lat/lon columns. Header was: {header}. "
                f"Run --inspect on this file and adjust LAT_COL_CANDIDATES/"
                f"LON_COL_CANDIDATES at the top of this script."
            )
        lats, lons = [], []
        for row in reader:
            try:
                lats.append(float(row[lat_i]))
                lons.append(float(row[lon_i]))
            except (ValueError, IndexError):
                continue
    if not lats:
        raise ValueError("No numeric lat/lon rows parsed from this file.")
    return min(lats), max(lats), min(lons), max(lons)


def product_name(geometry_csv_path):
    """Walk up from .../<product>/geometry/<date>/file.csv to <product>."""
    d = os.path.dirname(geometry_csv_path)
    while d and os.path.basename(d) != "geometry":
        d = os.path.dirname(d)
    return os.path.basename(os.path.dirname(d)) if d else os.path.basename(geometry_csv_path)


def bboxes_overlap(b1, b2):
    lat1_min, lat1_max, lon1_min, lon1_max = b1
    lat2_min, lat2_max, lon2_min, lon2_max = b2
    return (lat1_min <= lat2_max and lat2_min <= lat1_max
            and lon1_min <= lon2_max and lon2_min <= lon1_max)


def main():
    parser = argparse.ArgumentParser(description="Find overlapping Chandrayaan-2 scene pairs")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect one geometry CSV's columns")
    parser.add_argument("--scan", type=str, default=None, help="Scan a data folder for overlapping pairs")
    args = parser.parse_args()

    if args.inspect:
        inspect_csv(args.inspect)
        return

    if args.scan:
        csvs = find_geometry_csvs(args.scan)
        if not csvs:
            print(f"No geometry CSVs found under {args.scan}.")
            print("Expected each extracted product to contain a 'geometry' folder.")
            return
        print(f"Found {len(csvs)} geometry file(s).\n")

        bboxes = {}
        for path in csvs:
            try:
                bboxes[path] = get_bbox(path)
            except ValueError as e:
                print(f"  Skipping {os.path.basename(path)}: {e}")

        print(f"\nParsed {len(bboxes)} bounding box(es). Checking all pairs...\n")
        found_any = False
        for (p1, b1), (p2, b2) in combinations(bboxes.items(), 2):
            if bboxes_overlap(b1, b2):
                found_any = True
                print("OVERLAP CANDIDATE:")
                print(f"  A: {product_name(p1)}")
                print(f"     lat=[{b1[0]:.3f}, {b1[1]:.3f}]  lon=[{b1[2]:.3f}, {b1[3]:.3f}]")
                print(f"  B: {product_name(p2)}")
                print(f"     lat=[{b2[0]:.3f}, {b2[1]:.3f}]  lon=[{b2[2]:.3f}, {b2[3]:.3f}]")
                print()
        if not found_any:
            print("No overlapping bounding boxes found among these products.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
