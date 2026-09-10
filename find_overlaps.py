r"""
ChandraMatch - Geometry-Based Pair Discovery
==============================================
Finds every product under a data folder and determines its footprint,
using whichever of two methods that product actually provides:

  1. A geometry CSV (lat/lon sampled across the scene) - used by raw
     "calibrated" OHRC/TMC-2 products. This was the original method.
  2. Corner coordinates embedded directly in the PDS4 .xml label -
     used by "derived" products (e.g. TMC-2 ortho) that don't ship a
     separate geometry folder at all. Some labels include multiple
     coordinate blocks of differing reliability (System_Level, Refined,
     Corrected) - the most-corrected one available is used automatically.

Then reports which pairs of scenes have overlapping footprints -
candidates worth actually running through chandramatch_pipeline.py.

Usage:
    python find_overlaps.py --inspect "path\to\..._g_grd_gds.csv"
    python find_overlaps.py --inspect_xml "path\to\some_label.xml"
    python find_overlaps.py --scan "C:\...\chandra match\data"
"""

import argparse
import csv
import glob
import os
import xml.etree.ElementTree as ET
from itertools import combinations

# Adjust these if --inspect shows different column header names.
LAT_COL_CANDIDATES = ["Latitude", "Lat", "LATITUDE", "lat"]
LON_COL_CANDIDATES = ["Longitude", "Lon", "Long", "LONGITUDE", "lon"]

# Preferred order of embedded coordinate blocks (most reliable first).
# Seen in real data: System_Level and Refined can be internally
# inconsistent for some products; Corrected is the trustworthy one when
# present.
XML_COORD_BLOCK_PREFERENCE = [
    "Corrected_Corner_Coordinates",
    "Refined_Corner_Coordinates",
    "System_Level_Coordinates",
]
XML_CORNER_FIELDS = [
    "upper_left_latitude", "upper_left_longitude",
    "upper_right_latitude", "upper_right_longitude",
    "lower_left_latitude", "lower_left_longitude",
    "lower_right_latitude", "lower_right_longitude",
]


def _local_name(tag):
    return tag.split("}")[-1] if "}" in tag else tag


def _normalize_lon(lon):
    """Wraps longitude into -180..180, matching the convention geometry
    CSV data already uses, so both sources are directly comparable."""
    lon = lon % 360
    return lon - 360 if lon > 180 else lon


def find_geometry_csvs(root):
    pattern = os.path.join(root, "**", "geometry", "**", "*.csv")
    return glob.glob(pattern, recursive=True)


def find_data_xmls(root):
    """Finds every product's main PDS4 label under data/<any subfolder>/."""
    pattern = os.path.join(root, "**", "data", "**", "*.xml")
    return sorted(set(glob.glob(pattern, recursive=True)))


def product_root_from_geometry_csv(geometry_csv_path):
    d = os.path.dirname(geometry_csv_path)
    while d and os.path.basename(d) != "geometry":
        parent = os.path.dirname(d)
        if parent == d:
            return None  # reached filesystem root without finding "geometry"
        d = parent
    return os.path.dirname(d) if d else None


def product_root_from_xml(xml_path):
    d = os.path.dirname(xml_path)
    while d and os.path.basename(d) != "data":
        parent = os.path.dirname(d)
        if parent == d:
            return None
        d = parent
    return os.path.dirname(d) if d else None


def product_name(path):
    """Works for either a geometry csv path or a data xml path."""
    root = product_root_from_geometry_csv(path) or product_root_from_xml(path)
    return os.path.basename(root) if root else os.path.basename(path)


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


def inspect_xml(path):
    block, values = _find_xml_coord_block(path)
    if block is None:
        print(f"No recognized coordinate block found in {path}.")
        print(f"Looked for: {XML_COORD_BLOCK_PREFERENCE}")
        return
    print(f"Using block '{block}' from {os.path.basename(path)}:")
    for k, v in values.items():
        print(f"  {k}: {v}")


def pick_column(header, candidates):
    for c in candidates:
        if c in header:
            return header.index(c)
    for i, h in enumerate(header):
        if any(c.lower() in h.lower() for c in candidates):
            return i
    return None


def get_bbox_from_csv(path):
    """Returns (lat_min, lat_max, lon_min, lon_max) from a geometry CSV."""
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


def _find_xml_coord_block(xml_path):
    """Returns (block_name, {field: value}) for the best available
    coordinate block, or (None, {}) if none found."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for pref in XML_COORD_BLOCK_PREFERENCE:
        block_el = None
        for el in root.iter():
            if _local_name(el.tag) == pref:
                block_el = el
                break
        if block_el is None:
            continue
        values = {}
        for el in block_el:
            name = _local_name(el.tag)
            if name in XML_CORNER_FIELDS:
                try:
                    values[name] = float(el.text)
                except (TypeError, ValueError):
                    pass
        if len(values) == 8:
            return pref, values
    return None, {}


def get_bbox_from_xml(xml_path):
    """Returns (lat_min, lat_max, lon_min, lon_max) from corner
    coordinates embedded directly in a PDS4 .xml label, for products
    (e.g. derived/ortho) that don't ship a separate geometry CSV."""
    _, values = _find_xml_coord_block(xml_path)
    if not values:
        raise ValueError(
            f"No usable coordinate block found in {xml_path}. "
            f"Run --inspect_xml on this file to see what's available."
        )
    lats = [values["upper_left_latitude"], values["upper_right_latitude"],
            values["lower_left_latitude"], values["lower_right_latitude"]]
    lons = [_normalize_lon(values["upper_left_longitude"]), _normalize_lon(values["upper_right_longitude"]),
            _normalize_lon(values["lower_left_longitude"]), _normalize_lon(values["lower_right_longitude"])]
    return min(lats), max(lats), min(lons), max(lons)


def bboxes_overlap(b1, b2):
    lat1_min, lat1_max, lon1_min, lon1_max = b1
    lat2_min, lat2_max, lon2_min, lon2_max = b2
    return (lat1_min <= lat2_max and lat2_min <= lat1_max
            and lon1_min <= lon2_max and lon2_min <= lon1_max)


def collect_all_bboxes(scan_root):
    """Builds {product_name: bbox} across every product found, using a
    geometry CSV when available and falling back to the product's own
    XML-embedded coordinates otherwise. Returns (bboxes, sources) where
    sources maps product_name -> 'csv' or 'xml', useful for knowing
    which method was actually used per product."""
    bboxes, sources = {}, {}

    csvs = find_geometry_csvs(scan_root)
    print(f"Found {len(csvs)} geometry CSV file(s).")
    products_with_csv = set()
    for path in csvs:
        name = product_name(path)
        try:
            bboxes[name] = get_bbox_from_csv(path)
            sources[name] = "csv"
            products_with_csv.add(name)
        except ValueError as e:
            print(f"  Skipping {os.path.basename(path)}: {e}")

    xmls = find_data_xmls(scan_root)
    fallback_count = 0
    for path in xmls:
        name = product_name(path)
        if name in products_with_csv:
            continue  # already have a geometry CSV for this product
        try:
            bboxes[name] = get_bbox_from_xml(path)
            sources[name] = "xml"
            fallback_count += 1
        except ValueError:
            continue  # neither CSV nor usable embedded coordinates - skip

    if fallback_count:
        print(f"Found {fallback_count} more product(s) via embedded XML "
              f"coordinates (no geometry CSV available for these).")

    return bboxes, sources


def main():
    parser = argparse.ArgumentParser(description="Find overlapping Chandrayaan-2 scene pairs")
    parser.add_argument("--inspect", type=str, default=None, help="Inspect one geometry CSV's columns")
    parser.add_argument("--inspect_xml", type=str, default=None, help="Inspect one PDS4 .xml label's embedded coordinates")
    parser.add_argument("--scan", type=str, default=None, help="Scan a data folder for overlapping pairs")
    args = parser.parse_args()

    if args.inspect:
        inspect_csv(args.inspect)
        return
    if args.inspect_xml:
        inspect_xml(args.inspect_xml)
        return

    if args.scan:
        bboxes, sources = collect_all_bboxes(args.scan)
        print(f"\nTotal: {len(bboxes)} product(s) with a known footprint. Checking all pairs...\n")

        found_any = False
        for (name1, b1), (name2, b2) in combinations(bboxes.items(), 2):
            if bboxes_overlap(b1, b2):
                found_any = True
                print("OVERLAP CANDIDATE:")
                print(f"  A: {name1} [{sources[name1]}]")
                print(f"     lat=[{b1[0]:.3f}, {b1[1]:.3f}]  lon=[{b1[2]:.3f}, {b1[3]:.3f}]")
                print(f"  B: {name2} [{sources[name2]}]")
                print(f"     lat=[{b2[0]:.3f}, {b2[1]:.3f}]  lon=[{b2[2]:.3f}, {b2[3]:.3f}]")
                print()
        if not found_any:
            print("No overlapping footprints found among these products.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()