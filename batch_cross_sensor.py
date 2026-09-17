r"""
ChandraMatch - Automated Cross-Sensor Batch Registration
============================================================
Finds every overlapping pair across ANY sensor combination (OHRC,
TMC-2, IIRS) in one pass, tries both detectors, and for multi-band
products (IIRS) automatically sweeps a spread of bands instead of you
manually re-running --inspect/--band/--detector combinations by hand.

Each unique file is loaded from disk exactly ONCE regardless of how
many pairs/bands/detectors it's tested against - critical for IIRS,
where a single file can be ~1GB+.

Usage:
    python batch_cross_sensor.py --scan "C:\...\chandra match\data"
    python batch_cross_sensor.py --scan "C:\...\chandra match\data" --detector both --n_bands 12
"""

import argparse
import csv
import glob
import os
import time
from itertools import combinations

import cv2
import numpy as np

from find_overlaps import collect_all_bboxes, bboxes_overlap
from cross_sensor import estimate_gsd
from chandramatch_pipeline import (
    load_pds4_raw,
    num_bands,
    postprocess_array,
    preprocess,
    detect_and_describe,
    match_features,
    estimate_homography,
    evaluate,
    KNOWN_KEYS_BAND_AXIS,
)

FIELDNAMES = [
    "product_a", "product_b", "band_a", "band_b", "detector", "status", "scale_normalized",
    "num_features_img1", "num_features_img2", "num_candidate_matches",
    "num_inlier_matches", "inlier_ratio", "reprojection_rmse_px", "processing_time_s",
]


def find_xml_for_product(product_root, product_name=None):
    """Finds a product's main PDS4 label under any data/<subfolder>/,
    since different processing levels use different subfolder names
    (calibrated, raw, derived).

    If multiple .xml files exist (seen in practice: stale leftovers from
    an earlier mis-extraction sitting alongside a later correct one),
    prefer whichever filename actually starts with the product's own
    folder name, rather than silently taking whatever glob finds first."""
    pattern = os.path.join(product_root, "data", "**", "*.xml")
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    if len(matches) > 1:
        print(f"    WARNING: {len(matches)} .xml files found under {product_root} "
              f"- expected 1. Likely stale leftovers from a previous mis-extraction.")
        if product_name:
            preferred = [m for m in matches if os.path.basename(m).startswith(product_name)]
            if preferred:
                print(f"    Using the one matching the folder name: {os.path.basename(preferred[0])}")
                return preferred[0]
        print(f"    Could not confirm which is correct - using {os.path.basename(matches[0])}. "
              f"Verify this manually; consider deleting the stale file(s).")
    return matches[0]


def find_product_root(name, scan_root):
    """Locates a product's root folder by name under the scan root."""
    for candidate in glob.glob(os.path.join(scan_root, "**", name), recursive=True):
        if os.path.isdir(candidate):
            return candidate
    return None


def band_sample(n, count):
    """Evenly-spaced band indices across [0, n-1], including both ends.
    For n_bands=1 (plain 2D images), just returns [0]."""
    if n <= 1:
        return [0]
    count = min(count, n)
    if count == 1:
        return [0]
    step = (n - 1) / (count - 1)
    return sorted(set(round(i * step) for i in range(count)))


def find_geometry_csv_for_product(product_root):
    matches = glob.glob(os.path.join(product_root, "geometry", "**", "*.csv"), recursive=True)
    return matches[0] if matches else None


def spatial_shape(arr, key):
    """The 2D (height, width) shape after removing the band axis, if any."""
    if arr.ndim == 3:
        band_axis = KNOWN_KEYS_BAND_AXIS.get(key, int(np.argmin(arr.shape)))
        dims = [arr.shape[i] for i in range(3) if i != band_axis]
        return tuple(dims)
    return arr.shape[:2]


def effective_gsd(native_gsd, shape_2d, max_dim=4000):
    """postprocess_array() downsamples internally to max_dim - each
    remaining pixel then represents MORE ground distance than the
    native GSD. This computes that actual effective GSD after
    downsampling, which is what a scale-normalization factor must be
    based on, not the native GSD alone (using native GSD directly would
    silently compound with the internal downsample and over/under-shrink)."""
    h, w = shape_2d
    downsample_scale = min(1.0, max_dim / max(h, w))
    return native_gsd / downsample_scale


def run_one(arr1, key1, band1, arr2, key2, band2, detector, scale_pair=None):
    """scale_pair, if given, is (scale1, scale2): resize factors that
    bring both images to comparable physical ground resolution before
    matching. Without this, a large GSD gap (e.g. OHRC vs IIRS, ~300x)
    fails regardless of which band is chosen - a completely different
    failure mode than 'this band has no useful content', which matters
    for correctly interpreting results."""
    img1 = postprocess_array(arr1, key1, band_index=band1, verbose=False)
    img2 = postprocess_array(arr2, key2, band_index=band2, verbose=False)

    if scale_pair is not None:
        s1, s2 = scale_pair
        if s1 != 1.0:
            h, w = img1.shape[:2]
            img1 = cv2.resize(img1, (max(1, int(w * s1)), max(1, int(h * s1))), interpolation=cv2.INTER_AREA)
        if s2 != 1.0:
            h, w = img2.shape[:2]
            img2 = cv2.resize(img2, (max(1, int(w * s2)), max(1, int(h * s2))), interpolation=cv2.INTER_AREA)

    t0 = time.time()
    gray1, _ = preprocess(img1)
    gray2, _ = preprocess(img2)
    kp1, desc1 = detect_and_describe(gray1, detector)
    kp2, desc2 = detect_and_describe(gray2, detector)

    if desc1 is None or len(kp1) == 0:
        return {"status": "failed_zero_keypoints_A"}
    if desc2 is None or len(kp2) == 0:
        return {"status": "failed_zero_keypoints_B"}

    matches = match_features(desc1, desc2, detector)
    H, mask = estimate_homography(kp1, kp2, matches)
    if H is None:
        return {"status": "failed_no_homography",
                "num_features_img1": len(kp1), "num_features_img2": len(kp2)}

    metrics = evaluate(kp1, kp2, matches, mask, H, time.time() - t0)
    # Flag statistically weak fits explicitly rather than reporting a
    # misleadingly "perfect" RMSE - a homography needs >=4 points, so a
    # fit built from very few matches (e.g. exactly 4-5) can show 0
    # reprojection error by construction, not because it's accurate.
    if metrics["num_inlier_matches"] < 8:
        metrics["status"] = "ok_but_low_sample_unreliable"
    else:
        metrics["status"] = "ok"
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Automated cross-sensor batch registration")
    parser.add_argument("--scan", required=True)
    parser.add_argument("--detector", default="orb", choices=["orb", "sift", "both"])
    parser.add_argument("--n_bands", type=int, default=8,
                         help="How many bands to sample for multi-band products like IIRS (default 8, spread evenly)")
    parser.add_argument("--out", default="cross_sensor_batch_results.csv")
    args = parser.parse_args()

    bboxes, sources = collect_all_bboxes(args.scan)
    pairs = [(a, b) for (a, ba), (b, bb) in combinations(bboxes.items(), 2) if bboxes_overlap(ba, bb)]
    print(f"\n{len(pairs)} overlapping pair(s) found across all sensors.\n")
    if not pairs:
        print("Nothing to run.")
        return

    detectors = ["orb", "sift"] if args.detector == "both" else [args.detector]

    # Cache: load each unique product from disk exactly once, regardless
    # of how many pairs/bands/detectors use it.
    # Bounded cache: with an 8GB-RAM-class machine and real files often
    # 1-2GB+ each (confirmed in practice - a single TMC-2 array here
    # needed 2.06GB), caching every unique product touched across an
    # 86-pair run would exhaust memory well before the run finishes.
    # Keep only the few most recently used raw arrays in memory,
    # evicting the oldest when the limit is hit - a product evicted and
    # later revisited just gets reloaded from disk, which costs time
    # but not correctness, and is still far better than the original
    # per-band-per-detector reloading this script replaced.
    MAX_CACHED_PRODUCTS = 3
    raw_cache = {}
    cache_order = []  # oldest first

    def get_raw(name):
        if name in raw_cache:
            cache_order.remove(name)
            cache_order.append(name)
            return raw_cache[name]

        root = find_product_root(name, args.scan)
        xml = find_xml_for_product(root, product_name=name) if root else None
        if not xml:
            return None

        print(f"  (loading {name} - not cached, reading from disk)")
        try:
            arr, key = load_pds4_raw(xml)
            result = (arr, key)
        except (SystemExit, Exception) as e:
            # Broad catch deliberately: pdr can raise many different
            # internal errors for corrupted/incomplete/unexpected files
            # (seen in practice: a truncated download raising a raw
            # numpy reshape error, and a genuine out-of-memory error on
            # a 2GB+ array, both deep inside pdr's PDS4 reader). One bad
            # or too-large file should skip that product, not crash the
            # entire batch run.
            print(f"    Failed to load ({type(e).__name__}): {e}")
            print(f"    (this can mean an incomplete/corrupted download, "
                  f"or the file is too large for available RAM)")
            return None  # deliberately NOT cached - a transient OOM might
                         # succeed later if memory frees up in the meantime

        raw_cache[name] = result
        cache_order.append(name)
        if len(cache_order) > MAX_CACHED_PRODUCTS:
            oldest = cache_order.pop(0)
            del raw_cache[oldest]
            print(f"    (evicted {oldest} from cache to free memory)")
        return result

    total_runs = 0
    all_rows = []
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, restval="")
        writer.writeheader()
        f.flush()

        for i, (name_a, name_b) in enumerate(pairs, 1):
            print(f"[pair {i}/{len(pairs)}] {name_a} x {name_b}")
            ra = get_raw(name_a)
            rb = get_raw(name_b)
            if ra is None or rb is None:
                print("  Skipping - could not load one or both products.")
                continue
            arr_a, key_a = ra
            arr_b, key_b = rb

            # Try to estimate real ground-sample-distance for each side
            # from its geometry CSV, so images with very different
            # physical resolution (e.g. OHRC ~0.25m/px vs IIRS ~80m/px,
            # a ~300x gap) get scale-normalized before matching. Without
            # this, every band combination would fail from pure scale
            # mismatch, telling you nothing about which band is useful.
            scale_pair = None
            root_a = find_product_root(name_a, args.scan)
            root_b = find_product_root(name_b, args.scan)
            geo_a = find_geometry_csv_for_product(root_a) if root_a else None
            geo_b = find_geometry_csv_for_product(root_b) if root_b else None
            if geo_a and geo_b:
                try:
                    native_gsd_a = estimate_gsd(geo_a)
                    native_gsd_b = estimate_gsd(geo_b)
                    eff_gsd_a = effective_gsd(native_gsd_a, spatial_shape(arr_a, key_a))
                    eff_gsd_b = effective_gsd(native_gsd_b, spatial_shape(arr_b, key_b))
                    if eff_gsd_a < eff_gsd_b:
                        scale_pair = (eff_gsd_a / eff_gsd_b, 1.0)
                    elif eff_gsd_b < eff_gsd_a:
                        scale_pair = (1.0, eff_gsd_b / eff_gsd_a)
                    else:
                        scale_pair = (1.0, 1.0)
                    print(f"  Scale-normalized: effective GSD A={eff_gsd_a:.3f} B={eff_gsd_b:.3f} "
                          f"(after internal downsampling) -> scale factors {scale_pair}")
                except ValueError as e:
                    print(f"  Could not estimate GSD for scale normalization: {e}")
            else:
                print("  WARNING: no geometry CSV for one or both products - "
                      "proceeding WITHOUT scale normalization. If these have "
                      "very different resolutions, failures below may be due "
                      "to scale mismatch, not band content.")

            bands_a = band_sample(num_bands(arr_a, key_a), args.n_bands)
            bands_b = band_sample(num_bands(arr_b, key_b), args.n_bands)

            for band_a in bands_a:
                for band_b in bands_b:
                    for det in detectors:
                        total_runs += 1
                        result = run_one(arr_a, key_a, band_a, arr_b, key_b, band_b, det,
                                          scale_pair=scale_pair)
                        result.update({
                            "product_a": name_a, "product_b": name_b,
                            "band_a": band_a, "band_b": band_b, "detector": det,
                            "scale_normalized": scale_pair is not None,
                        })
                        status = result.get("status")
                        # Only print non-trivial outcomes, to keep console
                        # output readable across potentially hundreds of
                        # band combinations.
                        if status in ("ok", "ok_but_low_sample_unreliable"):
                            print(f"    band_a={band_a} band_b={band_b} ({det}): "
                                  f"{status}  inlier_ratio={result.get('inlier_ratio')}")
                        writer.writerow(result)
                        all_rows.append(result)
            f.flush()

    print(f"\nDone. {total_runs} combination(s) tested across {len(pairs)} pair(s).")
    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    print(f"{len(ok_rows)} produced a statistically reliable registration (status='ok').")
    print(f"Results written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()