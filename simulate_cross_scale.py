r"""
ChandraMatch - Simulated Cross-Scale Validation
==================================================
TMC-2's real archive coverage doesn't overlap your OHRC dates/region, so
finding a genuine OHRC<->TMC-2 pair isn't reliably possible right now.

This sidesteps that: take ONE real OHRC image you already have, and
derive a synthetic "TMC-2-like" version of it — downsampled ~20x to a
realistic TMC-2 ground-sample-distance, with a KNOWN rotation/scale/
shift applied (simulating a different orbit pass) and a brightness
change (simulating different illumination).

Because the transform is known, this measures registration accuracy
DIRECTLY against ground truth — something a real unlabeled pair can't
give you. This is standard practice for validating registration methods
when a matched real dataset isn't available (cite this reasoning in
your report — it's a legitimate methodological choice, not a shortcut).

Usage:
    python simulate_cross_scale.py --xml "C:\...\some_real_OHRC_scene.xml"
"""

import argparse
import time

import cv2
import numpy as np

from chandramatch_pipeline import (
    load_pds4_image,
    preprocess,
    detect_and_describe,
    match_features,
    estimate_homography,
    evaluate,
)


def load_pds4_crop(xml_path, crop_size=6000, band_index=0):
    """Loads a PDS4 product like load_pds4_image(), but instead of
    downsampling the whole (often extremely elongated, e.g. ~8:1 for
    OHRC) strip, takes a centered SQUARE crop at native resolution
    first. This avoids the failure mode where downsampling an already
    narrow strip further leaves too few pixels in the short dimension
    for any feature detector to work with.
    """
    try:
        import pdr
    except ImportError as e:
        raise SystemExit("Missing dependency. Install with: pip install pdr") from e

    product = pdr.read(xml_path)
    keys = list(product.keys())
    pick = None
    for candidate in ["IMAGE", "ARRAY_0", "ARRAY", "Image"]:
        if candidate in keys:
            pick = candidate
            break
    if pick is None:
        for k in keys:
            try:
                if np.asarray(product[k]).ndim >= 2:
                    pick = k
                    break
            except Exception:
                continue
    if pick is None:
        raise SystemExit(f"Could not find an image array in {xml_path}. Keys were: {keys}")

    arr = np.squeeze(np.asarray(product[pick]))
    if arr.ndim == 3:
        arr = arr[band_index] if arr.shape[0] < arr.shape[-1] else arr[..., band_index]

    h, w = arr.shape[:2]
    size = min(crop_size, h, w)
    y0, x0 = (h - size) // 2, (w - size) // 2
    crop = arr[y0:y0 + size, x0:x0 + size]
    print(f"Native shape: {arr.shape} -> centered square crop: {crop.shape}")

    crop = crop.astype(np.float64)
    valid = crop[np.isfinite(crop)]
    lo, hi = np.percentile(valid, [2, 98])
    crop = np.nan_to_num(np.clip(crop, lo, hi), nan=lo)
    crop8 = ((crop - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    return cv2.cvtColor(crop8, cv2.COLOR_GRAY2BGR)


def make_simulated_tmc2(real_ohrc_img, gsd_ratio=20, angle_deg=6.0, scale=0.92,
                         tx=20, ty=-15, brightness_alpha=1.15, brightness_beta=12, seed=0):
    """Derives a synthetic 'TMC-2-like' image from a real OHRC image, with
    a known applied transform, for ground-truth-based validation."""
    gray = cv2.cvtColor(real_ohrc_img, cv2.COLOR_BGR2GRAY) if real_ohrc_img.ndim == 3 else real_ohrc_img
    h, w = gray.shape

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(gray, M, (w, h), borderValue=45)
    warped = cv2.convertScaleAbs(warped, alpha=brightness_alpha, beta=brightness_beta)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, warped.shape).astype(np.int16)
    warped = np.clip(warped.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    sim_tmc2 = cv2.resize(
        warped, (max(1, int(w // gsd_ratio)), max(1, int(h // gsd_ratio))),
        interpolation=cv2.INTER_AREA,
    )
    return cv2.cvtColor(sim_tmc2, cv2.COLOR_GRAY2BGR), M


def run_simulation(xml_path, crop_size=6000, gsd_ratio=20.0, angle_deg=6.0, scale_gt=0.92,
                    tx=20, ty=-15, detector="orb", verbose=True):
    """Runs the full simulated cross-scale test on one real OHRC file and
    returns a dict with pipeline metrics AND ground-truth recovery
    accuracy (scale/rotation error vs. the known applied transform).
    This is the reusable core both main() and batch_simulate_cross_scale.py
    call — one implementation, not two copies to keep in sync.
    """
    if verbose:
        print(f"Loading real OHRC scene (native-resolution square crop): {xml_path}")
    img1 = load_pds4_crop(xml_path, crop_size=crop_size)
    img2, applied_M = make_simulated_tmc2(
        img1, gsd_ratio=gsd_ratio, angle_deg=angle_deg, scale=scale_gt, tx=tx, ty=ty,
    )
    if verbose:
        print(f"Derived simulated TMC-2-like image at {img2.shape[:2]} "
              f"({gsd_ratio}x coarser, known transform applied)")

    h, w = img1.shape[:2]
    rscale = 1.0 / gsd_ratio
    img1_resized = cv2.resize(img1, (max(1, int(w * rscale)), max(1, int(h * rscale))),
                               interpolation=cv2.INTER_AREA)
    if verbose:
        print(f"Resized real OHRC image to match simulated scale: {img1_resized.shape[:2]}")

    t0 = time.time()
    gray1, _ = preprocess(img1_resized)
    gray2, _ = preprocess(img2)
    kp1, desc1 = detect_and_describe(gray1, detector)
    kp2, desc2 = detect_and_describe(gray2, detector)
    matches = match_features(desc1, desc2, detector)
    H, mask = estimate_homography(kp1, kp2, matches)
    elapsed = time.time() - t0

    if H is None:
        return {"status": "failed_no_homography", "detector": detector}

    metrics = evaluate(kp1, kp2, matches, mask, H, elapsed)

    recovered_scale = float(np.sqrt(H[0, 0] ** 2 + H[1, 0] ** 2))
    recovered_angle = float(np.degrees(np.arctan2(-H[1, 0], H[0, 0])))
    metrics.update({
        "status": "ok",
        "detector": detector,
        "true_scale": scale_gt,
        "recovered_scale": round(recovered_scale, 4),
        "scale_error_pct": round(abs(recovered_scale - scale_gt) / scale_gt * 100, 3),
        "true_angle_deg": angle_deg,
        "recovered_angle_deg": round(recovered_angle, 3),
        "angle_error_deg": round(abs(recovered_angle - angle_deg), 3),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Simulated cross-scale (OHRC->TMC-2-like) validation")
    parser.add_argument("--xml", required=True, help="Path to a real OHRC .xml (PDS4 label)")
    parser.add_argument("--crop_size", type=int, default=6000, help="Native-resolution square crop size before simulating (default 6000)")
    parser.add_argument("--gsd_ratio", type=float, default=20.0, help="Simulated resolution gap (default 20x, matching real OHRC/TMC-2)")
    parser.add_argument("--detector", default="orb", choices=["orb", "sift"])
    args = parser.parse_args()

    result = run_simulation(
        args.xml, crop_size=args.crop_size, gsd_ratio=args.gsd_ratio, detector=args.detector,
    )

    if result.get("status") != "ok":
        print("\nRegistration FAILED — not enough matches even in this controlled test.")
        print("This itself is informative: note it honestly as a limitation.")
        return

    print("\n=== METRICS ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
