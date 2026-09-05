"""
ChandraMatch - Phase 0 / Phase 1 Starter Pipeline
==================================================
A minimal, fully-working implementation of the classical (non-ML) pipeline
described in the ChandraMatch PRD:

    Input -> Preprocess -> Detect features -> Describe features ->
    Match -> Reject outliers (RANSAC) -> Estimate transform ->
    Register -> Evaluate -> Visualize

Every function below is labelled with the PRD requirement it implements
(FR-02 ... FR-12) so you can map code directly back to the document.

WHY THIS EXISTS
----------------
You don't need to touch real lunar data (or even understand it) to prove
this pipeline works. Run this script with NO arguments and it will
auto-generate a synthetic "before/after" image pair (a rotated, scaled,
re-lit copy of a test pattern) and run the full pipeline on it. This is
literally PRD Phase 0: "A simple experiment demonstrating correspondence
between two ordinary images."

Once this works, swap in two real photos (Phase 1), then real
Chandrayaan-2 OHRC/TMC-2 imagery (Phase 2). Nothing else in the script
needs to change.

USAGE
-----
    python chandramatch_pipeline.py
    python chandramatch_pipeline.py --img1 photoA.jpg --img2 photoB.jpg
    python chandramatch_pipeline.py --img1 A.png --img2 B.png --detector sift

REQUIREMENTS
------------
    pip install opencv-python opencv-contrib-python numpy matplotlib
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Phase 0 helper: synthetic test pair generator (no real data needed)
# ---------------------------------------------------------------------------
def generate_synthetic_pair(size=600, seed=7):
    """
    Builds a textured 'terrain-like' base image, then derives a second
    image from it via a KNOWN perspective transform + brightness shift
    + noise, simulating scale/rotation/illumination differences between
    two real sensor passes. Because we know the ground-truth transform,
    this is also a great sanity check for FR-11 (evaluation).
    """
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60, dtype=np.uint8)

    # Simulate crater-like circular features + linear ridges so ORB/SIFT
    # have plenty of distinctive corners/blobs to lock onto.
    for _ in range(40):
        cx, cy = rng.integers(0, size, size=2)
        r = rng.integers(8, 45)
        shade = int(rng.integers(20, 220))
        cv2.circle(img, (cx, cy), r, shade, thickness=-1)
        cv2.circle(img, (cx, cy), r, max(shade - 40, 0), thickness=2)
    for _ in range(15):
        p1 = tuple(rng.integers(0, size, size=2))
        p2 = tuple(rng.integers(0, size, size=2))
        shade = int(rng.integers(30, 200))
        cv2.line(img, p1, p2, shade, thickness=rng.integers(1, 4))

    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Known ground-truth transform: rotate + scale + translate.
    angle_deg, scale, tx, ty = 12.0, 0.85, 35, -20
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img2 = cv2.warpAffine(img, M, (size, size), borderValue=40)

    # Simulate an illumination difference (different sun angle) + sensor noise.
    img2 = cv2.convertScaleAbs(img2, alpha=1.25, beta=25)
    noise = rng.normal(0, 6, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Phase 2 helper: load a real Chandrayaan-2 PDS4 product (OHRC/TMC-2/IIRS)
# ---------------------------------------------------------------------------
def load_pds4_image(xml_path, band_index=0, max_dim=4000):
    """
    Loads a PDS4 product via the `pdr` library and returns an 8-bit BGR
    image ready for the rest of the pipeline.

    Requires: pip install pdr
    """
    try:
        import pdr
    except ImportError as e:
        raise SystemExit("Missing dependency. Install with: pip install pdr") from e

    product = pdr.read(xml_path)
    keys = list(product.keys())

    # Real Chandrayaan-2 products expose pixels under "ARRAY_0" (confirmed
    # via --inspect), not "IMAGE" as pdr's docs example suggested. Try the
    # known names first, then fall back to the first genuinely 2D+ array.
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
    if arr.ndim == 3:  # multi-band cube (e.g. IIRS) -> take one band
        arr = arr[band_index] if arr.shape[0] < arr.shape[-1] else arr[..., band_index]

    print(f"Loaded '{pick}': shape={arr.shape} dtype={arr.dtype}")

    # Real OHRC/TMC-2 scenes can be 1B+ pixels. Downsample BEFORE any
    # float64 conversion, or you'll try to allocate 8-10GB+ RAM and crash.
    # Full resolution is a later optimization, not needed for a prototype.
    h, w = arr.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        resize_dtype = arr.dtype if arr.dtype in (np.uint8, np.uint16) else np.float32
        arr = cv2.resize(arr.astype(resize_dtype), (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)
        print(f"Downsampled to {arr.shape} (full-res is a later optimization, not needed now)")

    arr = arr.astype(np.float64)
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        raise SystemExit(f"{xml_path}: no finite pixel values found — wrong band/key?")
    lo, hi = np.percentile(valid, [2, 98])
    arr = np.nan_to_num(np.clip(arr, lo, hi), nan=lo)
    arr8 = ((arr - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr8, cv2.COLOR_GRAY2BGR)


def inspect_pds4(xml_path):
    """Debug helper: prints the structure of a PDS4 product so you can
    find the right key/band if load_pds4_image() guesses wrong."""
    import pdr
    product = pdr.read(xml_path)
    print(f"Keys in {xml_path}:")
    for k in product.keys():
        try:
            a = np.asarray(product[k])
            print(f"  {k}: shape={a.shape} dtype={a.dtype}")
        except Exception as e:
            print(f"  {k}: <could not array-ify: {e}>")


# ---------------------------------------------------------------------------
# FR-02: Preprocessing
# ---------------------------------------------------------------------------
def preprocess(img_bgr, max_dim=1024):
    """Grayscale conversion + resize cap + contrast normalization."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # CLAHE = adaptive contrast normalization. Helps a lot on low-contrast
    # imagery (e.g. lunar surface under low sun angle).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray, scale


# ---------------------------------------------------------------------------
# FR-03 + FR-04: Feature detection + description
# ---------------------------------------------------------------------------
def detect_and_describe(gray, method="orb", n_features=3000):
    method = method.lower()
    if method == "orb":
        detector = cv2.ORB_create(nfeatures=n_features)
    elif method == "sift":
        detector = cv2.SIFT_create(nfeatures=n_features)
    else:
        raise ValueError(f"Unknown detector '{method}'. Use 'orb' or 'sift'.")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


# ---------------------------------------------------------------------------
# FR-05: Feature matching (with Lowe's ratio test to pre-filter junk matches)
# ---------------------------------------------------------------------------
def match_features(desc1, desc2, method="orb", ratio_thresh=0.75):
    norm = cv2.NORM_HAMMING if method.lower() == "orb" else cv2.NORM_L2
    bf = cv2.BFMatcher(norm)
    raw_pairs = bf.knnMatch(desc1, desc2, k=2)

    good = []
    for pair in raw_pairs:
        if len(pair) != 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            good.append(m)
    return good


# ---------------------------------------------------------------------------
# FR-06 + FR-07: Outlier rejection (RANSAC) + geometric transform estimate
# ---------------------------------------------------------------------------
def estimate_homography(kp1, kp2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return H, mask


# ---------------------------------------------------------------------------
# FR-07: Registration (warp image2 into image1's frame)
# ---------------------------------------------------------------------------
def register_images(gray1, gray2, H):
    h, w = gray1.shape
    warped2 = cv2.warpPerspective(gray2, H, (w, h))
    return warped2


# ---------------------------------------------------------------------------
# FR-11: Evaluation metrics
# ---------------------------------------------------------------------------
def evaluate(kp1, kp2, matches, mask, H, elapsed_s):
    n_inliers = int(mask.sum()) if mask is not None else 0
    n_matches = len(matches)
    inlier_ratio = n_inliers / n_matches if n_matches else 0.0

    rmse = None
    if H is not None and mask is not None and n_inliers > 0:
        src = np.float32([kp1[m.queryIdx].pt for m, keep in zip(matches, mask.ravel()) if keep])
        dst = np.float32([kp2[m.trainIdx].pt for m, keep in zip(matches, mask.ravel()) if keep])
        src_h = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(src_h - dst, axis=1)
        rmse = float(np.sqrt(np.mean(errors ** 2)))

    return {
        "num_features_img1": len(kp1),
        "num_features_img2": len(kp2),
        "num_candidate_matches": n_matches,
        "num_inlier_matches": n_inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "reprojection_rmse_px": round(rmse, 3) if rmse is not None else None,
        "processing_time_s": round(elapsed_s, 3),
    }


# ---------------------------------------------------------------------------
# FR-12: Visualization
# ---------------------------------------------------------------------------
def visualize(img1, img2, kp1, kp2, matches, mask, gray1, warped2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mask_list = mask.ravel().tolist() if mask is not None else None

    # 1) Inlier vs outlier matches, side by side.
    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchesMask=mask_list,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(os.path.join(out_dir, "01_inlier_matches.png"), match_vis)

    # 2) Registered result: checkerboard blend of image1 vs warped image2,
    #    so misalignment shows up as visible seams.
    h, w = gray1.shape
    tile = 40
    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                checker[y:y + tile, x:x + tile] = 1
    blend = np.where(checker == 1, gray1, warped2)
    cv2.imwrite(os.path.join(out_dir, "02_registered_checkerboard.png"), blend)

    # 3) Absolute difference map (naive, most meaningful for same-sensor pairs).
    diff = cv2.absdiff(gray1, warped2)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(out_dir, "03_difference_map.png"), diff_color)

    # 4) One combined figure for quick viewing.
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Image A"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); axes[0, 1].set_title("Image B"); axes[0, 1].axis("off")
    axes[1, 0].imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)); axes[1, 0].set_title("Inlier matches (green)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(blend, cmap="gray"); axes[1, 1].set_title("Registered (checkerboard blend)"); axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "00_summary.png"), dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_pipeline(img1_bgr, img2_bgr, detector="orb", out_dir="outputs"):
    t0 = time.time()

    gray1, _ = preprocess(img1_bgr)                       # FR-02
    gray2, _ = preprocess(img2_bgr)                       # FR-02

    kp1, desc1 = detect_and_describe(gray1, detector)     # FR-03/04
    kp2, desc2 = detect_and_describe(gray2, detector)     # FR-03/04

    matches = match_features(desc1, desc2, detector)      # FR-05
    H, mask = estimate_homography(kp1, kp2, matches)      # FR-06/07

    if H is None:
        print("Registration FAILED: not enough good matches to estimate a transform.")
        return None

    warped2 = register_images(gray1, gray2, H)            # FR-07
    elapsed = time.time() - t0

    metrics = evaluate(kp1, kp2, matches, mask, H, elapsed)   # FR-11
    visualize(img1_bgr, img2_bgr, kp1, kp2, matches, mask, gray1, warped2, out_dir)  # FR-12

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="ChandraMatch Phase 0/1 pipeline")
    parser.add_argument("--img1", type=str, default=None, help="Path to first ordinary image")
    parser.add_argument("--img2", type=str, default=None, help="Path to second ordinary image")
    parser.add_argument("--pds4_1", type=str, default=None, help="Path to first PDS4 .xml label (real Chandrayaan-2 data)")
    parser.add_argument("--pds4_2", type=str, default=None, help="Path to second PDS4 .xml label")
    parser.add_argument("--inspect", type=str, default=None, help="Path to a PDS4 .xml file to inspect and exit (debug)")
    parser.add_argument("--detector", type=str, default="orb", choices=["orb", "sift"])
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    if args.inspect:
        inspect_pds4(args.inspect)
        return

    if args.pds4_1 and args.pds4_2:
        img1 = load_pds4_image(args.pds4_1)
        img2 = load_pds4_image(args.pds4_2)
        print(f"Loaded PDS4 images: {args.pds4_1}, {args.pds4_2}")
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, "pds4_img1.png"), img1)
        cv2.imwrite(os.path.join(args.out, "pds4_img2.png"), img2)
        print(f"Also saved as plain PNGs in {args.out}/ "
              f"(pds4_img1.png, pds4_img2.png) — upload these to the web demo, "
              f"no need to re-read the raw PDS4 file every time.")
    elif args.img1 and args.img2:
        img1 = cv2.imread(args.img1)
        img2 = cv2.imread(args.img2)
        if img1 is None or img2 is None:
            raise SystemExit("Could not read one of the input images. Check the paths.")
        print(f"Loaded real images: {args.img1}, {args.img2}")
    else:
        print("No --img1/--img2 given -> generating a synthetic test pair (Phase 0 mode).")
        img1, img2 = generate_synthetic_pair()
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, "synthetic_A.png"), img1)
        cv2.imwrite(os.path.join(args.out, "synthetic_B.png"), img2)

    metrics = run_pipeline(img1, img2, detector=args.detector, out_dir=args.out)

    if metrics:
        print("\n=== METRICS (FR-11) ===")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        print(f"\nVisual outputs written to: {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
