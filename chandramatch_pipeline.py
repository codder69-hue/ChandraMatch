"""
ChandraMatch - Phase 0 / Phase 1 Starter Pipeline
==================================================
Classical feature-based image correspondence & registration pipeline.
See README.md for full usage. This version fixes a real bug found
against actual IIRS data: multi-band cube axis selection previously
guessed the band axis by comparing dimension sizes, which is wrong
when the band count and one spatial dimension are close in size (IIRS:
256 bands vs 250 cross-track pixels - the old heuristic picked wrong).

REQUIREMENTS: pip install -r requirements.txt
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Known PDS4 array keys, in lookup priority order, with the axis their
# spectral/band dimension sits on when the array is 3D. Determined from
# real inspected data, not guessed:
#   - ARRAY_0 (OHRC/TMC-2): always 2D, no band axis.
#   - SPECTRAL_QUBE_OBJECT (IIRS): confirmed via real --inspect output,
#     shape (256, 8953, 250) - band count (256, matches IIRS's actual
#     spec) is axis 0. Cross-checked against IIRS's documented ~80m
#     resolution over a 20km swath (20000/80 = 250, matching axis 2) -
#     this is a physically grounded assignment, not a guess.
KNOWN_KEYS_BAND_AXIS = {
    "SPECTRAL_QUBE_OBJECT": 0,
}


# Phase 0 helper: synthetic test pair generator (no real data needed)
def generate_synthetic_pair(size=600, seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 60, dtype=np.uint8)
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

    angle_deg, scale, tx, ty = 12.0, 0.85, 35, -20
    center = (size / 2, size / 2)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    img2 = cv2.warpAffine(img, M, (size, size), borderValue=40)
    img2 = cv2.convertScaleAbs(img2, alpha=1.25, beta=25)
    noise = rng.normal(0, 6, img2.shape).astype(np.int16)
    img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)


# Phase 2 helper: load a real Chandrayaan-2 PDS4 product (OHRC/TMC-2/IIRS)
def load_pds4_image(xml_path, band_index=0, max_dim=4000):
    """
    Loads a PDS4 product via `pdr` and returns an 8-bit BGR image ready
    for the rest of the pipeline. Handles both plain 2D scenes
    (OHRC/TMC-2) and 3D hyperspectral cubes (IIRS).
    """
    try:
        import pdr
    except ImportError as e:
        raise SystemExit("Missing dependency. Install with: pip install pdr") from e

    product = pdr.read(xml_path)
    keys = list(product.keys())

    pick = None
    for candidate in ["IMAGE", "ARRAY_0", "ARRAY", "Image", "SPECTRAL_QUBE_OBJECT"]:
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
    print(f"Loaded '{pick}': shape={arr.shape} dtype={arr.dtype}")

    if arr.ndim == 3:
        if pick in KNOWN_KEYS_BAND_AXIS:
            band_axis = KNOWN_KEYS_BAND_AXIS[pick]
        else:
            # Fallback heuristic for unrecognized multi-band keys: assume
            # the smallest dimension is the band axis. This is fragile
            # (proven wrong for IIRS, where band count and one spatial
            # dimension are close in size) - if you hit a new multi-band
            # key, verify with --inspect and add it to
            # KNOWN_KEYS_BAND_AXIS above rather than trusting this guess.
            band_axis = int(np.argmin(arr.shape))
            print(f"WARNING: unrecognized multi-band key '{pick}' - guessing "
                  f"band axis={band_axis} by smallest-dimension heuristic. "
                  f"Verify this is correct before trusting results.")
        arr = np.take(arr, indices=band_index, axis=band_axis)
        print(f"Selected band {band_index} along axis {band_axis} -> shape={arr.shape}")

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
    import pdr
    product = pdr.read(xml_path)
    print(f"Keys in {xml_path}:")
    for k in product.keys():
        try:
            a = np.asarray(product[k])
            print(f"  {k}: shape={a.shape} dtype={a.dtype}")
        except Exception as e:
            print(f"  {k}: <could not array-ify: {e}>")


# FR-02: Preprocessing
def preprocess(img_bgr, max_dim=1024):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    return gray, scale


# FR-03 + FR-04: Feature detection + description
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


# FR-05: Feature matching
def match_features(desc1, desc2, method="orb", ratio_thresh=0.75):
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        print("WARNING: one image produced zero usable keypoints/descriptors - "
              "no matches are possible. This usually means that image had too "
              "little texture/contrast for the detector to find anything (for "
              "IIRS, try a different --band - not every wavelength has equally "
              "clear surface detail).")
        return []
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


# FR-06 + FR-07: Outlier rejection (RANSAC) + geometric transform estimate
def estimate_homography(kp1, kp2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return H, mask


def register_images(gray1, gray2, H):
    h, w = gray1.shape
    warped2 = cv2.warpPerspective(gray2, H, (w, h))
    return warped2


# FR-11: Evaluation metrics
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


# FR-12: Visualization
def visualize(img1, img2, kp1, kp2, matches, mask, gray1, warped2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mask_list = mask.ravel().tolist() if mask is not None else None

    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchesMask=mask_list,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(os.path.join(out_dir, "01_inlier_matches.png"), match_vis)

    h, w = gray1.shape
    tile = 40
    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                checker[y:y + tile, x:x + tile] = 1
    blend = np.where(checker == 1, gray1, warped2)
    cv2.imwrite(os.path.join(out_dir, "02_registered_checkerboard.png"), blend)

    diff = cv2.absdiff(gray1, warped2)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    cv2.imwrite(os.path.join(out_dir, "03_difference_map.png"), diff_color)

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)); axes[0, 0].set_title("Image A"); axes[0, 0].axis("off")
    axes[0, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)); axes[0, 1].set_title("Image B"); axes[0, 1].axis("off")
    axes[1, 0].imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)); axes[1, 0].set_title("Inlier matches (green)"); axes[1, 0].axis("off")
    axes[1, 1].imshow(blend, cmap="gray"); axes[1, 1].set_title("Registered (checkerboard blend)"); axes[1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "00_summary.png"), dpi=140)
    plt.close(fig)


# Orchestration
def run_pipeline(img1_bgr, img2_bgr, detector="orb", out_dir="outputs"):
    t0 = time.time()
    gray1, _ = preprocess(img1_bgr)
    gray2, _ = preprocess(img2_bgr)
    kp1, desc1 = detect_and_describe(gray1, detector)
    kp2, desc2 = detect_and_describe(gray2, detector)

    print(f"Image A: {len(kp1)} keypoints found")
    print(f"Image B: {len(kp2)} keypoints found")
    if desc1 is None or len(kp1) == 0:
        print("Registration FAILED: zero keypoints in Image A - the content "
              "is likely too smooth/low-texture for this detector to find "
              "any features at all. This can be a real, reportable finding "
              "(e.g. a spectral band with little surface contrast), not "
              "necessarily a bug. Try a different --band, or SIFT instead "
              "of ORB, before assuming something is broken.")
        return None
    if desc2 is None or len(kp2) == 0:
        print("Registration FAILED: zero keypoints in Image B - see note above.")
        return None

    matches = match_features(desc1, desc2, detector)
    H, mask = estimate_homography(kp1, kp2, matches)

    if H is None:
        print("Registration FAILED: not enough good matches to estimate a transform.")
        return None

    warped2 = register_images(gray1, gray2, H)
    elapsed = time.time() - t0
    metrics = evaluate(kp1, kp2, matches, mask, H, elapsed)
    visualize(img1_bgr, img2_bgr, kp1, kp2, matches, mask, gray1, warped2, out_dir)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="ChandraMatch pipeline")
    parser.add_argument("--img1", type=str, default=None)
    parser.add_argument("--img2", type=str, default=None)
    parser.add_argument("--pds4_1", type=str, default=None)
    parser.add_argument("--pds4_2", type=str, default=None)
    parser.add_argument("--inspect", type=str, default=None)
    parser.add_argument("--band", type=int, default=0, help="Band index for multi-band cubes like IIRS (default 0 = nearest visible/NIR, closest to OHRC/TMC-2's panchromatic range)")
    parser.add_argument("--detector", type=str, default="orb", choices=["orb", "sift"])
    parser.add_argument("--out", type=str, default="outputs")
    args = parser.parse_args()

    if args.inspect:
        inspect_pds4(args.inspect)
        return

    if args.pds4_1 and args.pds4_2:
        img1 = load_pds4_image(args.pds4_1, band_index=args.band)
        img2 = load_pds4_image(args.pds4_2, band_index=args.band)
        print(f"Loaded PDS4 images: {args.pds4_1}, {args.pds4_2}")
        os.makedirs(args.out, exist_ok=True)
        cv2.imwrite(os.path.join(args.out, "pds4_img1.png"), img1)
        cv2.imwrite(os.path.join(args.out, "pds4_img2.png"), img2)
        print(f"Also saved as plain PNGs in {args.out}/ for the web demo.")
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