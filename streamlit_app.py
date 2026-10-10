"""
ChandraMatch - Local Web Demo (Streamlit)
==========================================
Tabs:
  Live Run          upload two images, run the pipeline, compare ORB vs SIFT, download results
  Saved Results     pick any result you saved earlier
  Evaluation        batch_results.csv + report_figures/ (time-gap study, ORB vs SIFT)
  Cross-Scale       cross_scale_results.csv (ground-truth simulation, OHRC degraded to TMC-2 scale)
  Cross-Sensor      cross_sensor_batch_results.csv (OHRC/TMC-2 vs IIRS, with honest reliability flags)
  Method            pipeline stages and design decisions
  About             project summary, team, limitations

Every data tab degrades gracefully: if its file is missing it tells you which
command generates it. Nothing on this page is hard-coded as a "result"; every
number is computed from your CSVs or from the run you just did.

Run with:
    python -m streamlit run streamlit_app.py
"""

import csv
import io
import json
import os
import re
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from chandramatch_pipeline import (
    preprocess,
    detect_and_describe,
    match_features,
    estimate_homography,
    register_images,
    evaluate,
)

# ---------------------------------------------------------------------------
# Config: EDIT THIS BLOCK with your real details
# ---------------------------------------------------------------------------
TEAM = [
    # {"name": "Your Name", "role": "Pipeline, scaling, web app",
    #  "email": "you@example.com", "github": "https://github.com/codder69-hue",
    #  "linkedin": ""},
]
REPO_URL = "https://github.com/codder69-hue/ChandraMatch"
INSTITUTION = ""   # e.g. "Your College, Department, Semester"

RESULTS_DIR = "saved_results"
FEEDBACK_LOG = "feedback_log.csv"
CROSS_SCALE_CSV = "cross_scale_results.csv"
BATCH_CSV = "batch_results.csv"
CROSS_SENSOR_CSV = "cross_sensor_batch_results.csv"
FIGURES_DIR = "report_figures"
FAVICON_PATH = "assets/favicon.png"

# Fewer inliers than this means the homography is statistically weak.
MIN_RELIABLE_INLIERS = 8

os.makedirs(RESULTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="ChandraMatch",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling: flat colours, one accent, rectangular buttons, subtle fade only
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp { background-color: #12141a; color: #e8e8e8; }
    h1, h2, h3, h4 { color: #f2f2f0; font-weight: 600; }
    p, label, span, div { color: #cfcfcf; }
    .stButton > button, .stDownloadButton > button {
        background-color: #c9944a;
        color: #12141a;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #b8823a;
        color: #12141a;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 4px 4px 0 0; }
    [data-testid="stMetricValue"] { color: #c9944a; }
    .cm-fade-in { animation: cmFadeIn 0.4s ease-in; }
    @keyframes cmFadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .cm-card {
        border: 1px solid #2a2d35;
        border-left: 3px solid #c9944a;
        border-radius: 4px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0 1rem 0;
        background-color: #181b22;
    }
    .cm-stage {
        border: 1px solid #2a2d35;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        background-color: #181b22;
    }
    .cm-footer {
        border-top: 1px solid #2a2d35;
        margin-top: 3rem;
        padding-top: 1.5rem;
        color: #8a8a8a;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("ChandraMatch")
st.markdown(
    "Feature-based image correspondence and registration for Chandrayaan-2 "
    "imagery (OHRC, TMC-2, IIRS). Classical computer vision: ORB/SIFT feature "
    "matching, RANSAC outlier rejection and homography estimation, evaluated "
    "against real mission data."
)
st.divider()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_bgr(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def checkerboard_blend(gray1, warped2, tile_frac=15):
    h, w = gray1.shape
    tile = max(20, min(h, w) // tile_frac)
    checker = np.zeros((h, w), dtype=np.uint8)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            if ((x // tile) + (y // tile)) % 2 == 0:
                checker[y:y + tile, x:x + tile] = 1
    return np.where(checker == 1, gray1, warped2)


def run_full_pipeline(img1, img2, detector):
    """Returns (img1, img2, match_vis, blend, metrics) or None on failure."""
    t0 = time.time()
    gray1, _ = preprocess(img1)
    gray2, _ = preprocess(img2)
    kp1, desc1 = detect_and_describe(gray1, detector)
    kp2, desc2 = detect_and_describe(gray2, detector)
    if desc1 is None or desc2 is None or len(kp1) == 0 or len(kp2) == 0:
        return None
    matches = match_features(desc1, desc2, detector)
    H, mask = estimate_homography(kp1, kp2, matches)
    if H is None:
        return None
    warped2 = register_images(gray1, gray2, H)
    elapsed = time.time() - t0
    metrics = evaluate(kp1, kp2, matches, mask, H, elapsed)
    mask_list = mask.ravel().tolist()
    match_vis = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        matchesMask=mask_list, matchColor=(201, 148, 74),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    blend = checkerboard_blend(gray1, warped2)
    return img1, img2, match_vis, blend, metrics


def is_reliable(metrics):
    n_in = metrics.get("num_inlier_matches", 0)
    rmse = metrics.get("reprojection_rmse_px", None)
    if n_in < MIN_RELIABLE_INLIERS:
        return False
    if n_in == 4 and rmse == 0:
        return False
    return True


def png_bytes(arr_bgr_or_gray):
    ok, buf = cv2.imencode(".png", arr_bgr_or_gray)
    return buf.tobytes() if ok else b""


def render_result(img1, img2, match_vis, blend, metrics, key_prefix="r"):
    st.markdown('<div class="cm-fade-in">', unsafe_allow_html=True)
    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidate matches", metrics["num_candidate_matches"])
    m2.metric("Inlier matches", metrics["num_inlier_matches"])
    m3.metric("Inlier ratio", f"{metrics['inlier_ratio']*100:.1f}%")
    m4.metric("RMSE (px)", metrics["reprojection_rmse_px"])

    if is_reliable(metrics):
        st.success(
            f"Reliable fit: {metrics['num_inlier_matches']} inliers "
            f"(threshold: at least {MIN_RELIABLE_INLIERS})."
        )
    else:
        st.warning(
            f"Statistically weak fit: only {metrics['num_inlier_matches']} inliers "
            f"(threshold: at least {MIN_RELIABLE_INLIERS}). A homography needs 4 points "
            "to be solved exactly, so very few inliers can look perfect "
            "(RMSE near 0) without being meaningful. Treat this result as unreliable."
        )

    with st.expander("Full metrics"):
        st.json(metrics)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="Image A")
    c2.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="Image B")

    st.subheader("Inlier matches")
    st.caption("Lines connect points RANSAC accepted as consistent with a single transform.")
    st.image(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Registered result")
    st.caption(
        "Checkerboard blend of both images after alignment. Continuous features "
        "across tile edges indicate accurate registration."
    )
    st.image(blend, clamp=True, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    d1.download_button("Download metrics (JSON)", json.dumps(metrics, indent=2),
                       file_name="metrics.json", mime="application/json",
                       key=f"{key_prefix}_dl_json")
    d2.download_button("Download match image (PNG)", png_bytes(match_vis),
                       file_name="matches.png", mime="image/png",
                       key=f"{key_prefix}_dl_match")
    d3.download_button("Download registered blend (PNG)", png_bytes(blend),
                       file_name="registered_blend.png", mime="image/png",
                       key=f"{key_prefix}_dl_blend")
    st.markdown('</div>', unsafe_allow_html=True)


def save_result(label, img1, img2, match_vis, blend, metrics):
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in label.strip())
    folder = os.path.join(RESULTS_DIR, safe)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, "img1.png"), img1)
    cv2.imwrite(os.path.join(folder, "img2.png"), img2)
    cv2.imwrite(os.path.join(folder, "match_vis.png"), match_vis)
    cv2.imwrite(os.path.join(folder, "blend.png"), blend)
    with open(os.path.join(folder, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return safe


def load_result(label):
    folder = os.path.join(RESULTS_DIR, label)
    img1 = cv2.imread(os.path.join(folder, "img1.png"))
    img2 = cv2.imread(os.path.join(folder, "img2.png"))
    match_vis = cv2.imread(os.path.join(folder, "match_vis.png"))
    blend = cv2.imread(os.path.join(folder, "blend.png"), cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(folder, "metrics.json")) as f:
        metrics = json.load(f)
    if img1 is None or img2 is None or match_vis is None or blend is None:
        raise FileNotFoundError(f"Saved result '{label}' is incomplete.")
    return img1, img2, match_vis, blend, metrics


def list_saved():
    return sorted(
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
        and os.path.exists(os.path.join(RESULTS_DIR, d, "metrics.json"))
    )


def read_csv_safe(path):
    """Returns (df, error_message). Never raises."""
    if not os.path.exists(path):
        return None, "missing"
    try:
        df = pd.read_csv(path)
    except Exception as e:  # corrupt or empty file
        return None, f"could not read: {e}"
    if df.empty:
        return None, "empty"
    return df, None


def parse_stamp(text):
    """Pulls a YYYYMMDD[THHMMSS] timestamp out of a product name, else None."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})(?:T?(\d{2})(\d{2})(\d{2}))?", str(text))
    if not m:
        return None
    try:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
        hh = int(m[4] or 0)
        mi = int(m[5] or 0)
        ss = int(m[6] or 0)
        return datetime(y, mo, d, hh, mi, ss)
    except ValueError:
        return None


def show_figure(name, caption):
    path = os.path.join(FIGURES_DIR, name)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"{name} not found in {FIGURES_DIR}/. Run analyze_results.py to generate it.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Settings")
    detector = st.selectbox("Feature detector", ["orb", "sift"], index=0)
    st.caption("ORB: faster. SIFT: often more robust on low-texture terrain.")
    compare_both = st.checkbox("Also run the other detector and compare", value=False)
    st.divider()
    st.caption(f"Source code: {REPO_URL}")
    if os.path.exists(FEEDBACK_LOG):
        with open(FEEDBACK_LOG, "rb") as f:
            st.download_button("Download feedback log (CSV)", f.read(),
                               file_name="feedback_log.csv", mime="text/csv")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_live, tab_saved, tab_eval, tab_scale, tab_sensor, tab_method, tab_about = st.tabs(
    ["Live Run", "Saved Results", "Evaluation", "Cross-Scale",
     "Cross-Sensor", "Method", "About"]
)

# --- Live Run ---------------------------------------------------------------
with tab_live:
    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Image A", type=["png", "jpg", "jpeg"], key="img1")
    with col2:
        file2 = st.file_uploader("Image B", type=["png", "jpg", "jpeg"], key="img2")

    run_btn = st.button("Run pipeline", type="primary", disabled=not (file1 and file2))

    if run_btn:
        try:
            img1 = to_bgr(file1)
            img2 = to_bgr(file2)
            with st.spinner(f"Running with {detector.upper()}: preprocess, detect, match, RANSAC, register..."):
                result = run_full_pipeline(img1, img2, detector)
            other = "sift" if detector == "orb" else "orb"
            other_result = None
            if compare_both:
                with st.spinner(f"Running with {other.upper()} for comparison..."):
                    other_result = run_full_pipeline(img1, img2, other)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            result, other_result = None, None
        else:
            if result is None:
                st.error(
                    "Registration failed: not enough good matches or no features found. "
                    "These images may not overlap, or try the other detector."
                )
                st.session_state.pop("last_result", None)
                st.session_state.pop("compare", None)
            else:
                st.session_state["last_result"] = result
                if compare_both:
                    st.session_state["compare"] = {
                        "primary": (detector, result[4]),
                        "other": (other, other_result[4] if other_result else None),
                    }
                else:
                    st.session_state.pop("compare", None)

    if "last_result" in st.session_state:
        img1, img2, match_vis, blend, metrics = st.session_state["last_result"]

        cmp = st.session_state.get("compare")
        if cmp:
            st.subheader("Detector comparison")
            rows = []
            for tag in ("primary", "other"):
                name, m = cmp[tag]
                if m is None:
                    rows.append({"detector": name.upper(), "candidates": "-", "inliers": "-",
                                 "inlier ratio": "failed", "RMSE (px)": "-", "reliable": "no"})
                else:
                    rows.append({
                        "detector": name.upper(),
                        "candidates": m["num_candidate_matches"],
                        "inliers": m["num_inlier_matches"],
                        "inlier ratio": f"{m['inlier_ratio']*100:.1f}%",
                        "RMSE (px)": m["reprojection_rmse_px"],
                        "reliable": "yes" if is_reliable(m) else "no",
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Detail below is for the detector selected in the sidebar.")

        render_result(img1, img2, match_vis, blend, metrics, key_prefix="live")
        st.divider()
        label = st.text_input("Save this result as")
        if st.button("Save this result", disabled=not label.strip()):
            safe = save_result(label, img1, img2, match_vis, blend, metrics)
            st.success(f"Saved as '{safe}'. It now appears under Saved Results.")
    else:
        st.info("Upload two images, pick a detector, then click Run pipeline.")

# --- Saved Results ----------------------------------------------------------
with tab_saved:
    saved = list_saved()
    if not saved:
        st.info("Nothing saved yet. Run the pipeline in Live Run and use Save this result.")
    else:
        choice = st.selectbox("Saved result", saved)
        try:
            r = load_result(choice)
            render_result(*r, key_prefix="saved")
        except Exception as e:
            st.error(f"Could not load '{choice}': {e}")

# --- Evaluation -------------------------------------------------------------
with tab_eval:
    st.subheader("Batch evaluation on real OHRC pairs")
    st.caption(
        "Every overlapping OHRC scene pair found in the downloaded data, "
        "registered with both ORB and SIFT. Numbers below are computed from "
        f"{BATCH_CSV}."
    )
    df, err = read_csv_safe(BATCH_CSV)
    if df is None:
        st.info(
            f"{BATCH_CSV} not available ({err}). Run "
            "`python batch_run.py --scan <data folder>` to generate it."
        )
    elif "inlier_ratio" not in df.columns or "detector" not in df.columns:
        st.warning(f"{BATCH_CSV} is missing the expected columns (detector, inlier_ratio).")
    else:
        ok = df[df["status"] == "ok"] if "status" in df.columns else df
        if ok.empty:
            st.warning("No successful runs in the batch file.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Registrations", len(ok))
            if "pair_a" in ok.columns and "pair_b" in ok.columns:
                c2.metric("Scene pairs", len(ok[["pair_a", "pair_b"]].drop_duplicates()))
            c3.metric("Overall mean inlier ratio", f"{ok['inlier_ratio'].mean()*100:.1f}%")

            st.markdown("**By detector**")
            by_det = ok.groupby("detector")["inlier_ratio"].agg(["mean", "median", "std", "count"]).round(4)
            st.dataframe(by_det, use_container_width=True)

            if "pair_a" in ok.columns and "pair_b" in ok.columns:
                ok = ok.copy()
                gaps = []
                for _, row in ok.iterrows():
                    t1, t2 = parse_stamp(row["pair_a"]), parse_stamp(row["pair_b"])
                    gaps.append(abs((t2 - t1).total_seconds()) / 3600.0 if t1 and t2 else np.nan)
                ok["time_gap_hours"] = gaps
                g = ok.dropna(subset=["time_gap_hours"])
                if len(g) >= 4:
                    same = g[g["time_gap_hours"] < 24]
                    far = g[g["time_gap_hours"] >= 24]
                    st.markdown("**Effect of acquisition time gap**")
                    a, b, c = st.columns(3)
                    a.metric("Same day (<24 h)", f"{same['inlier_ratio'].mean()*100:.1f}%" if len(same) else "n/a",
                             help=f"n = {len(same)}")
                    b.metric("Larger gap (24 h or more)", f"{far['inlier_ratio'].mean()*100:.1f}%" if len(far) else "n/a",
                             help=f"n = {len(far)}")
                    corr = g["time_gap_hours"].corr(g["inlier_ratio"])
                    c.metric("Correlation (gap vs ratio)", f"{corr:.2f}" if pd.notna(corr) else "n/a",
                             help="Pearson correlation. Negative means quality drops as the gap grows.")
                    st.caption(
                        "Interpretation: matching quality falls as the time gap grows, "
                        "which is consistent with changing solar illumination and shadows "
                        "altering the appearance of the same terrain."
                    )

            f1, f2 = st.columns(2)
            with f1:
                show_figure("inlier_ratio_vs_time_gap.png", "Inlier ratio against time gap between scenes")
            with f2:
                show_figure("orb_vs_sift_per_pair.png", "ORB against SIFT, per scene pair")
            show_figure("inlier_ratio_distribution.png", "Distribution of inlier ratios")

            with st.expander("Raw results table"):
                st.dataframe(df, use_container_width=True)
            st.download_button("Download batch_results.csv", df.to_csv(index=False),
                               file_name="batch_results.csv", mime="text/csv")

# --- Cross-Scale ------------------------------------------------------------
with tab_scale:
    st.subheader("Cross-scale validation")
    st.caption(
        "Real OHRC scenes, each synthetically degraded to TMC-2's documented "
        "~20x coarser resolution under a known applied transform, then "
        "registered. Because the transform is known, this measures accuracy "
        "directly against ground truth."
    )
    df, err = read_csv_safe(CROSS_SCALE_CSV)
    if df is None:
        st.info(
            f"{CROSS_SCALE_CSV} not available ({err}). Run "
            "`python batch_simulate_cross_scale.py --scan <data folder>` "
            "to generate it, then reload."
        )
    else:
        df_ok = df[df["status"] == "ok"] if "status" in df.columns else df
        needed = {"inlier_ratio", "scale_error_pct", "angle_error_deg"}
        if df_ok.empty:
            st.warning("No successful runs in the cross-scale file.")
        elif not needed.issubset(df_ok.columns):
            st.warning(f"Expected columns missing: {sorted(needed - set(df_ok.columns))}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Scenes tested", len(df_ok))
            c2.metric("Mean inlier ratio", f"{df_ok['inlier_ratio'].mean()*100:.1f}%")
            c3.metric("Median scale error", f"{df_ok['scale_error_pct'].median():.2f}%")
            c4.metric("Median rotation error", f"{df_ok['angle_error_deg'].median():.2f} deg")

            if "true_angle_deg" in df_ok.columns and "recovered_angle_deg" in df_ok.columns:
                st.markdown("**Applied rotation against recovered rotation**")
                plot_df = df_ok[["true_angle_deg", "recovered_angle_deg"]].rename(
                    columns={"true_angle_deg": "applied (deg)", "recovered_angle_deg": "recovered (deg)"}
                )
                st.scatter_chart(plot_df, x="applied (deg)", y="recovered (deg)", color="#c9944a")
                st.caption("Points on the diagonal mean the pipeline recovered the applied rotation exactly.")

            st.dataframe(df_ok, use_container_width=True)
            st.download_button("Download cross_scale_results.csv", df_ok.to_csv(index=False),
                               file_name="cross_scale_results.csv", mime="text/csv")

# --- Cross-Sensor -----------------------------------------------------------
with tab_sensor:
    st.subheader("Cross-sensor matching (OHRC or TMC-2 against IIRS)")
    st.markdown(
        '<div class="cm-card">IIRS is a hyperspectral imager, so there is no single '
        "obvious band to compare against a panchromatic image. Raw IIRS data also "
        "contains push-broom striping. This tab reports what was actually found, "
        "including the failures.</div>",
        unsafe_allow_html=True,
    )
    df, err = read_csv_safe(CROSS_SENSOR_CSV)
    if df is None:
        st.info(
            f"{CROSS_SENSOR_CSV} not available ({err}). Run "
            "`python batch_cross_sensor.py` to generate it, then reload."
        )
    else:
        if "num_inlier_matches" in df.columns:
            n_in = pd.to_numeric(df["num_inlier_matches"], errors="coerce")
            total = len(df)
            reliable = int((n_in >= MIN_RELIABLE_INLIERS).sum())
            weak = int(((n_in < MIN_RELIABLE_INLIERS) & n_in.notna()).sum())
            failed = int(n_in.isna().sum())
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Configurations tried", total)
            c2.metric(f"Reliable ({MIN_RELIABLE_INLIERS}+ inliers)", reliable)
            c3.metric("Statistically weak", weak)
            c4.metric("No match / failed", failed)
            if reliable == 0:
                st.warning(
                    "No configuration produced a statistically reliable cross-sensor match "
                    "in this file. This is a genuine finding, not a bug: it identifies "
                    "band selection and striping removal as the next problems to solve."
                )
        st.dataframe(df, use_container_width=True)
        st.download_button("Download cross_sensor_batch_results.csv", df.to_csv(index=False),
                           file_name="cross_sensor_batch_results.csv", mime="text/csv")

    st.markdown("**Why weak fits are flagged**")
    st.markdown(
        "A homography has 8 degrees of freedom and is solved exactly from 4 point "
        "pairs. A fit built on exactly 4 inliers will therefore show a reprojection "
        "error of 0.0 regardless of whether it is correct. The pipeline treats fewer "
        f"than {MIN_RELIABLE_INLIERS} inliers as unreliable to avoid reporting such "
        "fits as successes."
    )

# --- Method -----------------------------------------------------------------
with tab_method:
    st.subheader("Pipeline")
    stages = [
        ("1. Load and preprocess",
         "PDS4 products are read with the pdr library. Large strips are downsampled "
         "before float conversion to keep memory bounded, then contrast-normalised."),
        ("2. Detect and describe",
         "ORB (fast, binary descriptors) or SIFT (scale-invariant, float descriptors) "
         "extract keypoints and descriptors from each image."),
        ("3. Match",
         "Brute-force k-nearest-neighbour matching, filtered with Lowe's ratio test "
         "(threshold 0.75) to discard ambiguous matches."),
        ("4. RANSAC homography",
         "cv2.findHomography with RANSAC (reprojection threshold 5 px) separates "
         "geometrically consistent matches (inliers) from outliers."),
        ("5. Register",
         "The second image is warped into the first image's frame with the estimated "
         "homography and shown as a checkerboard blend for visual verification."),
        ("6. Evaluate",
         "Inlier ratio and reprojection RMSE are computed for every run. Fits with "
         f"fewer than {MIN_RELIABLE_INLIERS} inliers are flagged as unreliable."),
    ]
    for title, body in stages:
        st.markdown(f'<div class="cm-stage"><b>{title}</b><br>{body}</div>', unsafe_allow_html=True)

    st.subheader("Design decisions")
    st.markdown(
        "- **Classical, not learned.** No labelled training data is needed, every "
        "stage can be inspected, and it runs on a laptop without a GPU.\n"
        "- **Ground-truth validation.** Real accuracy needs a known answer, so the "
        "cross-scale test applies a known transform to real data and measures recovery.\n"
        "- **Automation.** Overlap discovery from geometry files, batch registration, "
        "memory-aware caching and result analysis are all scripted.\n"
        "- **Honest reporting.** Weak fits are flagged, failures are listed, and the "
        "limitations are stated below."
    )

# --- About ------------------------------------------------------------------
with tab_about:
    st.subheader("About this project")
    st.markdown(
        "ChandraMatch implements a classical feature-based pipeline for "
        "correspondence and registration of Chandrayaan-2 imagery, with "
        "quantitative evaluation at every stage. It began as a hackathon "
        "prototype and is being extended as a college minor project, with a "
        "focus on automation and validation on real data."
    )
    if INSTITUTION:
        st.markdown(f"**Institution:** {INSTITUTION}")

    st.subheader("Known limitations")
    st.markdown(
        "- Reliable cross-sensor (OHRC or TMC-2 to IIRS) matching has not yet been "
        "demonstrated. Raw IIRS striping and band choice are open problems.\n"
        "- Matching quality drops as the time gap between scenes grows.\n"
        "- A single global homography assumes locally flat terrain; strong relief "
        "would need a more flexible model.\n"
        "- TMC-2 coverage is limited near the south pole, where OHRC is the main source."
    )

    st.subheader("Data and tools")
    st.markdown(
        "Chandrayaan-2 data from ISRO's ISSDC (PRADAN). Built with OpenCV, NumPy, "
        "pandas, pdr and Streamlit."
    )

    st.subheader("Team")
    if TEAM:
        for p in TEAM:
            st.markdown(f"**{p.get('name', '')}**, {p.get('role', '')}")
    else:
        st.caption("Set the TEAM list at the top of streamlit_app.py.")

# ---------------------------------------------------------------------------
# Footer: feedback form + contact
# ---------------------------------------------------------------------------
st.markdown('<div class="cm-footer">', unsafe_allow_html=True)
fcol1, fcol2 = st.columns(2)

with fcol1:
    st.markdown("**Feedback**")
    fb_name = st.text_input("Name (optional)", key="fb_name")
    fb_message = st.text_area("Comments or suggestions", key="fb_message", height=80)
    if st.button("Submit feedback", disabled=not fb_message.strip()):
        try:
            is_new = not os.path.exists(FEEDBACK_LOG)
            with open(FEEDBACK_LOG, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if is_new:
                    writer.writerow(["timestamp", "name", "message"])
                writer.writerow([datetime.now().isoformat(timespec="seconds"), fb_name, fb_message])
            st.success("Thank you, feedback recorded.")
        except OSError as e:
            st.error(f"Could not save feedback: {e}")

with fcol2:
    st.markdown("**Contact**")
    if TEAM:
        for p in TEAM:
            parts = [f"**{p.get('name', '')}**"]
            if p.get("email"):
                parts.append(f"[{p['email']}](mailto:{p['email']})")
            if p.get("github"):
                parts.append(f"[GitHub]({p['github']})")
            if p.get("linkedin"):
                parts.append(f"[LinkedIn]({p['linkedin']})")
            st.markdown(" | ".join(parts))
    else:
        st.caption("Contact details not set. Edit the TEAM list at the top of streamlit_app.py.")
    st.markdown(f"[Source code on GitHub]({REPO_URL})")

st.markdown('</div>', unsafe_allow_html=True)