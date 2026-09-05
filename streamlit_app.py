"""
ChandraMatch - Local Web Demo (Streamlit)
==========================================
Same tested backend as before (chandramatch_pipeline.py functions,
unchanged) with a restructured, restrained UI:
  - Flat colors only, no gradients
  - Rectangular buttons (not pill-shaped)
  - No emoji icons, no fake metrics/reviews/counters
  - A "Cross-Scale Validation" tab reading cross_scale_results.csv
  - A feedback form (writes to feedback_log.csv, local only)
  - A contact footer

NOTE ON THE CSS: Streamlit's internal DOM class names (things like
.stApp, .stButton) are stable across recent versions but not
guaranteed forever. If a style looks off after an update, that's the
part to check first — everything else (data, metrics, logic) is
independent of styling and unaffected by any CSS quirks.

Run with:
    streamlit run streamlit_app.py
"""

import csv
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np
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

RESULTS_DIR = "saved_results"
FEEDBACK_LOG = "feedback_log.csv"
CROSS_SCALE_CSV = "cross_scale_results.csv"
FAVICON_PATH = "assets/favicon.png"

os.makedirs(RESULTS_DIR, exist_ok=True)

st.set_page_config(
    page_title="ChandraMatch",
    page_icon=FAVICON_PATH if os.path.exists(FAVICON_PATH) else None,
    layout="wide",
)

# ---------------------------------------------------------------------------
# Restrained styling: flat colors, no gradients, rectangular buttons,
# a single accent color, subtle fade-in on new content only.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    .stApp {
        background-color: #12141a;
        color: #e8e8e8;
    }
    h1, h2, h3, h4 {
        color: #f2f2f0;
        font-weight: 600;
    }
    p, label, span, div {
        color: #cfcfcf;
    }
    .stButton > button {
        background-color: #c9944a;
        color: #12141a;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #b8823a;
        color: #12141a;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px 4px 0 0;
    }
    [data-testid="stMetricValue"] {
        color: #c9944a;
    }
    .cm-fade-in {
        animation: cmFadeIn 0.4s ease-in;
    }
    @keyframes cmFadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
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
# Header — factual, no hero copy
# ---------------------------------------------------------------------------
st.title("ChandraMatch")
st.markdown(
    "Feature-based image correspondence and registration for Chandrayaan-2 "
    "imagery (OHRC, TMC-2). Classical computer vision — ORB/SIFT feature "
    "matching, RANSAC outlier rejection, homography estimation — evaluated "
    "against real mission data."
)
st.divider()


# ---------------------------------------------------------------------------
# Shared pipeline helpers (unchanged from before)
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
    t0 = time.time()
    gray1, _ = preprocess(img1)
    gray2, _ = preprocess(img2)
    kp1, desc1 = detect_and_describe(gray1, detector)
    kp2, desc2 = detect_and_describe(gray2, detector)
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


def render_result(img1, img2, match_vis, blend, metrics):
    st.markdown('<div class="cm-fade-in">', unsafe_allow_html=True)
    st.subheader("Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidate matches", metrics["num_candidate_matches"])
    m2.metric("Inlier matches", metrics["num_inlier_matches"])
    m3.metric("Inlier ratio", f"{metrics['inlier_ratio']*100:.1f}%")
    m4.metric("RMSE (px)", metrics["reprojection_rmse_px"])
    with st.expander("Full metrics"):
        st.json(metrics)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), caption="Image A")
    c2.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), caption="Image B")

    st.subheader("Inlier matches")
    st.caption("Highlighted lines connect points RANSAC accepted as consistent with a single transform.")
    st.image(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB), use_container_width=True)

    st.subheader("Registered result")
    st.caption("Checkerboard blend of both images after alignment — continuous features across tile edges indicate accurate registration.")
    st.image(blend, clamp=True, use_container_width=True)
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
    return img1, img2, match_vis, blend, metrics


def list_saved():
    return sorted(
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
saved_labels = list_saved()
tab_titles = ["Live Run"] + saved_labels + ["Cross-Scale Validation", "About"]
tabs = st.tabs(tab_titles)

with tabs[0]:
    with st.sidebar:
        st.header("Settings")
        detector = st.selectbox("Feature detector", ["orb", "sift"], index=0)
        st.caption("ORB: faster. SIFT: often more robust on low-texture terrain.")

    col1, col2 = st.columns(2)
    with col1:
        file1 = st.file_uploader("Image A", type=["png", "jpg", "jpeg"], key="img1")
    with col2:
        file2 = st.file_uploader("Image B", type=["png", "jpg", "jpeg"], key="img2")

    run_btn = st.button("Run pipeline", type="primary", disabled=not (file1 and file2))

    if run_btn:
        img1 = to_bgr(file1)
        img2 = to_bgr(file2)
        with st.spinner("Running: preprocess, detect, match, RANSAC, register..."):
            result = run_full_pipeline(img1, img2, detector)
        if result is None:
            st.error("Registration failed — not enough good matches. These images may not overlap, or try the other detector.")
        else:
            st.session_state["last_result"] = result

    if "last_result" in st.session_state:
        img1, img2, match_vis, blend, metrics = st.session_state["last_result"]
        render_result(img1, img2, match_vis, blend, metrics)
        st.divider()
        label = st.text_input("Save this result as")
        if st.button("Save this result", disabled=not label.strip()):
            safe = save_result(label, img1, img2, match_vis, blend, metrics)
            st.success(f"Saved as '{safe}'.")
            st.rerun()
    else:
        st.info("Upload two images, pick a detector, then click Run pipeline.")

for i, label in enumerate(saved_labels, start=1):
    with tabs[i]:
        img1, img2, match_vis, blend, metrics = load_result(label)
        render_result(img1, img2, match_vis, blend, metrics)

# --- Cross-Scale Validation tab ---
with tabs[len(saved_labels) + 1]:
    st.subheader("Cross-Scale Validation")
    st.caption(
        "Real OHRC scenes, each synthetically degraded to TMC-2's documented "
        "~20x coarser resolution under a known applied transform, then "
        "registered. Because the transform is known, this measures accuracy "
        "directly against ground truth."
    )
    if os.path.exists(CROSS_SCALE_CSV):
        import pandas as pd
        df = pd.read_csv(CROSS_SCALE_CSV)
        df_ok = df[df["status"] == "ok"] if "status" in df.columns else df
        if len(df_ok):
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean inlier ratio", f"{df_ok['inlier_ratio'].mean()*100:.1f}%")
            c2.metric("Median scale error", f"{df_ok['scale_error_pct'].median():.2f}%")
            c3.metric("Median rotation error", f"{df_ok['angle_error_deg'].median():.2f} deg")
            st.dataframe(df_ok, use_container_width=True)
        else:
            st.warning("cross_scale_results.csv has no successful runs yet.")
    else:
        st.info(
            "No cross_scale_results.csv found yet. Run "
            "`python batch_simulate_cross_scale.py --scan <data folder>` "
            "to generate it, then reload this page."
        )

# --- About tab ---
with tabs[-1]:
    st.subheader("About this project")
    st.markdown(
        "ChandraMatch implements a classical feature-based pipeline for "
        "correspondence and registration of Chandrayaan-2 imagery: "
        "preprocessing, ORB/SIFT feature detection and description, "
        "brute-force matching with Lowe's ratio test, RANSAC-based outlier "
        "rejection, homography estimation, and registration, with "
        "quantitative evaluation (inlier ratio, reprojection RMSE) at every "
        "stage.\n\n"
        "Classical methods were used deliberately rather than a trained "
        "model: no labeled training data is required, results are "
        "inspectable at every stage, and the pipeline runs without a GPU."
    )
    st.subheader("Team")
    st.markdown("Add your team names and roles here.")


# ---------------------------------------------------------------------------
# Footer — feedback form + contact info
# ---------------------------------------------------------------------------
st.markdown('<div class="cm-footer">', unsafe_allow_html=True)
fcol1, fcol2 = st.columns(2)

with fcol1:
    st.markdown("**Feedback**")
    fb_name = st.text_input("Name (optional)", key="fb_name")
    fb_message = st.text_area("Comments or suggestions", key="fb_message", height=80)
    if st.button("Submit feedback", disabled=not fb_message.strip()):
        is_new = not os.path.exists(FEEDBACK_LOG)
        with open(FEEDBACK_LOG, "a", newline="") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "name", "message"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), fb_name, fb_message])
        st.success("Thanks — feedback recorded.")

with fcol2:
    st.markdown("**Contact**")
    st.markdown(
        "Add your name(s), email, and GitHub/LinkedIn links here so judges "
        "know who to reach."
    )

st.markdown('</div>', unsafe_allow_html=True)
