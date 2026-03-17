

import os
import tempfile
from pathlib import Path
import math

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from extract_features import run_pipeline
from collapse_features import collapse_to_vector
from run_models import (
    run_all_models,
    get_view,
    results_to_dataframe,
    ALL_PHYSICIANS,
    ALL_TARGETS,
    _model_id,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EEG Abnormality Prediction Pipeline",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@300;400;600;700&display=swap');

    :root {
        --fh-navy:#00456A;
        --fh-blue:#006699;
        --fh-teal:#0091B3;
        --fh-dark:#1B2A3D;
        --fh-orange:#E87722;
        --fh-orange-lt:#F5A623;
        --fh-soft:#DCECF5;
    }

    .stApp {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        background: linear-gradient(160deg,#E8F4F8 0%,#D0E8F2 30%,#C2DFF0 60%,#D6EDF5 100%) !important;
    }

    section[data-testid="stSidebar"] {
        background:#FFFFFF !important;
        border-right:1px solid #B8CCD8;
    }

    section[data-testid="stSidebar"] > div:first-child {
        padding-top:0.5rem;
    }

    section[data-testid="stSidebar"] .stMarkdown h4 {
        color:#00456A !important;
        font-size:0.9rem !important;
        font-weight:700 !important;
        margin-bottom:0.4rem !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] .stMarkdown label,
    section[data-testid="stSidebar"] .stMarkdown div {
        color:#2C3E50 !important;
    }

    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stRadio label span,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span,
    section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label p {
        color:#2C3E50 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSelectbox label span,
    section[data-testid="stSidebar"] .stFileUploader label,
    section[data-testid="stSidebar"] .stFileUploader label span,
    section[data-testid="stSidebar"] .stFileUploader label p {
        color:#00456A !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background:#DCECF5 !important;
        border:2px dashed #8CB4C8 !important;
        border-radius:12px !important;
        padding:1rem !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
        color:#00456A !important;
    }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
        background:linear-gradient(135deg,#006699,#0091B3) !important;
        color:white !important;
        border:none !important;
        font-weight:600 !important;
        border-radius:6px !important;
        width:100% !important;
        padding:0.5rem 1rem !important;
    }

    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    header[data-testid="stHeader"],
    .stDeployButton {
        display:none !important;
        height:0 !important;
        visibility:hidden !important;
    }

    .block-container {
        padding-top:0.5rem !important;
    }

    .stApp .stMarkdown h1,
    .stApp .stMarkdown h2,
    .stApp .stMarkdown h3,
    .stApp .stMarkdown h4,
    .stApp .stMarkdown h5,
    .stApp .stMarkdown h6 {
        color:#00456A !important;
    }

    .stApp .stMarkdown p {
        color:#1B2A3D !important;
    }

    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] label span,
    div[data-testid="stRadio"] label p,
    div[role="radiogroup"] label,
    div[role="radiogroup"] label span,
    div[role="radiogroup"] label p {
        color:#00456A !important;
        font-weight:600 !important;
    }

    .demo-scores {
        display:flex;
        gap:0.4rem;
        margin-top:0.4rem;
        flex-wrap:wrap;
    }

    .score-badge {
        background:#E8EEF2;
        color:#00456A;
        font-size:0.72rem;
        padding:0.15rem 0.5rem;
        border-radius:20px;
        font-weight:600;
    }

    .score-1 { background:#D4EDDA; color:#155724; }
    .score-2 { background:#FFF3CD; color:#856404; }
    .score-3 { background:#FFE0CC; color:#8B4513; }
    .score-4 { background:#F8D7DA; color:#721C24; }

    .status-ready {
        background:#D4EDDA;
        color:#155724;
        padding:0.5rem 1rem;
        border-radius:6px;
        font-size:0.85rem;
        font-weight:600;
        text-align:center;
    }

    .status-processing {
        background:#CCE5FF;
        color:#004085;
        padding:0.5rem 1rem;
        border-radius:6px;
        font-size:0.85rem;
        font-weight:600;
        text-align:center;
    }

    .status-empty {
        background:#F0F2F4;
        color:#5A6B7D;
        padding:0.5rem 1rem;
        border-radius:6px;
        font-size:0.85rem;
        text-align:center;
    }

    .fh-divider {
        height:2px;
        background:linear-gradient(90deg,#E87722 0%,#F5A623 40%,transparent 100%);
        border:none;
        margin:1rem 0;
        border-radius:2px;
    }

    .stButton > button {
        background:linear-gradient(135deg,#E87722,#F5A623) !important;
        color:white !important;
        border:none !important;
        font-weight:600 !important;
        border-radius:6px !important;
        padding:0.5rem 1.5rem !important;
        transition:all 0.2s !important;
    }

    .stButton > button:hover {
        transform:translateY(-1px) !important;
        box-shadow:0 4px 12px rgba(232,119,34,0.35) !important;
    }

    .stDownloadButton > button {
        background:linear-gradient(135deg,#006699,#0091B3) !important;
        color:white !important;
        border:none !important;
        font-weight:600 !important;
        border-radius:6px !important;
    }

    .stDownloadButton > button:hover {
        box-shadow:0 4px 12px rgba(0,102,153,0.3) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap:0px;
        border-bottom:2px solid #D6DCE2;
    }

    .stTabs [data-baseweb="tab"] {
        color:#5A6B7D;
        font-weight:600;
        padding:0.7rem 1.5rem;
    }

    .stTabs [aria-selected="true"] {
        color:#00456A !important;
        border-bottom:3px solid #E87722 !important;
    }

    .metric-card {
        background:white;
        border-radius:10px;
        padding:0.9rem 1rem;
        border:1px solid #D6DCE2;
        text-align:center;
    }

    /* ── Score table card ── */
    .score-table-card {
        background: white;
        border-radius: 12px;
        border: 1px solid #D6DCE2;
        overflow: hidden;
        margin-top: 0.5rem;
    }

    .score-table-header {
        background: linear-gradient(90deg, #00456A, #006699);
        color: white;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }

    /* ── SHAP bar inside explanation tab ── */
    .shap-pos { color: #B42318; font-weight: 700; }
    .shap-neg { color: #156A3A; font-weight: 700; }

    .footer-authors {
        background:rgba(255,255,255,0.7);
        border-top:1px solid #B8CCD8;
        border-radius:8px 8px 0 0;
        margin-top:3rem;
        padding:1rem 1.5rem 0.5rem 1.5rem;
        text-align:center;
        color:#8B9DB3;
        font-size:0.78rem;
        line-height:1.6;
    }

    .footer-authors a { color:#0091B3; text-decoration:none; }

    .footer-authors .author-line {
        color:#5A6B7D;
        font-size:0.82rem;
        font-weight:600;
    }

    #MainMenu, footer { visibility:hidden !important; display:none !important; }
    header { visibility:hidden !important; height:0 !important; display:none !important; }
</style>
""", unsafe_allow_html=True)

# ── Demo files ────────────────────────────────────────────────────────────────
DEMO_FILES = [
    {
        "label": "Normal EEG",
        "path": "hdf/normal/51d51b4e-da45-44d4-922f-b628b33c240a-ico-4-ltc.hdf5",
        "description": "All targets score 1 — no abnormality detected",
        "scores": {"Abnormality": 1, "Focal Epi": 1, "Focal Non-epi": 1, "Gen Epi": 1, "Gen Non-epi": 1},
    },
    {
        "label": "Mild Abnormality (1-1-2-2-2)",
        "path": "hdf/1_1_2_2_2/000dee81-8ae4-4275-bfdf-556658a2709f-ico-4-ltc.hdf5",
        "description": "Borderline focal/generalised non-epileptic features",
        "scores": {"Abnormality": 2, "Focal Epi": 1, "Focal Non-epi": 2, "Gen Epi": 1, "Gen Non-epi": 2},
    },
    {
        "label": "Focal Epi + Gen Non-Epi (3-1-1-4-4)",
        "path": "hdf/3_1_1_4_4/381da026-5cab-48b1-bf58-a02704659c68-ico-4-ltc.hdf5",
        "description": "Probable focal epileptic, definite generalised non-epileptic",
        "scores": {"Abnormality": 4, "Focal Epi": 3, "Focal Non-epi": 1, "Gen Epi": 1, "Gen Non-epi": 4},
    },
    {
        "label": "Definite Focal Epi (4-1-1-1-4)",
        "path": "hdf/4_1_1_1_4/575f3167-871c-4be8-a12f-48ceb16915ca-ico-4-ltc.hdf5",
        "description": "Definite focal epileptic features, definite overall abnormality",
        "scores": {"Abnormality": 4, "Focal Epi": 4, "Focal Non-epi": 1, "Gen Epi": 1, "Gen Non-epi": 1},
    },
]

# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in [
    ("selected_file", None),
    ("file_source", None),
    ("pipeline_status", "idle"),
    ("feature_df", None),
    ("params_df", None),
    ("all_results", None),
    ("pipeline_error", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
LOGO_PATH = "assets/fraser_health_logo.png"

with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)
    else:
        st.markdown(
            '<div style="text-align:center;background:#FFF8F2;border:1px dashed #E87722;'
            'border-radius:6px;padding:0.6rem;margin-bottom:0.3rem;">'
            '<p style="color:#E87722;font-size:0.75rem;margin:0;">'
            'Place logo at <code>assets/fraser_health_logo.png</code></p></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div style="text-align:center;padding:0 0 0.2rem 0;">'
        '<p style="color:#00456A;margin:0;font-size:1rem;font-weight:700;letter-spacing:-0.3px;">'
        '🧠 EEG Abnormality Prediction Pipeline</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📂 Data Source")
    source = st.radio("src", ["Upload HDF5 file", "Use demo recording"], label_visibility="collapsed")

    if source == "Upload HDF5 file":
        uploaded = st.file_uploader(
            "HDF5",
            type=["hdf5", "h5"],
            help="Beamformed EEG file with label_tcs and label_names datasets",
        )
        if uploaded is not None:
            st.session_state.selected_file = uploaded.name
            st.session_state.file_source = "upload"
            st.session_state._uploaded_bytes = uploaded.getvalue()
            name_trunc = uploaded.name[:40] + ("…" if len(uploaded.name) > 40 else "")
            st.markdown(
                f'<div class="status-ready">✅ Loaded: {name_trunc}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='color:#5A6B7D;font-size:0.82rem;margin-bottom:0.5rem;'>"
            "Select a pre-loaded recording with known labels:</p>",
            unsafe_allow_html=True,
        )
        demo_choice = st.selectbox(
            "demo",
            range(len(DEMO_FILES)),
            format_func=lambda i: DEMO_FILES[i]["label"],
            label_visibility="collapsed",
        )
        demo = DEMO_FILES[demo_choice]
        st.session_state.selected_file = demo["label"]
        st.session_state.file_source = "demo"
        st.session_state._demo_index = demo_choice
        badges = "".join(
            f'<span class="score-badge score-{s}">{t}: {s}</span> '
            for t, s in demo["scores"].items()
        )
        st.markdown(
            f'<div style="background:#F0F2F4;border-radius:8px;padding:0.8rem;'
            f'margin-top:0.5rem;border-left:3px solid #E87722;">'
            f'<p style="color:#5A6B7D;font-size:0.8rem;margin:0 0 0.5rem 0;">'
            f'{demo["description"]}</p>'
            f'<div class="demo-scores">{badges}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ Pipeline Controls")
    can_run = st.session_state.selected_file is not None

    if st.button("🚀 Run Pipeline", disabled=not can_run, use_container_width=True):
        st.session_state.pipeline_status = "processing"
        st.session_state.feature_df = None
        st.session_state.params_df = None
        st.session_state.all_results = None
        st.session_state.pipeline_error = None

        pbar = st.progress(0, text="Starting…")
        plabel = st.empty()

        def _progress(done, total):
            frac = done / total if total > 0 else 0
            pbar.progress(frac, text=f"Parcel {done}/{total}")
            plabel.markdown(
                f"<p style='color:#5A6B7D;font-size:0.78rem;text-align:center;margin:0;'>"
                f"{int(frac * 100)}% complete</p>",
                unsafe_allow_html=True,
            )

        tmp_path = None
        try:
            if st.session_state.file_source == "upload":
                raw = st.session_state.get("_uploaded_bytes")
                if not raw:
                    raise ValueError("Upload bytes missing — please re-upload.")
                suffix = Path(st.session_state.selected_file).suffix or ".hdf5"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t:
                    t.write(raw)
                    tmp_path = t.name
                h5_path = tmp_path
            else:
                demo = DEMO_FILES[st.session_state._demo_index]
                h5_path = demo["path"]
                if not os.path.exists(h5_path):
                    raise FileNotFoundError(f"Demo file not found: {h5_path}")

            pbar.progress(0, text="Extracting spectral features…")
            df = run_pipeline(h5_path=h5_path, progress_callback=_progress)
            if df.empty:
                raise ValueError("No alpha peaks detected — check HDF5 data.")
            st.session_state.feature_df = df

            pbar.progress(0.92, text="Collapsing to model input vector…")
            params_df = collapse_to_vector(df)
            st.session_state.params_df = params_df

            pbar.progress(0.96, text="Running 20 models…")
            all_results = run_all_models(params_df)
            st.session_state.all_results = all_results
            st.session_state.pipeline_status = "done"

        except Exception as exc:
            st.session_state.pipeline_status = "error"
            st.session_state.pipeline_error = str(exc)
        finally:
            pbar.empty()
            plabel.empty()
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    status = st.session_state.pipeline_status
    if status == "idle":
        html = (
            '<div class="status-ready">✅ Ready to process</div>'
            if can_run
            else '<div class="status-empty">Upload or select a file to begin</div>'
        )
        st.markdown(html, unsafe_allow_html=True)
    elif status == "processing":
        st.markdown('<div class="status-processing">⏳ Processing…</div>', unsafe_allow_html=True)
    elif status == "done":
        df = st.session_state.feature_df
        n_p = df["label"].nunique() if df is not None and "label" in df.columns else 0
        n_m = sum(
            1 for v in (st.session_state.all_results or {}).values()
            if v.get("available") and v.get("result")
        )
        st.markdown(
            f'<div class="status-ready">✅ {n_p} parcels · {n_m}/20 models</div>',
            unsafe_allow_html=True,
        )
    elif status == "error":
        st.markdown(
            '<div style="background:#F8D7DA;color:#721C24;padding:0.5rem 1rem;'
            'border-radius:6px;font-size:0.82rem;font-weight:600;text-align:center;">'
            '❌ Pipeline failed</div>',
            unsafe_allow_html=True,
        )
        if st.session_state.pipeline_error:
            st.markdown(
                f'<p style="color:#721C24;font-size:0.75rem;margin-top:0.4rem;'
                f'word-break:break-word;">{st.session_state.pipeline_error}</p>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📥 Downloads")
    is_done = status == "done"

    c1, c2 = st.columns(2)
    with c1:
        fcsv = (
            st.session_state.feature_df.to_csv(index=False).encode()
            if st.session_state.feature_df is not None
            else b""
        )
        st.download_button(
            "Feature CSV",
            data=fcsv,
            file_name="eeg_features.csv",
            mime="text/csv",
            disabled=not is_done,
            use_container_width=True,
        )
    with c2:
        pcsv = (
            st.session_state.params_df.to_csv(index=False).encode()
            if st.session_state.params_df is not None
            else b""
        )
        st.download_button(
            "Params Vector",
            data=pcsv,
            file_name="eeg_params.csv",
            mime="text/csv",
            disabled=not is_done,
            use_container_width=True,
        )

    if is_done and st.session_state.all_results:
        rcsv = results_to_dataframe(st.session_state.all_results).to_csv(index=False).encode()
        st.download_button(
            "Model Results",
            data=rcsv,
            file_name="eeg_model_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _classify_call(prob: float, thr: float, band: float = 0.06):
    prob = float(prob)
    thr = float(thr)
    lo = max(0.0, thr - band)
    hi = min(1.0, thr + band)
    if prob < lo:
        return "Normal", "#156A3A"
    elif prob <= hi:
        return "Borderline", "#9A6B00"
    else:
        return "Abnormal", "#B42318"


def _gauge_html(
    target: str,
    prob: float,
    thr: float,
    phys_prob: float = None,
    phys_thr: float = None,
    phys_name: str = None,
    band: float = 0.06,
) -> str:
    prob = max(0.0, min(1.0, float(prob)))
    thr = max(0.0, min(1.0, float(thr)))
    band = max(0.01, min(0.15, float(band)))

    lo = max(0.0, thr - band)
    hi = min(1.0, thr + band)

    call, call_col = _classify_call(prob, thr, band=band)

    W, H = 220, 188
    CX, CY = 110, 138
    R = 84

    def _polar(p, radius=R):
        deg = 180 - (p * 180)
        rad = math.radians(deg)
        x = CX + radius * math.cos(rad)
        y = CY - radius * math.sin(rad)
        return x, y

    def _arc(p0, p1, color, stroke=18):
        x0, y0 = _polar(p0)
        x1, y1 = _polar(p1)
        return (
            f'<path d="M {x0:.2f} {y0:.2f} '
            f'A {R} {R} 0 0 1 {x1:.2f} {y1:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-linecap="round"/>'
        )

    bg_arc     = _arc(0.0, 1.0, "#DDE5EC", stroke=20)
    green_arc  = _arc(0.0, lo, "#2ECC71", stroke=20)
    yellow_arc = _arc(lo, hi, "#F1C40F", stroke=20)
    red_arc    = _arc(hi, 1.0, "#E74C3C", stroke=20)

    ticks = []
    for i in range(0, 11):
        p = i / 10
        x1, y1 = _polar(p, R + 14)
        x2, y2 = _polar(p, R - 12)
        ticks.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#6E7F91" stroke-width="2"/>'
        )
    ticks_svg = "".join(ticks)

    tx1, ty1 = _polar(thr, R + 14)
    tx2, ty2 = _polar(thr, R - 14)
    thr_tick  = (
        f'<line x1="{tx1:.2f}" y1="{ty1:.2f}" x2="{tx2:.2f}" y2="{ty2:.2f}" '
        f'stroke="#0F172A" stroke-width="3"/>'
    )

    nx, ny = _polar(prob, R - 14)
    needle  = (
        f'<line x1="{CX}" y1="{CY}" x2="{nx:.2f}" y2="{ny:.2f}" '
        f'stroke="#22324A" stroke-width="5" stroke-linecap="round"/>'
        f'<circle cx="{CX}" cy="{CY}" r="9" fill="#22324A"/>'
        f'<circle cx="{CX}" cy="{CY}" r="4" fill="#93A4B7"/>'
    )

    phys_needle = ""
    phys_text   = ""
    if phys_prob is not None and phys_thr is not None and phys_name:
        phys_prob = max(0.0, min(1.0, float(phys_prob)))
        px, py    = _polar(phys_prob, R - 20)
        phys_needle = (
            f'<line x1="{CX}" y1="{CY}" x2="{px:.2f}" y2="{py:.2f}" '
            f'stroke="#E87722" stroke-width="3" stroke-dasharray="5 4" '
            f'stroke-linecap="round"/>'
        )
        phys_text = (
            f'<text x="{CX}" y="178" text-anchor="middle" font-size="10" '
            f'fill="#E87722" font-weight="600">{phys_name}: p={phys_prob:.2f}, thr={phys_thr:.2f}</text>'
        )

    return f"""
<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:240px;display:block;margin:auto;">
    <rect x="1" y="1" width="{W-2}" height="{H-2}" rx="14" fill="white" stroke="#D6DCE2"/>
    <text x="{CX}" y="22" text-anchor="middle" font-size="17" fill="#00456A" font-weight="700">{target}</text>
    {bg_arc}
    {green_arc}
    {yellow_arc}
    {red_arc}
    {ticks_svg}
    {thr_tick}
    {phys_needle}
    {needle}
    <text x="24" y="128" font-size="11" fill="#708090">0</text>
    <text x="194" y="128" font-size="11" fill="#708090">1</text>
    <text x="{CX}" y="158" text-anchor="middle" font-size="18" fill="{call_col}" font-weight="800">{call}</text>
    <text x="{CX}" y="170" text-anchor="middle" font-size="11" fill="#5A6B7D">p={prob:.2f} · thr={thr:.2f}</text>
    {phys_text}
</svg>
"""


def _spread_close_positions(xs, min_gap=78, x_min=40, x_max=780):
    if not xs:
        return []
    adjusted    = xs[:]
    adjusted[0] = max(x_min, min(x_max, adjusted[0]))
    for i in range(1, len(adjusted)):
        adjusted[i] = max(adjusted[i], adjusted[i - 1] + min_gap)
    if adjusted[-1] > x_max:
        shift    = adjusted[-1] - x_max
        adjusted = [x - shift for x in adjusted]
        for i in range(len(adjusted) - 2, -1, -1):
            adjusted[i] = min(adjusted[i], adjusted[i + 1] - min_gap)
        if adjusted[0] < x_min:
            adjusted[0] = x_min
            for i in range(1, len(adjusted)):
                adjusted[i] = max(adjusted[i], adjusted[i - 1] + min_gap)
    return adjusted


def _threshold_bar_html(bar_marks: list) -> str:
    W, H        = 820, 42
    TOP_PAD     = 18
    BOTTOM_PAD  = 120
    MAX_D       = 0.45

    segs = [
        (0.00, 0.35, "#2ECC71"),
        (0.35, 0.50, "#A8D94B"),
        (0.50, 0.60, "#F1C40F"),
        (0.60, 0.75, "#E67E22"),
        (0.75, 1.00, "#E74C3C"),
    ]

    rects = "".join(
        f'<rect x="{int(x0 * W)}" y="{TOP_PAD}" width="{max(1, int((x1 - x0) * W))}" height="{H}" fill="{c}"/>'
        for x0, x1, c in segs
    )

    cx_thr = W // 2
    centre = (
        f'<line x1="{cx_thr}" y1="{TOP_PAD - 10}" x2="{cx_thr}" y2="{TOP_PAD + H + 12}" '
        f'stroke="#2C3E50" stroke-width="2" stroke-dasharray="5 4"/>'
        f'<text x="{cx_thr}" y="{TOP_PAD + H + 26}" text-anchor="middle" '
        f'font-size="12" fill="#2C3E50" font-weight="700">threshold</text>'
    )

    mark_colours = ["#00456A", "#006699", "#0091B3", "#E87722", "#8B4513", "#721C24", "#7C3AED"]

    items = []
    for i, m in enumerate(bar_marks):
        rel = 0.5 + (m["prob"] - m["thr"]) / (2 * MAX_D)
        rel = max(0.02, min(0.98, rel))
        bx  = int(rel * W)
        items.append({
            "orig_x": bx,
            "label":  m["label"],
            "prob":   m["prob"],
            "thr":    m["thr"],
            "color":  mark_colours[i % len(mark_colours)],
        })

    items    = sorted(items, key=lambda z: z["orig_x"])
    spread_xs = _spread_close_positions([it["orig_x"] for it in items], min_gap=95, x_min=36, x_max=W-36)

    row_pattern = [0, 1, 2, 1, 0, 2, 1]
    marks_svg   = ""

    for i, (it, tx) in enumerate(zip(items, spread_xs)):
        bx    = it["orig_x"]
        col   = it["color"]
        row   = row_pattern[i % len(row_pattern)]
        label_y = TOP_PAD + H + 48 + row * 22
        prob_y  = label_y + 14
        marks_svg += (
            f'<line x1="{bx}" y1="{TOP_PAD - 6}" x2="{bx}" y2="{TOP_PAD + H + 6}" '
            f'stroke="{col}" stroke-width="2.5"/>'
            f'<circle cx="{bx}" cy="{TOP_PAD + H/2}" r="8" fill="{col}" stroke="white" stroke-width="2.5"/>'
            f'<line x1="{bx}" y1="{TOP_PAD + H + 8}" x2="{tx}" y2="{label_y - 12}" '
            f'stroke="{col}" stroke-width="1.5" opacity="0.8"/>'
            f'<text x="{tx}" y="{label_y}" text-anchor="middle" font-size="11.5" '
            f'fill="{col}" font-weight="700">{it["label"]}</text>'
            f'<text x="{tx}" y="{prob_y}" text-anchor="middle" font-size="10" '
            f'fill="{col}">p={it["prob"]:.2f} · thr={it["thr"]:.2f}</text>'
        )

    return f'''
<svg viewBox="-12 0 {W + 24} {TOP_PAD + H + BOTTOM_PAD}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:860px;display:block;overflow:visible;">
    <defs>
        <clipPath id="barclip">
            <rect x="0" y="{TOP_PAD}" width="{W}" height="{H}" rx="8"/>
        </clipPath>
    </defs>
    <g clip-path="url(#barclip)">{rects}</g>
    {centre}
    {marks_svg}
    <text x="0" y="{TOP_PAD + H + 96}" font-size="11" fill="#708090">← more normal</text>
    <text x="{W}" y="{TOP_PAD + H + 96}" text-anchor="end" font-size="11" fill="#708090">more abnormal →</text>
</svg>
'''


def _score_bars_html(rows: list[dict]) -> str:
    """
    Horizontal 0–100 score bars.

    Score = sigmoid(raw_log_odds) × 100.  50 = threshold boundary.
    Bar fills left→right; left half green zone, right half red zone.

    Each row dict:  physician, label_mode, target, raw_score, probability, threshold, call
    """
    import math as _math

    BAR_W   = 520          # total bar track width (0–100 maps here)
    ROW_H   = 36
    L_PAD   = 200          # label column
    R_PAD   = 58           # score number + badge
    TOTAL_W = L_PAD + BAR_W + R_PAD
    TOP_PAD = 16
    BAR_H   = 13
    SEP_H   = 10

    def _sigmoid100(logit: float) -> float:
        return 100.0 / (1.0 + _math.exp(-logit))

    def _bar_color(s100: float) -> str:
        """Colour based on distance from 50."""
        d = s100 - 50
        if d >= 30:  return "#A93226"   # very abnormal
        if d >= 10:  return "#E74C3C"   # abnormal
        if d >= 0:   return "#F1948A"   # just over
        if d >= -10: return "#82E0AA"   # just under
        if d >= -30: return "#2ECC71"   # normal
        return "#1A7A40"                # very normal

    def _num_color(s100: float) -> str:
        d = s100 - 50
        if d > 5:   return "#A93226"
        if d < -5:  return "#1A7A40"
        return "#9A6B00"

    call_badge = {
        "Normal":     ("#D4EDDA", "#155724"),
        "Borderline": ("#FFF3CD", "#856404"),
        "Abnormal":   ("#F8D7DA", "#721C24"),
    }

    # Group + sort
    gen_rows  = sorted([r for r in rows if r["physician"] == "general"],  key=lambda r: r["target"])
    phys_rows = sorted([r for r in rows if r["physician"] != "general"],  key=lambda r: (r["physician"], r["target"]))
    all_rows  = gen_rows + phys_rows
    n         = len(all_rows)
    has_sep   = bool(gen_rows and phys_rows)
    SVG_H     = TOP_PAD + n * ROW_H + (SEP_H if has_sep else 0) + 24

    def _px(s100: float) -> float:
        """Map 0–100 score to bar x pixel."""
        return L_PAD + (s100 / 100.0) * BAR_W

    lines = []

    # Axis tick lines and labels: 0, 25, 50, 75, 100
    for v in [0, 25, 50, 75, 100]:
        tx   = _px(v)
        is50 = (v == 50)
        col  = "#6E7F91" if is50 else "#C8D4DC"
        sw   = "1.5"     if is50 else "0.5"
        dash = "none"    if is50 else "3 3"
        lines.append(
            f'<line x1="{tx:.1f}" y1="{TOP_PAD - 10}" x2="{tx:.1f}" y2="{SVG_H - 14}" '
            f'stroke="{col}" stroke-width="{sw}" stroke-dasharray="{dash}"/>'
        )
        label = "threshold" if is50 else str(v)
        lines.append(
            f'<text x="{tx:.1f}" y="{SVG_H - 2}" text-anchor="middle" '
            f'font-size="9" fill="#8B9DB3">{label}</text>'
        )

    # Direction labels
    lines.append(
        f'<text x="{_px(12):.1f}" y="{TOP_PAD - 2}" text-anchor="middle" '
        f'font-size="9" fill="#2ECC71">← normal</text>'
    )
    lines.append(
        f'<text x="{_px(88):.1f}" y="{TOP_PAD - 2}" text-anchor="middle" '
        f'font-size="9" fill="#E74C3C">abnormal →</text>'
    )

    row_offset = 0
    for i, r in enumerate(all_rows):
        # separator between general and physician blocks
        if has_sep and i == len(gen_rows):
            row_offset += SEP_H
            sep_y = TOP_PAD + i * ROW_H + row_offset - SEP_H // 2
            lines.append(
                f'<line x1="{L_PAD - 10}" y1="{sep_y}" x2="{TOTAL_W - 4}" y2="{sep_y}" '
                f'stroke="#D6DCE2" stroke-width="0.5" stroke-dasharray="4 3"/>'
            )

        y     = TOP_PAD + i * ROW_H + row_offset
        mid_y = y + ROW_H / 2

        # zebra
        if i % 2 == 0:
            lines.append(
                f'<rect x="0" y="{y:.1f}" width="{TOTAL_W}" height="{ROW_H}" fill="#F7FAFC"/>'
            )

        # labels
        is_general = r["physician"] == "general"
        phys_label = "general" if is_general else r["physician"]
        tgt_short  = (r["target"]
                      .replace("Focal ", "F·")
                      .replace("Gen ",   "G·")
                      .replace(" Non-epi", " Nepi"))
        phys_col = "#00456A" if is_general else "#E87722"
        lines.append(
            f'<text x="{L_PAD - 8}" y="{mid_y - 5:.1f}" text-anchor="end" '
            f'font-size="11" fill="{phys_col}" font-weight="600" dominant-baseline="central">'
            f'{phys_label}</text>'
        )
        lines.append(
            f'<text x="{L_PAD - 8}" y="{mid_y + 8:.1f}" text-anchor="end" '
            f'font-size="10" fill="#5A6B7D" dominant-baseline="central">'
            f'{tgt_short}</text>'
        )

        # compute score
        s100    = _sigmoid100(r["raw_score"])
        bar_px  = _px(s100)
        bar_y   = mid_y - BAR_H / 2

        # green half of track (0–50)
        lines.append(
            f'<rect x="{L_PAD:.1f}" y="{bar_y:.1f}" width="{BAR_W/2:.1f}" '
            f'height="{BAR_H}" fill="#E8F5E9" rx="3"/>'
        )
        # red half of track (50–100)
        lines.append(
            f'<rect x="{_px(50):.1f}" y="{bar_y:.1f}" width="{BAR_W/2:.1f}" '
            f'height="{BAR_H}" fill="#FDECEA" rx="3"/>'
        )

        # filled bar from left edge to score position
        fill_w  = max(bar_px - L_PAD, 1.0)
        bar_col = _bar_color(s100)
        lines.append(
            f'<rect x="{L_PAD:.1f}" y="{bar_y:.1f}" width="{fill_w:.1f}" '
            f'height="{BAR_H}" fill="{bar_col}" rx="3" opacity="0.82"/>'
        )

        # marker dot at exact score position
        lines.append(
            f'<circle cx="{bar_px:.1f}" cy="{mid_y:.1f}" r="5" '
            f'fill="{bar_col}" stroke="white" stroke-width="1.5"/>'
        )

        # score number
        num_col = _num_color(s100)
        lines.append(
            f'<text x="{L_PAD + BAR_W + 7}" y="{mid_y + 1:.1f}" '
            f'font-size="12" fill="{num_col}" font-weight="700" dominant-baseline="central" '
            f'font-family="monospace">{s100:.0f}</text>'
        )

        # call badge
        c_bg, c_fg = call_badge.get(r["call"], ("#F0F0F0", "#333"))
        bx = TOTAL_W - 13
        lines.append(
            f'<rect x="{bx - 9}" y="{mid_y - 7:.1f}" width="18" height="14" '
            f'fill="{c_bg}" rx="3"/>'
        )
        lines.append(
            f'<text x="{bx:.1f}" y="{mid_y + 1:.1f}" text-anchor="middle" '
            f'font-size="9" fill="{c_fg}" font-weight="700" dominant-baseline="central">'
            f'{r["call"][0]}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 {TOTAL_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{TOTAL_W}px;display:block;">'
        + "".join(lines) +
        '</svg>'
    )

    caption = (
        '<p style="color:#8B9DB3;font-size:0.7rem;margin:5px 0 0 0;">'
        'Score = sigmoid(log-odds) × 100. &nbsp;'
        '0 = maximally normal · 50 = threshold · 100 = maximally abnormal. &nbsp;'
        '<strong style="color:#155724;">N</strong> Normal · '
        '<strong style="color:#856404;">B</strong> Borderline · '
        '<strong style="color:#721C24;">A</strong> Abnormal'
        '</p>'
    )

    return (
        '<div style="background:white;border-radius:10px;border:1px solid #D6DCE2;'
        'padding:10px 14px 8px;">'
        + svg + caption +
        '</div>'
    )


def _shap_waterfall_html(feature_names, shap_values, base_value, final_value,
                          n_display: int = 15) -> str:
    """
    Pure-SVG SHAP waterfall for a single sample.

    feature_names : list[str]
    shap_values   : np.ndarray of float, same length
    base_value    : float  (expected model output / base log-odds)
    final_value   : float  (raw score = base + sum(shap))
    """
    vals  = np.asarray(shap_values, dtype=float)
    names = list(feature_names)

    # Sort by |SHAP| descending, take top n_display
    order = np.argsort(np.abs(vals))[::-1][:n_display]
    vals  = vals[order]
    names = [names[i] for i in order]

    # Sort ascending so largest bar is at top
    order2 = np.argsort(vals)
    vals   = vals[order2]
    names  = [names[i] for i in order2]

    BAR_H   = 22
    GAP     = 4
    L_PAD   = 260      # left margin for feature name
    R_PAD   = 90       # right margin for value label
    BAR_W   = 380      # total width of the bar region
    TOP_PAD = 40
    BOT_PAD = 50
    n       = len(vals)
    SVG_H   = TOP_PAD + n * (BAR_H + GAP) + BOT_PAD
    SVG_W   = L_PAD + BAR_W + R_PAD

    max_abs = max(np.abs(vals).max(), 1e-6)
    scale   = (BAR_W * 0.45) / max_abs   # half the bar width for max value

    # Centre line at L_PAD + BAR_W/2
    cx = L_PAD + BAR_W / 2

    bars_svg  = ""
    texts_svg = ""

    running = base_value
    for i, (v, name) in enumerate(zip(vals, names)):
        y      = TOP_PAD + i * (BAR_H + GAP)
        bar_x  = cx if v >= 0 else cx + v * scale
        bar_w  = abs(v) * scale
        color  = "#E74C3C" if v >= 0 else "#2ECC71"
        # feature name
        texts_svg += (
            f'<text x="{L_PAD - 8}" y="{y + BAR_H/2 + 4:.0f}" '
            f'text-anchor="end" font-size="11" fill="#1B2A3D">'
            f'{name[:40]}</text>'
        )
        # bar
        bars_svg += (
            f'<rect x="{bar_x:.1f}" y="{y}" width="{max(bar_w,1):.1f}" height="{BAR_H}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
        )
        # value label
        label_x = bar_x + bar_w + 5 if v >= 0 else bar_x - 5
        anchor  = "start" if v >= 0 else "end"
        texts_svg += (
            f'<text x="{label_x:.1f}" y="{y + BAR_H/2 + 4:.0f}" '
            f'text-anchor="{anchor}" font-size="10" fill="{color}" font-weight="700">'
            f'{v:+.3f}</text>'
        )

    # Centre line
    line_h  = TOP_PAD - 10 + n * (BAR_H + GAP) + BOT_PAD
    centre_line = (
        f'<line x1="{cx:.1f}" y1="{TOP_PAD - 15}" x2="{cx:.1f}" y2="{line_h}" '
        f'stroke="#9AA5B1" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<text x="{cx:.1f}" y="{TOP_PAD - 20}" text-anchor="middle" '
        f'font-size="10" fill="#5A6B7D">E[f(x)]={base_value:+.3f}</text>'
    )

    # Final score annotation
    final_x   = cx + final_value * scale
    final_x   = max(L_PAD + 4, min(SVG_W - R_PAD - 4, final_x))
    final_ann = (
        f'<line x1="{final_x:.1f}" y1="{TOP_PAD - 10}" x2="{final_x:.1f}" y2="{line_h - 30}" '
        f'stroke="#00456A" stroke-width="2" stroke-dasharray="3 2"/>'
        f'<text x="{final_x:.1f}" y="{line_h - 15}" text-anchor="middle" '
        f'font-size="10" fill="#00456A" font-weight="700">f(x)={final_value:+.3f}</text>'
    )

    return (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{SVG_W}px;display:block;">'
        f'<rect width="{SVG_W}" height="{SVG_H}" fill="white" rx="10"/>'
        f'<text x="{SVG_W//2}" y="22" text-anchor="middle" font-size="13" '
        f'fill="#00456A" font-weight="700">SHAP Waterfall — feature contributions to score</text>'
        f'{centre_line}'
        f'{final_ann}'
        f'{bars_svg}'
        f'{texts_svg}'
        f'</svg>'
    )


def _shap_bar_html(feature_names, mean_abs_shap, n_display: int = 20) -> str:
    """Horizontal bar chart of mean |SHAP| values."""
    vals  = np.asarray(mean_abs_shap, dtype=float)
    names = list(feature_names)

    order = np.argsort(vals)[-n_display:]   # bottom n → ascending so top is at top in SVG
    vals  = vals[order]
    names = [names[i] for i in order]

    BAR_H   = 20
    GAP     = 4
    L_PAD   = 260
    R_PAD   = 70
    BAR_W   = 350
    TOP_PAD = 40
    BOT_PAD = 30
    n       = len(vals)
    SVG_H   = TOP_PAD + n * (BAR_H + GAP) + BOT_PAD
    SVG_W   = L_PAD + BAR_W + R_PAD

    scale = BAR_W * 0.9 / max(vals.max(), 1e-6)

    bars_svg  = ""
    texts_svg = ""
    for i, (v, name) in enumerate(zip(vals, names)):
        y = TOP_PAD + (n - 1 - i) * (BAR_H + GAP)   # flip so highest is at top
        texts_svg += (
            f'<text x="{L_PAD - 8}" y="{y + BAR_H/2 + 4:.0f}" '
            f'text-anchor="end" font-size="11" fill="#1B2A3D">{name[:40]}</text>'
        )
        bw = v * scale
        bars_svg += (
            f'<rect x="{L_PAD}" y="{y}" width="{bw:.1f}" height="{BAR_H}" '
            f'fill="#0091B3" rx="3" opacity="0.85"/>'
        )
        texts_svg += (
            f'<text x="{L_PAD + bw + 5:.1f}" y="{y + BAR_H/2 + 4:.0f}" '
            f'font-size="10" fill="#006699" font-weight="600">{v:.4f}</text>'
        )

    return (
        f'<svg viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{SVG_W}px;display:block;">'
        f'<rect width="{SVG_W}" height="{SVG_H}" fill="white" rx="10"/>'
        f'<text x="{SVG_W//2}" y="22" text-anchor="middle" font-size="13" '
        f'fill="#00456A" font-weight="700">Mean |SHAP| — feature importance for this model</text>'
        f'{bars_svg}'
        f'{texts_svg}'
        f'</svg>'
    )


def _get_shap_for_new_vector(entry: dict, model_dir: Path, params_df: pd.DataFrame):
    """
    Run SHAP on the current input vector using the saved explainer.
    Returns (shap_values_1d, base_value, raw_score, feature_names) or None on failure.
    """
    try:
        import joblib, xgboost as xgb

        explainer_path = model_dir / "shap_explainer.pkl"
        model_path     = model_dir / "model.json"
        if not explainer_path.exists() or not model_path.exists():
            return None

        final_feats = entry["meta"].get("final_features", [])
        if not final_feats:
            return None

        # Build input vector — align to model feature list
        available = [f for f in final_feats if f in params_df.columns]
        if not available:
            return None

        X_new = params_df[available].values.astype(np.float32)
        # Pad missing features with 0
        if len(available) < len(final_feats):
            full = np.zeros((1, len(final_feats)), dtype=np.float32)
            idx  = [final_feats.index(f) for f in available]
            full[0, idx] = X_new[0]
            X_new = full

        # Raw score via XGBoost DMatrix
        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        dmat       = xgb.DMatrix(X_new, feature_names=final_feats)
        raw_score  = float(model.get_booster().predict(dmat, output_margin=True)[0])

        # SHAP values
        explainer  = joblib.load(str(explainer_path))
        shap_out   = explainer(X_new)
        shap_vals  = shap_out.values
        if shap_vals.ndim == 3:          # multi-output
            shap_vals = shap_vals[:, :, 1]
        shap_1d    = shap_vals[0]

        base_value = float(explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[-1])

        return shap_1d, base_value, raw_score, final_feats

    except Exception as e:
        return None


def _empty(icon, title):
    st.markdown(
        f'<div style="text-align:center;padding:4rem 2rem;">'
        f'<span style="font-size:4rem;opacity:0.3;">{icon}</span>'
        f'<h3 style="color:#5A6B7D;font-weight:400;margin-top:1rem;">{title}</h3>'
        f'<p style="color:#8B9DB3;font-size:0.95rem;">'
        f'Upload or select an EEG recording and run the pipeline to see results.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS  (3 tabs now)
# ══════════════════════════════════════════════════════════════════════════════
tab_pred, tab_explain, tab_model = st.tabs([
    "📊  Predictions",
    "🔍  Explanations",
    "🔬  Model Details",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    if st.session_state.pipeline_status != "done" or st.session_state.all_results is None:
        _empty("📊", "No predictions yet")
    else:
        ar = st.session_state.all_results

        st.markdown('<h3 style="color:#00456A;">Model Selection</h3>', unsafe_allow_html=True)
        sc1, sc2 = st.columns([3, 1])

        with sc1:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Group</div>', unsafe_allow_html=True)
            sel_phys = st.radio(
                "Group",
                ["general"] + ALL_PHYSICIANS,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "🌐 General" if x == "general" else f"👤 {x}",
                key="sel_phys",
            )

        with sc2:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Mode</div>', unsafe_allow_html=True)
            sel_mode = st.radio(
                "Mode",
                ["strict", "relaxed"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "🔒 Strict" if x == "strict" else "🔓 Relaxed",
                key="sel_mode",
            )

        if sel_phys != "general":
            st.caption(
                f"ℹ️ Per-physician models cover Abnormality only. "
                f"Dr. {sel_phys} · {sel_mode} shown as orange dashed needle on Abnormality gauge."
            )

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        gview = get_view(ar, "general", sel_mode)

        phys_result = None
        if sel_phys != "general":
            pe = ar.get(f"physician__{sel_phys}__{sel_mode}__Abnormality", {})
            if pe.get("available") and pe.get("result"):
                phys_result = pe["result"]

        # ── Abnormality Spectrum bar ──────────────────────────────────────────
        st.markdown('<h4 style="color:#00456A;">Abnormality Spectrum</h4>', unsafe_allow_html=True)

        bar_marks = []
        for tgt, ent in gview.items():
            if ent.get("available") and ent.get("result"):
                bar_marks.append({
                    "label": tgt,
                    "prob":  ent["result"]["probability"],
                    "thr":   ent["result"]["threshold"],
                })
        if phys_result:
            bar_marks.append({
                "label": f"{sel_phys} physician",
                "prob":  phys_result["probability"],
                "thr":   phys_result["threshold"],
            })

        if bar_marks:
            bar_svg  = _threshold_bar_html(bar_marks)
            bar_html = f"""<!DOCTYPE html><html>
<body style="margin:0;padding:0;background:transparent;">
<div style="background:white;border-radius:12px;padding:18px 22px 14px;border:1px solid #D6DCE2;">
<p style="color:#5A6B7D;font-size:13px;line-height:1.45;margin:0 0 12px 0;">
Each marker shows where a model places this recording relative to its own threshold.
The dashed centre line marks the threshold. Left of centre = more normal. Right = more abnormal.
</p>
{bar_svg}
</div></body></html>"""
            components.html(bar_html, height=300, scrolling=False)

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        # ── Gauge cards ───────────────────────────────────────────────────────
        st.markdown('<h2 style="color:#00456A;margin-bottom:0.6rem;">Predictions by Category</h2>', unsafe_allow_html=True)

        gauge_svgs = []
        for tgt in ALL_TARGETS:
            ent = gview[tgt]
            if not ent.get("available") or not ent.get("result"):
                gauge_svgs.append(
                    f'<div style="flex:1;min-width:190px;text-align:center;'
                    f'background:white;border-radius:12px;padding:1rem;border:1px solid #D6DCE2;">'
                    f'<p style="color:#00456A;font-weight:700;font-size:13px;">{tgt}</p>'
                    f'<p style="color:#8B9DB3;font-size:12px;">Not available</p></div>'
                )
            else:
                r   = ent["result"]
                pp  = phys_result["probability"] if (phys_result and tgt == "Abnormality") else None
                pt  = phys_result["threshold"]   if (phys_result and tgt == "Abnormality") else None
                pn  = sel_phys                   if (phys_result and tgt == "Abnormality") else None
                svg = _gauge_html(tgt, r["probability"], r["threshold"],
                                  phys_prob=pp, phys_thr=pt, phys_name=pn, band=0.06)
                gauge_svgs.append(f'<div style="flex:1;min-width:190px;">{svg}</div>')

        gauges_html = f"""<!DOCTYPE html><html>
<body style="margin:0;padding:0;background:transparent;font-family:'Source Sans Pro',sans-serif;">
<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:nowrap;">
{''.join(gauge_svgs)}
</div>
<p style="color:#8B9DB3;font-size:11px;margin:10px 0 0 0;text-align:center;">
🟢 Normal zone &nbsp;·&nbsp; 🟡 Borderline band around threshold &nbsp;·&nbsp;
🔴 Abnormal zone &nbsp;·&nbsp;
<span style="color:#2C3E50;">solid needle = general model</span> &nbsp;·&nbsp;
<span style="color:#E87722;">dashed needle = physician model</span>
</p>
</body></html>"""
        components.html(gauges_html, height=275, scrolling=False)

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        # ── Raw Score Bars (below gauges, collapsed by default) ──────────────

        score_rows = []
        for mid_key, ent in ar.items():
            if not ent.get("available") or not ent.get("result"):
                continue
            r    = ent["result"]
            meta = ent.get("meta", {})
            phys = meta.get("physician", mid_key.split("__")[0])
            mode = meta.get("label_mode", "")
            tgt  = meta.get("target", "")
            # filter by the mode radio already selected by user
            if mode != sel_mode:
                continue
            prob   = float(r["probability"])
            thr    = float(r["threshold"])
            raw_sc = float(r.get("raw_score", math.log(max(prob, 1e-7) / max(1 - prob, 1e-7))))
            call, _ = _classify_call(prob, thr, band=0.06)
            score_rows.append({
                "physician":   phys,
                "label_mode":  mode,
                "target":      tgt,
                "raw_score":   raw_sc,
                "probability": prob,
                "threshold":   thr,
                "call":        call,
            })

        if score_rows:
            with st.expander(f"📊 Model Scores — {sel_mode} mode ({len(score_rows)} models)", expanded=False):
                bars_html = _score_bars_html(score_rows)
                components.html(
                    f"<!DOCTYPE html><html>"
                    f"<body style='margin:0;padding:0;background:transparent;'>"
                    f"{bars_html}"
                    f"</body></html>",
                    height=60 + len(score_rows) * 36 + 36,
                    scrolling=False,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — EXPLANATIONS  (new tab)
# ══════════════════════════════════════════════════════════════════════════════
with tab_explain:
    if st.session_state.pipeline_status != "done" or st.session_state.all_results is None:
        _empty("🔍", "No explanations yet")
    else:
        ar         = st.session_state.all_results
        params_df  = st.session_state.params_df

        st.markdown('<h3 style="color:#00456A;">Per-Recording Explanations</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#5A6B7D;font-size:0.9rem;margin-top:-0.5rem;">'
            'SHAP values computed live on <em>this specific recording</em> — '
            'showing exactly which features pushed the model toward or away from abnormality.</p>',
            unsafe_allow_html=True,
        )

        # Model selector
        ec1, ec2, ec3 = st.columns([2, 1, 1])
        with ec1:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Group</div>', unsafe_allow_html=True)
            ex_phys = st.radio(
                "Group",
                ["general"] + ALL_PHYSICIANS,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "🌐 General" if x == "general" else f"👤 {x}",
                key="ex_phys",
            )
        with ec2:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Mode</div>', unsafe_allow_html=True)
            ex_mode = st.radio(
                "Mode",
                ["strict", "relaxed"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "🔒 Strict" if x == "strict" else "🔓 Relaxed",
                key="ex_mode",
            )
        with ec3:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Target</div>', unsafe_allow_html=True)
            ex_tgt_choices = ALL_TARGETS if ex_phys == "general" else ["Abnormality"]
            ex_tgt = st.selectbox("Target", ex_tgt_choices, label_visibility="collapsed", key="ex_tgt")

        ex_mid   = _model_id(ex_phys, ex_mode, ex_tgt)
        ex_entry = ar.get(ex_mid, {"available": False})
        ex_dir   = Path("model_outputs") / ex_mid

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        if not ex_entry.get("available") or not ex_entry.get("result"):
            st.warning(f"Model **{ex_mid}** not available.")
        else:
            ex_result = ex_entry["result"]
            ex_meta   = ex_entry["meta"]
            ex_cv     = ex_meta.get("cv_metrics", {})
            prob      = float(ex_result["probability"])
            thr       = float(ex_result["threshold"])
            call, _   = _classify_call(prob, thr, band=0.06)

            # ── Summary card ─────────────────────────────────────────────────
            xc1, xc2, xc3, xc4 = st.columns(4)
            raw_sc = float(ex_result.get(
                "raw_score",
                math.log(max(prob, 1e-7) / max(1 - prob, 1e-7))
            ))

            def _xmc(col, lbl, val, sub="", val_color="#00456A"):
                col.markdown(
                    f'<div class="metric-card">'
                    f'<p style="color:#5A6B7D;font-size:0.75rem;margin:0 0 0.2rem 0;">{lbl}</p>'
                    f'<p style="color:{val_color};font-size:1.4rem;font-weight:800;margin:0;">{val}</p>'
                    f'<p style="color:#8B9DB3;font-size:0.72rem;margin:0;">{sub}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            call_color = {"Normal": "#156A3A", "Borderline": "#9A6B00", "Abnormal": "#B42318"}.get(call, "#00456A")
            _xmc(xc1, "Prediction", call, f"p={prob:.3f} · thr={thr:.3f}", val_color=call_color)

            sc_color = "#B42318" if raw_sc > 0.2 else ("#9A6B00" if raw_sc > -0.2 else "#156A3A")
            _xmc(xc2, "Raw Score (log-odds)", f"{raw_sc:+.3f}",
                 "sigmoid → probability", val_color=sc_color)
            _xmc(xc3, "CV AUPRC", f"{ex_cv.get('auprc_mean', 0):.3f}",
                 f"±{ex_cv.get('auprc_std', 0):.3f}")
            _xmc(xc4, "Features Used", str(ex_meta.get("n_features_final", "—")),
                 "after ReliefF consensus")

            # Interpretation sentence
            direction = "toward abnormal" if raw_sc > 0 else "toward normal"
            confidence = "high confidence" if abs(raw_sc) > 1.5 else \
                         "moderate confidence" if abs(raw_sc) > 0.5 else "near the decision boundary"
            st.markdown(
                f'<div style="background:#EBF5FB;border-left:4px solid #0091B3;'
                f'border-radius:6px;padding:0.8rem 1rem;margin:1rem 0;">'
                f'<p style="margin:0;color:#1B2A3D;font-size:0.9rem;">'
                f'<strong>Score interpretation:</strong> Raw score of <strong>{raw_sc:+.3f}</strong> '
                f'is <strong>{direction}</strong> with <strong>{confidence}</strong>. '
                f'The threshold for this model is <strong>{thr:.3f}</strong> (probability), '
                f'equivalent to a log-odds of <strong>{math.log(thr/(1-thr+1e-9)):.3f}</strong>.'
                f'</p></div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

            # ── SHAP section ──────────────────────────────────────────────────
            st.markdown('<h4 style="color:#00456A;">SHAP Feature Contributions — This Recording</h4>', unsafe_allow_html=True)

            with st.spinner("Computing SHAP values for this recording…"):
                shap_result = _get_shap_for_new_vector(ex_entry, ex_dir, params_df)

            if shap_result is None:
                st.warning(
                    "SHAP explainer not found for this model. "
                    "Ensure `shap_explainer.pkl` exists in the model output directory."
                )
            else:
                shap_1d, base_val, shap_raw_score, feat_names = shap_result

                # Waterfall
                n_disp = st.slider("Features to display in waterfall", 5, min(30, len(feat_names)), 15,
                                   key="ex_ndisp")
                wf_svg  = _shap_waterfall_html(feat_names, shap_1d, base_val, shap_raw_score, n_display=n_disp)
                components.html(
                    f"<!DOCTYPE html><html><body style='margin:0;padding:4px;background:transparent;'>"
                    f"{wf_svg}</body></html>",
                    height=60 + n_disp * 28,
                    scrolling=False,
                )

                st.caption(
                    "🔵 Green bars push toward **normal** (negative SHAP). "
                    "🔴 Red bars push toward **abnormal** (positive SHAP). "
                    "E[f(x)] = expected model output over training data. "
                    "f(x) = actual log-odds for this recording."
                )

                st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

                # Top contributors table
                st.markdown('<h4 style="color:#00456A;">Top Feature Contributions — Ranked</h4>', unsafe_allow_html=True)
                order_table = np.argsort(np.abs(shap_1d))[::-1][:30]
                rows_table  = []
                for idx in order_table:
                    v    = float(shap_1d[idx])
                    rows_table.append({
                        "Feature": feat_names[idx],
                        "SHAP Value": f"{v:+.4f}",
                        "Direction": "▲ abnormal" if v > 0 else "▼ normal",
                        "|SHAP|": round(abs(v), 4),
                    })
                shap_table_df = pd.DataFrame(rows_table)
                st.dataframe(
                    shap_table_df.style
                        .applymap(
                            lambda v: "color:#B42318;font-weight:700" if "▲" in str(v)
                                      else ("color:#156A3A;font-weight:700" if "▼" in str(v) else ""),
                            subset=["Direction"],
                        ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

                # Model-level mean |SHAP| bar (from saved CSV, for context)
                shap_csv = ex_dir / "shap_feature_importance.csv"
                if shap_csv.exists():
                    st.markdown(
                        '<h4 style="color:#00456A;">Model-Level Feature Importance (mean |SHAP| over training)</h4>',
                        unsafe_allow_html=True,
                    )
                    shap_df = pd.read_csv(shap_csv)
                    bar_svg = _shap_bar_html(
                        shap_df["feature"].tolist(),
                        shap_df["mean_abs_shap"].tolist(),
                        n_display=20,
                    )
                    components.html(
                        f"<!DOCTYPE html><html><body style='margin:0;padding:4px;background:transparent;'>"
                        f"{bar_svg}</body></html>",
                        height=60 + min(20, len(shap_df)) * 26,
                        scrolling=False,
                    )
                    st.caption(
                        "This shows which features are *generally* most predictive across training data. "
                        "Compare against the waterfall above to see if this recording follows the typical pattern."
                    )

            st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

            # ── Input vector snapshot ─────────────────────────────────────────
            st.markdown('<h4 style="color:#00456A;">Input Feature Vector — This Recording</h4>', unsafe_allow_html=True)
            final_feats = ex_meta.get("final_features", [])
            avail_feats = [f for f in final_feats if f in params_df.columns]

            if avail_feats:
                snap_df = params_df[avail_feats].T.reset_index()
                snap_df.columns = ["Feature", "Value"]
                snap_df["Value"] = snap_df["Value"].round(5)

                if shap_result is not None:
                    shap_map = {feat_names[i]: float(shap_1d[i]) for i in range(len(feat_names))}
                    snap_df["SHAP"] = snap_df["Feature"].map(shap_map).round(4)
                    snap_df["|SHAP|"] = snap_df["SHAP"].abs()
                    snap_df = snap_df.sort_values("|SHAP|", ascending=False).drop(columns=["|SHAP|"])

                with st.expander(f"📋 {len(snap_df)} model features for this recording", expanded=False):
                    st.dataframe(snap_df, use_container_width=True, hide_index=True)
            else:
                st.info("Feature vector not available (params_df columns don't overlap model features).")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    if st.session_state.pipeline_status != "done" or st.session_state.all_results is None:
        _empty("🔬", "No model details yet")
    else:
        ar = st.session_state.all_results

        st.markdown('<h3 style="color:#00456A;">Select Model to Inspect</h3>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns([2, 1, 1])

        with mc1:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Group</div>', unsafe_allow_html=True)
            md_phys = st.radio(
                "Group",
                ["general"] + ALL_PHYSICIANS,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "🌐 General" if x == "general" else f"👤 {x}",
                key="md_phys",
            )

        with mc2:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Mode</div>', unsafe_allow_html=True)
            md_mode = st.radio(
                "Mode",
                ["strict", "relaxed"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "🔒 Strict" if x == "strict" else "🔓 Relaxed",
                key="md_mode",
            )

        with mc3:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.2rem;">Target</div>', unsafe_allow_html=True)
            tgt_choices = ALL_TARGETS if md_phys == "general" else ["Abnormality"]
            md_tgt = st.selectbox("Target", tgt_choices, label_visibility="collapsed", key="md_tgt")

        mid = _model_id(md_phys, md_mode, md_tgt)
        entry = ar.get(mid, {"available": False})
        model_dir = Path("model_outputs") / mid

        if not entry.get("available") or not entry.get("result"):
            st.warning(f"Model {mid} was not found or failed to run.")
        else:
            meta   = entry["meta"]
            result = entry["result"]
            cv     = meta.get("cv_metrics", {})

            st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

            prob  = result["probability"]
            thr   = result["threshold"]
            call, _ = _classify_call(prob, thr, band=0.06)

            m1, m2, m3, m4, m5 = st.columns(5)

            def _mc(col, lbl, val, sub=""):
                col.markdown(
                    f'<div class="metric-card">'
                    f'<p style="color:#5A6B7D;font-size:0.75rem;margin:0 0 0.2rem 0;">{lbl}</p>'
                    f'<p style="color:#00456A;font-size:1.4rem;font-weight:800;margin:0;">{val}</p>'
                    f'<p style="color:#8B9DB3;font-size:0.72rem;margin:0;">{sub}</p>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            _mc(m1, "This Recording", call, f"p={prob:.2f} vs thr={thr:.2f}")
            _mc(m2, "Probability",    f"{prob:.2f}", f"threshold {thr:.2f}")
            _mc(m3, "CV AUPRC",       f"{cv.get('auprc_mean', 0):.3f}", f"±{cv.get('auprc_std', 0):.3f}")
            _mc(m4, "CV AUROC",       f"{cv.get('auroc_mean', 0):.3f}", f"±{cv.get('auroc_std', 0):.3f}")
            _mc(m5, "CV F1",          f"{cv.get('f1_mean', 0):.3f}",
                f"P {cv.get('precision_mean', 0):.2f} · R {cv.get('recall_mean', 0):.2f}")

            st.markdown("")
            plot_dir = model_dir / "plots"

            def _plot_or_placeholder(path, caption):
                if path.exists():
                    st.image(str(path), caption=caption, use_container_width=True)
                else:
                    st.markdown(
                        f'<div style="background:#F0F2F4;border-radius:8px;padding:1rem;'
                        f'text-align:center;color:#8B9DB3;font-size:0.8rem;">'
                        f'{caption}<br><small>not generated</small></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("#### 📈 Model Performance")
            perf = [
                ("pr_curve.png",         "Precision-Recall Curve"),
                ("roc_curve.png",        "ROC Curve"),
                ("fold_stability.png",   "Fold Stability"),
                ("prob_distribution.png","Probability Distribution"),
                ("confusion_matrix.png", "Confusion Matrix"),
            ]
            pc = st.columns(3)
            for i, (f, t) in enumerate(perf):
                with pc[i % 3]:
                    _plot_or_placeholder(plot_dir / f, t)

            st.markdown("#### 🧬 Feature Importance & Explainability")
            shap_plots = [
                ("shap_beeswarm.png",    "SHAP Beeswarm (top 20)"),
                ("shap_bar.png",         "SHAP Bar Chart"),
                ("native_importance.png","XGBoost Gain Importance"),
            ]
            sc = st.columns(3)
            for i, (f, t) in enumerate(shap_plots):
                with sc[i]:
                    _plot_or_placeholder(plot_dir / f, t)

            st.markdown("#### 🌊 SHAP Waterfall (Training Samples)")
            wc = st.columns(3)
            for i in range(3):
                with wc[i]:
                    _plot_or_placeholder(plot_dir / f"shap_waterfall_s{i}.png", f"Training sample {i}")
            st.caption(
                "⚠️ These waterfalls are from training samples. "
                "For the waterfall of **this specific recording**, see the Explanations tab."
            )

            st.markdown("#### 📉 Partial Dependence — Top 5 Features")
            _plot_or_placeholder(plot_dir / "pdp_top5.png", "PDP top 5 features")

            shap_csv = model_dir / "shap_feature_importance.csv"
            if shap_csv.exists():
                with st.expander("🔢 Top 20 Features by mean |SHAP|", expanded=False):
                    st.dataframe(pd.read_csv(shap_csv).head(20), use_container_width=True, hide_index=True)

            cv_csv = model_dir / "cv_results.csv"
            if cv_csv.exists():
                with st.expander("📋 Cross-Validation Results per Fold", expanded=False):
                    st.dataframe(pd.read_csv(cv_csv), use_container_width=True, hide_index=True)

            report = model_dir / "evaluation_report.txt"
            if report.exists():
                with st.expander("📝 Full Evaluation Report", expanded=False):
                    st.code(report.read_text(), language=None)


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer-authors">
    <div class="author-line">
        Iryna Gunko ¹ &nbsp;·&nbsp; Vasily A. Vakorin ¹ ² * &nbsp;·&nbsp;
        Alexander Moiseev ¹ &nbsp;·&nbsp; Sam M. Doesburg ¹ &nbsp;·&nbsp; George Medvedev ²
    </div>
    <div>
        ¹ Dept. Biomedical Physiology &amp; Kinesiology, Simon Fraser University, Burnaby, Canada &nbsp;&nbsp;
        ² Royal Columbian Hospital, Fraser Health Authority, New Westminster, Canada
    </div>
    <div style="margin-top:0.3rem;">
        * Correspondence:
        <a href="mailto:iryna_gunko@sfu.ca">iryna_gunko@sfu.ca</a> &nbsp;·&nbsp;
        <a href="mailto:vasily_vakorin@sfu.ca">vasily_vakorin@sfu.ca</a>
    </div>
    <div style="margin-top:0.5rem;color:#A0AEC0;font-size:0.72rem;">
        © 2026 Fraser Health Authority × Simon Fraser University &nbsp;|&nbsp; Research Use Only
    </div>
</div>
""", unsafe_allow_html=True)