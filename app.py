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
        margin:0.4rem 0;
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

    /* ── Clinical explainer panels ── */
    .clinical-explainer {
        background: #F0F7FF;
        border-left: 3px solid #0091B3;
        border-radius: 0 6px 6px 0;
        padding: 0.5rem 0.8rem;
        margin: 0.15rem 0 0.25rem 0;
        font-size: 0.66rem;
        color: #1B2A3D !important;
        line-height: 1.5;
    }

    /* ── Expander header ── */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary * {
        color: #00456A !important;
        fill: #00456A !important;
    }

    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] summary span,
    div[data-testid="stExpander"] summary div,
    details[data-testid="stExpander"] summary p,
    details[data-testid="stExpander"] summary span,
    details[data-testid="stExpander"] summary div {
        color: #00456A !important;
        margin: 0 !important;
    }

    div[data-testid="stExpander"] summary,
    details[data-testid="stExpander"] summary {
        background: #F0F7FF !important;
        border-radius: 5px !important;
        padding: 3px 8px !important;
        font-weight: 700 !important;
        font-size: 0.675rem !important;
    }

    div[data-testid="stExpander"] summary:hover,
    details[data-testid="stExpander"] summary:hover {
        color: #006699 !important;
    }

    div[data-testid="stExpander"] summary svg,
    details[data-testid="stExpander"] summary svg {
        color: #00456A !important;
        fill: #00456A !important;
    }

    /* shrink the expander's internal top/bottom gap */
    div[data-testid="stExpander"] > div[data-testid="stExpanderDetails"],
    details[data-testid="stExpander"] > div {
        padding-top: 0.25rem !important;
        padding-bottom: 0.1rem !important;
    }
    .clinical-explainer strong {
        color: #00456A;
    }

    .clinical-explainer .legend-row {
        display: flex;
        gap: 1.2rem;
        flex-wrap: wrap;
        margin-top: 0.5rem;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        font-size: 0.82rem;
    }

    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }

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

    /* ── Multiselect — light background, dark text ────────────────────── */
    /* Container / input box */
    [data-testid="stMultiSelect"] > div > div {
        background: #F0F7FF !important;
        border: 1px solid #B8CCD8 !important;
        border-radius: 8px !important;
    }
    /* Placeholder & typed text */
    [data-testid="stMultiSelect"] input {
        color: #1B2A3D !important;
    }
    [data-testid="stMultiSelect"] input::placeholder {
        color: #7A93A8 !important;
    }
    /* Selected tag chips */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
        background: #DCECF5 !important;
        border: 1px solid #8CB4C8 !important;
        border-radius: 20px !important;
    }
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] span {
        color: #00456A !important;
        font-weight: 600 !important;
    }
    /* X remove button on chips */
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] button svg {
        fill: #00456A !important;
    }
    /* Dropdown list */
    [data-baseweb="popover"] ul[role="listbox"] {
        background: #FFFFFF !important;
        border: 1px solid #B8CCD8 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="popover"] ul[role="listbox"] li {
        color: #1B2A3D !important;
        background: #FFFFFF !important;
    }
    [data-baseweb="popover"] ul[role="listbox"] li:hover,
    [data-baseweb="popover"] ul[role="listbox"] li[aria-selected="true"] {
        background: #DCECF5 !important;
        color: #00456A !important;
    }
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
        "label": "Definite Focal Epi (4-1-1-1-4)",
        "path": "hdf/4_1_1_1_4/575f3167-871c-4be8-a12f-48ceb16915ca-ico-4-ltc.hdf5",
        "description": "Definite focal epileptic features, definite overall abnormality",
        "scores": {"Abnormality": 4, "Focal Epi": 4, "Focal Non-epi": 1, "Gen Epi": 1, "Gen Non-epi": 1},
    }
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
    ("_csv_df", None),
    ("_csv_type", None),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

LOGO_PATH = "assets/fraser_health_logo.png"

with st.sidebar:

    st.markdown(
        '<div style="text-align:center;padding:0 0 0.2rem 0;">'
        '<p style="color:#00456A;margin:0;font-size:1rem;font-weight:700;letter-spacing:-0.3px;">'
        '🧠 EEG Abnormality Prediction Pipeline</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

    st.markdown("#### 📂 Data Source")
    source = st.radio(
        "src",
        ["Upload HDF5 file", "Upload CSV vector", "Use demo recording"],
        label_visibility="collapsed",
    )

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

    elif source == "Upload CSV vector":
        st.markdown(
            "<p style='color:#5A6B7D;font-size:0.78rem;margin:0 0 0.3rem 0;'>"
            "Upload a previously exported <strong>Params Vector</strong> or "
            "<strong>Feature CSV</strong> — skips HDF5 processing entirely.</p>",
            unsafe_allow_html=True,
        )
        uploaded_csv = st.file_uploader(
            "CSV",
            type=["csv"],
            help="Either the Params Vector (1-row wide CSV) or Feature CSV (parcel-level rows) "
                 "downloaded from a previous pipeline run.",
            key="csv_uploader",
        )
        if uploaded_csv is not None:
            try:
                _csv_df = pd.read_csv(uploaded_csv)
                # Detect type by column signature
                _is_feature_csv = all(c in _csv_df.columns for c in ("subject_id", "parcel", "network"))
                _is_params_vector = any(c.startswith(("net8_", "global_")) for c in _csv_df.columns)

                if _is_feature_csv:
                    _csv_type = "feature"
                elif _is_params_vector:
                    _csv_type = "params"
                else:
                    _csv_type = None

                if _csv_type is None:
                    st.markdown(
                        '<div style="background:#F8D7DA;color:#721C24;padding:0.4rem 0.8rem;'
                        'border-radius:6px;font-size:0.78rem;">⚠️ Unrecognised CSV format. '
                        'Please upload a Params Vector or Feature CSV from this pipeline.</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.session_state.selected_file = uploaded_csv.name
                    st.session_state.file_source = "csv"
                    st.session_state._csv_df   = _csv_df
                    st.session_state._csv_type = _csv_type
                    _label = "Params Vector" if _csv_type == "params" else "Feature CSV"
                    name_trunc = uploaded_csv.name[:38] + ("…" if len(uploaded_csv.name) > 38 else "")
                    st.markdown(
                        f'<div class="status-ready">✅ {_label} loaded: {name_trunc}</div>',
                        unsafe_allow_html=True,
                    )
            except Exception as _e:
                st.markdown(
                    f'<div style="background:#F8D7DA;color:#721C24;padding:0.4rem 0.8rem;'
                    f'border-radius:6px;font-size:0.78rem;">❌ Could not read CSV: {_e}</div>',
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
            if st.session_state.file_source == "csv":
                # ── CSV fast-path: skip HDF5 entirely ────────────────────────
                _csv_df   = st.session_state.get("_csv_df")
                _csv_type = st.session_state.get("_csv_type")
                if _csv_df is None:
                    raise ValueError("CSV data missing — please re-upload.")

                if _csv_type == "feature":
                    pbar.progress(0.5, text="Collapsing Feature CSV to model input vector…")
                    params_df = collapse_to_vector(_csv_df)
                    st.session_state.feature_df = _csv_df
                else:
                    # params vector — already collapsed, use directly
                    pbar.progress(0.5, text="Using Params Vector directly…")
                    params_df = _csv_df
                    st.session_state.feature_df = None   # no parcel-level data

                st.session_state.params_df = params_df
                pbar.progress(0.96, text="Running 20 models…")
                all_results = run_all_models(params_df)
                st.session_state.all_results = all_results
                st.session_state.pipeline_status = "done"

            else:
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
        n_m = sum(
            1 for v in (st.session_state.all_results or {}).values()
            if v.get("available") and v.get("result")
        )
        if df is not None and "label" in df.columns:
            n_p = df["label"].nunique()
            _done_label = f"✅ {n_p} parcels · {n_m}/20 models"
        elif st.session_state.get("_csv_type") == "params":
            _done_label = f"✅ Params Vector · {n_m}/20 models"
        else:
            _done_label = f"✅ {n_m}/20 models"
        st.markdown(
            f'<div class="status-ready">{_done_label}</div>',
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

def clinical_explainer(key: str, label: str, content_html: str):
    """
    Render a collapsible 'How to read this' panel.
    key    : unique Streamlit expander key
    label  : short title shown on the collapsed bar
    content_html : HTML string rendered inside .clinical-explainer div
    """
    with st.expander(f"ℹ️ How to read this — {label}", expanded=False):
        st.markdown(
            f'<div class="clinical-explainer">{content_html}</div>',
            unsafe_allow_html=True,
        )


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

    # Gradient arc: stops anchored to lo/hi (threshold band), so green/yellow/red
    # zones are proportional to the actual threshold, not fixed at 50%.
    # Within each zone the colour transitions smoothly.
    _GRAD_STEPS = 80
    def _lerp_color(t):
        # stops at 0, lo, lo, hi, hi, 1  — hard zone edges, smooth within
        G = (39,  174,  96)   # green
        Y = (249, 202,  36)   # yellow
        R = (231,  76,  60)   # red
        stops = [
            (0.0, G),
            (lo,  G),
            (lo,  Y),
            (hi,  Y),
            (hi,  R),
            (1.0, R),
        ]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t1 == t0:
                continue
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                rv = int(c0[0] + f * (c1[0] - c0[0]))
                gv = int(c0[1] + f * (c1[1] - c0[1]))
                bv = int(c0[2] + f * (c1[2] - c0[2]))
                return f"#{rv:02X}{gv:02X}{bv:02X}"
        r2, g2, b2 = R
        return f"#{r2:02X}{g2:02X}{b2:02X}"

    bg_arc = _arc(0.0, 1.0, "#DDE5EC", stroke=20)
    grad_arcs = ""
    for si in range(_GRAD_STEPS):
        p0 = si / _GRAD_STEPS
        p1 = (si + 1) / _GRAD_STEPS
        mid = (p0 + p1) / 2
        grad_arcs += _arc(p0, p1, _lerp_color(mid), stroke=20)

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
    {grad_arcs}
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

    grad_def = (
        '<defs><linearGradient id="barGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%"   stop-color="#27AE60"/>'
        '<stop offset="50%"  stop-color="#F9CA24"/>'
        '<stop offset="100%" stop-color="#E74C3C"/>'
        '</linearGradient></defs>'
    )
    rects = (
        grad_def +
        f'<rect x="0" y="{TOP_PAD}" width="{W}" height="{H}" fill="url(#barGrad)"/>'
    )

    centre = ""

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
    """
    import math as _math

    BAR_W   = 520
    ROW_H   = 36
    L_PAD   = 200
    R_PAD   = 58
    TOTAL_W = L_PAD + BAR_W + R_PAD
    TOP_PAD = 16
    BAR_H   = 13
    SEP_H   = 10

    def _sigmoid100(logit: float) -> float:
        return 100.0 / (1.0 + _math.exp(-logit))

    def _interp_color(s100: float) -> str:
        # 0→50→100 maps to green→yellow→red
        t = max(0.0, min(1.0, s100 / 100.0))
        stops = [
            (0.00, (39,  174,  96)),   # #27AE60 green
            (0.50, (249, 202,  36)),   # #F9CA24 yellow
            (1.00, (231,  76,  60)),   # #E74C3C red
        ]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t <= t1:
                f = (t - t0) / (t1 - t0)
                r = int(c0[0] + f * (c1[0] - c0[0]))
                g = int(c0[1] + f * (c1[1] - c0[1]))
                b = int(c0[2] + f * (c1[2] - c0[2]))
                return f"#{r:02X}{g:02X}{b:02X}"
        return "#E74C3C"

    def _bar_color(s100: float) -> str:
        return _interp_color(s100)

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

    gen_rows  = sorted([r for r in rows if r["physician"] == "general"],  key=lambda r: r["target"])
    phys_rows = sorted([r for r in rows if r["physician"] != "general"],  key=lambda r: (r["physician"], r["target"]))
    all_rows  = gen_rows + phys_rows
    n         = len(all_rows)
    has_sep   = bool(gen_rows and phys_rows)
    SVG_H     = TOP_PAD + n * ROW_H + (SEP_H if has_sep else 0) + 24

    def _px(s100: float) -> float:
        return L_PAD + (s100 / 100.0) * BAR_W

    lines = []

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
        if has_sep and i == len(gen_rows):
            row_offset += SEP_H
            sep_y = TOP_PAD + i * ROW_H + row_offset - SEP_H // 2
            lines.append(
                f'<line x1="{L_PAD - 10}" y1="{sep_y}" x2="{TOTAL_W - 4}" y2="{sep_y}" '
                f'stroke="#D6DCE2" stroke-width="0.5" stroke-dasharray="4 3"/>'
            )

        y     = TOP_PAD + i * ROW_H + row_offset
        mid_y = y + ROW_H / 2

        if i % 2 == 0:
            lines.append(
                f'<rect x="0" y="{y:.1f}" width="{TOTAL_W}" height="{ROW_H}" fill="#F7FAFC"/>'
            )

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

        s100    = _sigmoid100(r["raw_score"])
        bar_px  = _px(s100)
        bar_y   = mid_y - BAR_H / 2

        lines.append(
            f'<rect x="{L_PAD:.1f}" y="{bar_y:.1f}" width="{BAR_W:.1f}" '
            f'height="{BAR_H}" fill="url(#scoreTrackGrad)" rx="3" opacity="0.18"/>'
        )

        fill_w  = max(bar_px - L_PAD, 1.0)
        bar_col = _bar_color(s100)
        lines.append(
            f'<rect x="{L_PAD:.1f}" y="{bar_y:.1f}" width="{fill_w:.1f}" '
            f'height="{BAR_H}" fill="{bar_col}" rx="3" opacity="0.82"/>'
        )

        lines.append(
            f'<circle cx="{bar_px:.1f}" cy="{mid_y:.1f}" r="5" '
            f'fill="{bar_col}" stroke="white" stroke-width="1.5"/>'
        )

        num_col = _num_color(s100)
        lines.append(
            f'<text x="{L_PAD + BAR_W + 7}" y="{mid_y + 1:.1f}" '
            f'font-size="12" fill="{num_col}" font-weight="700" dominant-baseline="central" '
            f'font-family="monospace">{s100:.0f}</text>'
        )

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

    score_grad_defs = (
        '<defs><linearGradient id="scoreTrackGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
        '<stop offset="0%"   stop-color="#27AE60"/>'
        '<stop offset="50%"  stop-color="#F9CA24"/>'
        '<stop offset="100%" stop-color="#E74C3C"/>'
        '</linearGradient></defs>'
    )
    svg = (
        f'<svg viewBox="0 0 {TOTAL_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:{TOTAL_W}px;display:block;">'
        + score_grad_defs
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
    vals  = np.asarray(shap_values, dtype=float)
    names = list(feature_names)

    order = np.argsort(np.abs(vals))[::-1][:n_display]
    vals  = vals[order]
    names = [names[i] for i in order]

    order2 = np.argsort(vals)
    vals   = vals[order2]
    names  = [names[i] for i in order2]

    BAR_H   = 22
    GAP     = 4
    L_PAD   = 260
    R_PAD   = 90
    BAR_W   = 380
    TOP_PAD = 58
    BOT_PAD = 50
    n       = len(vals)
    SVG_H   = TOP_PAD + n * (BAR_H + GAP) + BOT_PAD
    SVG_W   = L_PAD + BAR_W + R_PAD

    max_abs = max(np.abs(vals).max(), 1e-6)
    scale   = (BAR_W * 0.45) / max_abs

    cx = L_PAD + BAR_W / 2

    bars_svg  = ""
    texts_svg = ""

    running = base_value
    for i, (v, name) in enumerate(zip(vals, names)):
        y      = TOP_PAD + i * (BAR_H + GAP)
        bar_x  = cx if v >= 0 else cx + v * scale
        bar_w  = abs(v) * scale
        color  = "#E74C3C" if v >= 0 else "#2ECC71"
        texts_svg += (
            f'<text x="{L_PAD - 8}" y="{y + BAR_H/2 + 4:.0f}" '
            f'text-anchor="end" font-size="11" fill="#1B2A3D">'
            f'{name[:40]}</text>'
        )
        bars_svg += (
            f'<rect x="{bar_x:.1f}" y="{y}" width="{max(bar_w,1):.1f}" height="{BAR_H}" '
            f'fill="{color}" rx="3" opacity="0.85"/>'
        )
        label_x = bar_x + bar_w + 5 if v >= 0 else bar_x - 5
        anchor  = "start" if v >= 0 else "end"
        texts_svg += (
            f'<text x="{label_x:.1f}" y="{y + BAR_H/2 + 4:.0f}" '
            f'text-anchor="{anchor}" font-size="10" fill="{color}" font-weight="700">'
            f'{v:+.3f}</text>'
        )

    line_h  = TOP_PAD - 10 + n * (BAR_H + GAP) + BOT_PAD
    centre_line = (
        f'<line x1="{cx:.1f}" y1="{TOP_PAD - 15}" x2="{cx:.1f}" y2="{line_h}" '
        f'stroke="#9AA5B1" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<text x="{cx:.1f}" y="{TOP_PAD - 20}" text-anchor="middle" '
        f'font-size="10" fill="#5A6B7D">E[f(x)]={base_value:+.3f}</text>'
    )

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
        f'<text x="{SVG_W//2}" y="16" text-anchor="middle" font-size="13" '
        f'fill="#00456A" font-weight="700">SHAP Waterfall — feature contributions to score</text>'
        f'{centre_line}'
        f'{final_ann}'
        f'{bars_svg}'
        f'{texts_svg}'
        f'</svg>'
    )


def _shap_bar_html(feature_names, mean_abs_shap, n_display: int = 20) -> str:
    vals  = np.asarray(mean_abs_shap, dtype=float)
    names = list(feature_names)

    order = np.argsort(vals)[-n_display:]
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
        y = TOP_PAD + (n - 1 - i) * (BAR_H + GAP)
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
    try:
        import joblib, xgboost as xgb

        explainer_path = model_dir / "shap_explainer.pkl"
        model_path     = model_dir / "model.json"
        if not explainer_path.exists() or not model_path.exists():
            return None

        final_feats = entry["meta"].get("final_features", [])
        if not final_feats:
            return None

        available = [f for f in final_feats if f in params_df.columns]
        if not available:
            return None

        X_new = params_df[available].values.astype(np.float32)
        if len(available) < len(final_feats):
            full = np.zeros((1, len(final_feats)), dtype=np.float32)
            idx  = [final_feats.index(f) for f in available]
            full[0, idx] = X_new[0]
            X_new = full

        model = xgb.XGBClassifier()
        model.load_model(str(model_path))
        dmat       = xgb.DMatrix(X_new, feature_names=final_feats)
        raw_score  = float(model.get_booster().predict(dmat, output_margin=True)[0])

        explainer  = joblib.load(str(explainer_path))
        shap_out   = explainer(X_new)
        shap_vals  = shap_out.values
        if shap_vals.ndim == 3:
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

tab_pred, tab_explain, tab_model = st.tabs([
    "📊  Predictions",
    "🔍  Explanations",
    "🔬  Model Details",
])

with tab_pred:
    if st.session_state.pipeline_status != "done" or st.session_state.all_results is None:
        _empty("📊", "No predictions yet")
    else:
        ar = st.session_state.all_results

        st.markdown('<h3 style="color:#00456A;margin:0 0 0.3rem 0;">Model Selection</h3>', unsafe_allow_html=True)

        st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Mode</div>', unsafe_allow_html=True)
        sel_mode = st.radio(
            "Mode",
            ["strict", "relaxed"],
            horizontal=True,
            label_visibility="collapsed",
            format_func=lambda x: "Strict" if x == "strict" else "Relaxed",
            key="sel_mode",
        )

        # ── Explainer: Model Groups & Modes ──────────────────────────────────
        clinical_explainer(
            key="pred_groups",
            label="Model Groups & Modes",
            content_html="""
<strong>General model</strong> was trained on EEG recordings from all physicians combined,
giving the broadest possible view of what abnormality looks like in this dataset.
<strong>Per-physician models</strong> were trained only on recordings labelled by that
specific physician — they capture each clinician's individual interpretation style and
threshold for calling a finding abnormal. Neither is definitively "correct"; they are
complementary perspectives.<br><br>
Both the <strong>Abnormality Spectrum</strong> and <strong>Predictions by Category</strong>
always show the general model (row 1) and all physician models (row 2) simultaneously.<br><br>
<strong>Strict mode</strong> uses only unambiguous cases: label 1 (clearly normal) and
label 4 (clearly abnormal). Labels 2 and 3 are excluded from training.<br>
<strong>Relaxed mode</strong> treats labels 1 &amp; 2 as normal and labels 3 &amp; 4 as abnormal,
including borderline cases in training. The model learns a broader, more inclusive concept of
abnormality and is better at detecting subtle findings.
""",
        )

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        gview = get_view(ar, "general", sel_mode)

        # ── Abnormality Spectrum bar ──────────────────────────────────────────
        st.markdown('<h4 style="color:#00456A;margin:0 0 0.2rem 0;">Abnormality Spectrum</h4>', unsafe_allow_html=True)

        # Row 1: general model — all targets
        gen_bar_marks = []
        for tgt, ent in gview.items():
            if ent.get("available") and ent.get("result"):
                gen_bar_marks.append({
                    "label": tgt,
                    "prob":  ent["result"]["probability"],
                    "thr":   ent["result"]["threshold"],
                })

        # Row 2: per-physician Abnormality only
        phys_bar_marks = []
        for phys in ALL_PHYSICIANS:
            pe = ar.get(f"physician__{phys}__{sel_mode}__Abnormality", {})
            if pe.get("available") and pe.get("result"):
                phys_bar_marks.append({
                    "label": f"👤 {phys}",
                    "prob":  pe["result"]["probability"],
                    "thr":   pe["result"]["threshold"],
                })

        if gen_bar_marks:
            row2_bar_html = ""
            if phys_bar_marks:
                row2_bar_html = f"""
<p style="color:#00456A;font-size:12px;font-weight:700;margin:6px 0 2px 0;">
  Abnormality — Per Physician
</p>
{_threshold_bar_html(phys_bar_marks)}"""

            bar_html = f"""<!DOCTYPE html><html>
<body style="margin:0;padding:0;background:transparent;">
<div style="background:white;border-radius:12px;padding:14px 22px 14px;border:1px solid #D6DCE2;">
<p style="color:#00456A;font-size:12px;font-weight:700;margin:0 0 4px 0;">General Model</p>
{_threshold_bar_html(gen_bar_marks)}
{row2_bar_html}
</div></body></html>"""
            spectrum_height = 220 + (214 if phys_bar_marks else 0)
            components.html(bar_html, height=spectrum_height, scrolling=False)

        # ── Explainer: Abnormality Spectrum bar ───────────────────────────────
        clinical_explainer(
            key="pred_spectrum",
            label="Abnormality Spectrum chart",
            content_html="""
<strong>What this shows:</strong> A horizontal gradient strip from green (normal) to red (abnormal).
Each dot represents one model; its position reflects how far the predicted probability sits
above or below that model's own threshold.<br>
<strong>Row 1</strong> — General model across all 5 EEG categories.<br>
<strong>Row 2</strong> — Each physician's Abnormality model, so you can compare how each
clinician's model reads the same recording.<br>
<strong>Colour zones:</strong>
🟢 strongly normal &nbsp;|&nbsp; 🟡 borderline &nbsp;|&nbsp; 🟠 mildly abnormal &nbsp;|&nbsp; 🔴 strongly abnormal<br>
<strong>Clinical takeaway:</strong> Markers clustering left = consensus normal; scattered across
the threshold = borderline, warrants careful human review.
""",
        )

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        # ── Gauge cards ───────────────────────────────────────────────────────
        st.markdown('<h2 style="color:#00456A;margin:0 0 0.2rem 0;">Predictions by Category</h2>', unsafe_allow_html=True)

        # Row 1: General model — all 5 targets
        gen_gauge_svgs = []
        for tgt in ALL_TARGETS:
            ent = gview[tgt]
            if not ent.get("available") or not ent.get("result"):
                gen_gauge_svgs.append(
                    f'<div style="flex:1;min-width:190px;text-align:center;'
                    f'background:white;border-radius:12px;padding:1rem;border:1px solid #D6DCE2;">'
                    f'<p style="color:#00456A;font-weight:700;font-size:13px;">{tgt}</p>'
                    f'<p style="color:#8B9DB3;font-size:12px;">Not available</p></div>'
                )
            else:
                r   = ent["result"]
                svg = _gauge_html(tgt, r["probability"], r["threshold"], band=0.06)
                gen_gauge_svgs.append(f'<div style="flex:1;min-width:190px;">{svg}</div>')

        # Row 2: Per-physician Abnormality gauges (one per physician)
        phys_gauge_svgs = []
        for phys in ALL_PHYSICIANS:
            pe = ar.get(f"physician__{phys}__{sel_mode}__Abnormality", {})
            if pe.get("available") and pe.get("result"):
                r   = pe["result"]
                svg = _gauge_html(f"👤 {phys}", r["probability"], r["threshold"], band=0.06)
                phys_gauge_svgs.append(f'<div style="flex:1;min-width:190px;">{svg}</div>')
            else:
                phys_gauge_svgs.append(
                    f'<div style="flex:1;min-width:190px;text-align:center;'
                    f'background:white;border-radius:12px;padding:1rem;border:1px solid #D6DCE2;">'
                    f'<p style="color:#00456A;font-weight:700;font-size:13px;">👤 {phys}</p>'
                    f'<p style="color:#8B9DB3;font-size:12px;">Not available</p></div>'
                )

        # Build combined HTML with two rows
        row2_html = ""
        if phys_gauge_svgs:
            row2_html = f"""
<p style="color:#00456A;font-size:12px;font-weight:700;margin:6px 0 4px 0;">
  Abnormality — Per Physician
</p>
<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:nowrap;">
{''.join(phys_gauge_svgs)}
</div>"""

        gauges_html = f"""<!DOCTYPE html><html>
<body style="margin:0;padding:0;background:transparent;font-family:'Source Sans Pro',sans-serif;">
<p style="color:#00456A;font-size:12px;font-weight:700;margin:0 0 6px 0;">
  General Model
</p>
<div style="display:flex;gap:12px;align-items:flex-start;flex-wrap:nowrap;">
{''.join(gen_gauge_svgs)}
</div>
{row2_html}
<p style="color:#8B9DB3;font-size:11px;margin:4px 0 0 0;text-align:center;">
🟢 Normal zone &nbsp;·&nbsp; 🟡 Borderline band around threshold &nbsp;·&nbsp;
🔴 Abnormal zone
</p>
</body></html>"""

        gauge_height = 230 + (222 if phys_gauge_svgs else 0)
        components.html(gauges_html, height=gauge_height, scrolling=False)

        # ── Explainer: Gauge dials ────────────────────────────────────────────
        clinical_explainer(
            key="pred_gauges",
            label="Gauge dials — Predictions by Category",
            content_html="""
<strong>Each dial represents one EEG category.</strong> Think of it like a speedometer:
the needle points to the model's estimated probability (0 = definitely normal, 1 = definitely
abnormal). The coloured arc is divided into three zones:<br><br>
&nbsp;&nbsp;🟢 <strong>Green (left arc)</strong> — probability is comfortably below the threshold → <em>Normal</em><br>
&nbsp;&nbsp;🟡 <strong>Yellow (centre band)</strong> — probability is close to the threshold → <em>Borderline</em> (±6% around threshold)<br>
&nbsp;&nbsp;🔴 <strong>Red (right arc)</strong> — probability is comfortably above the threshold → <em>Abnormal</em><br><br>
The <strong>black tick mark</strong> on the arc shows exactly where the threshold sits for this
model. The <strong>solid dark needle</strong> is the model's prediction.<br>
The first row shows the General model across all five EEG categories. The second row shows
per-physician Abnormality predictions — one gauge per physician — so you can compare how each
clinician's model reads the same recording.<br>
<strong>p =</strong> probability output by the model &nbsp;|&nbsp;
<strong>thr =</strong> decision threshold chosen during training to optimise F1 score on held-out data.
""",
        )

        st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

        # ── Raw Score Bars ────────────────────────────────────────────────────
        score_rows = []
        for mid_key, ent in ar.items():
            if not ent.get("available") or not ent.get("result"):
                continue
            r    = ent["result"]
            meta = ent.get("meta", {})
            phys = meta.get("physician", mid_key.split("__")[0])
            mode = meta.get("label_mode", "")
            tgt  = meta.get("target", "")
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

                # ── Explainer: Score Bars ────────────────────────────────────
                clinical_explainer(
                    key="pred_scorebars",
                    label="Model Score bars",
                    content_html="""
<strong>What the score means:</strong> Each horizontal bar represents one model's output
for this recording. The score shown (0–100) is derived from the model's internal
<em>log-odds</em> — a measure of how strongly the model leans toward abnormal versus normal
— converted to a 0–100 scale using the sigmoid (S-shaped) function:<br><br>
<code>Score = sigmoid(log-odds) × 100</code><br>0 = maximally normal · 100 = maximally abnormal. The threshold is Optuna-tuned. Its position on this scale = threshold × 100<br>
<strong>Badge letters:</strong> <strong style="color:#155724;">N</strong> = Normal &nbsp;|&nbsp;
<strong style="color:#856404;">B</strong> = Borderline &nbsp;|&nbsp;
<strong style="color:#721C24;">A</strong> = Abnormal<br><br>
General model rows appear in <strong style="color:#00456A;">navy</strong>;
physician-specific model rows appear in <strong style="color:#E87722;">orange</strong>.
""",
                )


with tab_explain:
    if st.session_state.pipeline_status != "done" or st.session_state.all_results is None:
        _empty("🔍", "No explanations yet")
    else:
        ar         = st.session_state.all_results
        params_df  = st.session_state.params_df

        st.markdown('<h3 style="color:#00456A;margin:0 0 0.2rem 0;">Per-Recording Explanations</h3>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#5A6B7D;font-size:0.85rem;margin:0 0 0.3rem 0;">'
            'SHAP values computed live on <em>this specific recording</em> — '
            'showing exactly which features pushed the model toward or away from abnormality.</p>',
            unsafe_allow_html=True,
        )

        # ── Explainer: What is SHAP? ──────────────────────────────────────────
        clinical_explainer(
            key="explain_shap_intro",
            label="What is SHAP and why does it matter?",
            content_html="""
<strong>SHAP (SHapley Additive exPlanations)</strong> is a method that answers the question:
<em>"For this specific patient's EEG, which signal features drove the model's decision — and by how much?"</em><br><br>
The underlying model (XGBoost) is a <strong>gradient-boosted decision tree</strong> — an ensemble
of hundreds of simple decision rules that each learn from the mistakes of the previous ones.
The final prediction is the weighted sum of all those rules, which makes it powerful but
opaque. SHAP opens the "black box" by assigning each input feature a credit (positive or
negative) for the model's final score, using a mathematically rigorous framework borrowed
from game theory (Shapley values).<br><br>
<strong>Key concept — the baseline:</strong> SHAP compares each feature's actual value
against what the model would predict <em>on average</em> across all training recordings
(called E[f(x)], the expected model output). A feature's SHAP value is how much it shifted
the prediction away from that average.<br><br>
<strong>Why this matters clinically:</strong> Rather than simply saying "abnormal",
the model can now show <em>which frequency bands, which brain regions, and which signal
characteristics</em> are most unusual in this recording — providing a starting point for
the reviewing neurophysiologist.
""",
        )

        ec1, ec2, ec3 = st.columns([2, 1, 1])
        with ec1:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Group</div>', unsafe_allow_html=True)
            ex_phys = st.radio(
                "Group",
                ["general"] + ALL_PHYSICIANS,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "General" if x == "general" else f"👤 {x}",
                key="ex_phys",
            )
        with ec2:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Mode</div>', unsafe_allow_html=True)
            ex_mode = st.radio(
                "Mode",
                ["strict", "relaxed"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "Strict" if x == "strict" else "Relaxed",
                key="ex_mode",
            )
        with ec3:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Target</div>', unsafe_allow_html=True)
            ex_tgt_choices = ALL_TARGETS if ex_phys == "general" else ["Abnormality"]
            _ex_tgt_default = ex_tgt_choices.index("Abnormality") if "Abnormality" in ex_tgt_choices else 0
            ex_tgt = st.selectbox("Target", ex_tgt_choices, index=_ex_tgt_default, label_visibility="collapsed", key="ex_tgt")

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

            # ── Explainer: Summary metric cards ──────────────────────────────
            clinical_explainer(
                key="explain_metrics_cards",
                label="Summary metric cards",
                content_html="""
<strong>Prediction</strong> — the model's verdict for this recording: Normal, Borderline, or Abnormal.
Borderline means the predicted probability falls within ±6% of the model's threshold; clinical
review is particularly important in these cases.<br><br>
<strong>Raw Score (log-odds)</strong> — the model's internal "voice" before it is
converted to a probability. Positive values lean abnormal; negative values lean normal.
A value of 0 means 50% probability (pure uncertainty). The magnitude matters:
+2.0 is far more confident than +0.2.<br>
&nbsp;&nbsp;Formula: <code>probability = 1 / (1 + e<sup>−raw_score</sup>)</code><br><br>
<strong>CV AUPRC (Area Under the Precision-Recall Curve)</strong> — the model's overall quality
measured during cross-validation. This metric is particularly informative for imbalanced datasets
(where abnormal EEGs are the minority). A value of 1.0 is perfect; 0.5 is no better than
chance for a balanced dataset. Values above 0.75 are generally considered good for clinical AI
screening tools.<br><br>
<strong>Features Used</strong> — how many EEG signal features the model actually relies on,
after feature selection. Features are alpha-band spectral parameters (peak frequency,
power, bandwidth, etc.) measured across brain regions (parcels). Fewer features means a
more parsimonious, generalisable model.
""",
            )

            direction = "toward abnormal" if raw_sc > 0 else "toward normal"
            confidence = "high confidence" if abs(raw_sc) > 1.5 else \
                         "moderate confidence" if abs(raw_sc) > 0.5 else "near the decision boundary"
            st.markdown(
                f'<div style="background:#EBF5FB;border-left:4px solid #0091B3;'
                f'border-radius:6px;padding:0.5rem 0.8rem;margin:0.3rem 0;">'
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
            st.markdown('<h4 style="color:#00456A;margin:0 0 0.2rem 0;">SHAP Feature Contributions — This Recording</h4>', unsafe_allow_html=True)

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

                # ── Explainer: SHAP Waterfall ─────────────────────────────────
                clinical_explainer(
                    key="explain_waterfall",
                    label="SHAP Waterfall chart — this recording",
                    content_html="""
<strong>What this shows:</strong> The waterfall chart explains <em>this specific recording's</em>
model output, feature by feature. It answers: "Which EEG signal features pushed the model toward
abnormal, and which pushed it toward normal — and by how much?"<br><br>
<strong>How to read it:</strong><br>
&nbsp;&nbsp;• The <strong>vertical dashed line labelled E[f(x)]</strong> is the model's average
predicted probability over a reference set of 200 training recordings — the baseline before any
features of this recording are considered.<br>
&nbsp;&nbsp;• Each <strong>horizontal bar</strong> shows one feature's contribution to the
predicted probability for this recording:<br>
&nbsp;&nbsp;&nbsp;&nbsp;— <span style="color:#E74C3C;">🔴 Red bar (positive value)</span> = this
feature <em>increased</em> the predicted probability, pushing toward abnormal<br>
&nbsp;&nbsp;&nbsp;&nbsp;— <span style="color:#2ECC71;">🟢 Green bar (negative value)</span> = this
feature <em>decreased</em> the predicted probability, pushing toward normal<br>
&nbsp;&nbsp;• The <strong>bar length</strong> shows the magnitude — a longer bar means a stronger
influence on the probability.<br>
&nbsp;&nbsp;• Features are sorted by signed SHAP value: the <strong>strongest normal-pushing
features appear at the top</strong> (most negative, green), the <strong>strongest
abnormal-pushing features appear at the bottom</strong> (most positive, red).<br>
&nbsp;&nbsp;• The <strong>blue dashed line labelled f(x)</strong> is the model's raw log-odds
score for this recording, shown for reference. Note: SHAP bars are in probability units
while f(x) is in log-odds — they are not directly additive.<br><br>
<strong>What are the features?</strong> Each feature is an alpha-band spectral parameter
extracted from the EEG signal and summarised across brain regions (parcels). Examples include:<br>
&nbsp;&nbsp;— <em>alpha_peak_cf</em>: centre frequency of the dominant alpha peak (typically 8–13 Hz)<br>
&nbsp;&nbsp;— <em>alpha_peak_pw</em>: power of the alpha peak above the aperiodic background<br>
&nbsp;&nbsp;— <em>alpha_peak_bw</em>: bandwidth (spectral width) of the alpha peak<br>
&nbsp;&nbsp;— <em>aperiodic_exponent</em>: slope of the 1/f background spectrum<br><br>
<strong>Clinical interpretation:</strong> The features with the longest bars — whether green
(top) or red (bottom) — had the largest influence on this recording's prediction. If a
posterior parcel shows a large red bar for alpha power, the model found that region's alpha
to be unusually low relative to what it learned from training — consistent with posterior
alpha attenuation. Use the highest-magnitude bars to direct your review to the most
informative channels and frequency bands in the raw EEG.
""",
                )

                st.caption(
                    "🟢 Green bars push toward **normal** (negative SHAP). "
                    "🔴 Red bars push toward **abnormal** (positive SHAP). "
                )

                st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

                # Top contributors table
                # Top contributors table — collapsible
                with st.expander("📋 Top Feature Contributions — Ranked", expanded=False):
                    order_table = np.argsort(np.abs(shap_1d))[::-1][:30]
                    rows_table  = []
                    for idx in order_table:
                        v = float(shap_1d[idx])
                        rows_table.append({
                            "Feature":    feat_names[idx],
                            "SHAP Value": f"{v:+.4f}",
                            "Direction":  "▲ abnormal" if v > 0 else "▼ normal",
                            "|SHAP|":     round(abs(v), 4),
                        })
                    shap_table_df = pd.DataFrame(rows_table)
                    st.dataframe(
                        shap_table_df.style
                            .set_properties(**{
                                "background-color": "#F8FAFC",
                                "color":            "#1B2A3D",
                                "font-size":        "13px",
                            })
                            .set_properties(
                                subset=["Feature"],
                                **{"font-weight": "600", "color": "#00456A"},
                            )
                            .applymap(
                                lambda v: "color:#B42318;font-weight:700;background-color:#FEF2F2"
                                        if "▲" in str(v)
                                        else ("color:#156A3A;font-weight:700;background-color:#F0FDF4"
                                                if "▼" in str(v) else ""),
                                subset=["Direction"],
                            )
                            .set_table_styles([{
                                "selector": "thead tr th",
                                "props":    [("background-color", "#E8F0F5"),
                                            ("color", "#00456A"),
                                            ("font-weight", "700"),
                                            ("font-size", "12px")],
                            }]),
                        use_container_width=True,
                        hide_index=True,
                    )

                # ── Explainer: Top contributions table ────────────────────────
                clinical_explainer(
                    key="explain_shap_table",
                    label="Top Feature Contributions table",
                    content_html="""
<strong>What this table shows:</strong> The same information as the waterfall chart but in tabular
form, ranked by the absolute size of each feature's influence (|SHAP|), largest first.<br><br>
<strong>Columns explained:</strong><br>
&nbsp;&nbsp;• <strong>Feature</strong> — the EEG signal parameter name.
The naming convention is: <code>[spectral_property]_[brain_region/parcel_number]</code>.
Brain regions (parcels) correspond to the Schaefer 400-parcel cortical atlas, which divides
the cortex into 400 regions of interest.<br>
&nbsp;&nbsp;• <strong>SHAP Value</strong> — the contribution in log-odds units. Adding up all SHAP
values plus the baseline E[f(x)] gives the final raw score f(x).<br>
&nbsp;&nbsp;• <strong>Direction</strong> — ▲ Abnormal means this feature pushes toward abnormal;
▼ Normal means it pushes toward normal.<br>
&nbsp;&nbsp;• <strong>|SHAP|</strong> — the absolute magnitude, used for ranking. A value of 0.5 means
this feature shifted the log-odds by 0.5 units (approximately 12% probability shift near the threshold).<br><br>
<strong>Practical use:</strong> Sort by |SHAP| to find the most influential features.
Cross-reference the top features with the raw EEG trace — the channel and frequency region
highlighted here is where the model found the most deviation from normal.
""",
                )

            st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)

            # ── Input vector snapshot ─────────────────────────────────────────
            st.markdown('<h4 style="color:#00456A;margin:0 0 0.2rem 0;">Input Feature Vector — This Recording</h4>', unsafe_allow_html=True)
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

                    # ── Explainer: Input feature vector ───────────────────────
                    clinical_explainer(
                        key="explain_input_vector",
                        label="Input Feature Vector table",
                        content_html="""
<strong>What this table shows:</strong> The raw numeric values of every feature the model
received as input for this specific recording, sorted by their |SHAP| importance.<br><br>
<strong>Value column:</strong> The actual measured value of each spectral parameter.
For example, <em>alpha_peak_freq_parcel_42 = 9.73</em> means the dominant alpha frequency
in parcel 42 is 9.73 Hz. Normal posterior alpha is typically 9–11 Hz in adults; values
outside this range may indicate pathology.<br><br>
<strong>SHAP column:</strong> How much this feature's specific value pushed the model's
score away from the average. A SHAP of +0.4 with a value of 7.2 Hz (for peak frequency)
would mean "the model sees this 7.2 Hz alpha as abnormally slow, and that contributed
significantly to the abnormality call."<br><br>
<strong>Features missing from the model:</strong> If a feature appears in the full EEG
feature set but not in this table, it was eliminated during the ReliefF feature
selection step — the model determined it had little predictive value across training folds
and excluded it to reduce overfitting.
""",
                    )
            else:
                st.info("Feature vector not available (params_df columns don't overlap model features).")

with tab_model:
    if st.session_state.pipeline_status != "done" or st.session_state.all_results is None:
        _empty("🔬", "No model details yet")
    else:
        ar = st.session_state.all_results

        st.markdown('<h3 style="color:#00456A;margin:0 0 0.3rem 0;">Select Model to Inspect</h3>', unsafe_allow_html=True)
        mc1, mc2, mc3 = st.columns([2, 1, 1])

        with mc1:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Group</div>', unsafe_allow_html=True)
            md_phys = st.radio(
                "Group",
                ["general"] + ALL_PHYSICIANS,
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "General" if x == "general" else f"👤 {x}",
                key="md_phys",
            )

        with mc2:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Mode</div>', unsafe_allow_html=True)
            md_mode = st.radio(
                "Mode",
                ["strict", "relaxed"],
                horizontal=True,
                label_visibility="collapsed",
                format_func=lambda x: "Strict" if x == "strict" else "Relaxed",
                key="md_mode",
            )

        with mc3:
            st.markdown('<div style="color:#00456A;font-weight:700;margin-bottom:0.1rem;">Target</div>', unsafe_allow_html=True)
            tgt_choices = ALL_TARGETS if md_phys == "general" else ["Abnormality"]
            _md_tgt_default = tgt_choices.index("Abnormality") if "Abnormality" in tgt_choices else 0
            md_tgt = st.selectbox("Target", tgt_choices, index=_md_tgt_default, label_visibility="collapsed", key="md_tgt")

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

            # ── Explainer: Model performance metrics ──────────────────────────
            clinical_explainer(
                key="model_perf_metrics",
                label="Model performance metrics",
                content_html="""
These five metrics summarise how well this model performed during <strong>cross-validation</strong>
— testing on held-out EEG recordings it had never seen during training. The ± value is the
standard deviation across folds (lower = more stable/consistent).<br><br>
<strong>CV AUPRC</strong> (Area Under the Precision-Recall Curve) — the primary metric for this
imbalanced classification task. It summarises the trade-off between:<br>
&nbsp;&nbsp;• <em>Precision</em>: of the recordings the model called abnormal, what fraction actually were?<br>
&nbsp;&nbsp;• <em>Recall (Sensitivity)</em>: of all truly abnormal recordings, what fraction did the model catch?<br>
A perfect model = 1.0. A random classifier on a dataset with 20% abnormal cases would score ≈0.20.
Values above 0.75 are generally considered useful for screening; above 0.85 is strong.<br><br>
<strong>CV AUROC</strong> (Area Under the ROC Curve) — measures the model's ability to rank abnormal
recordings above normal ones, at any threshold. 1.0 = perfect discrimination; 0.5 = random.
AUROC is less sensitive to class imbalance than AUPRC.<br><br>
<strong>CV F1</strong> — the harmonic mean of Precision and Recall at the chosen threshold.
F1 = 1.0 is perfect; F1 = 0.0 is useless. The threshold was chosen during Optuna hyperparameter
optimisation to maximise F1 on validation data.<br>
&nbsp;&nbsp;• <em>P (Precision)</em> — positive predictive value: how many flagged abnormal EEGs are truly abnormal<br>
&nbsp;&nbsp;• <em>R (Recall/Sensitivity)</em> — of all true abnormals, how many did the model detect
""",
            )

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

            # ── Performance plots ─────────────────────────────────────────────
            st.markdown('<h4 style="color:#00456A;margin:0.2rem 0 0.2rem 0;">📈 Model Performance</h4>', unsafe_allow_html=True)

            # ── Explainer: Model Performance (consolidated) ───────────────────
            clinical_explainer(
                key="model_performance_all",
                label="Model Performance",
                content_html="""
<strong>PR Curve</strong> — Precision (y) vs Recall (x) at every threshold. Curve hugging
the top-right = strong model. <strong>AUPRC</strong> summarises it in one number; higher is better.
Preferred over ROC here because abnormal EEGs are the minority class.<br><br>
<strong>ROC Curve</strong> — Sensitivity (y) vs False Positive Rate (x). Diagonal = random chance;
top-left corner = perfect. <strong>AUROC</strong> = probability the model ranks a true abnormal
above a true normal. 0.5 = random, 1.0 = perfect.<br><br>
<strong>Probability Distribution</strong> — Histograms of predicted probabilities for normal (blue)
and abnormal (orange) recordings. Good separation (blue near 0, orange near 1) = confident model.
Heavy overlap near the threshold = many borderline cases requiring human review.<br><br>
<strong>Confusion Matrix</strong> — At the chosen threshold:<br>
&nbsp;&nbsp;• <strong>TP</strong> ✅ caught abnormal &nbsp;·&nbsp; <strong>TN</strong> ✅ correctly normal<br>
&nbsp;&nbsp;• <strong>FN</strong> ❌ missed abnormal (most clinically dangerous) &nbsp;·&nbsp; <strong>FP</strong> ⚠️ unnecessary flag<br>
All plots use OOF (out-of-fold) predictions — every recording evaluated by a model that never saw it during training.
""",
            )

            perf = [
                ("pr_curve.png",         "Precision-Recall Curve"),
                ("prob_distribution.png","Probability Distribution"),
                ("roc_curve.png",        "ROC Curve"),
                ("confusion_matrix.png", "Confusion Matrix"),
            ]
            pc = st.columns(4)
            for i, (f, t) in enumerate(perf):
                with pc[i]:
                    _plot_or_placeholder(plot_dir / f, t)

            st.markdown('<h4 style="color:#00456A;margin:0.2rem 0 0.2rem 0;">🧬 Feature Importance & Explainability</h4>', unsafe_allow_html=True)

            # ── Explainer: SHAP Beeswarm ──────────────────────────────────────
            clinical_explainer(
                key="model_shap_beeswarm",
                label="SHAP Beeswarm plot",
                content_html="""
<strong>What this shows:</strong> The beeswarm (or summary) plot gives a global view of how
each feature influences the model across all training recordings simultaneously.<br><br>
<strong>How to read it:</strong><br>
&nbsp;&nbsp;• <strong>Y-axis</strong> — features ranked by importance (highest mean |SHAP| at the top)<br>
&nbsp;&nbsp;• <strong>X-axis</strong> — the SHAP value for each individual recording. Positive = pushed toward
abnormal; negative = pushed toward normal<br>
&nbsp;&nbsp;• <strong>Each dot</strong> = one training recording. Dots are spread vertically (the "beeswarm")
to avoid overlap.<br>
&nbsp;&nbsp;• <strong>Dot colour</strong> — represents the actual feature value for that recording:
<em>red = high value, blue = low value</em>. This reveals the direction of the relationship:
if red dots (high feature value) cluster on the right (positive SHAP), it means a high value of
that feature pushes toward abnormal.<br><br>
<strong>Example interpretation:</strong> If <em>alpha_power_parcel_162</em> shows red dots (high power)
on the left (negative SHAP) and blue dots (low power) on the right (positive SHAP), this means
<em>lower alpha power in parcel 162 drives the model toward calling the EEG abnormal</em> —
consistent with the known EEG finding of posterior alpha attenuation in encephalopathy.
""",
            )

            # ── Explainer: SHAP Bar ───────────────────────────────────────────
            clinical_explainer(
                key="model_shap_bar",
                label="SHAP Bar chart — global feature importance",
                content_html="""
<strong>What this shows:</strong> The mean absolute SHAP value for each feature, averaged across
all training recordings. This is the simplest global summary of feature importance:<br>
<em>"On average, how much does knowing this feature's value change the model's prediction?"</em><br><br>
<strong>How to read it:</strong> Longer bar = more important feature overall. Unlike the beeswarm,
this collapses direction — it does not tell you whether high or low values are abnormal,
just how much the feature matters.<br><br>
<strong>Feature naming convention:</strong><br>
&nbsp;&nbsp;• <code>alpha_cf_parcel_N</code> — centre frequency of the alpha peak in brain region N<br>
&nbsp;&nbsp;• <code>alpha_pw_parcel_N</code> — alpha peak power<br>
&nbsp;&nbsp;• <code>alpha_bw_parcel_N</code> — alpha peak bandwidth (spectral width)<br>
&nbsp;&nbsp;• <code>alpha_exp_parcel_N</code> — aperiodic exponent (slope of 1/f background)<br>
&nbsp;&nbsp;• <code>alpha_offset_parcel_N</code> — aperiodic offset (overall power level)<br><br>
<strong>Clinical correlate:</strong> Features from posterior parcels (occipital, parietal) typically
dominate for alpha-rhythm disorders. Features from frontal parcels may dominate for diffuse slowing.
A heavily frontal-dominant importance profile may suggest the model has learned frontal encephalopathy patterns.
""",
            )

            shap_plots = [
                ("shap_beeswarm.png",    "SHAP Beeswarm (top 20)"),
                ("shap_bar.png",         "SHAP Bar Chart"),
                ("native_importance.png","XGBoost Gain Importance"),
            ]
            sc = st.columns(3)
            for i, (f, t) in enumerate(shap_plots):
                with sc[i]:
                    _plot_or_placeholder(plot_dir / f, t)

            # ── Explainer: XGBoost Native Importance ──────────────────────────
            clinical_explainer(
                key="model_xgb_gain",
                label="XGBoost Native Gain Importance",
                content_html="""
<strong>What this shows:</strong> XGBoost's own built-in measure of feature importance,
based on <em>gain</em> — the average improvement in prediction accuracy brought by a feature
each time it is used as a split point in a decision tree.<br><br>
<strong>Gain</strong> measures how much each feature reduces uncertainty (impurity) at the
decision nodes where it is used. A high gain means the feature consistently helps the model
make better predictions when it appears in a tree split.<br><br>
<strong>Difference from SHAP:</strong> XGBoost native importance is a property of the model
structure (which features appear in tree splits) rather than a measure of each feature's
contribution to a specific prediction. SHAP is generally preferred for clinical explanation
because it is fairer to correlated features and directly additive. Native importance is
provided here for reference and cross-validation of SHAP findings.<br><br>
<strong>Consistency check:</strong> If the top features by gain roughly match the top features
by SHAP importance, the model is internally consistent. Large discrepancies may suggest
correlated features where SHAP is redistributing credit between highly correlated brain regions.
""",
            )

            report = model_dir / "evaluation_report.txt"
            if report.exists():
                with st.expander("📝 Full Evaluation Report", expanded=False):
                    st.code(report.read_text(), language=None)

                    clinical_explainer(
                        key="model_eval_report",
                        label="Full Evaluation Report",
                        content_html="""
<strong>What this shows:</strong> A complete text summary of everything about this model:
training configuration, cross-validation performance, per-fold breakdown, hyperparameters
selected by Optuna, and the top features by both SHAP and XGBoost Gain importance.<br><br>
<strong>Key sections to review:</strong><br>
&nbsp;&nbsp;• <em>Stability</em> section — explicitly flags if AUPRC standard deviation exceeds 0.10
(a threshold indicating high variance and potential unreliability)<br>
&nbsp;&nbsp;• <em>Best hyperparameters</em> — the XGBoost settings selected across folds, including
<em>scale_pos_weight</em> (the class imbalance correction factor), <em>max_depth</em> (tree complexity),
and <em>learning_rate</em><br>
&nbsp;&nbsp;• <em>Top 10 features by mean |SHAP|</em> — the global most important features<br><br>
<strong>scale_pos_weight</strong> deserves special attention: it equals (number of normal recordings) ÷ (number of abnormal recordings).
A value of 4.0 means the model trains on a 4:1 normal-to-abnormal ratio and upweights abnormal
cases to compensate. A very high value (>10) may indicate extreme class imbalance and the
model's performance on abnormal cases should be interpreted with particular care.
""",
                    )

            # ── SHAP Dependence Explorer ──────────────────────────────────────
            st.markdown('<div class="fh-divider"></div>', unsafe_allow_html=True)
            st.markdown('<h4 style="color:#00456A;margin:0.2rem 0 0.2rem 0;">🔎 SHAP Dependence Explorer</h4>', unsafe_allow_html=True)

            clinical_explainer(
                key="model_shap_dependence",
                label="SHAP Dependence Explorer",
                content_html="""
<strong>What this shows:</strong> For the selected feature, every training sample is plotted as a dot:
<em>X-axis</em> = the actual measured value of that feature, <em>Y-axis</em> = its SHAP contribution
(how much it pushed the model toward abnormal or normal for that sample).<br><br>
<strong>How to read it:</strong><br>
&nbsp;&nbsp;• Dots <strong>above zero</strong> (red) = this feature value pushed the model toward <em>abnormal</em><br>
&nbsp;&nbsp;• Dots <strong>below zero</strong> (green) = pushed toward <em>normal</em><br>
&nbsp;&nbsp;• The <strong>shape of the cloud</strong> tells you the relationship: a downward slope means
higher feature values → more normal; an upward slope → more abnormal; flat → feature value doesn't matter much<br>
&nbsp;&nbsp;• The <strong>blue dashed line</strong> marks the value for <em>this recording</em> (if available)<br><br>
<strong>Clinical use:</strong> "If this patient's alpha peak frequency were higher, would the model
be less concerned?" — read directly off the chart.
""",
            )

            _shap_csv_path = model_dir / "shap_values_top30.csv"
            _bg_npy_path   = model_dir / "shap_background.npy"

            if _shap_csv_path.exists() and _bg_npy_path.exists():
                @st.cache_data(show_spinner=False)
                def _load_shap_dependence(shap_path: str, bg_path: str, feat_list: tuple):
                    _sv  = pd.read_csv(shap_path)          # N_samples × 30 SHAP cols
                    _bg  = np.load(bg_path)                 # (200, 55) feature values
                    _bg_df = pd.DataFrame(_bg, columns=list(feat_list))
                    # keep only features present in both
                    _common = [f for f in _sv.columns if f in _bg_df.columns]
                    return _sv[_common], _bg_df[_common]

                _feat_list = tuple(meta.get("final_features", []))
                _shap_df, _feat_df = _load_shap_dependence(
                    str(_shap_csv_path), str(_bg_npy_path), _feat_list
                )

                # Feature selector — ordered by mean |SHAP|
                _feat_order = _shap_df.abs().mean().sort_values(ascending=False).index.tolist()
                _sel_feat = st.selectbox(
                    "Select feature to explore",
                    _feat_order,
                    key=f"dep_feat_{mid}",
                )

                if _sel_feat and _sel_feat in _feat_df.columns:
                    _x_vals  = _feat_df[_sel_feat].values        # feature values (200 bg samples)
                    _y_vals  = _shap_df[_sel_feat].values        # SHAP values (N training samples)

                    # X values may have fewer rows than SHAP if bg is a subset — use the shorter
                    _n      = min(len(_x_vals), len(_y_vals))
                    _x_vals = _x_vals[:_n]
                    _y_vals = _y_vals[:_n]

                    # Current recording value for this feature
                    _cur_val = None
                    if st.session_state.params_df is not None:
                        _pdf = st.session_state.params_df
                        if _sel_feat in _pdf.columns:
                            _cur_val = float(_pdf[_sel_feat].iloc[0])

                    # ── Build SVG scatter ─────────────────────────────────────
                    SVG_W, SVG_H = 820, 340
                    L, R, T, B   = 62, 20, 20, 50   # margins

                    _xmin, _xmax = float(np.nanmin(_x_vals)), float(np.nanmax(_x_vals))
                    _ymin, _ymax = float(np.nanmin(_y_vals)), float(np.nanmax(_y_vals))
                    _xpad = (_xmax - _xmin) * 0.04 or 0.1
                    _ypad = (_ymax - _ymin) * 0.08 or 0.1
                    _xmin -= _xpad;  _xmax += _xpad
                    _ymin -= _ypad;  _ymax += _ypad

                    PW = SVG_W - L - R   # plot width
                    PH = SVG_H - T - B   # plot height

                    def _px(v, lo, hi, size):
                        return L + (v - lo) / (hi - lo) * PW if size == "x" else T + (1 - (v - lo) / (hi - lo)) * PH

                    def _dot_color(shap_v):
                        # green (negative SHAP) → white (zero) → red (positive SHAP)
                        if shap_v >= 0:
                            t = min(1.0, shap_v / max(abs(_ymax), 1e-6))
                            r = int(220 + (231 - 220) * t); g = int(220 * (1 - t)); b = int(220 * (1 - t))
                        else:
                            t = min(1.0, abs(shap_v) / max(abs(_ymin), 1e-6))
                            r = int(220 * (1 - t)); g = int(180 + (39 - 180) * t + 220 * (1 - t)); b = int(220 * (1 - t))
                        return f"#{max(0,min(255,r)):02X}{max(0,min(255,g)):02X}{max(0,min(255,b)):02X}"

                    # Zero line Y position
                    _zero_y = _px(0.0, _ymin, _ymax, "y")

                    dots = []
                    for xi, yi in zip(_x_vals, _y_vals):
                        if np.isnan(xi) or np.isnan(yi):
                            continue
                        cx = _px(xi, _xmin, _xmax, "x")
                        cy = _px(yi, _ymin, _ymax, "y")
                        col = _dot_color(yi)
                        dots.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="{col}" opacity="0.55"/>')

                    # Axis ticks
                    x_ticks = ""; y_ticks = ""
                    for ti in range(6):
                        tv = _xmin + ti * (_xmax - _xmin) / 5
                        tx = _px(tv, _xmin, _xmax, "x")
                        x_ticks += (f'<line x1="{tx:.1f}" y1="{T+PH}" x2="{tx:.1f}" y2="{T+PH+5}" stroke="#AAB8C2" stroke-width="1"/>'
                                    f'<text x="{tx:.1f}" y="{T+PH+17}" text-anchor="middle" font-size="10" fill="#5A6B7D">{tv:.2f}</text>')
                    for ti in range(6):
                        tv = _ymin + ti * (_ymax - _ymin) / 5
                        ty = _px(tv, _ymin, _ymax, "y")
                        y_ticks += (f'<line x1="{L-5}" y1="{ty:.1f}" x2="{L}" y2="{ty:.1f}" stroke="#AAB8C2" stroke-width="1"/>'
                                    f'<text x="{L-8}" y="{ty+4:.1f}" text-anchor="end" font-size="10" fill="#5A6B7D">{tv:.2f}</text>')

                    # Current recording vertical line
                    cur_line = ""
                    if _cur_val is not None and _xmin <= _cur_val <= _xmax:
                        cx2 = _px(_cur_val, _xmin, _xmax, "x")
                        cur_line = (
                            f'<line x1="{cx2:.1f}" y1="{T}" x2="{cx2:.1f}" y2="{T+PH}" '
                            f'stroke="#0091B3" stroke-width="2" stroke-dasharray="6 3"/>'
                            f'<text x="{cx2:.1f}" y="{T-5}" text-anchor="middle" font-size="10" '
                            f'fill="#0091B3" font-weight="700">this recording ({_cur_val:.3f})</text>'
                        )

                    dep_svg = f"""
<svg viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:{SVG_W}px;display:block;">
  <rect width="{SVG_W}" height="{SVG_H}" fill="white" rx="10"/>
  <!-- plot area border -->
  <rect x="{L}" y="{T}" width="{PW}" height="{PH}" fill="#F8FAFC" stroke="#D6DCE2" stroke-width="1"/>
  <!-- zero line -->
  <line x1="{L}" y1="{_zero_y:.1f}" x2="{L+PW}" y2="{_zero_y:.1f}"
        stroke="#888" stroke-width="1.2" stroke-dasharray="4 3"/>
  <text x="{L+4}" y="{_zero_y-4:.1f}" font-size="9" fill="#888">SHAP = 0</text>
  <!-- above zero label -->
  <text x="{L+PW-4}" y="{T+14}" text-anchor="end" font-size="10" fill="#B42318" font-weight="600">▲ pushes abnormal</text>
  <!-- below zero label -->
  <text x="{L+PW-4}" y="{T+PH-6}" text-anchor="end" font-size="10" fill="#156A3A" font-weight="600">▼ pushes normal</text>
  {cur_line}
  {''.join(dots)}
  {x_ticks}
  {y_ticks}
  <!-- axis labels -->
  <text x="{L + PW/2:.1f}" y="{SVG_H-4}" text-anchor="middle" font-size="11" fill="#00456A" font-weight="600">{_sel_feat}</text>
  <text x="12" y="{T + PH/2:.1f}" text-anchor="middle" font-size="11" fill="#00456A" font-weight="600"
        transform="rotate(-90, 12, {T + PH/2:.1f})">SHAP value</text>
</svg>"""

                    components.html(
                        f"<!DOCTYPE html><html><body style='margin:0;padding:6px;background:transparent;'>"
                        f"{dep_svg}</body></html>",
                        height=SVG_H + 20,
                        scrolling=False,
                    )

                    # Summary stats below the chart
                    _mean_shap = float(np.mean(_y_vals))
                    _corr      = float(np.corrcoef(_x_vals, _y_vals)[0, 1]) if len(_x_vals) > 2 else 0.0
                    _direction = "higher value → more abnormal" if _corr > 0.05 else ("higher value → more normal" if _corr < -0.05 else "no clear linear trend")
                    st.caption(
                        f"**{_sel_feat}** · {len(_x_vals)} training samples · "
                        f"mean SHAP = {_mean_shap:+.3f} · feature↔SHAP correlation r = {_corr:.2f} · "
                        f"*{_direction}*"
                    )
            else:
                st.info("SHAP dependence data not available for this model (shap_values_top30.csv or shap_background.npy missing).")

st.markdown("""
<div class="footer-authors">
    <div class="author-line">
        Iryna Gunko  &nbsp;·&nbsp; Vasily A. Vakorin
    </div>
    <div>
        Dept. Biomedical Physiology &amp; Kinesiology, Simon Fraser University, Burnaby, Canada
    </div>
</div>
""", unsafe_allow_html=True)