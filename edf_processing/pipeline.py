import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import mne
import numpy as np

from config import BEM, SUBJECTS_DIR, SRC_ICO5, TRANS

logger = logging.getLogger(__name__)

PARC_NAME = "Schaefer2018_400Parcels_17Networks_order"


def load_raw(
    file_bytes: bytes, filename: str, montage_name: str = "standard_1020"
) -> mne.io.Raw:
    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        if suffix == ".fif":
            raw = mne.io.read_raw_fif(tmp_path, preload=True, verbose=False)
        else:
            raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
            if "Status" in raw.ch_names:
                raw.drop_channels(["Status"])
            montage = mne.channels.make_standard_montage(montage_name)
            raw.set_montage(montage, on_missing="warn")
            mne.datasets.eegbci.standardize(raw)
    finally:
        os.unlink(tmp_path)
    logger.info("Loaded %s — %d ch, %.1f s @ %.0f Hz (montage: %s)",
                filename, len(raw.ch_names), raw.times[-1], raw.info["sfreq"], montage_name)
    return raw


def apply_notch(raw: mne.io.Raw, freq: float, method: str) -> mne.io.Raw:
    raw.notch_filter(freqs=freq, picks="eeg", method=method, verbose=False)
    logger.info("Notch filter applied at %.1f Hz (%s)", freq, method)
    return raw


def apply_bandpass(
    raw: mne.io.Raw, l_freq: float, h_freq: float, method: str
) -> mne.io.Raw:
    raw.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", method=method, verbose=False)
    logger.info("Bandpass %.1f–%.1f Hz (%s)", l_freq, h_freq, method)
    return raw


def apply_resample(raw: mne.io.Raw, target_sfreq: float) -> mne.io.Raw:
    raw.resample(target_sfreq, verbose=False)
    logger.info("Resampled to %.0f Hz", target_sfreq)
    return raw


def apply_pyprep(
    raw: mne.io.Raw, line_noise: float, montage_name: str = "standard_1020"
) -> mne.io.Raw:
    from pyprep.prep_pipeline import PrepPipeline

    sample_rate = raw.info["sfreq"]
    line_freqs = np.arange(line_noise, sample_rate / 2, line_noise)
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": line_freqs,
        "max_iterations": 4,
    }
    montage = mne.channels.make_standard_montage(montage_name)
    raw.set_montage(montage, on_missing="warn")
    prep = PrepPipeline(
        raw,
        prep_params,
        montage,
        ransac=True,
        channel_wise=False,
        random_state=12345,
        filter_kwargs={"method": "fir"},
        matlab_strict=False,
    )
    prep.fit()
    raw = prep.raw
    logger.info("PyPrep completed")
    return raw


def make_forward_solution(raw: mne.io.Raw) -> mne.Forward:
    trans = mne.read_trans(TRANS)
    src = mne.read_source_spaces(SRC_ICO5, verbose=False)
    bem = mne.read_bem_solution(BEM, verbose=False)
    fwd = mne.make_forward_solution(
        raw.info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        ignore_ref=True,
        mindist=5.0,
        verbose=False,
    )
    logger.info("Forward solution computed")
    return fwd


def apply_lcmv_beamformer(raw: mne.io.Raw, fwd: mne.Forward) -> mne.SourceEstimate:
    noise_cov = mne.compute_raw_covariance(
        raw, tmin=0, tmax=None, method="auto", verbose=False
    )
    noise_cov_reg = mne.cov.regularize(
        noise_cov, raw.info, mag=0.05, grad=0.05, eeg=0.1, verbose=False
    )
    filters = mne.beamformer.make_lcmv(
        raw.info,
        fwd,
        noise_cov_reg,
        reg=0.05,
        pick_ori="max-power",
        weight_norm="unit-noise-gain",
        rank=None,
        verbose=False,
    )
    stc = mne.beamformer.apply_lcmv_raw(raw, filters, verbose=False)
    logger.info("LCMV beamformer applied")
    return stc


def get_schaefer_labels() -> List[mne.Label]:
    labels = mne.read_labels_from_annot(
        "fsaverage",
        parc=PARC_NAME,
        subjects_dir=SUBJECTS_DIR,
        verbose=False,
    )
    # drop the unknown/medial-wall label MNE appends
    labels = [lb for lb in labels if "unknown" not in lb.name.lower()]
    logger.info("Loaded %d Schaefer labels", len(labels))
    return labels


def run_edf_pipeline(
    file_bytes: bytes,
    filename: str,
    notch_enabled: bool,
    notch_freq: float,
    notch_method: str,
    bp_enabled: bool,
    bp_lfreq: float,
    bp_hfreq: float,
    bp_method: str,
    resample_enabled: bool,
    resample_target: float,
    pyprep_enabled: bool,
    pyprep_line_noise: float,
    montage_name: str = "standard_1020",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, List[str], float]:
    """
    Returns (parcel_signals, label_names, sfreq).
    parcel_signals shape: (n_parcels, n_timepoints)
    """

    def _step(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    _step("Loading raw file…")
    raw = load_raw(file_bytes, filename, montage_name=montage_name)

    if notch_enabled:
        _step(f"Applying notch filter at {notch_freq} Hz…")
        apply_notch(raw, notch_freq, notch_method)

    if bp_enabled:
        _step(f"Applying bandpass filter {bp_lfreq}–{bp_hfreq} Hz…")
        apply_bandpass(raw, bp_lfreq, bp_hfreq, bp_method)

    if resample_enabled:
        _step(f"Resampling to {resample_target} Hz…")
        apply_resample(raw, resample_target)

    if pyprep_enabled:
        _step("Running PyPrep (bad channel detection)…")
        raw = apply_pyprep(raw, pyprep_line_noise, montage_name=montage_name)

    _step("Setting average reference…")
    raw.set_eeg_reference("average", projection=True, verbose=False)
    raw.apply_proj()

    _step("Computing forward solution…")
    fwd = make_forward_solution(raw)

    _step("Applying LCMV beamformer…")
    stc = apply_lcmv_beamformer(raw, fwd)

    _step("Loading Schaefer 400 parcels…")
    labels = get_schaefer_labels()
    src = mne.read_source_spaces(SRC_ICO5, verbose=False)

    _step("Extracting parcel time courses…")
    parcel_signals = mne.extract_label_time_course(
        stc, labels, src=src, mode="pca_flip", verbose=False
    )
    label_names = [lb.name for lb in labels]

    sfreq = raw.info["sfreq"]
    logger.info(
        "Pipeline done — %d parcels × %d timepoints @ %.0f Hz",
        parcel_signals.shape[0],
        parcel_signals.shape[1],
        sfreq,
    )
    return parcel_signals, label_names, sfreq
