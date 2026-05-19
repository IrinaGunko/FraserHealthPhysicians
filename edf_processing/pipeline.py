
import logging
import os
import os.path as op
import sys
import tempfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import mne
import numpy as np

# ── add Moiseev's source dirs to path ─────────────────────────────────────────
_HERE      = op.dirname(op.abspath(__file__))
_AMOISEEV  = op.abspath(op.join(_HERE, '..', 'amoiseev'))
_BEAM_PY   = op.abspath(op.join(_HERE, '..', 'beam-python'))
for _p in (_AMOISEEV, _BEAM_PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from amoiseev.edf_preprocessing import PreProcessing
from amoiseev.do_src_reconstr import (
    compute_beamformer_stc,
    compute_roi_time_courses,
)

from config import BEM, SUBJECTS_DIR, SRC_ICO5, TRANS  # noqa: E402

logger = logging.getLogger(__name__)

PARC_NAME = "Schaefer2018_400Parcels_17Networks_order"


# ── configuration helpers ──────────────────────────────────────────────────────

def _preproc_conf(notch_freq: float, bp_lfreq: float, bp_hfreq: float,
                  resample_target: float) -> dict:

    return {
        "target_channels": [],          # empty → no mandatory channels; accept any EDF
        "opt_channels": [
            "ECG1", "ECG2", "EKG", "EKG1", "EKG2",
            "EOG 1", "EOG 2", "EOG1", "EOG2", "L EOG", "R EOG", "PG1", "PG2",
            "A1", "A2",
        ],
        "eog_channels": ["EOG 1", "EOG 2", "EOG1", "EOG2", "L EOG", "R EOG", "PG1", "PG2"],
        "ecg_channels": ["ECG1", "ECG2", "EKG", "EKG1", "EKG2"],
        "exclude_channels": [
            "AUX1", "AUX2", "AUX3", "AUX4", "AUX5", "AUX6", "AUX7", "AUX8",
            "DC1",  "DC2",  "DC3",  "DC4",  "DIF1", "DIF2", "DIF3", "DIF4",
            "Patient Event", "Photic", "Trigger Event", "x1", "x2",
            "PATIENT EVENT", "PHOTIC", "TRIGGER EVENT", "X1", "X2",
        ],
        "rename_channels": {"L EOG": "EOG1", "R EOG": "EOG2", "EKG": "EKG1"},
        "print_opt_channels": False,
        "discard_at_start_seconds": 0,
        "target_frequency":  resample_target,
        "target_band":       [bp_lfreq, bp_hfreq],
        "powerline_frq":     notch_freq,
        "allow_upsampling":  True,
        "max_rec_length":    7200,
        "target_segments":   1,
        "target_length":     7200,
        "flat_parms": {
            "flat_max_ptp": 1e-6,
            "bad_percent":  50.0,
            "min_duration": 10.0,
        },
        "HV_regexp":          r"H.*V.*\d+\s*[MmIiNn]{3}",
        "HV_end":             "END HV",
        "hv_pad_interval":    30,
        "photic_starts":      ["Hz"],
        "photic_ends":        ["Off"],
        "photic_pad_interval": 30,
        "max_isi":            360,
    }


# ── internal helpers ───────────────────────────────────────────────────────────

def _make_forward(raw: mne.io.Raw) -> mne.Forward:
    trans = mne.read_trans(TRANS, verbose=False)
    src   = mne.read_source_spaces(SRC_ICO5, verbose=False)
    bem   = mne.read_bem_solution(BEM, verbose=False)
    fwd = mne.make_forward_solution(
        raw.info, trans=trans, src=src, bem=bem,
        eeg=True, meg=False, ignore_ref=True, mindist=5.0, verbose=False,
    )
    logger.info("Forward solution: %d sources", fwd['nsource'])
    return fwd


def _apply_pyprep_ica(raw: mne.io.Raw, line_noise: float,
                      montage_name: str) -> mne.io.Raw:
    from pyprep.prep_pipeline import PrepPipeline as _PrepPipeline
    from mne.preprocessing import ICA

    montage = mne.channels.make_standard_montage(montage_name)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage(montage, on_missing='warn', verbose=False)

    sample_rate = raw.info['sfreq']
    line_freqs  = np.arange(line_noise, sample_rate / 2, line_noise)
    prep_params = {
        "ref_chs": "eeg", "reref_chs": "eeg",
        "line_freqs": line_freqs,
        "max_iterations": 4,
    }
    prep = _PrepPipeline(
        raw, prep_params, montage,
        ransac=True, channel_wise=False, max_chunk_size=None,
        random_state=12345, filter_kwargs={"method": "fir"}, matlab_strict=False,
    )
    prep.fit()
    raw = prep.raw
    logger.info("PyPREP — interpolated: %s", prep.interpolated_channels)

    ica = ICA(n_components=0.99999, random_state=12345, method='fastica',
              max_iter='auto', verbose=False)
    ica.fit(raw, picks='eeg', tstep=2.0, verbose=False)

    eog_chs = [ch for ch in raw.ch_names
               if raw.get_channel_types([ch])[0] == 'eog']
    ecg_chs = [ch for ch in raw.ch_names
               if raw.get_channel_types([ch])[0] == 'ecg']

    if eog_chs:
        try:
            eog_idx, _ = ica.find_bads_eog(raw, measure='zscore', threshold=3.0, verbose=False)
            ica.exclude += eog_idx
            logger.info("ICA — EOG components excluded: %s", eog_idx)
        except Exception as exc:
            logger.warning("ICA EOG detection failed: %s", exc)

    if ecg_chs:
        try:
            ecg_idx, _ = ica.find_bads_ecg(raw, method='correlation', measure='zscore',
                                            threshold=3.0, verbose=False)
            ica.exclude += ecg_idx
            logger.info("ICA — ECG components excluded: %s", ecg_idx)
        except Exception as exc:
            logger.warning("ICA ECG detection failed: %s", exc)

    ica.apply(raw, verbose=False)
    return raw


# ── public entry point ─────────────────────────────────────────────────────────

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

    def _step(msg: str):
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    # ── 1. write bytes to temp file ────────────────────────────────────────────
    suffix = Path(filename).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(file_bytes)
    tmp.close()
    tmp_path = tmp.name

    try:
        # ── 2. Moiseev PreProcessing: notch → bandpass → resample ─────────────
        _step("Loading raw file…")
        eff_notch = notch_freq     if notch_enabled   else 10_000.0
        eff_lfreq = bp_lfreq       if bp_enabled       else 0.1
        eff_hfreq = bp_hfreq       if bp_enabled       else 200.0
        eff_sfreq = resample_target if resample_enabled else 256.0

        conf = _preproc_conf(eff_notch, eff_lfreq, eff_hfreq, eff_sfreq)
        pp   = PreProcessing(tmp_path, conf_dict=conf)

        if pp.skip_it:
            raise ValueError(
                "EDF was rejected by Moiseev preprocessing "
                "(recording too long or sampling rate too low)."
            )

        raw_seg = pp.raw

        montage     = mne.channels.make_standard_montage(montage_name)
        montage_set = set(montage.ch_names)
        drop = [ch for ch in raw_seg.ch_names if ch not in montage_set]
        if drop:
            logger.info("Dropping %d non-montage channels: %s", len(drop), drop)
            raw_seg.drop_channels(drop)
        raw_seg.set_montage(montage, on_missing='warn', verbose=False)

        if len(raw_seg.ch_names) == 0:
            raise ValueError(
                "No channels matched the montage. "
                "Try a different montage or check the EDF channel names."
            )

        # ── 5. optional PyPREP + ICA (Moiseev's Pipeline logic) ───────────────
        if pyprep_enabled:
            _step("Running PyPrep + ICA…")
            raw_seg = _apply_pyprep_ica(raw_seg, pyprep_line_noise, montage_name)

        # ── 6. average reference ───────────────────────────────────────────────
        _step("Setting average reference…")
        raw_seg.set_eeg_reference("average", projection=True, verbose=False)
        raw_seg.apply_proj()

        # ── 7. forward solution ────────────────────────────────────────────────
        _step("Computing forward solution…")
        fwd = _make_forward(raw_seg)

        # ── 8. Moiseev beamformer weights (no full STC needed) ─────────────────
        _step("Applying Moiseev beamformer…")
        _, data_cov, W, _U, pz = compute_beamformer_stc(raw_seg, fwd, return_stc=False)
        logger.info("Beamformer: pz=%.4f  W=%s", pz, W.shape)

        # ── 9. load Schaefer parcels, restrict to forward source space ─────────
        _step("Loading Schaefer 400 parcels…")
        all_labels = mne.read_labels_from_annot(
            "fsaverage", parc=PARC_NAME,
            subjects_dir=SUBJECTS_DIR, verbose=False,
        )
        labels = [
            lb.restrict(fwd['src'])
            for lb in all_labels
            if "unknown" not in lb.name.lower()
        ]
        labels = [lb for lb in labels if len(lb.vertices)]
        logger.info("Schaefer labels after restriction: %d", len(labels))

        # ── 10. label time courses: W_label.T @ sensor_data ───────────────────
        #  This is done inside beam_extract_label_time_course (called by compute_roi_time_courses)
        _step("Extracting parcel time courses…")
        eeg_data = raw_seg.get_data(picks='eeg')   # (nchans, ntimes)
        label_tcs, _ = compute_roi_time_courses(
            'beam', labels, fwd,
            sensor_data=eeg_data,
            cov=data_cov,
            W=W,
        )
        label_names = [lb.name for lb in labels]
        sfreq = raw_seg.info["sfreq"]

        logger.info(
            "Pipeline done — %d labels × %d timepoints @ %.0f Hz",
            label_tcs.shape[0], label_tcs.shape[1], sfreq,
        )
        return label_tcs, label_names, sfreq

    finally:
        os.unlink(tmp_path)
