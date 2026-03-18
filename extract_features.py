import logging
import os
import re

import h5py
import numpy as np
import pandas as pd
from specparam import SpectralModel

from features.psd import compute_psd
from config import AlphaAnalysisConfig, S_A_AXIS_RANKS
from utils.label_utils import parse_label_name
import specparam as _specparam

logger = logging.getLogger(__name__)

_SPECPARAM_V2 = int(_specparam.__version__.split('.')[0]) >= 2
logger.debug("specparam version: %s (v2=%s)", _specparam.__version__, _SPECPARAM_V2)


_DESIRED_ORDER = [
    "subject_id",
    "network", "parcel", "label",
    "alpha_peak_index", "alpha_peak_cf", "alpha_peak_pw", "alpha_peak_bw",
    "aperiodic_offset", "aperiodic_exponent", "r_squared", "fit_error",
    "T1T2ratio", "G1.fMRI", "Evolution.Expansion", "AllometricScaling.PNC20mm",
    "PET.AG", "CBF", "PC1.AHBA", "PC1.Neurosynth", "BigBrain.Histology",
    "Cortical.Thickness", "finalrank.wholebrain", "averagerank.hemisphere",
    "finalrank.hemisphere", "averagerank.wholebrain",
]

_GRADIENT_COLS = [
    "T1T2ratio", "G1.fMRI", "Evolution.Expansion", "AllometricScaling.PNC20mm",
    "PET.AG", "CBF", "PC1.AHBA", "PC1.Neurosynth", "BigBrain.Histology",
    "Cortical.Thickness", "finalrank.wholebrain", "averagerank.hemisphere",
    "finalrank.hemisphere", "averagerank.wholebrain",
]


def debug_hdf5(h5_path: str) -> dict:

    logger.info("=" * 60)
    logger.info("HDF5 DEBUG: %s", h5_path)
    logger.info("=" * 60)

    info = {"path": h5_path, "datasets": {}, "signals_shape": None,
            "label_sample": [], "sfreq_found": None}

    with h5py.File(h5_path, "r") as f:
        logger.info("Top-level keys: %s", list(f.keys()))

        for key in f.keys():
            ds = f[key]
            if hasattr(ds, "shape"):
                dtype = str(ds.dtype)
                shape = ds.shape
                info["datasets"][key] = {"shape": shape, "dtype": dtype}
                logger.info("  [%s]  shape=%s  dtype=%s", key, shape, dtype)

                if ds.size == 1:
                    val = ds[()]
                    logger.info("    value: %s", val)
                    info["sfreq_found"] = val
                elif ds.size <= 10:
                    logger.info("    values: %s", ds[:])

        if "label_tcs" in f:
            sig = f["label_tcs"][:]
            info["signals_shape"] = sig.shape
            n0, n1 = sig.shape
            logger.info("")
            logger.info("label_tcs shape: %s", sig.shape)
            if n0 < n1:
                logger.info("  -> interpretation: (%d parcels, %d timepoints)", n0, n1)
            else:
                logger.info("  -> interpretation: (%d timepoints, %d parcels) — will be transposed", n0, n1)
            logger.info("  min=%.4f  max=%.4f  mean=%.4f  std=%.4f",
                        sig.min(), sig.max(), sig.mean(), sig.std())
            logger.info("  any NaN: %s  any Inf: %s",
                        np.isnan(sig).any(), np.isinf(sig).any())

        if "label_names" in f:
            raw = f["label_names"][:]
            decoded = [
                x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
                for x in raw
            ]
            info["label_sample"] = decoded[:5]
            logger.info("")
            logger.info("label_names: %d labels", len(decoded))
            logger.info("  first 5: %s", decoded[:5])
            logger.info("  last  5: %s", decoded[-5:])

    logger.info("=" * 60)
    return info


def debug_psd(psd: np.ndarray, freqs: np.ndarray,
              label_names: list, n_samples: int = 5,
              save_path: str | None = None):

    logger.info("")
    logger.info("PSD DEBUG")
    logger.info("  psd shape  : %s  (should be n_parcels x n_freqs)", psd.shape)
    logger.info("  freqs shape: %s", freqs.shape)
    logger.info("  freq range : %.2f - %.2f Hz", freqs.min(), freqs.max())
    logger.info("  freq resolution: %.4f Hz", freqs[1] - freqs[0])
    logger.info("  psd min=%.6f  max=%.6f  mean=%.6f", psd.min(), psd.max(), psd.mean())
    logger.info("  any NaN: %s  any Inf: %s",
                np.isnan(psd).any(), np.isinf(psd).any())

    alpha_mask = (freqs >= 7) & (freqs <= 14)
    alpha_psd  = psd[:, alpha_mask]
    logger.info("  alpha band (7-14 Hz): %d freq bins", alpha_mask.sum())
    logger.info("  alpha psd mean=%.6f  max=%.6f", alpha_psd.mean(), alpha_psd.max())

    alpha_power = alpha_psd.mean(axis=1)
    top5_idx    = np.argsort(alpha_power)[-5:][::-1]
    logger.info("  top 5 parcels by mean alpha power:")
    for idx in top5_idx:
        name = label_names[idx] if idx < len(label_names) else f"parcel_{idx}"
        logger.info("    [%d] %s  alpha_power=%.6f", idx, name, alpha_power[idx])

    if save_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            indices = np.linspace(0, len(psd) - 1, min(n_samples, len(psd)), dtype=int)
            fig, axes = plt.subplots(n_samples, 1, figsize=(10, 3 * n_samples), sharex=True)
            if n_samples == 1:
                axes = [axes]

            for ax, idx in zip(axes, indices):
                name = label_names[idx] if idx < len(label_names) else f"parcel_{idx}"
                ax.semilogy(freqs, psd[idx], color="#2166ac", lw=1.5)
                ax.axvspan(7, 14, alpha=0.15, color="orange", label="alpha (7-14 Hz)")
                ax.set_ylabel("Power")
                ax.set_title(f"Parcel {idx}: {name}", fontsize=9)
                ax.legend(fontsize=7, frameon=False)
                ax.spines[["top", "right"]].set_visible(False)

            axes[-1].set_xlabel("Frequency (Hz)")
            fig.suptitle("PSD Debug — sample parcels", fontsize=11, y=1.01)
            plt.tight_layout()
            plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            logger.info("  PSD plot saved: %s", save_path)
        except Exception as e:
            logger.warning("  PSD plot failed: %s", e)


def clean_label(label: str) -> str:
    """'17Networks_LH_ContA_Cingm_1-lh'  ->  'lh_17Networks_LH_ContA_Cingm_1'"""
    return re.sub(r'(.+)-(lh|rh)$', r'\2_\1', label)


def load_fraser_hdf5(h5_path: str):

    with h5py.File(h5_path, "r") as f:
        signals         = f["label_tcs"][:]
        raw_label_names = f["label_names"][:]

    label_names = [
        x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else str(x)
        for x in raw_label_names
    ]

    base_name  = os.path.basename(h5_path)
    stem       = base_name.replace(".hdf5", "").replace(".h5", "")
    subject_id = stem.split("-ico-")[0] if "-ico-" in stem else stem

    metadata = {"subject_id": subject_id}
    return signals, label_names, metadata


def _resolve_sa_path(sa_csv_path: str | None) -> str:
    """Return the SA axis CSV path, searching in standard locations."""
    if sa_csv_path:
        return sa_csv_path

    _here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(_here, "S-A_Axis", "schaefer400_sa_axis_merged.csv")
    if os.path.exists(candidate):
        return candidate

    if os.path.exists(S_A_AXIS_RANKS):
        logger.warning("Using fallback SA path from config: %s", S_A_AXIS_RANKS)
        return S_A_AXIS_RANKS

    raise FileNotFoundError(
        f"S-A axis CSV not found.\n"
        f"Expected: {candidate}\n"
        f"Make sure schaefer400_sa_axis_merged.csv is in the S-A_Axis/ folder."
    )


def run_pipeline(
    h5_path,
    config=None,
    sa_csv_path=None,
    save_csv_path=None,
    progress_callback=None,
    debug=False,
    debug_plot_path=None,
):

    if config is None:
        config = AlphaAnalysisConfig()

    sa_path = _resolve_sa_path(sa_csv_path)
    logger.info("SA axis CSV: %s", sa_path)

    if debug:
        debug_hdf5(h5_path)

    signals, label_names, metadata = load_fraser_hdf5(h5_path)
    n_parcels = len(label_names)

    if debug:
        logger.info("After load_fraser_hdf5:")
        logger.info("  signals.shape : %s", signals.shape)
        logger.info("  n_parcels     : %d", n_parcels)
        logger.info("  config.sfreq  : %s", config.sfreq)

    logger.info("Computing PSD for %d parcels ...", n_parcels)
    psd, freqs = compute_psd(signals, config)

    if debug:
        clean_names = [clean_label(l) for l in label_names]
        debug_psd(psd, freqs, clean_names, n_samples=5, save_path=debug_plot_path)

    logger.info("Fitting SpectralModel (alpha %d-%d Hz) ...",
                config.alpha_start, config.alpha_end)

    all_rows     = []
    n_no_peaks   = 0
    n_no_alpha   = 0
    n_fit_failed = 0

    for i, roi_psd in enumerate(psd):
        roi_label = clean_label(label_names[i])
        parsed    = parse_label_name(roi_label)
        network   = parsed["network_name"]
        parcel    = parsed["parcel_name"]

        model = SpectralModel(
            peak_width_limits = config.peak_width_limits,
            max_n_peaks       = config.max_n_peaks,
            min_peak_height   = config.min_peak_height,
            aperiodic_mode    = config.aperiodic_mode,
            peak_threshold    = config.peak_threshold,
            verbose           = False,
        )

        try:
            if _SPECPARAM_V2:
                model.fit(freqs, roi_psd, [freqs[0], freqs[-1]])
                ap_params   = model.results.params.aperiodic.params
                peak_params = model.results.params.periodic.params
                r_squared   = model.results.metrics.results['gof_rsquared']
                fit_error   = model.results.metrics.results['error_mae']
            else:
                model.fit(freqs, roi_psd)
                ap_params, peak_params, r_squared, fit_error, _ = model.get_results()
        except Exception as e:
            n_fit_failed += 1
            if n_fit_failed <= 3:
                logger.warning("  [%d] %s — fit FAILED: %s: %s",
                               i, roi_label, type(e).__name__, e)
            elif debug:
                logger.warning("  [%d] %s — fit FAILED: %s", i, roi_label, e)
            if progress_callback:
                progress_callback(i + 1, n_parcels)
            continue

        if peak_params is None:
            peak_list = []
        elif isinstance(peak_params, np.ndarray):
            peak_list = [tuple(row) for row in peak_params]
        else:
            peak_list = list(peak_params)

        if len(peak_list) == 0:
            n_no_peaks += 1
            if progress_callback:
                progress_callback(i + 1, n_parcels)
            continue

        alpha_peaks = [
            (cf, pw, bw)
            for cf, pw, bw in peak_list
            if config.alpha_start <= cf <= config.alpha_end
        ]

        if not alpha_peaks:
            n_no_alpha += 1
            if debug and i < 10:
                all_cfs = [cf for cf, pw, bw in peak_list]
                logger.info("  [%d] %s — peaks found but NONE in alpha range %d-%d Hz. "
                            "All peak CFs: %s",
                            i, roi_label, config.alpha_start, config.alpha_end,
                            [f"{cf:.2f}" for cf in all_cfs])
            if progress_callback:
                progress_callback(i + 1, n_parcels)
            continue

        for alpha_index, (cf, pw, bw) in enumerate(alpha_peaks, start=1):
            row = {
                "label":              roi_label,
                "network":            network,
                "parcel":             parcel,
                "alpha_peak_index":   alpha_index,
                "alpha_peak_cf":      cf,
                "alpha_peak_pw":      pw,
                "alpha_peak_bw":      bw,
                "aperiodic_offset":   ap_params[0] if len(ap_params) > 0 else np.nan,
                "aperiodic_exponent": ap_params[1] if len(ap_params) > 1 else np.nan,
                "r_squared":          r_squared,
                "fit_error":          fit_error,
            }
            row.update(metadata)
            all_rows.append(row)

        if progress_callback:
            progress_callback(i + 1, n_parcels)

    # -- Specparam summary ----------------------------------------------------
    logger.info("Specparam summary:")
    logger.info("  total parcels             : %d", n_parcels)
    logger.info("  fit failed                : %d", n_fit_failed)
    logger.info("  no peaks at all           : %d", n_no_peaks)
    logger.info("  peaks outside alpha range : %d", n_no_alpha)
    logger.info("  rows collected            : %d", len(all_rows))

    if not all_rows:
        logger.warning("No alpha peaks found in %s", h5_path)
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    sa_df = pd.read_csv(sa_path)
    logger.info("SA CSV columns: %s", list(sa_df.columns))

    sa_cols = [c for c in sa_df.columns if c not in ("row_index", "name")]

    already_present = [c for c in _GRADIENT_COLS if c in df.columns]
    if already_present:
        logger.warning("Dropping pre-existing gradient cols before merge: %s", already_present)
        df = df.drop(columns=already_present)

    df = df.merge(sa_df[sa_cols], on="label", how="left")

    n_matched = df["averagerank.wholebrain"].notna().sum()
    logger.info("Gradient merge: %d/%d rows matched", n_matched, len(df))

    if n_matched == 0:
        logger.warning("No rows matched in gradient merge — check label format in SA CSV vs HDF5")

    ordered = [c for c in _DESIRED_ORDER if c in df.columns]
    rest    = [c for c in df.columns if c not in _DESIRED_ORDER]
    df      = df[ordered + rest]

    logger.info("Output columns (%d): %s", len(df.columns), list(df.columns))

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or ".", exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        logger.info("Saved: %s", save_csv_path)

    logger.info("Done — %d rows, %d parcels with alpha peaks",
                len(df), df["label"].nunique())
    return df