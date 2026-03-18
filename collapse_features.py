
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MIN_ROWS = 40

MAJOR = {
    'Control A': 'Executive Control', 'Control B': 'Executive Control', 'Control C': 'Executive Control',
    'Default A': 'Default Mode',      'Default B': 'Default Mode',      'Default C': 'Default Mode',
    'Dorsal Attention A': 'Dorsal Attention', 'Dorsal Attention B': 'Dorsal Attention',
    'Salience/Ventral Attention A': 'Salience/Ventral Attention',
    'Salience/Ventral Attention B': 'Salience/Ventral Attention',
    'Limbic A': 'Limbic', 'Limbic B': 'Limbic',
    'Somatomotor A': 'Somatomotor', 'Somatomotor B': 'Somatomotor',
    'Visual A': 'Visual', 'Visual B': 'Visual',
    'Temporal Parietal': 'Temporal Parietal',
}

MAJOR_NETWORKS = [
    'Executive Control', 'Default Mode', 'Dorsal Attention',
    'Salience/Ventral Attention', 'Limbic', 'Somatomotor',
    'Visual', 'Temporal Parietal',
]

FEATURE_COLS = [
    'alpha_peak_cf', 'alpha_peak_pw', 'alpha_peak_bw',
    'aperiodic_offset', 'aperiodic_exponent',
    'T1T2ratio', 'G1.fMRI', 'Evolution.Expansion',
    'AllometricScaling.PNC20mm', 'PET.AG', 'CBF',
    'PC1.AHBA', 'PC1.Neurosynth', 'BigBrain.Histology',
    'Cortical.Thickness', 'averagerank.wholebrain',
]

STATS = ['median', 'IQR', 'p5', 'p95']

def _safe_name(net_name: str) -> str:
    return net_name.replace('/', '_').replace(' ', '_')


def _compute_stats(values: np.ndarray) -> dict:
    v = values[~np.isnan(values)]
    n = len(v)
    if n == 0:
        return {s: np.nan for s in STATS}
    if n == 1:
        val = float(v[0])
        return {'median': val, 'IQR': 0.0, 'p5': val, 'p95': val}
    med = float(np.median(v))
    q25 = float(np.percentile(v, 25, method='median_unbiased'))
    q75 = float(np.percentile(v, 75, method='median_unbiased'))
    p5  = float(np.percentile(v,  5, method='median_unbiased'))
    p95 = float(np.percentile(v, 95, method='median_unbiased'))
    return {'median': med, 'IQR': q75 - q25, 'p5': p5, 'p95': p95}


def _build_net_columns(net_label: str, grp: pd.DataFrame, n_total: int) -> dict:
    prefix = f"net8_{_safe_name(net_label)}"
    row = {}
    n_net = len(grp)
    row[f"{prefix}_n"]          = n_net
    row[f"{prefix}_proportion"] = n_net / n_total if n_total > 0 else np.nan
    for col in FEATURE_COLS:
        if col not in grp.columns:
            for stat in STATS:
                row[f"{prefix}_{col}_{stat}"] = np.nan
            continue
        vals = grp[col].values.astype(float)
        for stat_name, stat_val in _compute_stats(vals).items():
            row[f"{prefix}_{col}_{stat_name}"] = stat_val
    return row


def _build_global_columns(df_alpha: pd.DataFrame) -> dict:
    row = {}
    for col in FEATURE_COLS:
        if col not in df_alpha.columns:
            for stat in STATS:
                row[f"global_{col}_{stat}"] = np.nan
            continue
        vals = df_alpha[col].values.astype(float)
        for stat_name, stat_val in _compute_stats(vals).items():
            row[f"global_{col}_{stat_name}"] = stat_val
    return row


def collapse_to_vector(feature_df: pd.DataFrame) -> pd.DataFrame:

    required = {'alpha_peak_index', 'alpha_peak_cf', 'network'}
    missing  = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"Feature DataFrame is missing required columns: {missing}")

    missing_feat = [c for c in FEATURE_COLS if c not in feature_df.columns]
    if missing_feat:
        raise ValueError(
            f"Feature DataFrame is missing gradient/feature columns: {missing_feat}\n"
            "Make sure the S-A axis CSV was merged correctly during extraction."
        )


    df_alpha = feature_df[
        (feature_df['alpha_peak_index'] == 1) &
        (feature_df['alpha_peak_cf']    >  0)
    ].copy()

    n_total = len(df_alpha)
    logger.info("After alpha filter: %d rows (need >= %d)", n_total, MIN_ROWS)

    if n_total < MIN_ROWS:
        raise ValueError(
            f"Only {n_total} parcels with a valid primary alpha peak detected "
            f"(minimum required: {MIN_ROWS}). "
            "The recording may be too noisy or too short for reliable feature extraction."
        )

    df_alpha['major_network'] = df_alpha['network'].map(MAJOR)

    unmapped = df_alpha['major_network'].isna().sum()
    if unmapped > 0:
        unknown = df_alpha.loc[df_alpha['major_network'].isna(), 'network'].unique().tolist()
        logger.warning("%d parcels have unrecognised network labels: %s", unmapped, unknown)

    out = {'n_total': n_total}

    for net in MAJOR_NETWORKS:
        grp = df_alpha[df_alpha['major_network'] == net]
        out.update(_build_net_columns(net, grp, n_total))

    out.update(_build_global_columns(df_alpha))

    vector_df = pd.DataFrame([out])

    logger.info(
        "Collapsed to vector: %d feature columns",
        len(vector_df.columns),
    )
    return vector_df