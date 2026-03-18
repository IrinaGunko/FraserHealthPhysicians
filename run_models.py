

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

logger = logging.getLogger(__name__)

ALL_PHYSICIANS = ["Sophia", "Eleni", "Athina", "Maria", "Zoe"]
ALL_TARGETS    = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
LABEL_MODES    = ["strict", "relaxed"]

MODEL_OUTPUTS_DIR = Path(__file__).parent / "model_outputs"


def _model_id(physician: str, label_mode: str, target: str) -> str:
    t = target.replace(" ", "_").replace("-", "_")
    return (f"general__{label_mode}__{t}" if physician == "general"
            else f"physician__{physician}__{label_mode}__{t}")


def _expected_model_ids() -> list[str]:
    ids = []
    for lm in LABEL_MODES:
        for t in ALL_TARGETS:
            ids.append(_model_id("general", lm, t))
    for p in ALL_PHYSICIANS:
        for lm in LABEL_MODES:
            ids.append(_model_id(p, lm, "Abnormality"))
    return ids  # 20 total


def _prob_to_score(prob: float) -> int:

    if prob < 0.25:
        return 1
    elif prob < 0.50:
        return 2
    elif prob < 0.75:
        return 3
    else:
        return 4


def discover_models(model_root: Optional[Path] = None) -> Dict[str, Path]:

    root = model_root or MODEL_OUTPUTS_DIR
    found = {}
    if not root.exists():
        logger.warning("model_outputs/ not found at %s", root)
        return found
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / "model.json").exists() and (d / "metadata.json").exists():
            found[d.name] = d
    logger.info("Discovered %d models in %s", len(found), root)
    return found


def load_model(folder: Path) -> tuple[xgb.XGBClassifier, dict]:
    with open(folder / "metadata.json") as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(str(folder / "model.json"))
    return model, meta


def run_single_model(
    model: xgb.XGBClassifier,
    meta: dict,
    params_df: pd.DataFrame,
) -> dict:

    final_feats = meta["final_features"]
    threshold   = float(meta["best_threshold"])
    cv_metrics  = meta.get("cv_metrics", {})

    # Align columns — use NaN for any missing feature
    missing = [f for f in final_feats if f not in params_df.columns]
    if missing:
        logger.warning("Model %s: %d missing features, filling with NaN",
                       meta["model_id"], len(missing))

    X = pd.DataFrame(
        {f: params_df[f].values if f in params_df.columns else [np.nan]
         for f in final_feats}
    )

    prob  = float(model.predict_proba(X)[0, 1])
    binary = int(prob >= threshold)
    score  = _prob_to_score(prob)

    return {
        "probability":      round(prob, 4),
        "binary":           binary,
        "threshold":        round(threshold, 4),
        "score":            score,
        "n_features_used":  len(final_feats),
        "n_features_avail": len(final_feats) - len(missing),
        "missing_features": missing,
        "cv_metrics":       cv_metrics,
    }


def run_all_models(
    params_df: pd.DataFrame,
    model_root: Optional[Path] = None,
) -> Dict[str, dict]:

    available = discover_models(model_root)
    expected  = _expected_model_ids()
    output    = {}

    for mid in expected:
        if mid not in available:
            output[mid] = {"available": False, "result": None, "meta": None}
            logger.warning("Model not found: %s", mid)
            continue

        try:
            model, meta = load_model(available[mid])
            result = run_single_model(model, meta, params_df)
            output[mid] = {
                "available": True,
                "result":    result,
                "meta":      meta,
            }
            logger.info(
                "%s  prob=%.3f  score=%d  binary=%d  (thr=%.2f)",
                mid, result["probability"], result["score"],
                result["binary"], result["threshold"],
            )
        except Exception as e:
            logger.error("Error running model %s: %s", mid, e)
            output[mid] = {
                "available": False,
                "result":    None,
                "meta":      None,
                "error":     str(e),
            }

    n_ok  = sum(1 for v in output.values() if v["available"] and v["result"])
    n_miss = sum(1 for v in output.values() if not v["available"])
    logger.info("Inference complete: %d ok, %d missing/failed", n_ok, n_miss)
    return output


def results_to_dataframe(all_results: Dict[str, dict]) -> pd.DataFrame:

    rows = []
    for mid, entry in all_results.items():
        row = {"model_id": mid, "available": entry["available"]}
        if entry["available"] and entry["result"]:
            row.update(entry["result"])
            row.pop("missing_features", None)   # list — not CSV-friendly
            row.pop("cv_metrics", None)
            # flatten cv_metrics
            for k, v in (entry["result"].get("cv_metrics") or {}).items():
                row[f"cv_{k}"] = v
            # flatten meta
            for k in ("physician", "label_mode", "target", "n_samples",
                      "n_features_final", "best_threshold"):
                row[k] = entry.get("meta", {}).get(k, "")
        rows.append(row)
    return pd.DataFrame(rows)


def get_view(
    all_results: Dict[str, dict],
    physician: str,     # "general" or physician name
    label_mode: str,    # "strict" or "relaxed"
) -> Dict[str, dict]:

    view = {}
    for target in ALL_TARGETS:
        mid = _model_id(physician, label_mode, target)
        view[target] = all_results.get(mid, {"available": False, "result": None, "meta": None})
    return view