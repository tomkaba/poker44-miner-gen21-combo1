#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Dict, Sequence


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _linear_ramp(value: float, start: float, end: float) -> float:
    if end <= start:
        return 1.0 if value >= end else 0.0
    if value <= start:
        return 0.0
    if value >= end:
        return 1.0
    return (value - start) / (end - start)


def _inverse_ramp(value: float, good: float, bad: float) -> float:
    return 1.0 - _linear_ramp(value, good, bad)


def _alignment_score(value: float, target: float, contrast: float) -> float:
    scale = max(abs(contrast - target), 1e-9)
    return _clamp01(1.0 - abs(value - target) / scale)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def load_model(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_chunk_row(row: Dict[str, object], model: Dict[str, object]) -> Dict[str, object]:
    impossible_rate = _safe_float(row.get("preflop_impossible_hand_rate"))
    feasible_rate = _safe_float(row.get("feasible_preflop_hand_rate"), 1.0)
    log_surprise = _safe_float(row.get("mean_preflop_log_surprise_per_hand"))
    mean_bucket_consistency = _safe_float(row.get("mean_hand_bucket_consistency"))
    std_bucket_consistency = _safe_float(row.get("std_hand_bucket_consistency"))
    profile_dispersion = _safe_float(row.get("mean_hand_profile_dispersion"))
    std_cheap_fold = _safe_float(row.get("std_hand_cheap_fold_rate"))
    std_expensive_call = _safe_float(row.get("std_hand_expensive_call_rate"))
    std_passive_checked = _safe_float(row.get("std_hand_passive_when_checked_to_rate"))

    impossible_good = _safe_float(model.get("impossible_good_rate"), 0.18)
    impossible_fade = _safe_float(model.get("impossible_fade_rate"), 0.34)
    feasible_start = _safe_float(model.get("feasible_start_rate"), 0.66)
    feasible_full = _safe_float(model.get("feasible_full_rate"), 0.76)
    surprise_good = _safe_float(model.get("surprise_good"), 1.5)
    surprise_bad = _safe_float(model.get("surprise_bad"), 2.8)
    consistency_target = _safe_float(model.get("consistency_target"), 0.35)
    consistency_contrast = _safe_float(model.get("consistency_contrast"), 0.46)
    std_consistency_target = _safe_float(model.get("std_consistency_target"), 0.40)
    std_consistency_contrast = _safe_float(model.get("std_consistency_contrast"), 0.36)
    dispersion_target = _safe_float(model.get("dispersion_target"), 0.22)
    dispersion_contrast = _safe_float(model.get("dispersion_contrast"), 0.10)
    edge_std_target = _safe_float(model.get("edge_std_target"), 0.32)
    edge_std_contrast = _safe_float(model.get("edge_std_contrast"), 0.03)

    impossible_component = _inverse_ramp(impossible_rate, impossible_good, impossible_fade)
    feasible_component = _linear_ramp(feasible_rate, feasible_start, feasible_full)
    surprise_component = _inverse_ramp(log_surprise, surprise_good, surprise_bad)
    consistency_component = _alignment_score(mean_bucket_consistency, consistency_target, consistency_contrast)
    stability_component = _mean(
        [
            _alignment_score(std_bucket_consistency, std_consistency_target, std_consistency_contrast),
            _alignment_score(profile_dispersion, dispersion_target, dispersion_contrast),
        ]
    )
    edge_component = _mean(
        [
            _alignment_score(std_cheap_fold, edge_std_target, edge_std_contrast),
            _alignment_score(std_expensive_call, edge_std_target, edge_std_contrast),
            _alignment_score(std_passive_checked, edge_std_target, edge_std_contrast),
        ]
    )

    weights = model.get("weights") or {}
    boost = _clamp01(
        _safe_float(weights.get("impossible_component"), 0.24) * impossible_component
        + _safe_float(weights.get("feasible_component"), 0.20) * feasible_component
        + _safe_float(weights.get("surprise_component"), 0.12) * surprise_component
        + _safe_float(weights.get("consistency_component"), 0.20) * consistency_component
        + _safe_float(weights.get("stability_component"), 0.14) * stability_component
        + _safe_float(weights.get("edge_component"), 0.10) * edge_component
    )

    if boost >= _safe_float(model.get("high_boost_threshold"), 0.72):
        band = "high"
    elif boost >= _safe_float(model.get("medium_boost_threshold"), 0.45):
        band = "medium"
    else:
        band = "low"

    return {
        "gen17_math_1_pf_boost": boost,
        "gen17_math_1_pf_boost_band": band,
        "gen17_math_1_pf_boost_impossible_component": impossible_component,
        "gen17_math_1_pf_boost_feasible_component": feasible_component,
        "gen17_math_1_pf_boost_surprise_component": surprise_component,
        "gen17_math_1_pf_boost_consistency_component": consistency_component,
        "gen17_math_1_pf_boost_stability_component": stability_component,
        "gen17_math_1_pf_boost_edge_component": edge_component,
    }


def apply_boost(base_score: float, boost: float, boost_weight: float = 0.25) -> float:
    return _clamp01(base_score + (1.0 - base_score) * boost_weight * boost)


def summarize_distribution(rows: Sequence[Dict[str, object]], field: str) -> Dict[str, float]:
    values = [_safe_float(row.get(field)) for row in rows]
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p10": _quantile(values, 0.10),
        "p25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "p75": _quantile(values, 0.75),
        "p90": _quantile(values, 0.90),
    }


def evaluate_boost_model(rows: Sequence[Dict[str, object]], model: Dict[str, object], boost_threshold: float) -> Dict[str, float]:
    bot_rows = [row for row in rows if int(row.get("truth_value", 0)) == 1]
    human_rows = [row for row in rows if int(row.get("truth_value", 0)) == 0]
    bot_good = sum(1 for row in bot_rows if score_chunk_row(row, model)["gen17_math_1_pf_boost"] >= boost_threshold)
    human_good = sum(1 for row in human_rows if score_chunk_row(row, model)["gen17_math_1_pf_boost"] < boost_threshold)
    bot_recall = bot_good / max(len(bot_rows), 1)
    human_recall = human_good / max(len(human_rows), 1)
    return {
        "boost_threshold": boost_threshold,
        "bot_recall": bot_recall,
        "human_recall": human_recall,
        "balanced_accuracy": 0.5 * (bot_recall + human_recall),
    }