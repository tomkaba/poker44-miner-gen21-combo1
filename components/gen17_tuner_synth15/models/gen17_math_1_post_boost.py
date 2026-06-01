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


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_model(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_model(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def score_chunk_row(row: Dict[str, object], model: Dict[str, object]) -> Dict[str, object]:
    turn_like = _safe_float(row.get("turn_barrel_like_rate"))
    river_like = _safe_float(row.get("river_barrel_like_rate"))
    river_miss = _safe_float(row.get("river_barrel_miss_rate"))
    river_aggression = _safe_float(row.get("river_aggression_rate"))
    expensive_call = _safe_float(row.get("expensive_call_rate"))
    bucket_consistency = _safe_float(row.get("mean_hand_bucket_consistency"))
    std_bucket_consistency = _safe_float(row.get("std_hand_bucket_consistency"))
    profile_dispersion = _safe_float(row.get("mean_hand_profile_dispersion"))
    action_entropy = _safe_float(row.get("mean_hand_action_entropy"))
    std_entropy = _safe_float(row.get("std_hand_action_entropy"))

    barrel_start = _safe_float(model.get("barrel_like_start"), 0.28)
    barrel_full = _safe_float(model.get("barrel_like_full"), 0.52)
    river_miss_good = _safe_float(model.get("river_miss_good"), 0.10)
    river_miss_bad = _safe_float(model.get("river_miss_bad"), 0.36)
    river_aggr_start = _safe_float(model.get("river_aggression_start"), 0.12)
    river_aggr_full = _safe_float(model.get("river_aggression_full"), 0.32)
    expensive_call_good = _safe_float(model.get("expensive_call_good"), 0.02)
    expensive_call_bad = _safe_float(model.get("expensive_call_bad"), 0.12)
    consistency_target = _safe_float(model.get("consistency_target"), 0.36)
    consistency_contrast = _safe_float(model.get("consistency_contrast"), 0.46)
    std_consistency_target = _safe_float(model.get("std_consistency_target"), 0.40)
    std_consistency_contrast = _safe_float(model.get("std_consistency_contrast"), 0.45)
    dispersion_target = _safe_float(model.get("dispersion_target"), 0.20)
    dispersion_contrast = _safe_float(model.get("dispersion_contrast"), 0.10)
    mean_entropy_target = _safe_float(model.get("entropy_target"), 0.38)
    mean_entropy_contrast = _safe_float(model.get("entropy_contrast"), 0.45)
    std_entropy_target = _safe_float(model.get("std_entropy_target"), 0.18)
    std_entropy_contrast = _safe_float(model.get("std_entropy_contrast"), 0.08)

    aggression_component = _mean([
        _linear_ramp(turn_like, barrel_start, barrel_full),
        _linear_ramp(river_like, barrel_start, barrel_full),
        _linear_ramp(river_aggression, river_aggr_start, river_aggr_full),
    ])
    miss_component = _mean([
        _inverse_ramp(river_miss, river_miss_good, river_miss_bad),
        _inverse_ramp(expensive_call, expensive_call_good, expensive_call_bad),
    ])
    entropy_component = _inverse_ramp(action_entropy, mean_entropy_target, mean_entropy_contrast)
    consistency_component = _alignment_score(bucket_consistency, consistency_target, consistency_contrast)
    stability_component = _mean([
        _alignment_score(std_bucket_consistency, std_consistency_target, std_consistency_contrast),
        _alignment_score(profile_dispersion, dispersion_target, dispersion_contrast),
        _alignment_score(std_entropy, std_entropy_target, std_entropy_contrast),
    ])

    weights = model.get("weights") or {}
    boost = _clamp01(
        _safe_float(weights.get("aggression_component"), 0.28) * aggression_component
        + _safe_float(weights.get("miss_component"), 0.24) * miss_component
        + _safe_float(weights.get("entropy_component"), 0.14) * entropy_component
        + _safe_float(weights.get("consistency_component"), 0.18) * consistency_component
        + _safe_float(weights.get("stability_component"), 0.16) * stability_component
    )

    if boost >= _safe_float(model.get("high_boost_threshold"), 0.72):
        band = "high"
    elif boost >= _safe_float(model.get("medium_boost_threshold"), 0.45):
        band = "medium"
    else:
        band = "low"

    return {
        "gen17_math_1_post_boost": boost,
        "gen17_math_1_post_boost_band": band,
        "gen17_math_1_post_aggression_component": aggression_component,
        "gen17_math_1_post_miss_component": miss_component,
        "gen17_math_1_post_entropy_component": entropy_component,
        "gen17_math_1_post_consistency_component": consistency_component,
        "gen17_math_1_post_stability_component": stability_component,
    }


def apply_boost(base_score: float, boost: float, boost_weight: float = 0.15) -> float:
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


def evaluate_boost_model(rows: Sequence[Dict[str, object]], model: Dict[str, object], threshold: float) -> Dict[str, float]:
    bot_rows = [row for row in rows if int(row.get("truth_value", 0)) == 1]
    human_rows = [row for row in rows if int(row.get("truth_value", 0)) == 0]
    bot_good = sum(1 for row in bot_rows if score_chunk_row(row, model)["gen17_math_1_post_boost"] >= threshold)
    human_good = sum(1 for row in human_rows if score_chunk_row(row, model)["gen17_math_1_post_boost"] < threshold)
    bot_recall = bot_good / max(len(bot_rows), 1)
    human_recall = human_good / max(len(human_rows), 1)
    return {
        "boost_threshold": threshold,
        "bot_recall": bot_recall,
        "human_recall": human_recall,
        "balanced_accuracy": 0.5 * (bot_recall + human_recall),
    }