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


def _inverse_ramp(value: float, bad: float, good: float) -> float:
    return 1.0 - _linear_ramp(value, bad, good)


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


def score_chunk_row(row: Dict[str, object], model: Dict[str, object]) -> Dict[str, object]:
    turn_like = _safe_float(row.get("turn_barrel_like_rate"))
    river_miss = _safe_float(row.get("river_barrel_miss_rate"))
    river_aggression = _safe_float(row.get("river_aggression_rate"))
    expensive_call = _safe_float(row.get("expensive_call_rate"))
    action_entropy = _safe_float(row.get("mean_hand_action_entropy"))

    turn_low_bad = _safe_float(model.get("turn_like_low_bad"), 0.06)
    turn_good = _safe_float(model.get("turn_like_good"), 0.20)
    river_miss_soft = _safe_float(model.get("river_miss_soft"), 0.18)
    river_miss_hard = _safe_float(model.get("river_miss_hard"), 0.40)
    river_aggr_low_bad = _safe_float(model.get("river_aggression_low_bad"), 0.04)
    river_aggr_good = _safe_float(model.get("river_aggression_good"), 0.18)
    expensive_soft = _safe_float(model.get("expensive_call_soft"), 0.28)
    expensive_hard = _safe_float(model.get("expensive_call_hard"), 0.48)
    entropy_soft = _safe_float(model.get("action_entropy_soft"), 0.42)
    entropy_hard = _safe_float(model.get("action_entropy_hard"), 0.54)

    turn_component = _inverse_ramp(turn_like, turn_low_bad, turn_good)
    river_component = _linear_ramp(river_miss, river_miss_soft, river_miss_hard)
    river_aggr_component = _inverse_ramp(river_aggression, river_aggr_low_bad, river_aggr_good)
    expensive_component = _linear_ramp(expensive_call, expensive_soft, expensive_hard)
    entropy_component = _linear_ramp(action_entropy, entropy_soft, entropy_hard)

    weights = model.get("weights") or {}
    penalty = _clamp01(
        _safe_float(weights.get("turn_component"), 0.22) * turn_component
        + _safe_float(weights.get("river_component"), 0.24) * river_component
        + _safe_float(weights.get("river_aggr_component"), 0.20) * river_aggr_component
        + _safe_float(weights.get("expensive_component"), 0.20) * expensive_component
        + _safe_float(weights.get("entropy_component"), 0.14) * entropy_component
    )

    veto = bool(
        river_miss >= _safe_float(model.get("river_miss_veto"), river_miss_hard)
        or expensive_call >= _safe_float(model.get("expensive_call_veto"), expensive_hard)
        or river_aggression <= _safe_float(model.get("river_aggression_veto"), river_aggr_low_bad)
    )

    if veto or penalty >= 0.75:
        band = "red"
    elif penalty >= 0.30:
        band = "amber"
    else:
        band = "green"

    return {
        "gen17_math_1_post_penalty": penalty,
        "gen17_math_1_post_band": band,
        "gen17_math_1_post_veto": veto,
        "gen17_math_1_post_turn_component": turn_component,
        "gen17_math_1_post_river_component": river_component,
        "gen17_math_1_post_river_aggr_component": river_aggr_component,
        "gen17_math_1_post_expensive_component": expensive_component,
        "gen17_math_1_post_entropy_component": entropy_component,
    }


def apply_penalty(base_score: float, penalty: float, penalty_weight: float = 1.0) -> float:
    return _clamp01(base_score * (1.0 - penalty_weight * penalty))


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


def evaluate_penalty_model(rows: Sequence[Dict[str, object]], model: Dict[str, object], threshold: float) -> Dict[str, float]:
    bot_rows = [row for row in rows if int(row.get("truth_value", 0)) == 1]
    human_rows = [row for row in rows if int(row.get("truth_value", 0)) == 0]
    bot_good = sum(1 for row in bot_rows if score_chunk_row(row, model)["gen17_math_1_post_penalty"] < threshold)
    human_good = sum(1 for row in human_rows if score_chunk_row(row, model)["gen17_math_1_post_penalty"] >= threshold)
    bot_recall = bot_good / max(len(bot_rows), 1)
    human_recall = human_good / max(len(human_rows), 1)
    return {
        "penalty_threshold": threshold,
        "bot_recall": bot_recall,
        "human_recall": human_recall,
        "balanced_accuracy": 0.5 * (bot_recall + human_recall),
    }