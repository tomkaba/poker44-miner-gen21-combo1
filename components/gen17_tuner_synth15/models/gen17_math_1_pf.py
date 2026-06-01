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
    impossible_rate = _safe_float(row.get("preflop_impossible_hand_rate"))
    feasible_rate = _safe_float(row.get("feasible_preflop_hand_rate"), 1.0)
    low_support_count = _safe_float(row.get("total_preflop_low_support_count"))

    soft_impossible = _safe_float(model.get("soft_impossible_rate"), 0.25)
    hard_impossible = _safe_float(model.get("hard_impossible_rate"), 0.3556)
    soft_feasible = _safe_float(model.get("soft_feasible_rate"), 0.60)
    hard_feasible = _safe_float(model.get("hard_feasible_rate"), 0.42)
    impossible_weight = _safe_float(model.get("impossible_weight"), 0.85)
    feasible_weight = _safe_float(model.get("feasible_weight"), 0.15)
    veto_impossible = _safe_float(model.get("veto_impossible_rate"), hard_impossible)

    impossible_component = _linear_ramp(impossible_rate, soft_impossible, hard_impossible)
    feasible_component = _linear_ramp(soft_feasible - feasible_rate, 0.0, max(soft_feasible - hard_feasible, 1e-9))
    penalty = _clamp01(impossible_weight * impossible_component + feasible_weight * feasible_component)

    if penalty >= 0.75 or impossible_rate > veto_impossible:
        band = "red"
    elif penalty >= 0.30:
        band = "amber"
    else:
        band = "green"

    return {
        "gen17_math_1_pf_impossible_rate": impossible_rate,
        "gen17_math_1_pf_feasible_rate": feasible_rate,
        "gen17_math_1_pf_low_support_count": int(round(low_support_count)),
        "gen17_math_1_pf_impossible_component": impossible_component,
        "gen17_math_1_pf_feasible_component": feasible_component,
        "gen17_math_1_pf_penalty": penalty,
        "gen17_math_1_pf_band": band,
        "gen17_math_1_pf_veto": bool(impossible_rate > veto_impossible),
    }


def apply_penalty(base_score: float, penalty: float, penalty_weight: float = 1.0) -> float:
    return _clamp01(base_score * (1.0 - penalty_weight * penalty))


def best_impossible_threshold(bot_rows: Sequence[Dict[str, object]], human_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    candidates = sorted({
        _safe_float(row.get("preflop_impossible_hand_rate"))
        for row in [*bot_rows, *human_rows]
    })
    best = None
    for threshold in candidates:
        bot_recall = sum(1 for row in bot_rows if _safe_float(row.get("preflop_impossible_hand_rate")) <= threshold) / max(len(bot_rows), 1)
        human_recall = sum(1 for row in human_rows if _safe_float(row.get("preflop_impossible_hand_rate")) > threshold) / max(len(human_rows), 1)
        balanced_accuracy = 0.5 * (bot_recall + human_recall)
        item = {
            "threshold": threshold,
            "bot_recall": bot_recall,
            "human_recall": human_recall,
            "balanced_accuracy": balanced_accuracy,
        }
        if best is None or item["balanced_accuracy"] > best["balanced_accuracy"]:
            best = item
    return best or {"threshold": 0.3556, "bot_recall": 0.0, "human_recall": 0.0, "balanced_accuracy": 0.0}


def best_full_recall_impossible_threshold(bot_rows: Sequence[Dict[str, object]], human_rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    candidates = sorted({
        _safe_float(row.get("preflop_impossible_hand_rate"))
        for row in [*bot_rows, *human_rows]
    })
    chosen = None
    for threshold in candidates:
        human_recall = sum(1 for row in human_rows if _safe_float(row.get("preflop_impossible_hand_rate")) > threshold) / max(len(human_rows), 1)
        if human_recall < 0.999:
            continue
        bot_recall = sum(1 for row in bot_rows if _safe_float(row.get("preflop_impossible_hand_rate")) <= threshold) / max(len(bot_rows), 1)
        item = {
            "threshold": threshold,
            "bot_recall": bot_recall,
            "human_recall": human_recall,
            "balanced_accuracy": 0.5 * (bot_recall + human_recall),
        }
        if chosen is None or item["bot_recall"] > chosen["bot_recall"]:
            chosen = item
    return chosen or best_impossible_threshold(bot_rows, human_rows)


def evaluate_penalty_model(rows: Sequence[Dict[str, object]], model: Dict[str, object], penalty_threshold: float) -> Dict[str, float]:
    bot_rows = [row for row in rows if int(row.get("truth_value", 0)) == 1]
    human_rows = [row for row in rows if int(row.get("truth_value", 0)) == 0]
    bot_good = sum(1 for row in bot_rows if score_chunk_row(row, model)["gen17_math_1_pf_penalty"] < penalty_threshold)
    human_good = sum(1 for row in human_rows if score_chunk_row(row, model)["gen17_math_1_pf_penalty"] >= penalty_threshold)
    bot_recall = bot_good / max(len(bot_rows), 1)
    human_recall = human_good / max(len(human_rows), 1)
    return {
        "penalty_threshold": penalty_threshold,
        "bot_recall": bot_recall,
        "human_recall": human_recall,
        "balanced_accuracy": 0.5 * (bot_recall + human_recall),
    }


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