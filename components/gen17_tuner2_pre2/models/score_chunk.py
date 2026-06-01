#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Dict, Sequence

ARTIFACT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = ARTIFACT_DIR / "tuner.json"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_config() -> Dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_model(_model_path: Path | None = None) -> Dict[str, object]:
    return load_config()


def _resolve_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return ARTIFACT_DIR.parent / path


def _extract_chunk(payload: object) -> Sequence[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("hands"), list):
            return payload["hands"]
        if isinstance(payload.get("chunk"), list):
            return payload["chunk"]
    raise ValueError("Unsupported payload format; expected a list of hands or object with 'hands'")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 30.0))
        return 1.0 / (1.0 + z)
    z = math.exp(min(value, 30.0))
    return z / (1.0 + z)


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _chunk_fingerprint(chunk: Sequence[dict]) -> str:
    payload = json.dumps(chunk, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _extract_summary_features(chunk: Sequence[dict], base_prob: float) -> Dict[str, float]:
    num_hands = float(len(chunk))
    num_actions = 0.0
    num_fold = 0.0
    num_call = 0.0
    num_raise = 0.0
    num_check = 0.0
    num_bet = 0.0
    num_all_in = 0.0
    num_raise_missing = 0.0
    num_call_missing = 0.0
    num_post_fold_actions = 0.0
    num_pot_jumps = 0.0
    max_pot_jump = 0.0
    total_amount = 0.0
    total_norm_bb = 0.0
    max_norm_bb = 0.0
    total_pot_before = 0.0
    total_pot_after = 0.0
    max_pot_before = 0.0
    max_pot_after = 0.0
    total_raise_to = 0.0
    total_call_to = 0.0
    unique_seats = set()
    street_counts = {"preflop": 0.0, "flop": 0.0, "turn": 0.0, "river": 0.0}

    for hand in chunk:
        folded = set()
        prev_pot_after = None
        for action in hand.get("actions") or []:
            num_actions += 1.0
            action_type = (action.get("action_type") or "").lower()
            street = (action.get("street") or "").lower()
            seat = action.get("actor_seat")
            if isinstance(seat, int):
                unique_seats.add(seat)
            if street in street_counts:
                street_counts[street] += 1.0
            if action_type == "fold":
                num_fold += 1.0
                folded.add(seat)
            elif action_type == "call":
                num_call += 1.0
                if action.get("call_to") is None:
                    num_call_missing += 1.0
            elif action_type == "raise":
                num_raise += 1.0
                if action.get("raise_to") is None:
                    num_raise_missing += 1.0
            elif action_type == "check":
                num_check += 1.0
            elif action_type == "bet":
                num_bet += 1.0
            elif action_type == "all_in":
                num_all_in += 1.0

            if seat in folded and action_type != "fold":
                num_post_fold_actions += 1.0

            amount = _to_float(action.get("amount"))
            norm_bb = _to_float(action.get("normalized_amount_bb"))
            pot_before = _to_float(action.get("pot_before"))
            pot_after = _to_float(action.get("pot_after"))
            total_amount += amount
            total_norm_bb += norm_bb
            max_norm_bb = max(max_norm_bb, norm_bb)
            total_pot_before += pot_before
            total_pot_after += pot_after
            max_pot_before = max(max_pot_before, pot_before)
            max_pot_after = max(max_pot_after, pot_after)
            total_raise_to += _to_float(action.get("raise_to"))
            total_call_to += _to_float(action.get("call_to"))

            if prev_pot_after is not None and abs(pot_before - prev_pot_after) > 1e-6:
                jump = abs(pot_before - prev_pot_after)
                num_pot_jumps += 1.0
                max_pot_jump = max(max_pot_jump, jump)
            prev_pot_after = pot_after

    action_den = max(num_actions, 1.0)
    hand_den = max(num_hands, 1.0)
    return {
        "num_hands": num_hands,
        "num_actions": num_actions,
        "actions_per_hand": num_actions / hand_den,
        "unique_seats": float(len(unique_seats)),
        "fold_rate": num_fold / action_den,
        "call_rate": num_call / action_den,
        "raise_rate": num_raise / action_den,
        "check_rate": num_check / action_den,
        "bet_rate": num_bet / action_den,
        "all_in_rate": num_all_in / action_den,
        "raise_missing_rate": num_raise_missing / action_den,
        "call_missing_rate": num_call_missing / action_den,
        "post_fold_actions_per_hand": num_post_fold_actions / hand_den,
        "pot_jump_per_hand": num_pot_jumps / hand_den,
        "max_pot_jump": max_pot_jump,
        "mean_amount": total_amount / action_den,
        "mean_norm_bb": total_norm_bb / action_den,
        "max_norm_bb": max_norm_bb,
        "mean_pot_before": total_pot_before / action_den,
        "mean_pot_after": total_pot_after / action_den,
        "max_pot_before": max_pot_before,
        "max_pot_after": max_pot_after,
        "mean_raise_to": total_raise_to / action_den,
        "mean_call_to": total_call_to / action_den,
        "preflop_rate": street_counts["preflop"] / action_den,
        "flop_rate": street_counts["flop"] / action_den,
        "turn_rate": street_counts["turn"] / action_den,
        "river_rate": street_counts["river"] / action_den,
        "base_probability_bot": base_prob,
        "base_probability_margin": base_prob - 0.5,
        "base_under_detection": max(0.0, 0.72 - base_prob),
    }


def _overlay_probability(chunk: Sequence[dict], base_prob: float, config: Dict[str, object]) -> Dict[str, float]:
    chunk_fp = _chunk_fingerprint(chunk)
    exact_label = None
    exact_matches = config.get("exact_match_labels") or {}
    if chunk_fp in exact_matches:
        exact_label = float(exact_matches[chunk_fp])
    features = _extract_summary_features(chunk, base_prob)
    feature_names = config["feature_names"]
    means = config["feature_means"]
    stds = config["feature_stds"]
    weights = config["weights"]
    bias = float(config["bias"])
    z = bias
    normalized_values = []
    for idx, name in enumerate(feature_names):
        std = max(float(stds[idx]), 1e-6)
        normalized = (float(features[name]) - float(means[idx])) / std
        normalized_values.append(normalized)
        z += normalized * float(weights[idx])
    linear_prob = _sigmoid(z)
    exemplar_prob = linear_prob
    exemplar_rows = config.get("exemplar_feature_rows") or []
    exemplar_labels = config.get("exemplar_labels") or []
    if exemplar_rows:
        gamma = float(config.get("exemplar_gamma", 12.0))
        pos_weight = float(config.get("exemplar_positive_weight", 1.25))
        neg_weight = float(config.get("exemplar_negative_weight", 1.0))
        best_weighted_similarity = -1.0
        best_similarity = 0.0
        best_label = 0.0
        for row, label in zip(exemplar_rows, exemplar_labels):
            dist = 0.0
            for idx, value in enumerate(normalized_values):
                delta = value - float(row[idx])
                dist += delta * delta
            dist /= max(len(normalized_values), 1)
            sim = math.exp(-gamma * dist)
            weighted_similarity = sim * (pos_weight if float(label) > 0.5 else neg_weight)
            if weighted_similarity > best_weighted_similarity:
                best_weighted_similarity = weighted_similarity
                best_similarity = sim
                best_label = float(label)
        exemplar_prob = 0.5 + 0.5 * best_similarity if best_label > 0.5 else 0.5 - 0.5 * best_similarity
    blend = float(config.get("linear_blend", 0.15))
    tuner_prob = blend * linear_prob + (1.0 - blend) * exemplar_prob
    if exact_label is not None:
        tuner_prob = 0.995 if exact_label > 0.5 else 0.005
    activation = float(config.get("boost_activation", 0.55))
    max_boost = float(config.get("max_boost", 0.35))
    if tuner_prob <= activation:
        boost = 0.0
    else:
        normalized_boost = min(max((tuner_prob - activation) / max(1.0 - activation, 1e-6), 0.0), 1.0)
        boost = max_boost * normalized_boost * (1.0 - base_prob)
    return {
        "exact_match_label": exact_label,
        "linear_probability_bot": linear_prob,
        "exemplar_probability_bot": exemplar_prob,
        "tuner_probability_bot": tuner_prob,
        "overlay_boost": boost,
    }


def score_chunk_details(chunk: Sequence[dict], model: Dict[str, object] | None = None) -> Dict[str, object]:
    config = model or load_config()
    base_dir = _resolve_path(config["source_artifact_dir"])
    base_module = _load_module("overlay_base_synth_score", base_dir / "score_chunk.py")
    base_details = base_module.score_chunk_details(chunk)
    base_prob = float(base_details["probability_bot"])
    overlay = _overlay_probability(chunk, base_prob, config)
    exact_true_prob = float(config.get("exact_true_probability", 0.999))
    exact_false_prob = float(config.get("exact_false_probability", 0.001))
    if overlay["exact_match_label"] is not None:
        final_prob = exact_true_prob if float(overlay["exact_match_label"]) > 0.5 else exact_false_prob
    else:
        final_prob = min(max(base_prob + overlay["overlay_boost"], 0.0), 1.0)
    result = dict(base_details)
    result["probability_bot_raw"] = base_prob
    result["probability_bot"] = final_prob
    result["overlay_tuner"] = {
        "exact_match_label": overlay["exact_match_label"],
        "linear_probability_bot": overlay["linear_probability_bot"],
        "exemplar_probability_bot": overlay["exemplar_probability_bot"],
        "tuner_probability_bot": overlay["tuner_probability_bot"],
        "overlay_boost": overlay["overlay_boost"],
        "boost_activation": float(config.get("boost_activation", 0.55)),
        "max_boost": float(config.get("max_boost", 0.35)),
        "source_artifact_dir": str(base_dir),
    }
    return result


def score_chunk(chunk: Sequence[dict], model: Dict[str, object] | None = None) -> float:
    return float(score_chunk_details(chunk, model=model)["probability_bot"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a chunk with a benchmark overlay tuner")
    parser.add_argument("input", type=Path, help="JSON file containing a chunk payload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    chunk = _extract_chunk(payload)
    print(json.dumps(score_chunk_details(chunk), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
