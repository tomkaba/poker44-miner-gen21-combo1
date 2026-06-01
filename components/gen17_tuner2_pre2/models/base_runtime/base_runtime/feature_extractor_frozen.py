#!/usr/bin/env python3
import argparse
import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


AGGRESSIVE_ACTIONS = {"bet", "raise", "all_in", "all-in"}
PASSIVE_ACTIONS = {"call", "check"}
STREET_ORDER = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _bucket_stack(stack_bb: float) -> str:
    if stack_bb <= 12.0:
        return "short"
    if stack_bb <= 40.0:
        return "medium"
    return "deep"


def _bucket_players(player_count: int) -> str:
    if player_count <= 2:
        return "hu"
    if player_count <= 4:
        return "mid"
    return "full"


def _bucket_price(price_over_pot: float) -> str:
    if price_over_pot <= 0.0:
        return "none"
    if price_over_pot <= 0.25:
        return "cheap"
    if price_over_pot <= 0.75:
        return "medium"
    return "expensive"


def _bucket_size(size_over_pot: float) -> str:
    if size_over_pot <= 0.0:
        return "none"
    if size_over_pot < 0.33:
        return "tiny"
    if size_over_pot < 0.75:
        return "small"
    if size_over_pot <= 1.25:
        return "medium"
    if size_over_pot <= 2.0:
        return "large"
    return "overbet"


def _source_date_from_path(source_file: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", source_file)
    return match.group(1) if match else ""


def _previous_street(street: str) -> str:
    if street == "flop":
        return "preflop"
    if street == "turn":
        return "flop"
    if street == "river":
        return "turn"
    return ""


def iter_benchmark_chunks(conn: sqlite3.Connection, limit: Optional[int] = None) -> Iterator[Tuple[str, int, str, str, str, List[dict]]]:
    query = (
        "SELECT t.chunk_hash, COALESCE(t.truth_value, 0), COALESCE(t.truth_label, ''), t.source_file, d.chunk_raw "
        "FROM chunk_truth t JOIN chunk_dedup d ON d.chunk_hash = t.chunk_hash "
        "ORDER BY t.id"
    )
    params: Tuple[object, ...] = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    for chunk_hash, truth_value, truth_label, source_file, chunk_raw in conn.execute(query, params):
        try:
            chunk = json.loads(chunk_raw)
        except Exception:
            continue
        if not isinstance(chunk, list) or not chunk:
            continue
        source_file_str = str(source_file or "")
        yield str(chunk_hash), int(truth_value), str(truth_label), source_file_str, _source_date_from_path(source_file_str), chunk


def build_decision_records(hand: dict, hand_index: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    hero_seat = metadata.get("hero_seat")
    if hero_seat is None:
        return [], {"integrity_missing_hero": 1.0}

    hero_player = next((player for player in players if player.get("seat") == hero_seat), None)
    if hero_player is None:
        return [], {"integrity_missing_hero": 1.0}

    bb = _safe_float(metadata.get("bb"), 0.0)
    if bb <= 0:
        bb = 0.02
    hero_stack_bb = _safe_float(hero_player.get("starting_stack")) / bb
    player_count = len(players)

    decision_records: List[Dict[str, object]] = []
    last_action_by_street: Dict[str, dict] = {}
    last_aggressive_by_street: Dict[str, dict] = {}
    street_action_counts: Dict[str, int] = defaultdict(int)
    street_aggressive_counts: Dict[str, int] = defaultdict(int)
    street_passive_counts: Dict[str, int] = defaultdict(int)
    hero_aggressive_streets = set()
    folded_seats = set()

    for action_index, action in enumerate(actions):
        street = str(action.get("street") or "preflop").lower()
        actor_seat = action.get("actor_seat")
        action_type = str(action.get("action_type") or "").lower()
        amount_bb = _safe_float(action.get("normalized_amount_bb"))
        pot_before = _safe_float(action.get("pot_before"))
        pot_after = _safe_float(action.get("pot_after"))
        price_bb_proxy = 0.0
        facing_aggression = 0.0
        prev_aggr = last_aggressive_by_street.get(street)
        if prev_aggr is not None and prev_aggr.get("actor_seat") != hero_seat:
            facing_aggression = 1.0
            price_bb_proxy = _safe_float(prev_aggr.get("price_bb_proxy"))

        if actor_seat == hero_seat:
            active_players = max(1, player_count - len(folded_seats))
            price_over_pot = price_bb_proxy / max(pot_before / bb, 1e-9) if pot_before > 0 else 0.0
            size_over_pot = amount_bb / max(pot_before / bb, 1e-9) if pot_before > 0 else 0.0
            prev_street = _previous_street(street)
            preflop_open_opportunity = 1.0 if street == "preflop" and street_aggressive_counts[street] == 0 and street_passive_counts[street] == 0 else 0.0
            checked_to_opportunity = 1.0 if street in {"flop", "turn", "river"} and street_aggressive_counts[street] == 0 else 0.0
            hero_prev_street_aggressive = 1.0 if prev_street and prev_street in hero_aggressive_streets else 0.0
            min_equity_required_proxy = price_over_pot / (1.0 + price_over_pot) if price_over_pot > 0 else 0.0
            aggressive_now = 1.0 if action_type in AGGRESSIVE_ACTIONS else 0.0
            decision_records.append(
                {
                    "hand_index": hand_index,
                    "street": street,
                    "street_index": STREET_ORDER.get(street, 0),
                    "action_index": action_index,
                    "hero_action_type": action_type,
                    "hero_amount_bb": amount_bb,
                    "hero_stack_bb": hero_stack_bb,
                    "player_count": player_count,
                    "active_players": active_players,
                    "filled_ratio": player_count / max(_safe_int(metadata.get("max_seats"), 6), 1),
                    "facing_aggression": facing_aggression,
                    "price_bb_proxy": price_bb_proxy,
                    "price_over_pot_proxy": price_over_pot,
                    "min_equity_required_proxy": min_equity_required_proxy,
                    "size_over_pot": size_over_pot,
                    "stack_bucket": _bucket_stack(hero_stack_bb),
                    "players_bucket": _bucket_players(active_players),
                    "price_bucket": _bucket_price(price_over_pot),
                    "size_bucket": _bucket_size(size_over_pot),
                    "prior_street_action_count": street_action_counts[street],
                    "prior_street_aggressive_count": street_aggressive_counts[street],
                    "prior_street_passive_count": street_passive_counts[street],
                    "prev_action_type": str((last_action_by_street.get(street) or {}).get("action_type") or ""),
                    "prev_actor_is_hero": 1.0 if (last_action_by_street.get(street) or {}).get("actor_seat") == hero_seat else 0.0,
                    "preflop_open_opportunity": preflop_open_opportunity,
                    "checked_to_opportunity": checked_to_opportunity,
                    "hero_prev_street_aggressive": hero_prev_street_aggressive,
                    "preflop_open_raise": 1.0 if preflop_open_opportunity > 0 and aggressive_now > 0 else 0.0,
                    "preflop_open_call": 1.0 if preflop_open_opportunity > 0 and action_type == "call" else 0.0,
                    "preflop_defense_call": 1.0 if street == "preflop" and facing_aggression > 0 and action_type == "call" else 0.0,
                    "preflop_reraise": 1.0 if street == "preflop" and facing_aggression > 0 and aggressive_now > 0 else 0.0,
                    "flop_cbet_opportunity": 1.0 if street == "flop" and checked_to_opportunity > 0 and hero_prev_street_aggressive > 0 else 0.0,
                    "flop_cbet_like": 1.0 if street == "flop" and checked_to_opportunity > 0 and hero_prev_street_aggressive > 0 and aggressive_now > 0 else 0.0,
                    "turn_barrel_opportunity": 1.0 if street == "turn" and checked_to_opportunity > 0 and hero_prev_street_aggressive > 0 else 0.0,
                    "turn_barrel_like": 1.0 if street == "turn" and checked_to_opportunity > 0 and hero_prev_street_aggressive > 0 and aggressive_now > 0 else 0.0,
                    "river_barrel_opportunity": 1.0 if street == "river" and checked_to_opportunity > 0 and hero_prev_street_aggressive > 0 else 0.0,
                    "river_barrel_like": 1.0 if street == "river" and checked_to_opportunity > 0 and hero_prev_street_aggressive > 0 and aggressive_now > 0 else 0.0,
                    "river_aggression": 1.0 if street == "river" and aggressive_now > 0 else 0.0,
                    "river_call_facing_aggression": 1.0 if street == "river" and facing_aggression > 0 and action_type == "call" else 0.0,
                    "river_overbet": 1.0 if street == "river" and aggressive_now > 0 and size_over_pot > 1.25 else 0.0,
                }
            )
            last_aggressive_by_street.pop(street, None)

        if action_type == "fold" and actor_seat is not None:
            folded_seats.add(actor_seat)

        if action_type in AGGRESSIVE_ACTIONS:
            price_candidate = _safe_float(action.get("raise_to"))
            if price_candidate <= 0:
                price_candidate = _safe_float(action.get("call_to"))
            if price_candidate <= 0:
                price_candidate = amount_bb
            last_aggressive_by_street[street] = {
                "actor_seat": actor_seat,
                "action_type": action_type,
                "price_bb_proxy": price_candidate if price_candidate > 0 else amount_bb,
            }
            street_aggressive_counts[street] += 1
            if actor_seat == hero_seat:
                hero_aggressive_streets.add(street)
        elif action_type in PASSIVE_ACTIONS:
            street_passive_counts[street] += 1

        last_action_by_street[street] = {
            "actor_seat": actor_seat,
            "action_type": action_type,
        }
        street_action_counts[street] += 1

    if not decision_records:
        return [], {"integrity_missing_hero": 0.0, "hand_without_hero_actions": 1.0}

    return decision_records, {"integrity_missing_hero": 0.0, "hand_without_hero_actions": 0.0}


def compute_hand_features(hand: dict, hand_index: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    decision_records, flags = build_decision_records(hand, hand_index)
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}
    bb = _safe_float(metadata.get("bb"), 0.02)
    if bb <= 0:
        bb = 0.02
    hero_seat = metadata.get("hero_seat")
    hero_player = next((player for player in players if player.get("seat") == hero_seat), None)
    hero_stack_bb = (_safe_float(hero_player.get("starting_stack")) / bb) if hero_player else 0.0
    action_counter = Counter(record["hero_action_type"] for record in decision_records)
    price_values = [float(record["price_over_pot_proxy"]) for record in decision_records if float(record["price_over_pot_proxy"]) > 0]
    size_values = [float(record["size_over_pot"]) for record in decision_records if float(record["size_over_pot"]) > 0]
    facing_records = [record for record in decision_records if float(record["facing_aggression"]) > 0]
    cheap_folds = sum(1 for record in decision_records if record["hero_action_type"] == "fold" and float(record["price_over_pot_proxy"]) <= 0.25 and float(record["facing_aggression"]) > 0)
    expensive_calls = sum(1 for record in decision_records if record["hero_action_type"] == "call" and float(record["price_over_pot_proxy"]) >= 0.75)
    overbets = sum(1 for record in decision_records if float(record["size_over_pot"]) > 1.25)
    tiny_bets = sum(1 for record in decision_records if record["hero_action_type"] in AGGRESSIVE_ACTIONS and 0.0 < float(record["size_over_pot"]) < 0.33)
    jam_like = sum(1 for record in decision_records if float(record["hero_amount_bb"]) >= max(0.75 * hero_stack_bb, 0.0) and float(record["hero_amount_bb"]) > 0)

    hand_features: Dict[str, object] = {
        "hand_index": hand_index,
        "hero_seat": hero_seat,
        "hero_stack_bb": hero_stack_bb,
        "player_count": len(players),
        "hand_action_count": len(actions),
        "hero_decision_count": len(decision_records),
        "hero_fold_count": action_counter.get("fold", 0),
        "hero_call_count": action_counter.get("call", 0),
        "hero_check_count": action_counter.get("check", 0),
        "hero_bet_count": action_counter.get("bet", 0),
        "hero_raise_count": action_counter.get("raise", 0),
        "hero_allin_count": action_counter.get("all_in", 0) + action_counter.get("all-in", 0),
        "hero_facing_aggression_count": len(facing_records),
        "hero_fold_facing_aggression_count": sum(1 for record in facing_records if record["hero_action_type"] == "fold"),
        "hero_call_facing_aggression_count": sum(1 for record in facing_records if record["hero_action_type"] == "call"),
        "hero_raise_facing_aggression_count": sum(1 for record in facing_records if record["hero_action_type"] == "raise"),
        "cheap_fold_count": cheap_folds,
        "expensive_call_count": expensive_calls,
        "overbet_count": overbets,
        "tiny_bet_count": tiny_bets,
        "jam_like_count": jam_like,
        "mean_price_over_pot_proxy": _mean(price_values),
        "p90_price_over_pot_proxy": _quantile(price_values, 0.90),
        "mean_size_over_pot": _mean(size_values),
        "p90_size_over_pot": _quantile(size_values, 0.90),
        "showdown": 1.0 if bool(outcome.get("showdown")) else 0.0,
        "street_depth": float(max((STREET_ORDER.get(str(record["street"]), 0) for record in decision_records), default=0) + 1 if decision_records else 0),
        "integrity_missing_hero": float(flags.get("integrity_missing_hero", 0.0)),
        "hand_without_hero_actions": float(flags.get("hand_without_hero_actions", 0.0)),
    }
    return hand_features, decision_records


def action_entropy(decision_records: Sequence[Dict[str, object]]) -> float:
    if not decision_records:
        return 0.0
    counter = Counter(str(record["hero_action_type"]) for record in decision_records)
    total = float(sum(counter.values()))
    entropy = 0.0
    for count in counter.values():
        p = count / total
        entropy -= p * math.log(p + 1e-12)
    return entropy


def bucket_consistency(decision_records: Sequence[Dict[str, object]]) -> Tuple[float, float]:
    by_bucket: Dict[Tuple[object, ...], List[str]] = defaultdict(list)
    for record in decision_records:
        bucket = (
            record["street"],
            record["facing_aggression"],
            record["stack_bucket"],
            record["players_bucket"],
            record["price_bucket"],
        )
        by_bucket[bucket].append(str(record["hero_action_type"]))

    consistencies: List[float] = []
    entropies: List[float] = []
    for actions in by_bucket.values():
        if len(actions) < 2:
            continue
        counter = Counter(actions)
        total = float(len(actions))
        consistencies.append(max(counter.values()) / total)
        entropy = 0.0
        for count in counter.values():
            p = count / total
            entropy -= p * math.log(p + 1e-12)
        entropies.append(entropy)
    return _mean(consistencies), _mean(entropies)


def aggregate_chunk_features(hand_features: Sequence[Dict[str, object]], decision_records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    stack_values = [float(row["hero_stack_bb"]) for row in hand_features]
    decision_counts = [float(row["hero_decision_count"]) for row in hand_features]
    cheap_folds = [float(row["cheap_fold_count"]) for row in hand_features]
    expensive_calls = [float(row["expensive_call_count"]) for row in hand_features]
    overbets = [float(row["overbet_count"]) for row in hand_features]
    action_counter = Counter(str(record["hero_action_type"]) for record in decision_records)
    total_decisions = float(len(decision_records))
    consistency_mean, bucket_entropy = bucket_consistency(decision_records)

    def _rate(action_name: str) -> float:
        return action_counter.get(action_name, 0) / total_decisions if total_decisions > 0 else 0.0

    facing_aggression_count = sum(1 for record in decision_records if float(record["facing_aggression"]) > 0)
    fold_small_price_rate = (
        sum(1 for record in decision_records if record["hero_action_type"] == "fold" and float(record["facing_aggression"]) > 0 and float(record["price_over_pot_proxy"]) <= 0.25) / max(facing_aggression_count, 1)
    )
    expensive_call_rate = (
        sum(1 for record in decision_records if record["hero_action_type"] == "call" and float(record["price_over_pot_proxy"]) >= 0.75) / max(action_counter.get("call", 0), 1)
    )
    rare_size_rate = (
        sum(1 for record in decision_records if str(record["size_bucket"]) in {"tiny", "overbet"}) / total_decisions
        if total_decisions > 0
        else 0.0
    )
    preflop_open_opportunities = sum(float(record["preflop_open_opportunity"]) for record in decision_records)
    preflop_defense_spots = sum(1.0 for record in decision_records if record["street"] == "preflop" and float(record["facing_aggression"]) > 0)
    flop_cbet_opportunities = sum(float(record["flop_cbet_opportunity"]) for record in decision_records)
    turn_barrel_opportunities = sum(float(record["turn_barrel_opportunity"]) for record in decision_records)
    river_barrel_opportunities = sum(float(record["river_barrel_opportunity"]) for record in decision_records)
    river_decisions = sum(1.0 for record in decision_records if record["street"] == "river")
    river_facing_aggression = sum(1.0 for record in decision_records if record["street"] == "river" and float(record["facing_aggression"]) > 0)

    return {
        "chunk_hand_count": len(hand_features),
        "chunk_decision_count": int(total_decisions),
        "mean_hero_stack_bb": _mean(stack_values),
        "std_hero_stack_bb": _std(stack_values),
        "mean_hero_decision_count": _mean(decision_counts),
        "p90_hero_decision_count": _quantile(decision_counts, 0.90),
        "fold_rate": _rate("fold"),
        "call_rate": _rate("call"),
        "check_rate": _rate("check"),
        "bet_rate": _rate("bet"),
        "raise_rate": _rate("raise"),
        "allin_rate": _rate("all_in") + _rate("all-in"),
        "facing_aggression_rate": facing_aggression_count / total_decisions if total_decisions > 0 else 0.0,
        "fold_small_price_rate": fold_small_price_rate,
        "expensive_call_rate": expensive_call_rate,
        "overbet_rate": sum(overbets) / total_decisions if total_decisions > 0 else 0.0,
        "mean_cheap_fold_count_per_hand": _mean(cheap_folds),
        "mean_expensive_call_count_per_hand": _mean(expensive_calls),
        "action_entropy": action_entropy(decision_records),
        "bucket_consistency_mean": consistency_mean,
        "bucket_entropy_mean": bucket_entropy,
        "rare_size_rate": rare_size_rate,
        "mean_price_over_pot_proxy": _mean([float(record["price_over_pot_proxy"]) for record in decision_records]),
        "p90_price_over_pot_proxy": _quantile([float(record["price_over_pot_proxy"]) for record in decision_records], 0.90),
        "mean_min_equity_required_proxy": _mean([float(record["min_equity_required_proxy"]) for record in decision_records]),
        "p90_min_equity_required_proxy": _quantile([float(record["min_equity_required_proxy"]) for record in decision_records], 0.90),
        "mean_size_over_pot": _mean([float(record["size_over_pot"]) for record in decision_records]),
        "p90_size_over_pot": _quantile([float(record["size_over_pot"]) for record in decision_records], 0.90),
        "preflop_open_raise_rate": sum(float(record["preflop_open_raise"]) for record in decision_records) / max(preflop_open_opportunities, 1.0),
        "preflop_open_call_rate": sum(float(record["preflop_open_call"]) for record in decision_records) / max(preflop_open_opportunities, 1.0),
        "preflop_defense_call_rate": sum(float(record["preflop_defense_call"]) for record in decision_records) / max(preflop_defense_spots, 1.0),
        "preflop_reraise_rate": sum(float(record["preflop_reraise"]) for record in decision_records) / max(preflop_defense_spots, 1.0),
        "flop_cbet_like_rate": sum(float(record["flop_cbet_like"]) for record in decision_records) / max(flop_cbet_opportunities, 1.0),
        "turn_barrel_like_rate": sum(float(record["turn_barrel_like"]) for record in decision_records) / max(turn_barrel_opportunities, 1.0),
        "river_barrel_like_rate": sum(float(record["river_barrel_like"]) for record in decision_records) / max(river_barrel_opportunities, 1.0),
        "river_aggression_rate": sum(float(record["river_aggression"]) for record in decision_records) / max(river_decisions, 1.0),
        "river_call_facing_aggression_rate": sum(float(record["river_call_facing_aggression"]) for record in decision_records) / max(river_facing_aggression, 1.0),
        "river_overbet_rate": sum(float(record["river_overbet"]) for record in decision_records) / max(river_decisions, 1.0),
        "integrity_missing_hero_hands": sum(float(row["integrity_missing_hero"]) for row in hand_features),
        "hands_without_hero_actions": sum(float(row["hand_without_hero_actions"]) for row in hand_features),
    }


def export_dataset(db_path: Path, output_dir: Path, limit: Optional[int]) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    hand_output = output_dir / "hand_rows.jsonl"
    chunk_output = output_dir / "chunk_rows.jsonl"
    stats = {
        "chunk_rows": 0,
        "hand_rows": 0,
        "decision_rows": 0,
        "skipped_chunks": 0,
    }

    conn = sqlite3.connect(str(db_path))
    try:
        with hand_output.open("w", encoding="utf-8") as hand_handle, chunk_output.open("w", encoding="utf-8") as chunk_handle:
            for chunk_hash, truth_value, truth_label, source_file, source_date, chunk in iter_benchmark_chunks(conn, limit=limit):
                all_hand_features: List[Dict[str, object]] = []
                all_decision_records: List[Dict[str, object]] = []
                for hand_index, hand in enumerate(chunk):
                    if not isinstance(hand, dict):
                        continue
                    hand_features, decision_records = compute_hand_features(hand, hand_index)
                    hand_row = {
                        "chunk_hash": chunk_hash,
                        "truth_value": truth_value,
                        "truth_label": truth_label,
                        "source_file": source_file,
                        "source_date": source_date,
                        **hand_features,
                    }
                    hand_handle.write(json.dumps(hand_row, ensure_ascii=True) + "\n")
                    all_hand_features.append(hand_features)
                    all_decision_records.extend(decision_records)
                    stats["hand_rows"] += 1
                if not all_hand_features:
                    stats["skipped_chunks"] += 1
                    continue
                chunk_row = {
                    "chunk_hash": chunk_hash,
                    "truth_value": truth_value,
                    "truth_label": truth_label,
                    "source_file": source_file,
                    "source_date": source_date,
                    **aggregate_chunk_features(all_hand_features, all_decision_records),
                }
                chunk_handle.write(json.dumps(chunk_row, ensure_ascii=True) + "\n")
                stats["chunk_rows"] += 1
                stats["decision_rows"] += len(all_decision_records)
    finally:
        conn.close()

    manifest = {
        "db_path": str(db_path),
        "output_dir": str(output_dir),
        "limit": limit,
        **stats,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export benchmark-driven hero-centric Phase 1 dataset for gen17")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("/home/tk/training_gen15/log_management/miner_logs.db"),
        help="Path to miner_logs.db",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/tk/training_gen17/artifacts/phase1_dataset"),
        help="Directory for exported JSONL datasets",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional chunk limit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = export_dataset(args.db, args.output_dir, args.limit)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())