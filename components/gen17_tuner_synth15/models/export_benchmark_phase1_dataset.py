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


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


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


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _position_info(hero_seat: object, button_seat: object, players: Sequence[dict]) -> Tuple[str, str]:
    try:
        hero = int(hero_seat)
        button = int(button_seat)
    except Exception:
        return "unknown", "unknown"
    seats = sorted({int(player.get("seat")) for player in players if player.get("seat") is not None})
    if hero not in seats or button not in seats or not seats:
        return "unknown", "unknown"
    button_idx = seats.index(button)
    rotated = seats[button_idx:] + seats[:button_idx]
    hero_offset = rotated.index(hero)
    player_count = len(rotated)
    if player_count == 2:
        exact = "btn_sb" if hero_offset == 0 else "bb"
    else:
        mapping = {
            0: "btn",
            1: "sb",
            2: "bb",
            3: "utg",
            4: "hj",
            5: "co",
        }
        exact = mapping.get(hero_offset, f"p{hero_offset}")

    if exact in {"sb", "bb", "btn_sb"}:
        group = "blind"
    elif exact in {"btn", "co"}:
        group = "late"
    elif exact == "hj":
        group = "mid"
    elif exact == "utg":
        group = "early"
    else:
        group = "unknown"
    return exact, group


def _preflop_spot_type(record: Dict[str, object]) -> str:
    prior_aggr = int(record["prior_street_aggressive_count"])
    prior_passive = int(record["prior_street_passive_count"])
    if prior_aggr <= 0 and prior_passive <= 0:
        return "unopened"
    if prior_aggr <= 0 and prior_passive > 0:
        return "limped"
    if prior_aggr == 1:
        return "facing_open"
    if prior_aggr == 2:
        return "facing_3bet"
    return "facing_4bet_plus"


def _preflop_action_support_mass(record: Dict[str, object]) -> float:
    if str(record.get("street")) != "preflop":
        return 1.0

    action = str(record.get("hero_action_type") or "")
    position_exact = str(record.get("preflop_position_exact") or "unknown")
    position_group = str(record.get("preflop_position_group") or "unknown")
    spot_type = _preflop_spot_type(record)
    stack_bb = float(record.get("hero_stack_bb") or 0.0)
    price = float(record.get("price_over_pot_proxy") or 0.0)

    price_factor = _clamp(price, 0.0, 1.5)
    deep_bonus = 0.08 if stack_bb >= 40.0 else 0.03 if stack_bb >= 20.0 else -0.05
    short_bonus = 0.10 if stack_bb <= 15.0 else 0.0

    if spot_type == "unopened":
        if action in AGGRESSIVE_ACTIONS:
            base = 0.16
            if position_group == "late":
                base += 0.10
            elif position_group == "mid":
                base += 0.03
            elif position_group == "early":
                base -= 0.04
            elif position_group == "blind":
                base += 0.04
            return _clamp(base + deep_bonus * 0.25 + short_bonus * 0.10, 0.04, 0.55)
        if action == "call":
            if position_exact in {"sb", "btn_sb"}:
                return 0.18
            if position_exact == "bb":
                return 0.08
            return 0.01
        if action == "check":
            return 1.0 if position_exact == "bb" else 0.0
        if action == "fold":
            base = 0.70
            if position_group == "late":
                base -= 0.14
            elif position_group == "blind":
                base -= 0.18
            return _clamp(base, 0.12, 0.85)

    if spot_type == "limped":
        if action == "check":
            return 1.0 if position_exact == "bb" else 0.0
        if action in AGGRESSIVE_ACTIONS:
            base = 0.18 + (0.08 if position_group in {"late", "blind"} else 0.02)
            return _clamp(base + deep_bonus * 0.25, 0.05, 0.45)
        if action == "call":
            base = 0.20 if position_group == "blind" else 0.10
            return _clamp(base, 0.02, 0.30)
        if action == "fold":
            base = 0.50 if position_group != "blind" else 0.20
            return _clamp(base, 0.10, 0.70)

    if spot_type == "facing_open":
        if action == "fold":
            return _clamp(0.38 + 0.22 * price_factor - deep_bonus * 0.20, 0.18, 0.85)
        if action == "call":
            pos_bonus = 0.08 if position_group in {"late", "blind"} else 0.0
            return _clamp(0.22 - 0.10 * price_factor + pos_bonus + deep_bonus, 0.03, 0.42)
        if action in AGGRESSIVE_ACTIONS:
            pos_bonus = 0.06 if position_group in {"late", "blind"} else 0.0
            return _clamp(0.10 + pos_bonus + short_bonus + deep_bonus * 0.20, 0.02, 0.28)
        if action == "check":
            return 0.0

    if spot_type == "facing_3bet":
        if action == "fold":
            return _clamp(0.58 + 0.20 * price_factor + short_bonus * 0.20, 0.30, 0.92)
        if action == "call":
            return _clamp(0.12 + deep_bonus - 0.08 * price_factor, 0.01, 0.25)
        if action in AGGRESSIVE_ACTIONS:
            return _clamp(0.05 + short_bonus * 0.50 + (0.03 if position_group == "late" else 0.0), 0.01, 0.16)
        if action == "check":
            return 0.0

    if spot_type == "facing_4bet_plus":
        if action == "fold":
            return _clamp(0.76 + 0.10 * price_factor, 0.50, 0.98)
        if action == "call":
            return _clamp(0.03 + deep_bonus * 0.50, 0.0, 0.12)
        if action in AGGRESSIVE_ACTIONS:
            return _clamp(0.02 + short_bonus * 0.50, 0.0, 0.10)
        if action == "check":
            return 0.0

    return 0.0


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
    preflop_position_exact, preflop_position_group = _position_info(hero_seat, metadata.get("button_seat"), players)

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
                    "preflop_position_exact": preflop_position_exact,
                    "preflop_position_group": preflop_position_group,
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
            if street == "preflop":
                record = decision_records[-1]
                support_mass = _preflop_action_support_mass(record)
                record["preflop_spot_type"] = _preflop_spot_type(record)
                record["preflop_action_support_mass"] = support_mass
                record["preflop_log_surprise"] = -math.log(max(support_mass, 1e-9))
                record["preflop_low_support_flag"] = 1.0 if support_mass < 0.05 else 0.0
                record["preflop_impossible_flag"] = 1.0 if support_mass <= 0.0 else 0.0
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
    preflop_records = [record for record in decision_records if str(record["street"]) == "preflop"]
    preflop_supports = [float(record.get("preflop_action_support_mass", 1.0)) for record in preflop_records]
    preflop_log_surprises = [float(record.get("preflop_log_surprise", 0.0)) for record in preflop_records]
    preflop_low_support = sum(int(float(record.get("preflop_low_support_flag", 0.0)) > 0) for record in preflop_records)
    preflop_impossible = sum(int(float(record.get("preflop_impossible_flag", 0.0)) > 0) for record in preflop_records)
    preflop_line_support_geom_mean = math.exp(_mean([math.log(max(value, 1e-9)) for value in preflop_supports])) if preflop_supports else 1.0
    preflop_line_feasible = 1.0 if (not preflop_supports or min(preflop_supports) >= 0.03) else 0.0
    checked_to_spots = sum(1 for record in decision_records if float(record["checked_to_opportunity"]) > 0)
    passive_when_checked_to = sum(1 for record in decision_records if float(record["checked_to_opportunity"]) > 0 and record["hero_action_type"] in {"check", "call"})
    cbet_opportunities = sum(1 for record in decision_records if float(record["flop_cbet_opportunity"]) > 0)
    cbet_misses = sum(1 for record in decision_records if float(record["flop_cbet_opportunity"]) > 0 and float(record["flop_cbet_like"]) <= 0)
    turn_barrel_opportunities = sum(1 for record in decision_records if float(record["turn_barrel_opportunity"]) > 0)
    turn_barrel_misses = sum(1 for record in decision_records if float(record["turn_barrel_opportunity"]) > 0 and float(record["turn_barrel_like"]) <= 0)
    river_barrel_opportunities = sum(1 for record in decision_records if float(record["river_barrel_opportunity"]) > 0)
    river_barrel_misses = sum(1 for record in decision_records if float(record["river_barrel_opportunity"]) > 0 and float(record["river_barrel_like"]) <= 0)
    aggressive_decisions = sum(1 for record in decision_records if record["hero_action_type"] in AGGRESSIVE_ACTIONS)
    passive_decisions = sum(1 for record in decision_records if record["hero_action_type"] in PASSIVE_ACTIONS)
    hand_consistency, hand_bucket_entropy = bucket_consistency(decision_records)

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
        "cheap_fold_rate": _safe_div(float(cheap_folds), float(len(facing_records))),
        "expensive_call_count": expensive_calls,
        "expensive_call_rate": _safe_div(float(expensive_calls), float(action_counter.get("call", 0))),
        "overbet_count": overbets,
        "overbet_rate": _safe_div(float(overbets), float(max(len(decision_records), 1))),
        "tiny_bet_count": tiny_bets,
        "tiny_bet_rate": _safe_div(float(tiny_bets), float(max(aggressive_decisions, 1))),
        "jam_like_count": jam_like,
        "jam_like_rate": _safe_div(float(jam_like), float(max(aggressive_decisions, 1))),
        "preflop_decision_count": len(preflop_records),
        "preflop_support_mean": _mean(preflop_supports),
        "preflop_support_min": min(preflop_supports) if preflop_supports else 1.0,
        "preflop_log_surprise_sum": sum(preflop_log_surprises),
        "preflop_line_support_geom_mean": preflop_line_support_geom_mean,
        "preflop_low_support_count": preflop_low_support,
        "preflop_impossible_count": preflop_impossible,
        "preflop_line_feasible": preflop_line_feasible,
        "checked_to_opportunity_count": checked_to_spots,
        "passive_when_checked_to_count": passive_when_checked_to,
        "passive_when_checked_to_rate": _safe_div(float(passive_when_checked_to), float(checked_to_spots)),
        "flop_cbet_opportunity_count": cbet_opportunities,
        "flop_cbet_miss_count": cbet_misses,
        "flop_cbet_miss_rate": _safe_div(float(cbet_misses), float(cbet_opportunities)),
        "turn_barrel_opportunity_count": turn_barrel_opportunities,
        "turn_barrel_miss_count": turn_barrel_misses,
        "turn_barrel_miss_rate": _safe_div(float(turn_barrel_misses), float(turn_barrel_opportunities)),
        "river_barrel_opportunity_count": river_barrel_opportunities,
        "river_barrel_miss_count": river_barrel_misses,
        "river_barrel_miss_rate": _safe_div(float(river_barrel_misses), float(river_barrel_opportunities)),
        "aggression_ratio": _safe_div(float(aggressive_decisions), float(max(passive_decisions, 1))),
        "aggressive_decision_rate": _safe_div(float(aggressive_decisions), float(max(len(decision_records), 1))),
        "passive_decision_rate": _safe_div(float(passive_decisions), float(max(len(decision_records), 1))),
        "mean_price_over_pot_proxy": _mean(price_values),
        "p90_price_over_pot_proxy": _quantile(price_values, 0.90),
        "mean_size_over_pot": _mean(size_values),
        "p90_size_over_pot": _quantile(size_values, 0.90),
        "hand_action_entropy": action_entropy(decision_records),
        "hand_bucket_consistency": hand_consistency,
        "hand_bucket_entropy": hand_bucket_entropy,
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
    tiny_bets = [float(row["tiny_bet_count"]) for row in hand_features]
    jam_like = [float(row["jam_like_count"]) for row in hand_features]
    hand_action_entropies = [float(row["hand_action_entropy"]) for row in hand_features]
    hand_bucket_consistencies = [float(row["hand_bucket_consistency"]) for row in hand_features]
    hand_bucket_entropies = [float(row["hand_bucket_entropy"]) for row in hand_features]
    preflop_support_means = [float(row["preflop_support_mean"]) for row in hand_features]
    preflop_support_mins = [float(row["preflop_support_min"]) for row in hand_features]
    preflop_log_surprises = [float(row["preflop_log_surprise_sum"]) for row in hand_features]
    preflop_line_supports = [float(row["preflop_line_support_geom_mean"]) for row in hand_features]
    feasible_preflop_hands = [float(row["preflop_line_feasible"]) for row in hand_features]
    impossible_preflop_hands = [1.0 if float(row["preflop_impossible_count"]) > 0 else 0.0 for row in hand_features]
    action_counter = Counter(str(record["hero_action_type"]) for record in decision_records)
    total_decisions = float(len(decision_records))
    consistency_mean, bucket_entropy = bucket_consistency(decision_records)

    def _hand_rate(field: str) -> List[float]:
        return [float(row[field]) for row in hand_features]

    hand_fold_rates = [_safe_div(float(row["hero_fold_count"]), float(max(row["hero_decision_count"], 1))) for row in hand_features]
    hand_call_rates = [_safe_div(float(row["hero_call_count"]), float(max(row["hero_decision_count"], 1))) for row in hand_features]
    hand_check_rates = [_safe_div(float(row["hero_check_count"]), float(max(row["hero_decision_count"], 1))) for row in hand_features]
    hand_bet_rates = [_safe_div(float(row["hero_bet_count"]), float(max(row["hero_decision_count"], 1))) for row in hand_features]
    hand_raise_rates = [_safe_div(float(row["hero_raise_count"] + row["hero_allin_count"]), float(max(row["hero_decision_count"], 1))) for row in hand_features]
    hand_profiles = list(zip(hand_fold_rates, hand_call_rates, hand_check_rates, hand_bet_rates, hand_raise_rates, _hand_rate("cheap_fold_rate"), _hand_rate("expensive_call_rate"), _hand_rate("overbet_rate"), _hand_rate("passive_when_checked_to_rate"), _hand_rate("aggressive_decision_rate")))
    if hand_profiles:
        profile_means = [sum(values[idx] for values in hand_profiles) / len(hand_profiles) for idx in range(len(hand_profiles[0]))]
        profile_dispersion = [sum(abs(values[idx] - profile_means[idx]) for idx in range(len(profile_means))) / len(profile_means) for values in hand_profiles]
    else:
        profile_dispersion = []

    dominant_action_ratio = 0.0
    if hand_features:
        dominant_actions = []
        for row in hand_features:
            per_hand = {
                "fold": float(row["hero_fold_count"]),
                "call": float(row["hero_call_count"]),
                "check": float(row["hero_check_count"]),
                "bet": float(row["hero_bet_count"]),
                "raise": float(row["hero_raise_count"] + row["hero_allin_count"]),
            }
            dominant_actions.append(max(per_hand.items(), key=lambda item: (item[1], item[0]))[0])
        dominant_counter = Counter(dominant_actions)
        dominant_action_ratio = _safe_div(float(max(dominant_counter.values())), float(len(dominant_actions)))

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
        "tiny_bet_rate": sum(tiny_bets) / total_decisions if total_decisions > 0 else 0.0,
        "jam_like_rate": sum(jam_like) / total_decisions if total_decisions > 0 else 0.0,
        "mean_cheap_fold_count_per_hand": _mean(cheap_folds),
        "mean_expensive_call_count_per_hand": _mean(expensive_calls),
        "action_entropy": action_entropy(decision_records),
        "bucket_consistency_mean": consistency_mean,
        "bucket_entropy_mean": bucket_entropy,
        "mean_preflop_support_mean": _mean(preflop_support_means),
        "mean_preflop_support_min": _mean(preflop_support_mins),
        "mean_preflop_log_surprise_per_hand": _mean(preflop_log_surprises),
        "mean_preflop_line_support_geom_mean": _mean(preflop_line_supports),
        "feasible_preflop_hand_count": int(sum(feasible_preflop_hands)),
        "feasible_preflop_hand_rate": _safe_div(sum(feasible_preflop_hands), float(max(len(hand_features), 1))),
        "preflop_impossible_hand_count": int(sum(impossible_preflop_hands)),
        "preflop_impossible_hand_rate": _safe_div(sum(impossible_preflop_hands), float(max(len(hand_features), 1))),
        "total_preflop_low_support_count": int(sum(float(row["preflop_low_support_count"]) for row in hand_features)),
        "mean_hand_action_entropy": _mean(hand_action_entropies),
        "std_hand_action_entropy": _std(hand_action_entropies),
        "mean_hand_bucket_consistency": _mean(hand_bucket_consistencies),
        "std_hand_bucket_consistency": _std(hand_bucket_consistencies),
        "mean_hand_bucket_entropy": _mean(hand_bucket_entropies),
        "std_hand_bucket_entropy": _std(hand_bucket_entropies),
        "std_hand_fold_rate": _std(hand_fold_rates),
        "std_hand_call_rate": _std(hand_call_rates),
        "std_hand_check_rate": _std(hand_check_rates),
        "std_hand_bet_rate": _std(hand_bet_rates),
        "std_hand_raise_rate": _std(hand_raise_rates),
        "std_hand_cheap_fold_rate": _std(_hand_rate("cheap_fold_rate")),
        "std_hand_expensive_call_rate": _std(_hand_rate("expensive_call_rate")),
        "std_hand_overbet_rate": _std(_hand_rate("overbet_rate")),
        "std_hand_passive_when_checked_to_rate": _std(_hand_rate("passive_when_checked_to_rate")),
        "mean_hand_profile_dispersion": _mean(profile_dispersion),
        "p90_hand_profile_dispersion": _quantile(profile_dispersion, 0.90),
        "dominant_action_ratio_across_hands": dominant_action_ratio,
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
        "flop_cbet_miss_rate": sum(1.0 for record in decision_records if float(record["flop_cbet_opportunity"]) > 0 and float(record["flop_cbet_like"]) <= 0) / max(flop_cbet_opportunities, 1.0),
        "turn_barrel_like_rate": sum(float(record["turn_barrel_like"]) for record in decision_records) / max(turn_barrel_opportunities, 1.0),
        "turn_barrel_miss_rate": sum(1.0 for record in decision_records if float(record["turn_barrel_opportunity"]) > 0 and float(record["turn_barrel_like"]) <= 0) / max(turn_barrel_opportunities, 1.0),
        "river_barrel_like_rate": sum(float(record["river_barrel_like"]) for record in decision_records) / max(river_barrel_opportunities, 1.0),
        "river_barrel_miss_rate": sum(1.0 for record in decision_records if float(record["river_barrel_opportunity"]) > 0 and float(record["river_barrel_like"]) <= 0) / max(river_barrel_opportunities, 1.0),
        "passive_when_checked_to_rate": sum(1.0 for record in decision_records if float(record["checked_to_opportunity"]) > 0 and record["hero_action_type"] in {"check", "call"}) / max(sum(1.0 for record in decision_records if float(record["checked_to_opportunity"]) > 0), 1.0),
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
    decision_output = output_dir / "decision_rows.jsonl"
    stats = {
        "chunk_rows": 0,
        "hand_rows": 0,
        "decision_rows": 0,
        "skipped_chunks": 0,
    }

    conn = sqlite3.connect(str(db_path))
    try:
        with hand_output.open("w", encoding="utf-8") as hand_handle, chunk_output.open("w", encoding="utf-8") as chunk_handle, decision_output.open("w", encoding="utf-8") as decision_handle:
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
                    for decision_index, decision_record in enumerate(decision_records):
                        decision_row = {
                            "chunk_hash": chunk_hash,
                            "truth_value": truth_value,
                            "truth_label": truth_label,
                            "source_file": source_file,
                            "source_date": source_date,
                            "hand_index": hand_index,
                            "decision_index": decision_index,
                            **decision_record,
                        }
                        decision_handle.write(json.dumps(decision_row, ensure_ascii=True) + "\n")
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