"""Runtime chunk scoring."""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_MODEL_PATH = REPO_ROOT / "weights" / "gen20_tens1_7k_vote101_hardened.ts"

SUPPORTED_ACTION_TYPES = {"bet", "call", "check", "fold", "raise"}
SUPPORTED_STREETS = {"flop", "preflop", "river", "turn"}
DROP_NUMERIC_FIELDS = {"call_to", "raise_to"}
MAX_ACTIONS_PER_HAND = 12
SCALING_REFERENCE_MAX = {
    "amount": 2.52,
    "raise_to": 1.7624,
    "call_to": 1.1466,
    "normalized_amount_bb": 126.0,
    "pot_before": 2.52,
    "pot_after": 2.52,
}

ACTION_MAP = {
    "fold": 1,
    "call": 2,
    "raise": 3,
    "check": 4,
    "bet": 5,
    "all_in": 6,
}

STREET_MAP = {
    "preflop": 1,
    "flop": 2,
    "turn": 3,
    "river": 4,
}

_RUNTIME_MODEL: Optional[torch.jit.ScriptModule] = None
_RUNTIME_AVAILABLE = False
_RUNTIME_LOAD_ERROR: Optional[str] = None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _to_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _scale_numeric_value(value: float, reference_max: float) -> float:
    if reference_max <= 0.0:
        return value
    if value <= 0.0:
        return 0.0
    return float(math.log1p(value) / math.log1p(reference_max))


def _preprocess_runtime_chunk(chunk: List[dict]) -> List[dict]:
    transformed_hands: List[dict] = []
    for hand in chunk:
        raw_actions = hand.get("actions") or []
        raw_actions = raw_actions[:MAX_ACTIONS_PER_HAND]

        transformed_actions: List[dict] = []
        for action in raw_actions:
            action_type = str(action.get("action_type") or "").lower()
            street = str(action.get("street") or "").lower()
            if action_type not in SUPPORTED_ACTION_TYPES:
                continue
            if street and street not in SUPPORTED_STREETS:
                continue

            transformed_action = dict(action)
            for field in DROP_NUMERIC_FIELDS:
                transformed_action.pop(field, None)

            for field, reference_max in SCALING_REFERENCE_MAX.items():
                if field in DROP_NUMERIC_FIELDS:
                    continue
                value = _to_float(transformed_action.get(field))
                if value is None:
                    continue
                transformed_action[field] = _scale_numeric_value(value, reference_max)

            transformed_actions.append(transformed_action)

        if not transformed_actions:
            continue

        transformed_hand = dict(hand)
        transformed_hand["actions"] = transformed_actions
        transformed_hands.append(transformed_hand)

    return transformed_hands


def _runtime_shape() -> Tuple[int, int]:
    max_hands = int(os.getenv("POKER44_MODEL_MAX_HANDS", "128"))
    max_actions = int(os.getenv("POKER44_MODEL_MAX_ACTIONS", "32"))
    return max(1, max_hands), max(1, max_actions)


def _encode_chunk(chunk: List[dict], max_hands: int, max_actions: int) -> Dict[str, np.ndarray]:
    shape = (max_hands, max_actions)

    arr_action_type = np.zeros(shape, dtype=np.int64)
    arr_street = np.zeros(shape, dtype=np.int64)
    arr_actor_seat = np.zeros(shape, dtype=np.int64)

    arr_amount = np.zeros(shape, dtype=np.float32)
    arr_raise_to = np.zeros(shape, dtype=np.float32)
    arr_call_to = np.zeros(shape, dtype=np.float32)
    arr_norm_bb = np.zeros(shape, dtype=np.float32)
    arr_pot_before = np.zeros(shape, dtype=np.float32)
    arr_pot_after = np.zeros(shape, dtype=np.float32)

    arr_raise_miss = np.zeros(shape, dtype=np.float32)
    arr_call_miss = np.zeros(shape, dtype=np.float32)
    arr_valid = np.zeros(shape, dtype=np.float32)

    for h_i, hand in enumerate(chunk[:max_hands]):
        actions = hand.get("actions") or []
        for a_i, action in enumerate(actions[:max_actions]):
            t = (action.get("action_type") or "").lower()
            s = (action.get("street") or "").lower()
            seat = action.get("actor_seat")

            arr_action_type[h_i, a_i] = ACTION_MAP.get(t, 0)
            arr_street[h_i, a_i] = STREET_MAP.get(s, 0)
            arr_actor_seat[h_i, a_i] = int(seat) + 1 if isinstance(seat, int) and seat >= 0 else 0

            arr_amount[h_i, a_i] = _safe_float(action.get("amount"))
            rto = action.get("raise_to")
            cto = action.get("call_to")
            arr_raise_miss[h_i, a_i] = 1.0 if rto is None else 0.0
            arr_call_miss[h_i, a_i] = 1.0 if cto is None else 0.0
            arr_raise_to[h_i, a_i] = _safe_float(rto)
            arr_call_to[h_i, a_i] = _safe_float(cto)
            arr_norm_bb[h_i, a_i] = _safe_float(action.get("normalized_amount_bb"))
            arr_pot_before[h_i, a_i] = _safe_float(action.get("pot_before"))
            arr_pot_after[h_i, a_i] = _safe_float(action.get("pot_after"))
            arr_valid[h_i, a_i] = 1.0

    return {
        "action_type": arr_action_type,
        "street": arr_street,
        "actor_seat": arr_actor_seat,
        "amount": arr_amount,
        "raise_to": arr_raise_to,
        "call_to": arr_call_to,
        "norm_amount_bb": arr_norm_bb,
        "pot_before": arr_pot_before,
        "pot_after": arr_pot_after,
        "raise_to_missing": arr_raise_miss,
        "call_to_missing": arr_call_miss,
        "valid_mask": arr_valid,
    }


def _load_runtime_model() -> bool:
    global _RUNTIME_MODEL, _RUNTIME_AVAILABLE, _RUNTIME_LOAD_ERROR

    if _RUNTIME_AVAILABLE and _RUNTIME_MODEL is not None:
        return True
    if _RUNTIME_LOAD_ERROR is not None:
        return False

    try:
        _RUNTIME_MODEL = torch.jit.load(str(RUNTIME_MODEL_PATH), map_location="cpu")
        _RUNTIME_MODEL.eval()
        _RUNTIME_AVAILABLE = True
        _RUNTIME_LOAD_ERROR = None
        return True
    except Exception as exc:
        _RUNTIME_MODEL = None
        _RUNTIME_AVAILABLE = False
        _RUNTIME_LOAD_ERROR = str(exc)
        return False


def score_chunk_runtime_with_route(chunk: List[dict]) -> Tuple[float, str]:
    if not chunk:
        return 0.5, "empty_chunk"

    if not _load_runtime_model() or _RUNTIME_MODEL is None:
        return 0.5, "runtime_unavailable"

    try:
        max_hands, max_actions = _runtime_shape()
        preprocessed_chunk = _preprocess_runtime_chunk(chunk)
        if not preprocessed_chunk:
            return 0.5, "empty_after_preprocess"

        enc = _encode_chunk(preprocessed_chunk, max_hands=max_hands, max_actions=max_actions)

        x = {}
        for k, v in enc.items():
            t = torch.from_numpy(v).unsqueeze(0)
            if k in {"action_type", "street", "actor_seat"}:
                x[k] = t.long()
            else:
                x[k] = t.float()

        with torch.no_grad():
            y = _RUNTIME_MODEL(
                x["action_type"],
                x["street"],
                x["actor_seat"],
                x["amount"],
                x["raise_to"],
                x["call_to"],
                x["norm_amount_bb"],
                x["pot_before"],
                x["pot_after"],
                x["raise_to_missing"],
                x["call_to_missing"],
                x["valid_mask"],
            )
        return round(_clamp01(float(y.item())), 6), "runtime"
    except Exception:
        return 0.5, "runtime_error"


def score_chunk(chunk: List[dict]) -> float:
    score, _route = score_chunk_runtime_with_route(chunk)
    return score


def get_chunk_scorer_startup_check(scorer: str) -> Dict[str, object]:
    scorer_norm = (scorer or "").strip().lower()
    info: Dict[str, object] = {
        "scorer": scorer_norm,
        "active": scorer_norm == "runtime",
        "ok": True,
        "error": None,
        "details": {},
    }

    if scorer_norm != "runtime":
        return info

    info["details"] = {
        "artifact_path": str(RUNTIME_MODEL_PATH),
        "artifact_exists": RUNTIME_MODEL_PATH.exists(),
        "shape": _runtime_shape(),
    }

    ok = _load_runtime_model()
    info["ok"] = ok
    if not ok:
        info["error"] = _RUNTIME_LOAD_ERROR

    return info
