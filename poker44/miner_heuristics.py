"""Runtime combo chunk scoring for gen21combo1."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPONENTS_DIR = REPO_ROOT / "components"
TIEBREAKER_COMPONENT = "ml20tens1"

COMPONENT_CONFIGS: List[Dict[str, object]] = [
    {
        "name": "ml17_pre3",
        "threshold": 0.5,
        "module_path": COMPONENTS_DIR / "ml17_pre3" / "poker44" / "miner_heuristics.py",
    },
    {
        "name": "gen17_tuner_pre6",
        "threshold": 0.5,
        "module_path": COMPONENTS_DIR / "gen17_tuner_pre6" / "poker44" / "miner_heuristics.py",
    },
    {
        "name": "gen17_tuner2_pre2",
        "threshold": 0.5,
        "module_path": COMPONENTS_DIR / "gen17_tuner2_pre2" / "poker44" / "miner_heuristics.py",
    },
    {
        "name": "gen17_tuner_synth15",
        "threshold": 0.5,
        "module_path": COMPONENTS_DIR / "gen17_tuner_synth15" / "poker44" / "miner_heuristics.py",
    },
    {
        "name": "ml20tens1",
        "threshold": 0.5,
        "module_path": COMPONENTS_DIR / "ml20tens1" / "poker44" / "miner_heuristics.py",
    },
    {
        "name": "gen18_1",
        "threshold": 0.207742,
        "module_path": COMPONENTS_DIR / "gen18_1" / "poker44" / "miner_heuristics.py",
    },
]

_COMPONENT_MODULES: Dict[str, Any] = {}
_COMPONENT_LOAD_ERRORS: Dict[str, str] = {}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _component_config_map() -> Dict[str, Dict[str, object]]:
    return {str(cfg["name"]): cfg for cfg in COMPONENT_CONFIGS}


def _load_component_module(name: str) -> Any:
    if name in _COMPONENT_MODULES:
        return _COMPONENT_MODULES[name]
    if name in _COMPONENT_LOAD_ERRORS:
        raise RuntimeError(_COMPONENT_LOAD_ERRORS[name])

    cfg = _component_config_map()[name]
    module_path = Path(cfg["module_path"])
    if not module_path.exists():
        error = f"missing component module: {module_path}"
        _COMPONENT_LOAD_ERRORS[name] = error
        raise RuntimeError(error)

    spec = importlib.util.spec_from_file_location(f"poker44_gen21combo1_{name}", module_path)
    if spec is None or spec.loader is None:
        error = f"cannot load component module from {module_path}"
        _COMPONENT_LOAD_ERRORS[name] = error
        raise RuntimeError(error)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _COMPONENT_MODULES[name] = module
    return module


def _score_component_chunk(name: str, chunk: List[dict]) -> Dict[str, object]:
    cfg = _component_config_map()[name]
    threshold = float(cfg["threshold"])

    try:
        module = _load_component_module(name)
        score, route = module.score_chunk_runtime_with_route(chunk)
        score = round(_clamp01(float(score)), 6)
        prediction = score >= threshold
        return {
            "name": name,
            "score": score,
            "threshold": threshold,
            "prediction": prediction,
            "route": str(route),
            "error": None,
        }
    except Exception as exc:
        return {
            "name": name,
            "score": 0.5,
            "threshold": threshold,
            "prediction": 0.5 >= threshold,
            "route": "component_error",
            "error": str(exc),
        }


def score_chunk_runtime_bundle(chunk: List[dict]) -> Dict[str, object]:
    components: Dict[str, Dict[str, object]] = {}
    true_votes = 0

    for cfg in COMPONENT_CONFIGS:
        name = str(cfg["name"])
        result = _score_component_chunk(name, chunk)
        components[name] = result
        if bool(result["prediction"]):
            true_votes += 1

    total_votes = len(COMPONENT_CONFIGS)
    false_votes = total_votes - true_votes

    if true_votes > false_votes:
        combo_prediction = True
        decision_mode = "majority_true"
    elif false_votes > true_votes:
        combo_prediction = False
        decision_mode = "majority_false"
    else:
        combo_prediction = bool(components[TIEBREAKER_COMPONENT]["prediction"])
        decision_mode = f"tie_break_{TIEBREAKER_COMPONENT}"

    combo_score = round(true_votes / total_votes if total_votes else 0.5, 6)
    route = f"combo:{decision_mode}:{true_votes}T/{false_votes}F"

    return {
        "score": combo_score,
        "prediction": combo_prediction,
        "route": route,
        "true_votes": true_votes,
        "false_votes": false_votes,
        "components": components,
    }


def score_chunk_runtime_with_route(chunk: List[dict]) -> Tuple[float, str]:
    bundle = score_chunk_runtime_bundle(chunk)
    return float(bundle["score"]), str(bundle["route"])


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

    component_details: Dict[str, Dict[str, object]] = {}
    errors: List[str] = []
    for cfg in COMPONENT_CONFIGS:
        name = str(cfg["name"])
        module_path = Path(cfg["module_path"])
        detail: Dict[str, object] = {
            "module_path": str(module_path),
            "module_exists": module_path.exists(),
            "threshold": float(cfg["threshold"]),
        }
        try:
            module = _load_component_module(name)
            if hasattr(module, "get_chunk_scorer_startup_check"):
                detail["startup"] = module.get_chunk_scorer_startup_check("runtime")
        except Exception as exc:
            detail["error"] = str(exc)
            errors.append(f"{name}: {exc}")
        component_details[name] = detail

    info["ok"] = not errors
    info["details"] = {
        "components": component_details,
        "tie_breaker": TIEBREAKER_COMPONENT,
    }
    if errors:
        info["error"] = "; ".join(errors)
    return info
