"""Runtime chunk scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNER_CONFIG_PATH = REPO_ROOT / "models" / "tuner.json"
BASE_RUNTIME_DIR = REPO_ROOT / "models" / "base_runtime"
HAND_MODEL_PATH = BASE_RUNTIME_DIR / "hand_model.npz"
CHUNK_AGGREGATOR_MODEL_PATH = BASE_RUNTIME_DIR / "chunk_aggregator_model.npz"
RUNTIME_SCORER_PATH = REPO_ROOT / "models" / "score_chunk.py"
RUNTIME_FEATURE_EXTRACTOR_PATH = BASE_RUNTIME_DIR / "feature_extractor_frozen.py"

_RUNTIME_MODEL: Optional[Dict[str, Any]] = None
_RUNTIME_SCORER: Optional[Any] = None
_RUNTIME_AVAILABLE = False
_RUNTIME_LOAD_ERROR: Optional[str] = None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _load_runtime_scorer() -> Any:
    global _RUNTIME_SCORER

    if _RUNTIME_SCORER is not None:
        return _RUNTIME_SCORER

    spec = importlib.util.spec_from_file_location("poker44_gen17_tuner_pre6_scorer", RUNTIME_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load runtime scorer from {RUNTIME_SCORER_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _RUNTIME_SCORER = module
    return module


def _load_runtime_model() -> bool:
    global _RUNTIME_MODEL, _RUNTIME_AVAILABLE, _RUNTIME_LOAD_ERROR

    if _RUNTIME_AVAILABLE and _RUNTIME_MODEL is not None:
        return True
    if _RUNTIME_LOAD_ERROR is not None:
        return False

    try:
        scorer = _load_runtime_scorer()
        _RUNTIME_MODEL = scorer.load_model()
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
        scorer = _load_runtime_scorer()
        probability = float(scorer.score_chunk(chunk, model=_RUNTIME_MODEL))
        return round(_clamp01(probability), 6), "runtime"
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
        "tuner_config_path": str(TUNER_CONFIG_PATH),
        "tuner_config_exists": TUNER_CONFIG_PATH.exists(),
        "base_runtime_dir": str(BASE_RUNTIME_DIR),
        "base_runtime_exists": BASE_RUNTIME_DIR.exists(),
        "hand_model_path": str(HAND_MODEL_PATH),
        "hand_model_exists": HAND_MODEL_PATH.exists(),
        "chunk_aggregator_model_path": str(CHUNK_AGGREGATOR_MODEL_PATH),
        "chunk_aggregator_model_exists": CHUNK_AGGREGATOR_MODEL_PATH.exists(),
        "scorer_path": str(RUNTIME_SCORER_PATH),
        "scorer_exists": RUNTIME_SCORER_PATH.exists(),
        "feature_extractor_path": str(RUNTIME_FEATURE_EXTRACTOR_PATH),
        "feature_extractor_exists": RUNTIME_FEATURE_EXTRACTOR_PATH.exists(),
    }

    ok = _load_runtime_model()
    info["ok"] = ok
    if not ok:
        info["error"] = _RUNTIME_LOAD_ERROR

    return info
