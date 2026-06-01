"""Runtime chunk scoring."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
TUNER_CONFIG_PATH = REPO_ROOT / "models" / "tuner.json"
BASE_RUNTIME_DIR = REPO_ROOT / "models" / "base_runtime"
BASE_SYNTH_MANIFEST_PATH = BASE_RUNTIME_DIR / "synth_manifest.json"
PRE_ARTIFACT_DIR = BASE_RUNTIME_DIR / "pre_artifact"
PF_PENALTY_MODEL_PATH = BASE_RUNTIME_DIR / "pf_penalty_artifact" / "model.json"
PF_BOOST_MODEL_PATH = BASE_RUNTIME_DIR / "pf_boost_artifact" / "model.json"
POST_PENALTY_MODEL_PATH = BASE_RUNTIME_DIR / "post_penalty_artifact" / "model.json"
POST_BOOST_MODEL_PATH = BASE_RUNTIME_DIR / "post_boost_artifact" / "model.json"
RUNTIME_SCORER_PATH = REPO_ROOT / "models" / "score_chunk.py"

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

    spec = importlib.util.spec_from_file_location("poker44_gen17_synth_runtime_scorer", RUNTIME_SCORER_PATH)
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
        "base_synth_manifest_path": str(BASE_SYNTH_MANIFEST_PATH),
        "base_synth_manifest_exists": BASE_SYNTH_MANIFEST_PATH.exists(),
        "pre_artifact_dir": str(PRE_ARTIFACT_DIR),
        "pre_artifact_exists": PRE_ARTIFACT_DIR.exists(),
        "pf_penalty_model_path": str(PF_PENALTY_MODEL_PATH),
        "pf_penalty_model_exists": PF_PENALTY_MODEL_PATH.exists(),
        "pf_boost_model_path": str(PF_BOOST_MODEL_PATH),
        "pf_boost_model_exists": PF_BOOST_MODEL_PATH.exists(),
        "post_penalty_model_path": str(POST_PENALTY_MODEL_PATH),
        "post_penalty_model_exists": POST_PENALTY_MODEL_PATH.exists(),
        "post_boost_model_path": str(POST_BOOST_MODEL_PATH),
        "post_boost_model_exists": POST_BOOST_MODEL_PATH.exists(),
        "scorer_path": str(RUNTIME_SCORER_PATH),
        "scorer_exists": RUNTIME_SCORER_PATH.exists(),
    }

    ok = _load_runtime_model()
    info["ok"] = ok
    if not ok:
        info["error"] = _RUNTIME_LOAD_ERROR

    return info
