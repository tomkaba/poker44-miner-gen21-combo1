#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

ARTIFACT_DIR = Path(__file__).resolve().parent
REPO_ROOT = ARTIFACT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from export_benchmark_phase1_dataset import aggregate_chunk_features, compute_hand_features
from gen17_math_1_pf import apply_penalty as apply_pf_penalty, load_model as load_pf_penalty_model, score_chunk_row as score_pf_penalty_row
from gen17_math_1_pf_boost import apply_boost as apply_pf_boost, load_model as load_pf_boost_model, score_chunk_row as score_pf_boost_row
from gen17_math_1_post import apply_penalty as apply_post_penalty, load_model as load_post_penalty_model, score_chunk_row as score_post_penalty_row
from gen17_math_1_post_boost import apply_boost as apply_post_boost, load_model as load_post_boost_model, score_chunk_row as score_post_boost_row


CONFIG_PATH = ARTIFACT_DIR / "synth_manifest.json"


def load_config() -> Dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_model(_model_path: Path | None = None) -> Dict[str, object]:
    return load_config()


def _resolve_path(value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_chunk(payload: object) -> Sequence[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("hands"), list):
            return payload["hands"]
        if isinstance(payload.get("chunk"), list):
            return payload["chunk"]
    raise ValueError("Unsupported payload format; expected a list of hands or object with 'hands'")


def compute_math_chunk_row(chunk: Sequence[dict]) -> Dict[str, float]:
    all_hand_features = []
    all_decision_records = []
    for hand_index, hand in enumerate(chunk):
        if not isinstance(hand, dict):
            continue
        hand_features, decision_records = compute_hand_features(hand, hand_index)
        all_hand_features.append(hand_features)
        all_decision_records.extend(decision_records)
    if not all_hand_features:
        raise ValueError("Chunk has no valid hands for math scoring")
    return aggregate_chunk_features(all_hand_features, all_decision_records)


def score_chunk_details(chunk: Sequence[dict], config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    config = config or load_config()
    pre_dir = _resolve_path(config["pre_artifact_dir"])
    pre_module = _load_module("gen17_synth_pre_score", pre_dir / "score_chunk.py")
    pre_probability = float(pre_module.score_chunk(chunk))

    math_chunk_row = compute_math_chunk_row(chunk)
    adjusted = pre_probability
    result: Dict[str, object] = {
        "probability_bot_pre": pre_probability,
        "probability_bot": pre_probability,
        "chunk_hand_count": len(chunk),
        "components": {
            "pre_artifact": str(pre_dir),
        },
    }

    pf_penalty_dir = config.get("pf_penalty_artifact_dir") or config.get("penalty_artifact_dir")
    if pf_penalty_dir:
        pf_penalty_dir_path = _resolve_path(pf_penalty_dir)
        pf_penalty_model = load_pf_penalty_model(pf_penalty_dir_path / "model.json")
        pf_penalty_info = score_pf_penalty_row(math_chunk_row, pf_penalty_model)
        pf_penalty_weight = float(config.get("pf_penalty_weight") or config.get("penalty_weight") or pf_penalty_model.get("recommended_penalty_weight", 1.0))
        adjusted = apply_pf_penalty(adjusted, float(pf_penalty_info["gen17_math_1_pf_penalty"]), pf_penalty_weight)
        result.update(pf_penalty_info)
        result["probability_after_pf_penalty"] = adjusted
        result["components"]["pf_penalty_artifact"] = str(pf_penalty_dir_path)
        result["components"]["pf_penalty_weight"] = pf_penalty_weight

    pf_boost_dir = config.get("pf_boost_artifact_dir") or config.get("boost_artifact_dir")
    if pf_boost_dir:
        pf_boost_dir_path = _resolve_path(pf_boost_dir)
        pf_boost_model = load_pf_boost_model(pf_boost_dir_path / "model.json")
        pf_boost_info = score_pf_boost_row(math_chunk_row, pf_boost_model)
        pf_boost_weight = float(config.get("pf_boost_weight") or config.get("boost_weight") or pf_boost_model.get("recommended_boost_weight", 0.25))
        adjusted = apply_pf_boost(adjusted, float(pf_boost_info["gen17_math_1_pf_boost"]), pf_boost_weight)
        result.update(pf_boost_info)
        result["probability_after_pf_boost"] = adjusted
        result["components"]["pf_boost_artifact"] = str(pf_boost_dir_path)
        result["components"]["pf_boost_weight"] = pf_boost_weight

    post_penalty_dir = config.get("post_penalty_artifact_dir")
    if post_penalty_dir:
        post_penalty_dir_path = _resolve_path(post_penalty_dir)
        post_penalty_model = load_post_penalty_model(post_penalty_dir_path / "model.json")
        post_penalty_info = score_post_penalty_row(math_chunk_row, post_penalty_model)
        post_penalty_weight = float(config.get("post_penalty_weight") or post_penalty_model.get("recommended_penalty_weight", 1.0))
        adjusted = apply_post_penalty(adjusted, float(post_penalty_info["gen17_math_1_post_penalty"]), post_penalty_weight)
        result.update(post_penalty_info)
        result["probability_after_post_penalty"] = adjusted
        result["components"]["post_penalty_artifact"] = str(post_penalty_dir_path)
        result["components"]["post_penalty_weight"] = post_penalty_weight

    post_boost_dir = config.get("post_boost_artifact_dir")
    if post_boost_dir:
        post_boost_dir_path = _resolve_path(post_boost_dir)
        post_boost_model = load_post_boost_model(post_boost_dir_path / "model.json")
        post_boost_info = score_post_boost_row(math_chunk_row, post_boost_model)
        post_boost_weight = float(config.get("post_boost_weight") or post_boost_model.get("recommended_boost_weight", 0.25))
        adjusted = apply_post_boost(adjusted, float(post_boost_info["gen17_math_1_post_boost"]), post_boost_weight)
        result.update(post_boost_info)
        result["probability_after_post_boost"] = adjusted
        result["components"]["post_boost_artifact"] = str(post_boost_dir_path)
        result["components"]["post_boost_weight"] = post_boost_weight

    result["probability_bot"] = adjusted
    result["synth_name"] = str(config.get("artifact_name") or ARTIFACT_DIR.name)
    return result


def score_chunk(chunk: Sequence[dict], model: Optional[Dict[str, object]] = None) -> float:
    return float(score_chunk_details(chunk, config=model or load_config())["probability_bot"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a chunk with a synthesized gen17 pre+math artifact")
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
