#!/usr/bin/env python3
import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


ARTIFACT_DIR = Path(__file__).resolve().parent
FEATURE_EXTRACTOR_PATH = ARTIFACT_DIR / "feature_extractor_frozen.py"
HAND_MODEL_PATH = ARTIFACT_DIR / "hand_model.npz"
CHUNK_MODEL_PATH = ARTIFACT_DIR / "chunk_aggregator_model.npz"


def _load_feature_module():
    spec = importlib.util.spec_from_file_location("gen17_stage1_hl_feature_extractor", FEATURE_EXTRACTOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load feature extractor from {FEATURE_EXTRACTOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FEATURE_MODULE = _load_feature_module()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def load_npz_model(model_path: Path) -> Dict[str, object]:
    payload = np.load(model_path, allow_pickle=True)
    feature_names = [str(item) for item in payload["feature_names"].tolist()]
    return {
        "weights": payload["weights"].astype(np.float64),
        "bias": float(payload["bias"][0]),
        "mean": payload["mean"].astype(np.float64),
        "std": payload["std"].astype(np.float64),
        "feature_names": feature_names,
    }


def load_model(_model_path: Path | None = None) -> Dict[str, Dict[str, object]]:
    return {
        "hand_model": load_npz_model(HAND_MODEL_PATH),
        "chunk_model": load_npz_model(CHUNK_MODEL_PATH),
    }


def _standardize(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = std.copy()
    safe_std[safe_std < 1e-8] = 1.0
    return (values - mean) / safe_std


def _extract_hand_rows(chunk: Sequence[dict]) -> List[Dict[str, object]]:
    hand_rows: List[Dict[str, object]] = []
    for hand_index, hand in enumerate(chunk):
        if not isinstance(hand, dict):
            continue
        hand_features, _ = _FEATURE_MODULE.compute_hand_features(hand, hand_index)
        hand_rows.append(hand_features)
    if not hand_rows:
        raise ValueError("Chunk has no valid hands for hand-level scoring")
    return hand_rows


def score_chunk_details(
    chunk: Sequence[dict],
    hand_model: Dict[str, object] | None = None,
    chunk_model: Dict[str, object] | None = None,
) -> Dict[str, object]:
    hand_model = hand_model or load_npz_model(HAND_MODEL_PATH)
    chunk_model = chunk_model or load_npz_model(CHUNK_MODEL_PATH)

    hand_rows = _extract_hand_rows(chunk)
    hand_matrix = np.array(
        [[float(row[name]) for name in hand_model["feature_names"]] for row in hand_rows],
        dtype=np.float64,
    )
    hand_matrix = _standardize(hand_matrix, hand_model["mean"], hand_model["std"])
    hand_probs = sigmoid(hand_matrix @ hand_model["weights"] + hand_model["bias"])

    vals = np.array(hand_probs, dtype=np.float64)
    ordered = np.sort(vals)
    top3 = ordered[-3:] if len(ordered) >= 3 else ordered
    bottom3 = ordered[:3]
    chunk_feature_map = {
        "hand_count": float(len(vals)),
        "prob_mean": float(vals.mean()),
        "prob_median": float(np.median(vals)),
        "prob_std": float(vals.std()) if len(vals) > 1 else 0.0,
        "prob_min": float(vals.min()),
        "prob_max": float(vals.max()),
        "prob_p10": float(np.quantile(vals, 0.10)),
        "prob_p90": float(np.quantile(vals, 0.90)),
        "prob_top3_mean": float(top3.mean()),
        "prob_bottom3_mean": float(bottom3.mean()),
    }
    chunk_vector = np.array([float(chunk_feature_map[name]) for name in chunk_model["feature_names"]], dtype=np.float64)
    chunk_vector = _standardize(chunk_vector, chunk_model["mean"], chunk_model["std"])
    chunk_probability = float(sigmoid(chunk_vector @ chunk_model["weights"] + chunk_model["bias"]))

    return {
        "probability_bot": chunk_probability,
        "hand_count": int(len(vals)),
        "hand_prob_mean": float(vals.mean()),
        "hand_prob_std": float(vals.std()) if len(vals) > 1 else 0.0,
    }


def score_chunk(chunk: Sequence[dict], model: Dict[str, Dict[str, object]] | None = None) -> float:
    model = model or load_model()
    result = score_chunk_details(
        chunk,
        hand_model=model["hand_model"],
        chunk_model=model["chunk_model"],
    )
    return float(result["probability_bot"])


def _extract_chunk(payload: object) -> Sequence[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("hands"), list):
            return payload["hands"]
        if isinstance(payload.get("chunk"), list):
            return payload["chunk"]
    raise ValueError("Unsupported payload format; expected a list of hands or object with 'hands'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a chunk with gen17-preprod-stage1-hl")
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