"""Poker44 miner release."""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

try:
    from dotenv import load_dotenv

    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.miner_heuristics import get_chunk_scorer_startup_check, score_chunk_runtime_bundle
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


FORCED_VALIDATOR_HOTKEYS = {
    "5GgnyzhZ6ozkdnQumwRuEaULggvMr2np4SS3N7eDCMMrXoMC",
}

EXTRA_ALLOWED_VALIDATOR_HOTKEYS = {
    "5FZD47WhA1UaVicYAr7pGnWb2YQLMD7uViipDYN2r1AJ5ggD",
}

def _runtime_implementation_files(repo_root: Path) -> List[Path]:
    component_files = sorted(
        path
        for path in (repo_root / "components").rglob("*")
        if path.is_file()
    )
    return [
        Path(__file__).resolve(),
        repo_root / "start_miner.sh",
        repo_root / "README.md",
        repo_root / "poker44" / "__init__.py",
        repo_root / "poker44" / "base" / "miner.py",
        repo_root / "poker44" / "base" / "neuron.py",
        repo_root / "poker44" / "miner_heuristics.py",
        repo_root / "poker44" / "utils" / "config.py",
        repo_root / "poker44" / "utils" / "misc.py",
        repo_root / "poker44" / "utils" / "model_manifest.py",
        repo_root / "poker44" / "validator" / "synapse.py",
    ] + component_files


def _load_env_file():
    """Load .env file from current directory or parent directories."""
    if not HAS_DOTENV:
        return

    search_path = Path.cwd()
    for _ in range(5):
        env_file = search_path / ".env"
        if env_file.exists():
            try:
                load_dotenv(env_file, override=False)
                print(f"[.env] Loaded environment from {env_file}")
                return
            except Exception as e:
                print(f"[.env] Warning: Failed to load {env_file}: {e}")
                return
        search_path = search_path.parent
        if search_path == search_path.parent:
            break


class Miner(BaseMinerNeuron):
    """Deterministic runtime chunk scorer."""

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("Poker44 Miner started")

        chunk_scorer = "runtime"
        bt.logging.info("[init] POKER44_CHUNK_SCORER=runtime (hardcoded)")

        scorer_check = get_chunk_scorer_startup_check(chunk_scorer)
        if scorer_check.get("active"):
            details = scorer_check.get("details") or {}
            if scorer_check.get("ok"):
                bt.logging.info(
                    "[init] Chunk scorer startup check: ok "
                    f"scorer={scorer_check.get('scorer')} details={details}"
                )
            else:
                bt.logging.error(
                    "[init] Chunk scorer startup check: FAILED "
                    f"scorer={scorer_check.get('scorer')} "
                    f"error={scorer_check.get('error')} details={details}"
                )

        bt.logging.info(f"Axon created: {self.axon}")
        bt.logging.info(f"Build timestamp: {datetime.now(timezone.utc).isoformat()}")

        self._project_root = Path(__file__).resolve().parent.parent
        repo_root = Path(__file__).resolve().parents[1]

        try:
            git_commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
        except Exception:
            git_commit = os.getenv("POKER44_MODEL_REPO_COMMIT", "")

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=_runtime_implementation_files(repo_root),
            defaults={
                "model_name": "gen21combo1",
                "model_version": "21.1",
                "framework": "python-local-ensemble",
                "license": "MIT",
                "repo_url": "https://github.com/tomkaba/poker44-miner-gen21-combo1",
                "repo_commit": git_commit,
                "notes": "gen21combo1 majority-vote ensemble over ml17_pre3, gen17_tuner_pre6, gen17_tuner2_pre2, gen17_tuner_synth15, ml20tens1, and gen18_1 with per-model thresholds preserved and ml20tens1 as the 3:3 tie-breaker.",
                "open_source": True,
                "inference_mode": "local",
                "training_data_statement": "Uses copied local scorer artifacts from the selected gen17, gen18, and ml20 runtime releases, preserving each release threshold inside the ensemble.",
                "private_data_attestation": "This miner does not train on validator-private human data.",
                "data_attestation": "This miner does not train on validator-private human data.",
            },
        )

        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest full={json.dumps(self.model_manifest, ensure_ascii=True, sort_keys=True)}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            f"Implementation sha256={self.model_manifest.get('implementation_sha256', '')} "
            f"files={len(self.model_manifest.get('implementation_files', []) or [])}"
        )
        bt.logging.info(f"Project root: {repo_root}")

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        chunks: List[List[dict]] = synapse.chunks or []

        scores = []
        routes = []
        predictions = []
        for chunk in chunks:
            result = score_chunk_runtime_bundle(chunk)
            scores.append(float(result["score"]))
            routes.append(str(result["route"]))
            predictions.append(bool(result["prediction"]))

        chunk_sizes = [len(chunk or []) for chunk in chunks]

        def _preview(values, limit=8):
            if values is None:
                return []
            if len(values) <= limit:
                return values
            return values[:limit] + [f"...(+{len(values) - limit} more)"]

        bt.logging.debug(f"[miner] Received {len(chunks)} chunk(s); first sizes={_preview(chunk_sizes)}")

        synapse.risk_scores = scores
        synapse.predictions = predictions
        synapse.model_manifest = dict(self.model_manifest)
        bt.logging.info(
            f"[miner] Request result | chunks={len(chunks)} "
            f"scores={_preview(scores)} routes={_preview(routes)} "
            f"predictions={_preview(synapse.predictions)} "
            f"manifest={json.dumps(synapse.model_manifest, ensure_ascii=True, sort_keys=True)}"
        )

        bt.logging.debug(
            f"[DEBUG] Before sending: synapse.risk_scores={synapse.risk_scores}, "
            f"type={type(synapse.risk_scores)}, len={len(synapse.risk_scores) if synapse.risk_scores is not None else 'None'}"
        )
        bt.logging.debug(
            f"[miner] Responding with scores={_preview(scores)} "
            f"routes={_preview(routes)} predictions={_preview(synapse.predictions)}"
        )

        source_hotkey = getattr(getattr(synapse, "dendrite", None), "hotkey", "unknown")
        self._append_request_log(
            validator_hotkey=source_hotkey,
            chunk_sizes=chunk_sizes,
            chunk_routes=routes,
            scores=scores,
            predictions=synapse.predictions,
            chunks=chunks,
        )

        bt.logging.info(f"Scored {len(chunks)} chunks with scorer gen21combo1.")
        return synapse

    @staticmethod
    def _flag_enabled(config_section, attr, default=None):
        value = getattr(config_section, attr, default) if config_section else default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    def _allowed_validator_hotkeys(self) -> set[str]:
        cfg = getattr(self.config, "blacklist", None)
        allowed = set(FORCED_VALIDATOR_HOTKEYS) | set(EXTRA_ALLOWED_VALIDATOR_HOTKEYS)

        def _normalize(value) -> set[str]:
            if value is None:
                return set()
            if isinstance(value, (list, tuple, set)):
                iterable = value
            else:
                iterable = str(value).split(",")
            return {str(item).strip() for item in iterable if str(item).strip()}

        allowed |= _normalize(getattr(cfg, "forced_validator_hotkey", None))
        allowed |= _normalize(getattr(cfg, "forced_validator_hotkeys", None))
        allowed |= _normalize(getattr(cfg, "extra_validator_hotkeys", None))
        return allowed

    def score_chunk(self, chunk: list[dict]) -> float:
        return score_chunk_runtime_bundle(chunk)["score"]

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            bt.logging.warning("Received a request without a dendrite or hotkey.")
            return True, "Missing dendrite or hotkey"

        allow_non_registered = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "allow_non_registered",
            False,
        )
        force_validator_permit = self._flag_enabled(
            getattr(self.config, "blacklist", None),
            "force_validator_permit",
            True,
        )
        allowed_hotkeys = self._allowed_validator_hotkeys()

        if synapse.dendrite.hotkey in allowed_hotkeys:
            return False, "Validator allowlist"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            if not allow_non_registered:
                return True, "Unrecognized hotkey"
            return False, "Unregistered hotkey allowed"

        uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
        if force_validator_permit and not self.metagraph.validator_permit[uid]:
            return True, "Non-validator hotkey"

        return False, "Hotkey recognized!"

    async def priority(self, synapse: DetectionSynapse) -> float:
        return self.caller_priority(synapse)

    def _get_log_path(self) -> Path:
        uid = getattr(self, "uid", None)
        suffix = uid if uid is not None else "unknown"
        return self._project_root / f"miner_{suffix}.log"

    def _append_request_log(
        self,
        validator_hotkey,
        chunk_sizes,
        chunk_routes,
        scores,
        predictions,
        chunks,
    ) -> None:
        entry = {
            "timestamp": time.time(),
            "validator_hotkey": validator_hotkey,
            "miner_hotkey": getattr(self.wallet.hotkey, "ss58_address", "unknown"),
            "chunk_count": len(chunk_sizes),
            "chunk_sizes": chunk_sizes,
            "chunk_routes": chunk_routes,
            "scores": scores,
            "predictions": predictions,
            "chunks": chunks,
        }
        try:
            with self._get_log_path().open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as log_error:
            bt.logging.warning(f"Failed to append miner request log: {log_error}")

    def _dump_request_payload(self, *args, **kwargs):
        return


if __name__ == "__main__":
    _load_env_file()

    with Miner() as miner:
        bt.logging.info("Miner running...")
        while True:
            bt.logging.info(
                f"Miner UID: {miner.uid} | Incentive: {float(miner.metagraph.I[miner.uid])} | Scorer: gen21combo1"
            )
            time.sleep(60)
