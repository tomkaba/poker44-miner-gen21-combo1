# gen21combo1

Minimal release repository for Poker44 miner runtime scoring.

This repository is a standalone Poker44 miner release for `gen21combo1`, a 6-model local ensemble. The miner loads copied component artifacts locally, applies each component's original runtime threshold, and emits the combo prediction via majority vote with `ml20tens1` as the 3:3 tie-breaker.

## Quick start

```bash
git clone https://github.com/tomkaba/poker44-miner-gen21-combo1.git
cd poker44-miner-gen21-combo1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run Miner

```bash
python neurons/miner.py
```

or legacy wrapper:

```bash
./start_miner.sh HOTKEY_ID[,HOTKEY_ID2,...]
```

## Implementation

- Launcher: start_miner.sh
- Scorer entrypoint: poker44/miner_heuristics.py
- Entry point: neurons/miner.py
- Components: components/ml17_pre3, components/gen17_tuner_pre6, components/gen17_tuner2_pre2, components/gen17_tuner_synth15, components/ml20tens1, components/gen18_1

Component thresholds preserved in the combo runtime:

- ml17_pre3: 0.5
- gen17_tuner_pre6: 0.5
- gen17_tuner2_pre2: 0.5
- gen17_tuner_synth15: 0.5
- ml20tens1: 0.5
- gen18_1: 0.207742

Combo decision rule:

- evaluate all 6 component scorers locally
- convert each score to a boolean using that component's own threshold
- return the majority vote as the combo prediction
- if votes split 3:3, use `ml20tens1` as the tie-breaker

Published repo: https://github.com/tomkaba/poker44-miner-gen21-combo1

Base release lineage: copied local component artifacts wired into the `gen21combo1` ensemble runtime.

Manifest implementation SHA256 is computed from:

- start_miner.sh
- neurons/miner.py
- poker44/__init__.py
- poker44/base/miner.py
- poker44/base/neuron.py
- poker44/miner_heuristics.py
- poker44/utils/config.py
- poker44/utils/misc.py
- poker44/utils/model_manifest.py
- poker44/validator/synapse.py
- all files under components/

The generated release manifest in `models/model_manifest.json` is built from this miner flow and lists the implementation files that participate in runtime scoring and response generation.
