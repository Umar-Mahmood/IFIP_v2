# NetPolAgent

Reference implementation of NetPolAgent, a self-correcting multi-agent
pipeline that translates natural language network security intents into
verified Kubernetes `NetworkPolicy` YAML using small, locally-hosted LLMs.

- `policy_agent/` — the six-stage pipeline (Extractor, Generator, Critic,
  Verifier, Semantic Verifier, Refiner)
- `run_batch.py` — batch runner used for all experiments in the paper

## Usage

```
python3 run_batch.py kubernetes_policies.csv --mode full
```

Requires [Ollama](https://ollama.com) running locally with the models
configured in `policy_agent/config.py`, and a reachable Kubernetes cluster
for the Verifier's live connectivity tests.

## Dataset

https://github.com/Umar-Mahmood/IFIP_v2_dataset
# IFIP_v2
