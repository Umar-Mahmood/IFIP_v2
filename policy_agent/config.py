"""Central configuration for the multi-agent NetworkPolicy pipeline."""

OLLAMA_BASE_URL = "http://localhost:11434"

# Model assignments per agent role
EXTRACTOR_MODEL = "llama3.1:8b-instruct-fp16"
GENERATOR_MODEL = "llama3.1:8b-instruct-fp16"
CRITIC_MODEL    = "llama3.1:8b-instruct-fp16"
REFINER_MODEL   = "deepseek-coder-v2:latest"
EMBEDDING_MODEL = "nomic-embed-text"
# Note: EMBEDDING_MODEL/round_trip_similarity is a logged diagnostic only.
# calibrate_round_trip_threshold.py showed no similarity threshold reliably
# separates correct from corrupted extractions (a single flipped word or
# CIDR digit barely moves cosine similarity) — the actual round-trip trust
# signal is the LLM judge in extractor.py:_llm_same_meaning.

# Pipeline settings
MAX_ITERATIONS   = 5
OLLAMA_TIMEOUT   = 300   # seconds per LLM call

# Kubernetes test settings
TEST_NAMESPACE        = "netpol-verify"
TEST_NAMESPACE_CLIENT = "netpol-client"
POD_READY_TIMEOUT     = 60   # seconds to wait for pods
CONNECTIVITY_TIMEOUT  = 5    # seconds for curl/nc test
