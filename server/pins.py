"""Pinned model and server settings shared by the Modal apps. Log these with every run."""

MODEL_ID = "Qwen/Qwen3.6-27B"
# Exact Hugging Face commit; vLLM loads the local snapshot for this sha, offline.
MODEL_REVISION = "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
# Must contain "qwen": experiments/llm.py picks its reasoning-replay dialect by name.
SERVED_MODEL_NAME = "qwen3.6-27b"

VLLM_VERSION = "0.31.0"
VLLM_IMAGE = (
    "vllm/vllm-openai:v0.31.0"
    "@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
)
HF_HUB_VERSION = "2.1.1"

# 64,000 output tokens (paper setting) + up to 67,072 prompt tokens.
# vLLM rejects prompt + max_tokens > max_model_len; the game treats that rejection
# as a context-window error and drops the agent's oldest episode from memory.
MAX_MODEL_LEN = 131_072

# Each running sequence holds one Mamba (linear-attention) state block; on an H100 with
# this max_model_len only ~361 fit, below vLLM's default of 1024. 8 runner processes
# need at most ~16 in flight, so 64 leaves headroom without starving the KV cache.
MAX_NUM_SEQS = 64

# "H100!" pins an 80 GB H100 and stops Modal's silent upgrade to H200,
# so timings stay comparable across runs.
GPU = "H100!"

WEIGHTS_VOLUME = "qwen36-27b-weights"
WEIGHTS_DIR = "/models"


def snapshot_path() -> str:
    repo_dir = "models--" + MODEL_ID.replace("/", "--")
    return f"{WEIGHTS_DIR}/hub/{repo_dir}/snapshots/{MODEL_REVISION}"
