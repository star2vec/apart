"""Serve Qwen3.6-27B (bf16) with vLLM on Modal as an OpenAI-compatible endpoint.

Usage (from the repo root):
    modal run server/download_weights.py       # once, CPU only: fills the weights Volume
    modal deploy server/serve_qwen.py          # registers the GPU endpoint (scales to zero)
    modal app stop qwen36-vllm                 # stops any running container

Secret: vllm-api-key -> VLLM_API_KEY (clients send it as the bearer token)
"""

import json
import os
import subprocess

import modal

from pins import (
    GPU,
    MAX_MODEL_LEN,
    MODEL_ID,
    MODEL_REVISION,
    SERVED_MODEL_NAME,
    VLLM_IMAGE,
    VLLM_VERSION,
    WEIGHTS_DIR,
    WEIGHTS_VOLUME,
    snapshot_path,
)

MINUTES = 60
REQUEST_TIMEOUT = 60 * MINUTES  # hard cap per function input (one HTTP request)
STARTUP_TIMEOUT = 30 * MINUTES  # cold start: load ~52 GiB, compile, capture graphs
SCALEDOWN_WINDOW = 5 * MINUTES  # idle time before the GPU container shuts down
VLLM_PORT = 8000

# --- Storage ---------------------------------------------------------------------------

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME)
# torch.compile / CUDA-graph artefacts, so later cold starts are shorter.
vllm_cache_volume = modal.Volume.from_name("vllm-compile-cache", create_if_missing=True)


# --- Images ----------------------------------------------------------------------------

serve_image = (
    # The image's Python is uv-managed and invisible to Modal's version probe, so Modal
    # adds its own interpreter for the launcher; `vllm` still runs in the image's env.
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .env(
        {
            "HF_HOME": WEIGHTS_DIR,
            "HF_HUB_OFFLINE": "1",
            "VLLM_CACHE_ROOT": "/root/.cache/vllm",
        }
    )
    .add_local_python_source("pins")
)

app = modal.App("qwen36-vllm")


# --- Functions -------------------------------------------------------------------------


def vllm_command() -> list[str]:
    return [
        "vllm",
        "serve",
        snapshot_path(),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(MAX_MODEL_LEN),
        # Text-only game: skip loading the vision tower; text path is unchanged.
        "--language-model-only",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        # Unset sampling fields fall back to the model's generation_config.json
        # (top_p=0.95, top_k=20); the game sets temperature and max_tokens itself.
        "--generation-config",
        "auto",
        "--api-key",
        os.environ["VLLM_API_KEY"],
    ]


@app.function(
    image=serve_image,
    gpu=GPU,
    volumes={
        WEIGHTS_DIR: weights_volume,
        "/root/.cache/vllm": vllm_cache_volume,
    },
    secrets=[modal.Secret.from_name("vllm-api-key")],
    timeout=REQUEST_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=0,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
@modal.web_server(port=VLLM_PORT, startup_timeout=STARTUP_TIMEOUT)
def serve() -> None:
    if not os.path.isdir(snapshot_path()):
        raise RuntimeError(
            f"Weights for {MODEL_ID}@{MODEL_REVISION} are not in the Volume; "
            "run `modal run server/download_weights.py` first."
        )
    print(json.dumps(server_config()))
    cmd = vllm_command()
    redacted = [("<redacted>" if a == os.environ["VLLM_API_KEY"] else a) for a in cmd]
    print("launching:", " ".join(redacted), flush=True)
    subprocess.Popen(cmd)


def server_config() -> dict:
    """Everything a run log needs to identify the server. No secrets."""
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "served_model_name": SERVED_MODEL_NAME,
        "dtype": "bfloat16",
        "quantization": None,
        "vllm_version": VLLM_VERSION,
        "vllm_image": VLLM_IMAGE,
        "max_model_len": MAX_MODEL_LEN,
        "gpu": GPU,
        "reasoning_parser": "qwen3",
        "tool_call_parser": "qwen3_coder",
        "language_model_only": True,
        "request_timeout_s": REQUEST_TIMEOUT,
        "scaledown_window_s": SCALEDOWN_WINDOW,
    }

