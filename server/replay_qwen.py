"""Replay rebuilt prompts through Qwen3.6-27B with transformers and save residual-stream
vectors. A second Modal GPU function next to the vLLM server; same weights Volume.

    modal run server/replay_qwen.py --jobs results/replay/jobs.jsonl --tag pilot          # plan only
    modal run server/replay_qwen.py --jobs results/replay/jobs.jsonl --tag pilot --go     # starts a GPU

Outputs go to the Volume qwen36-activations under /<tag>/; fetch them with
    modal volume get qwen36-activations /<tag> <local_dir>
"""

import json
import subprocess
from pathlib import Path

import modal

from pins import GPU, MODEL_REVISION, VLLM_IMAGE, WEIGHTS_DIR, WEIGHTS_VOLUME, snapshot_path

MINUTES = 60
ACTS_DIR = "/acts"
FLA_VERSION = "0.5.2"

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME)
acts_volume = modal.Volume.from_name("qwen36-activations", create_if_missing=True)

image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    # Fast Gated DeltaNet kernels for transformers' linear-attention layers; --no-deps
    # keeps the image's torch/triton untouched.
    .run_commands(
        "uv pip install --system --python /usr/bin/python3 --no-deps "
        f"flash-linear-attention=={FLA_VERSION} fla-core=={FLA_VERSION}"
    )
    .env({"HF_HOME": WEIGHTS_DIR, "HF_HUB_OFFLINE": "1"})
    .add_local_file(Path(__file__).with_name("replay_worker.py"), "/root/replay_worker.py")
    .add_local_python_source("pins")
)

app = modal.App("qwen36-replay")


@app.function(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights_volume, ACTS_DIR: acts_volume},
    timeout=60 * MINUTES,
    max_containers=1,
)
def capture(jobs: list[dict], tag: str, layers: str) -> list[dict]:
    jobs_path = Path("/tmp/jobs.json")
    jobs_path.write_text(json.dumps(jobs))
    out = Path(ACTS_DIR) / tag
    subprocess.run(
        ["/usr/bin/python3", "/root/replay_worker.py", "--model", snapshot_path(),
         "--jobs", str(jobs_path), "--out", str(out), "--layers", layers],
        check=True,
    )
    acts_volume.commit()
    return [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]


@app.local_entrypoint()
def main(jobs: str, tag: str, layers: str = "all", go: bool = False) -> None:
    batch = [json.loads(line) for line in open(jobs)]
    for job in batch:
        if job.get("model_revision") != MODEL_REVISION:
            raise SystemExit(f"{job['id']} was prepared for revision {job.get('model_revision')}")
    tokens = sum(len(j["token_ids"]) for j in batch)
    positions = sum(len({p for ps in j["positions"].values() for p in ps}) for j in batch)
    print(f"{len(batch)} jobs, {tokens} tokens, {positions} positions, layers={layers}, "
          f"output qwen36-activations:/{tag}/ on {GPU}")
    if not go:
        print("plan only; pass --go to start the GPU")
        return
    for meta in capture.remote(batch, tag, layers):
        print(json.dumps({k: meta[k] for k in ("id", "n_tokens", "forward_s", "peak_gpu_gib", "tensors")}))
