"""Replay rebuilt prompts through Qwen3.6-27B with transformers and save residual-stream
vectors. A second Modal GPU function next to the vLLM server; same weights Volume.

    modal run server/replay_qwen.py --jobs <jobs.jsonl> --tag cache --layers 3,7,...      # plan only
    modal run server/replay_qwen.py --jobs <jobs.jsonl> --tag cache --layers 3,7,... \\
        --timeout-min 90 --go                                                            # starts a GPU

--go uploads the jobs to the Volume qwen36-activations under /<tag>/inputs/, runs the worker
(longest pass first; whole passes unless one runs out of memory, then chunks with the
model's state carried across them), commits finished passes every 2 minutes so a timeout
keeps them, and stops starting new passes 3 minutes before the hard timeout. Rerunning the
same command resumes: finished passes are skipped. Fetch outputs with
    modal volume get qwen36-activations /<tag>/acts <local_dir>
"""

import gzip
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import modal

from pins import GPU, MODEL_REVISION, VLLM_IMAGE, WEIGHTS_DIR, WEIGHTS_VOLUME, snapshot_path

MINUTES = 60
ACTS_DIR = "/acts"
ACTS_VOLUME = "qwen36-activations"
FLA_VERSION = "0.5.2"
ACCELERATE_VERSION = "1.15.0"  # transformers needs it for device_map loading
HOURLY_USD = 3.95  # Modal H100 list price
COMMIT_EVERY_S = 120
STOP_MARGIN_S = 180

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME)
acts_volume = modal.Volume.from_name(ACTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    # Fast Gated DeltaNet kernels for transformers' linear-attention layers, and accelerate
    # for device_map loading; --no-deps keeps the image's torch/triton untouched.
    .run_commands(
        "uv pip install --system --python /usr/bin/python3 --no-deps "
        f"flash-linear-attention=={FLA_VERSION} fla-core=={FLA_VERSION} accelerate=={ACCELERATE_VERSION}"
    )
    .env({"HF_HOME": WEIGHTS_DIR, "HF_HUB_OFFLINE": "1",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # long passes fragment memory
    .add_local_file(Path(__file__).with_name("replay_worker.py"), "/root/replay_worker.py")
    .add_local_python_source("pins")
)

app = modal.App("qwen36-replay")


@app.function(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS_DIR: weights_volume, ACTS_DIR: acts_volume},
    timeout=60 * MINUTES,  # replaced per run by --timeout-min
    max_containers=1,
)
def capture(tag: str, layers: str, chunk: int, timeout_s: int, check_chunking: bool) -> dict:
    started = time.time()
    acts_volume.reload()
    jobs_path = Path(ACTS_DIR) / tag / "inputs" / "jobs.jsonl.gz"
    out = Path(ACTS_DIR) / tag / "acts"
    out.mkdir(parents=True, exist_ok=True)

    stop = threading.Event()

    def commit_loop():
        while not stop.wait(COMMIT_EVERY_S):
            acts_volume.commit()

    committer = threading.Thread(target=commit_loop, daemon=True)
    committer.start()
    deadline = timeout_s - (time.time() - started) - STOP_MARGIN_S
    cmd = ["/usr/bin/python3", "/root/replay_worker.py", "--model", snapshot_path(),
           "--jobs", str(jobs_path), "--out", str(out), "--layers", layers,
           "--chunk", str(chunk), "--deadline-s", str(int(deadline))]
    if check_chunking:
        cmd.append("--check-chunking")
    code = subprocess.run(cmd).returncode
    stop.set()
    committer.join()
    acts_volume.commit()
    logs = [json.loads(p.read_text()) for p in sorted(out.glob("_run_*.json"))]
    return {"returncode": code, "container_s": round(time.time() - started, 1),
            "passes_done": len([p for p in out.glob("*.json") if not p.name.startswith("_run_")]),
            "run_log": logs[-1] if logs else None}


@app.local_entrypoint()
def main(jobs: str, tag: str, layers: str = "all", go: bool = False, chunk: int = 0,
         timeout_min: int = 60, check_chunking: bool = False, tokens_per_s: float = 5000.0) -> None:
    batch = [json.loads(line) for line in open(jobs)]
    for job in batch:
        if job.get("model_revision") != MODEL_REVISION:
            raise SystemExit(f"{job['id']} was prepared for revision {job.get('model_revision')}")
    tokens = sum(len(j["token_ids"]) for j in batch)
    positions = sum(len({p for ps in j["positions"].values() for p in ps}) for j in batch)
    n_layers = 64 if layers == "all" else len(layers.split(","))
    gib = positions * n_layers * 5120 * 2 / 2**30
    est_min = tokens / tokens_per_s / 60
    print(f"{len(batch)} passes, {tokens:,} tokens (longest {max(len(j['token_ids']) for j in batch):,}), "
          f"{positions:,} positions x {n_layers} layers = {gib:.2f} GiB, output {ACTS_VOLUME}:/{tag}/acts on {GPU}")
    print(f"forward time at {tokens_per_s:,.0f} tok/s: {est_min:.0f} min (+ image pull and weight load); "
          f"hard timeout {timeout_min} min = ${timeout_min / 60 * HOURLY_USD:.2f} at most")
    if not go:
        print("plan only; pass --go to start the GPU")
        return

    with tempfile.TemporaryDirectory() as tmp:
        packed = Path(tmp) / "jobs.jsonl.gz"
        with open(jobs, "rb") as src, gzip.open(packed, "wb") as dst:
            shutil.copyfileobj(src, dst)
        with acts_volume.batch_upload(force=True) as upload:
            upload.put_file(str(packed), f"/{tag}/inputs/jobs.jsonl.gz")
    print(f"uploaded jobs to {ACTS_VOLUME}:/{tag}/inputs/jobs.jsonl.gz", flush=True)

    t0 = time.time()
    result = capture.with_options(timeout=timeout_min * MINUTES).remote(
        tag, layers, chunk, timeout_min * MINUTES, check_chunking)
    wall_min = (time.time() - t0) / 60
    log = result.pop("run_log") or {}
    print(json.dumps(result))
    for key in ("environment", "load_s", "mode", "chunking_check", "events", "total_s"):
        if key in log:
            print(key, json.dumps(log[key]))
    done = log.get("jobs", [])
    if done:
        tok = sum(j["n_tokens"] for j in done)
        fwd = sum(j["forward_s"] for j in done)
        print(f"this run: {len(done)} passes, {tok:,} tokens in {fwd / 60:.1f} min forward "
              f"({tok / fwd:,.0f} tok/s); peak GPU memory {max(j['peak_gpu_gib'] or 0 for j in done)} GiB "
              f"(longest pass {done[0]['n_tokens']:,} tokens: {done[0]['peak_gpu_gib']} GiB)")
    print(f"GPU: container {result['container_s'] / 60:.1f} min, call wall {wall_min:.1f} min "
          f"~ ${wall_min / 60 * HOURLY_USD:.2f} (Modal dashboard has the billed figure)")
