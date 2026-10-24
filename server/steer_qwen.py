"""Steering runs with transformers on Modal (one GPU function, same image and weights as the
replay app). Plan only unless --go.

    modal run server/steer_qwen.py --jobs <steer_jobs.json> --tag steer-pilot --samples 4 --timeout-min 45 --go

Uploads the jobs and the two direction files to qwen36-activations:/<tag>/inputs/, runs
steer_worker.py, commits outputs every 2 minutes, stops starting new calls 3 minutes before
the hard timeout. Fetch with: modal volume get qwen36-activations /<tag>/out <dir>
"""

import json
import subprocess
import threading
import time
from pathlib import Path

import modal

from pins import GPU, VLLM_IMAGE, WEIGHTS_DIR, WEIGHTS_VOLUME, snapshot_path

MINUTES = 60
ACTS_DIR = "/acts"
HOURLY_USD = 3.95
FLA_VERSION = "0.5.2"  # same kernels and image as replay_qwen.py
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME)
acts_volume = modal.Volume.from_name("qwen36-activations", create_if_missing=True)
image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .run_commands("uv pip install --system --python /usr/bin/python3 --no-deps "
                  f"flash-linear-attention=={FLA_VERSION} fla-core=={FLA_VERSION}")
    .env({"HF_HOME": WEIGHTS_DIR, "HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_file(Path(__file__).with_name("steer_worker.py"), "/root/steer_worker.py")
    .add_local_file(Path(__file__).with_name("steer_batched_worker.py"), "/root/steer_batched_worker.py")
    .add_local_python_source("pins")
)
app = modal.App("qwen36-steer")


@app.function(image=image, gpu=GPU, volumes={WEIGHTS_DIR: weights_volume, ACTS_DIR: acts_volume},
              timeout=60 * MINUTES, max_containers=1)
def steer(tag: str, layer: int, strengths: str, samples: int, max_new_tokens: int, c_only: bool,
          timeout_s: int, unsteered_only: bool = False, batched: bool = False, strength: float = 4.0,
          in_batch_unsteered: bool = False) -> dict:
    started = time.time()
    acts_volume.reload()
    inp = Path(ACTS_DIR) / tag / "inputs"
    out = Path(ACTS_DIR) / tag / "out"
    out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def commit_loop():
        while not stop.wait(120):
            acts_volume.commit()

    th = threading.Thread(target=commit_loop, daemon=True)
    th.start()
    cmd = ["/usr/bin/python3", "/root/steer_worker.py", "--model", snapshot_path(), "--jobs", str(inp / "jobs.json"),
           "--direction", str(inp / "direction.npz"), "--random-direction", str(inp / "random_direction.npz"),
           "--layer", str(layer), "--strengths", strengths, "--samples", str(samples),
           "--max-new-tokens", str(max_new_tokens), "--out", str(out),
           "--deadline-s", str(int(timeout_s - (time.time() - started) - 180))]
    if c_only:
        cmd.append("--c-only")
    if unsteered_only:
        cmd.append("--unsteered-only")
    if batched:  # all conditions of a call share generate() batches (steer_batched_worker.py)
        cmd = ["/usr/bin/python3", "/root/steer_batched_worker.py", "--model", snapshot_path(),
               "--jobs", str(inp / "jobs.json"), "--direction", str(inp / "direction.npz"),
               "--random-direction", str(inp / "random_direction.npz"), "--layer", str(layer),
               "--strength", str(strength), "--samples", str(samples), "--max-new-tokens", str(max_new_tokens),
               "--out", str(out), "--deadline-s", str(int(timeout_s - (time.time() - started) - 180))]
        if in_batch_unsteered:
            cmd.append("--unsteered")
    code = subprocess.run(cmd).returncode
    stop.set()
    th.join()
    acts_volume.commit()
    log = json.loads((out / "_log.json").read_text()) if (out / "_log.json").exists() else None
    return {"returncode": code, "container_s": round(time.time() - started, 1), "log": log}


@app.local_entrypoint()
def main(jobs: str, tag: str, direction: str, random_direction: str, layer: int = 47, strengths: str = "2,4",
         samples: int = 4, max_new_tokens: int = 6144, c_only: bool = False, timeout_min: int = 45,
         unsteered_only: bool = False, batched: bool = False, strength: float = 4.0,
         in_batch_unsteered: bool = False, go: bool = False) -> None:
    batch = json.loads(Path(jobs).read_text())
    n_cond = (1 if unsteered_only else 3 + (1 if in_batch_unsteered else 0) if batched
              else 1 + 2 * len(strengths.split(",")) + 1 + (1 if c_only else 0))
    print(f"{len(batch)} calls x {n_cond} conditions x {samples} samples, layer {layer}, strengths {strengths} SD; "
          f"longest prompt {max(len(j['token_ids']) + len(j.get('reply_ids', [])) for j in batch):,} tokens; hard timeout {timeout_min} min "
          f"= ${timeout_min / 60 * HOURLY_USD:.2f} at most")
    if not go:
        print("plan only; pass --go to start the GPU")
        return
    with acts_volume.batch_upload(force=True) as up:
        up.put_file(jobs, f"/{tag}/inputs/jobs.json")
        up.put_file(direction, f"/{tag}/inputs/direction.npz")
        up.put_file(random_direction, f"/{tag}/inputs/random_direction.npz")
    t0 = time.time()
    res = steer.with_options(timeout=timeout_min * MINUTES).remote(tag, layer, strengths, samples, max_new_tokens,
                                                                    c_only, timeout_min * MINUTES, unsteered_only,
                                                                    batched, strength, in_batch_unsteered)
    wall = (time.time() - t0) / 60
    log = res.pop("log") or {}
    print(json.dumps(res))
    for t in log.get("teacher_forcing", []):
        print("teacher forcing", json.dumps(t))
    for c in log.get("calls", []):
        print(json.dumps({k: c.get(k) for k in ("id", "n_prompt_tokens", "prefill_s", "peak_gpu_gib", "batches")}))
        conds = c.get("conditions", {})
        for cond in (conds.items() if isinstance(conds, dict) else enumerate(conds)):
            print("   ", json.dumps(cond))
    print(f"GPU: container {res['container_s'] / 60:.1f} min, call wall {wall:.1f} min ~ ${wall / 60 * HOURLY_USD:.2f}")
