"""Fetch the pinned Qwen3.6-27B revision into a Modal Volume. CPU only, never a GPU.

Usage (from the repo root):
    modal run server/download_weights.py            # download
    modal run server/download_weights.py::verify    # check snapshot symlinks resolve

Secret: huggingface -> HF_TOKEN
"""

import json

import modal

from pins import HF_HUB_VERSION, MODEL_ID, MODEL_REVISION, WEIGHTS_DIR, WEIGHTS_VOLUME

weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"huggingface_hub[hf_xet]=={HF_HUB_VERSION}")
    .env({"HF_HOME": WEIGHTS_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("pins")
)

app = modal.App("qwen36-weights")


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: weights_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
    cpu=4,
)
def download() -> dict:
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
    weights_volume.commit()
    manifest = {"model_id": MODEL_ID, "revision": MODEL_REVISION, "path": path}
    print(json.dumps(manifest))
    return manifest


@app.function(image=image, volumes={WEIGHTS_DIR: weights_volume}, timeout=10 * 60)
def verify() -> dict:
    """Resolve every snapshot symlink and check the shards listed in the index exist."""
    import os

    from pins import snapshot_path

    root = snapshot_path()
    files = {}
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        target = os.path.realpath(path)
        files[name] = os.path.getsize(target) if os.path.isfile(target) else None
    with open(os.path.join(root, "model.safetensors.index.json")) as f:
        shards = set(json.load(f)["weight_map"].values())
    report = {
        "snapshot": root,
        "files": len(files),
        "unresolved": [n for n, size in files.items() if size is None],
        "shards_in_index": len(shards),
        "shards_missing": sorted(s for s in shards if not files.get(s)),
        "total_bytes": sum(size or 0 for size in files.values()),
    }
    print(json.dumps(report, indent=2))
    return report


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(download.remote(), indent=2))
