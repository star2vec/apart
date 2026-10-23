"""Step 1 (activation cache), part 4 (Mac, after `modal volume get`): check the downloaded passes against
the jobs and the index, and pack one array per layer in index order.

    tools/.venv/bin/python tools/step1_cache/pack.py --acts <downloaded acts dir> --out agent-collusion/results/step1_cache/packed

Writes layer_<L>.npy, shape [n_index_rows, 5120], dtype uint16 holding the bfloat16 bits
exactly as saved (numpy has no bfloat16; `load_layer` below widens to float32), plus
check.json. Row i of every layer is row i of index.jsonl.
"""

import argparse
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "agent-collusion" / "results" / "step1_cache"


def read_bf16_safetensors(path):
    """{name: uint16 array} from a safetensors file of bfloat16 tensors (no torch needed)."""
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if spec["dtype"] != "BF16":
            raise ValueError(f"{path.name}:{name} is {spec['dtype']}")
        s, e = spec["data_offsets"]
        out[name] = np.frombuffer(raw, dtype="<u2", count=(e - s) // 2, offset=8 + n + s).reshape(spec["shape"])
    return out


def load_layer(packed_dir, layer):
    """float32 [n_index_rows, hidden] for one layer."""
    bits = np.load(Path(packed_dir) / f"layer_{layer:02d}.npy")
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", required=True, type=Path)
    parser.add_argument("--out", default=CACHE / "packed", type=Path)
    parser.add_argument("--jobs", default=CACHE / "jobs.jsonl", type=Path)
    parser.add_argument("--index", default=CACHE / "index.jsonl", type=Path)
    args = parser.parse_args()

    expected = {}
    for line in open(args.jobs):
        j = json.loads(line)
        expected[j["id"]] = (j["sha256_token_ids"], sorted({p for ps in j["positions"].values() for p in ps}))
    index = [json.loads(l) for l in open(args.index)]
    missing = sorted(set(expected) - {p.stem for p in args.acts.glob("*.json") if not p.name.startswith("_run_")})
    problems, layers = [], None
    tensors_by_job = {}
    for job_id, (sha, rows) in expected.items():
        meta_path = args.acts / f"{job_id}.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        if meta["sha256_token_ids"] != sha or meta["rows"] != rows:
            problems.append(f"{job_id}: hash or rows differ from the job")
            continue
        t = read_bf16_safetensors(args.acts / f"{job_id}.safetensors")
        layers = layers or sorted(int(k.split(".")[1]) for k in t if k.startswith("resid_post."))
        for k, v in t.items():
            if v.shape[0] != len(rows):
                problems.append(f"{job_id}:{k} has {v.shape[0]} rows, expected {len(rows)}")
            f = (v.astype(np.uint32) << 16).view(np.float32)
            if not np.isfinite(f).all():
                problems.append(f"{job_id}:{k} has non-finite values")
        tensors_by_job[job_id] = t

    args.out.mkdir(parents=True, exist_ok=True)
    norms = {}
    for layer in layers or []:
        hidden = next(iter(tensors_by_job.values()))[f"resid_post.{layer}"].shape[1]
        arr = np.zeros((len(index), hidden), dtype=np.uint16)
        have = np.zeros(len(index), dtype=bool)
        for i, r in enumerate(index):
            t = tensors_by_job.get(r["job"])
            if t is not None:
                arr[i] = t[f"resid_post.{layer}"][r["row"]]
                have[i] = True
        np.save(args.out / f"layer_{layer:02d}.npy", arr)
        f = (arr[have].astype(np.uint32) << 16).view(np.float32)
        norms[layer] = round(float(np.linalg.norm(f, axis=1).mean()), 1)
    check = {"passes_expected": len(expected), "passes_found": len(tensors_by_job), "missing": missing,
             "problems": problems, "layers": layers, "index_rows": len(index),
             "index_rows_with_vectors": int(sum(r["job"] in tensors_by_job for r in index)),
             "mean_norm_by_layer": norms}
    (args.out / "check.json").write_text(json.dumps(check, indent=2))
    print(json.dumps({k: v for k, v in check.items() if k != "missing"} | {"missing": len(missing)}, indent=1))


if __name__ == "__main__":
    main()
