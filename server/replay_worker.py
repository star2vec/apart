"""Residual-stream capture with transformers. Runs inside the vLLM image under its own
Python (/usr/bin/python3: torch 2.13, transformers 5.17), launched by replay_qwen.py.
Also runs on CPU with a small random model for offline tests (--device cpu).

For each job (longest first): one no-grad forward pass of the text model over the job's
token ids, forward hooks on the chosen decoder layers (and a pre-hook on layer 0 for the
embedding output when layer -1 is asked for), keeping only the job's positions. Saves
<out>/<job>.safetensors (tensor "resid_post.<layer>" or "resid_pre.0", shape
[n_positions, hidden], bfloat16 as computed; rows = sorted unique positions) and <job>.json.

Passes run whole (--chunk 0) unless a pass runs out of GPU memory; then that pass and all
later ones are split into chunks of --fallback-chunk tokens, with the model's cache carrying
attention keys/values and the Gated DeltaNet conv and recurrent states across chunks.
Before any chunked output is kept, a whole-vs-chunked comparison on the shortest job must
agree (cosine >= --min-cosine at every saved position and layer); --check-chunking runs that
comparison at start regardless.

Jobs whose outputs already exist with the same token hash are skipped (resume). The worker
stops starting new passes once the predicted finish would pass --deadline-s.
"""

import argparse
import gzip
import hashlib
import json
import platform
import re
import time
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file


def decoder_layers(model, n_layers):
    candidates = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList) and len(module) == n_layers
        and name.endswith("layers") and "visual" not in name and "mtp" not in name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one decoder ModuleList, found {[n for n, _ in candidates]}")
    return candidates[0]


def text_model(model):
    """The decoder stack that owns the layers (Qwen3_5TextModel inside the multimodal wrapper)."""
    for name in ("model.language_model", "language_model", "model"):
        mod = model
        try:
            for part in name.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if hasattr(mod, "layers") and hasattr(mod, "embed_tokens"):
            return mod
    raise RuntimeError("no text model with layers and embed_tokens")


def safe_name(job_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", job_id)


def load_jobs(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        text = f.read()
    text = text.strip()
    return json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines()]


class Capture:
    """Hooks that keep the residual stream at given absolute positions, chunk by chunk."""

    def __init__(self, layers, chosen, want_embed):
        self.layers, self.chosen, self.want_embed = layers, chosen, want_embed
        self.handles = []

    def __enter__(self):
        self.handles = []
        if self.want_embed:
            self.handles.append(self.layers[0].register_forward_pre_hook(self._pre))
        for i in self.chosen:
            self.handles.append(self.layers[i].register_forward_hook(self._post(f"resid_post.{i}")))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()

    def start(self, positions, start, end, store):
        """Prepare to capture rows of `positions` in [start, end) into `store`."""
        sel = [k for k, p in enumerate(positions) if start <= p < end]
        self.rows = sel
        self.local = torch.tensor([positions[k] - start for k in sel], dtype=torch.long)
        self.store = store

    def _keep(self, name, hidden):
        if not self.rows:
            return
        picked = hidden[0].index_select(0, self.local.to(hidden.device)).detach().cpu()
        buf = self.store.setdefault(name, {})
        for r, k in enumerate(self.rows):
            buf[k] = picked[r]

    def _pre(self, module, args, kwargs=None):
        self._keep("resid_pre.0", args[0] if args else kwargs["hidden_states"])

    def _post(self, name):
        def hook(module, args, output):
            self._keep(name, output[0] if isinstance(output, tuple) else output)
        return hook


def forward(lm, head, ids, positions, capture, chunk, device):
    """Run the text model over ids (whole, or in chunks with a carried cache).
    Returns {tensor_name: [n_positions, hidden]} and the final token's log-probs."""
    store = {}
    n = len(ids)
    step = n if chunk <= 0 else chunk
    cache = None
    if step < n:
        cache = transformers.DynamicCache(config=lm.config)
    last = None
    with torch.inference_mode():
        for start in range(0, n, step):
            end = min(n, start + step)
            capture.start(positions, start, end, store)
            x = torch.tensor([ids[start:end]], device=device)
            out = lm(input_ids=x, past_key_values=cache, use_cache=cache is not None)
            last = out.last_hidden_state[0, -1]
            del out, x
        logprobs = torch.log_softmax(head(last).float(), -1)
    tensors = {name: torch.stack([rows[k] for k in range(len(positions))]) for name, rows in store.items()}
    return tensors, logprobs


def compare(a, b):
    worst, worst_rel = 1.0, 0.0
    for name in a:
        x, y = a[name].float(), b[name].float()
        cos = torch.nn.functional.cosine_similarity(x, y, dim=-1)
        rel = (x - y).norm(dim=-1) / x.norm(dim=-1).clamp_min(1e-6)
        worst, worst_rel = min(worst, float(cos.min())), max(worst_rel, float(rel.max()))
    return {"min_cosine": round(worst, 6), "max_rel_diff": round(worst_rel, 6)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated indices; -1 = embedding output")
    parser.add_argument("--chunk", type=int, default=0, help="0 = whole pass first")
    parser.add_argument("--fallback-chunk", type=int, default=16384)
    parser.add_argument("--check-chunking", action="store_true")
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--deadline-s", type=float, default=0, help="0 = none")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()
    t_start = time.time()
    cuda = args.device == "cuda"

    from transformers.utils import import_utils

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(0) if cuda else args.device,
        "flash_linear_attention": getattr(import_utils, "is_flash_linear_attention_available", lambda: None)(),
        "causal_conv1d": getattr(import_utils, "is_causal_conv1d_available", lambda: None)(),
    }
    print(json.dumps(environment), flush=True)

    config = transformers.AutoConfig.from_pretrained(args.model)
    model_cls = getattr(transformers, config.architectures[0])
    model = model_cls.from_pretrained(args.model, dtype=getattr(torch, args.dtype), device_map=args.device,
                                      attn_implementation="sdpa")
    model.eval()
    lm = text_model(model)
    head = model.get_output_embeddings()
    n_layers = config.get_text_config().num_hidden_layers
    layers_name, layers = decoder_layers(model, n_layers)
    chosen = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    want_embed = -1 in chosen
    chosen = [i for i in chosen if i >= 0]
    load_s = round(time.time() - t_start, 1)
    print(f"loaded {config.architectures[0]} in {load_s}s; hooking {layers_name} {chosen}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = sorted(load_jobs(args.jobs), key=lambda j: -len(j["token_ids"]))
    run_log = {"environment": environment, "load_s": load_s, "layers": chosen, "embed": want_embed,
               "mode": "whole" if args.chunk <= 0 else f"chunk{args.chunk}", "jobs": [], "events": []}
    log_path = out / f"_run_{int(t_start)}.json"

    def save_log():
        log_path.write_text(json.dumps(run_log, indent=2))

    def done(job):
        meta = out / f"{safe_name(job['id'])}.json"
        return meta.exists() and json.loads(meta.read_text()).get("sha256_token_ids") == job["sha256_token_ids"]

    def positions_of(job):
        return sorted({p for ps in job["positions"].values() for p in ps})

    chunk = args.chunk
    chunk_checked = None

    def check_chunking():
        small = min(jobs, key=lambda j: len(j["token_ids"]))
        ids, pos = small["token_ids"], positions_of(small)
        piece = max(1, len(ids) // 3)
        with Capture(layers, chosen, want_embed) as cap:
            whole, lp_w = forward(lm, head, ids, pos, cap, 0, args.device)
            parts, lp_c = forward(lm, head, ids, pos, cap, piece, args.device)
        res = {"job": small["id"], "n_tokens": len(ids), "chunk": piece, **compare(whole, parts),
               "last_token_argmax_equal": int(lp_w.argmax()) == int(lp_c.argmax())}
        res["ok"] = res["min_cosine"] >= args.min_cosine
        run_log["chunking_check"] = res
        save_log()
        print("chunking check", json.dumps(res), flush=True)
        return res["ok"]

    if args.check_chunking:
        chunk_checked = check_chunking()
        if not chunk_checked:
            raise SystemExit("chunked forward does not match the whole forward")

    rate = None  # tokens per second, measured
    for job in jobs:
        ids = job["token_ids"]
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != job["sha256_token_ids"]:
            raise ValueError(f"{job['id']}: token ids do not match their hash")
        if done(job):
            run_log["events"].append({"skip_done": job["id"]})
            continue
        elapsed = time.time() - t_start
        if args.deadline_s and rate and elapsed + len(ids) / rate > args.deadline_s:
            run_log["events"].append({"stopped_for_deadline": job["id"], "elapsed_s": round(elapsed)})
            print(f"deadline: stopping before {job['id']}", flush=True)
            break
        pos = positions_of(job)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        oom = False
        try:
            with Capture(layers, chosen, want_embed) as cap:
                tensors, logprobs = forward(lm, head, ids, pos, cap, chunk, args.device)
        except torch.OutOfMemoryError:
            if chunk > 0:
                raise
            oom = True  # handled below, after the except block has released the failed pass
        if oom:
            peak = round(torch.cuda.max_memory_allocated() / 2**30, 1) if cuda else None
            run_log["events"].append({"oom_whole_pass": job["id"], "n_tokens": len(ids), "peak_gpu_gib": peak})
            print(f"OOM on whole pass of {job['id']} ({len(ids)} tokens); switching to chunks", flush=True)
            if cuda:
                torch.cuda.empty_cache()
            if chunk_checked is None:
                chunk_checked = check_chunking()
            if not chunk_checked:
                save_log()
                raise SystemExit("chunked forward does not match the whole forward; not continuing")
            chunk = args.fallback_chunk
            run_log["mode"] = f"chunk{chunk} from {job['id']}"
            if cuda:
                torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with Capture(layers, chosen, want_embed) as cap:
                tensors, logprobs = forward(lm, head, ids, pos, cap, chunk, args.device)
        forward_s = time.time() - t0
        rate = len(ids) / forward_s if rate is None else 0.7 * rate + 0.3 * len(ids) / forward_s
        top = torch.topk(logprobs, 10)
        name = safe_name(job["id"])
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / f"{name}.safetensors"))
        meta = {
            "id": job["id"], "rows": pos, "positions": job["positions"], "n_tokens": len(ids),
            "sha256_token_ids": job["sha256_token_ids"], "model_revision": job.get("model_revision"),
            "layers": chosen, "tensors": sorted(tensors), "hidden_size": int(next(iter(tensors.values())).shape[-1]),
            "dtype": str(next(iter(tensors.values())).dtype).replace("torch.", ""),
            "chunk": chunk, "last_token_top10": [[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)],
            "forward_s": round(forward_s, 2),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1) if cuda else None,
            "environment": environment,
        }
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        run_log["jobs"].append({k: meta[k] for k in ("id", "n_tokens", "chunk", "forward_s", "peak_gpu_gib")})
        save_log()
        print(json.dumps(run_log["jobs"][-1]), flush=True)
    run_log["total_s"] = round(time.time() - t_start, 1)
    save_log()


if __name__ == "__main__":
    main()
