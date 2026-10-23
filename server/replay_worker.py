"""Residual-stream capture with transformers. Runs inside the vLLM image under its own
Python (/usr/bin/python3: torch 2.13, transformers 5.17), launched by replay_qwen.py.

For each job: one no-grad forward pass over the job's token ids, forward hooks on every
decoder layer (and a pre-hook on layer 0 for the embedding output), keeping only the
requested positions. Saves <out>/<job>.safetensors (tensor "resid_post.<layer>" or
"resid_pre.0", shape [n_positions, hidden], bfloat16 as computed) and <job>.json.
"""

import argparse
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
        and name.endswith("layers") and "visual" not in name
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one decoder ModuleList, found {[n for n, _ in candidates]}")
    return candidates[0]


def safe_name(job_id):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", job_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated indices")
    args = parser.parse_args()

    from transformers.utils import import_utils

    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "flash_linear_attention": getattr(import_utils, "is_flash_linear_attention_available", lambda: None)(),
        "causal_conv1d": getattr(import_utils, "is_causal_conv1d_available", lambda: None)(),
    }
    print(json.dumps(environment), flush=True)

    started = time.time()
    config = transformers.AutoConfig.from_pretrained(args.model)
    model_cls = getattr(transformers, config.architectures[0])
    model = model_cls.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                      attn_implementation="sdpa")
    model.eval()
    n_layers = config.get_text_config().num_hidden_layers
    layers_name, layers = decoder_layers(model, n_layers)
    chosen = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    print(f"loaded {config.architectures[0]} in {time.time() - started:.0f}s; hooking {layers_name}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(args.jobs).read_text())
    for job in jobs:
        ids = job["token_ids"]
        if hashlib.sha256(json.dumps(ids).encode()).hexdigest() != job["sha256_token_ids"]:
            raise ValueError(f"{job['id']}: token ids do not match their hash")
        order = sorted({p for ps in job["positions"].values() for p in ps})
        index = torch.tensor(order, device="cuda")
        captured = {}

        def keep(name):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[name] = hidden[0].index_select(0, index).detach().cpu()
            return hook

        def keep_input(module, inputs):
            captured["resid_pre.0"] = inputs[0][0].index_select(0, index).detach().cpu()

        handles = [layers[0].register_forward_pre_hook(keep_input)]
        handles += [layers[i].register_forward_hook(keep(f"resid_post.{i}")) for i in chosen]
        t0 = time.time()
        try:
            with torch.inference_mode():
                result = model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False, logits_to_keep=1)
        finally:
            for h in handles:
                h.remove()
        logits = result.logits[0, -1].float()
        top = torch.topk(torch.log_softmax(logits, -1), 10)
        name = safe_name(job["id"])
        save_file({k: v.contiguous() for k, v in captured.items()}, str(out / f"{name}.safetensors"))
        meta = {
            "id": job["id"],
            "rows": order,
            "positions": job["positions"],
            "n_tokens": len(ids),
            "n_prompt_tokens": job.get("n_prompt_tokens"),
            "sha256_token_ids": job["sha256_token_ids"],
            "model_revision": job.get("model_revision"),
            "layers": chosen,
            "tensors": sorted(captured),
            "hidden_size": int(next(iter(captured.values())).shape[-1]),
            "dtype": "bfloat16",
            "last_token_top10": [[int(i), round(float(v), 4)] for v, i in zip(top.values, top.indices)],
            "forward_s": round(time.time() - t0, 1),
            "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "environment": environment,
        }
        (out / f"{name}.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({k: meta[k] for k in ("id", "n_tokens", "forward_s", "peak_gpu_gib")}), flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
