"""Batched steering: all conditions of a call in shared generate() batches, each row with its own
intervention at one decoder layer. Runs inside the vLLM image (or on CPU with a small random
model for offline tests), launched by steer_qwen.py --batched.

Per call: one prefill up to (not including) C, cached. Rows = conditions x samples; rows are
packed into batches as large as memory allows (the prompt's cache is copied per row): batch
size = floor(memory budget / per-row cost), the per-row cost measured from the prefilled cache
and the expected generation length; an out-of-memory batch is retried at half size.
Each batch: a deep copy of the cache expanded to the batch, one generate(). A forward hook on
the layer applies per row, at every position from C on:
  add       h <- h + alpha * v          (target direction, or a random unit direction)
  mean_abl  h <- h - ((h . u) - m) u    (projection on the target direction set to m)
  none      unchanged
Sampling: temperature 1.0, top_p 0.95, top_k 20 (the served generation config). Verdict read
from <parameter=verdict>; invalid if it hit max_new_tokens or has no verdict.
"""

import argparse
import copy
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
import transformers

VERDICT_RE = re.compile(r"<parameter=verdict>\s*(accept|reject)\b", re.I)


def text_model(model):
    for name in ("model.language_model", "language_model", "model"):
        mod = model
        try:
            for part in name.split("."):
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if hasattr(mod, "layers") and hasattr(mod, "embed_tokens"):
            return mod
    raise RuntimeError("no text model")


class RowSteer:
    """Per-row additive offsets and mean-ablation masks for the current batch."""

    def __init__(self):
        self.offsets, self.abl_mask, self.u, self.m = None, None, None, 0.0

    def set(self, offsets=None, abl_mask=None, u=None, m=0.0):
        self.offsets, self.abl_mask, self.u, self.m = offsets, abl_mask, u, m

    def __call__(self, module, args, output):
        if self.offsets is None and self.abl_mask is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        if self.offsets is not None:
            h = h + self.offsets.to(h.dtype)[:, None, :]
        if self.abl_mask is not None and bool(self.abl_mask.any()):
            u = self.u.to(h.dtype)
            proj = h @ u                                    # [B, T]
            shift = (proj - self.m) * self.abl_mask.to(h.dtype)[:, None]
            h = h - shift.unsqueeze(-1) * u
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--direction", required=True, help="npz: unit, sd, abl_mean")
    p.add_argument("--random-direction", required=True)
    p.add_argument("--layer", type=int, default=47)
    p.add_argument("--strength", type=float, default=4.0, help="multiple of the projection SD at C")
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--unsteered", action="store_true", help="include an in-batch unsteered condition")
    p.add_argument("--max-new-tokens", type=int, default=6144)
    p.add_argument("--mem-budget-gib", type=float, default=76.0, help="peak memory to aim for")
    p.add_argument("--expected-new-tokens", type=int, default=2500, help="for the per-row memory estimate")
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deadline-s", type=float, default=0)
    a = p.parse_args()
    t_start = time.time()
    cuda = a.device == "cuda"
    if cuda:  # the cuDNN attention kernel failed mid-run (mha_graph.execute not good); use flash / mem-efficient
        torch.backends.cuda.enable_cudnn_sdp(False)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    config = transformers.AutoConfig.from_pretrained(a.model)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        a.model, dtype=getattr(torch, a.dtype), device_map=a.device, attn_implementation="sdpa")
    model.eval()
    lm = text_model(model)
    tok = transformers.AutoTokenizer.from_pretrained(a.model)
    d = np.load(a.direction)
    u = torch.tensor(d["unit"], dtype=torch.float32, device=a.device)
    sd, m = float(d["sd"]), float(d["abl_mean"])
    rnd = torch.tensor(np.load(a.random_direction)["unit"], dtype=torch.float32, device=a.device)
    steer = RowSteer()
    handle = lm.layers[a.layer].register_forward_hook(steer)
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    gen = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=a.max_new_tokens,
               pad_token_id=eos[-1])
    log = {"load_s": round(time.time() - t_start, 1), "strength_sd": a.strength, "sd": sd, "abl_mean": m,
           "layer": a.layer, "samples": a.samples, "calls": []}
    jobs = json.loads(Path(a.jobs).read_text())
    for job in jobs:
        if a.deadline_s and time.time() - t_start > a.deadline_s:
            log["stopped_for_deadline"] = job["id"]
            break
        ids, c, sign = job["token_ids"], job["c_token"], job["target_sign"]
        alpha = sign * a.strength * sd
        conds = ([("unsteered", "none")] if a.unsteered else []) + [
            (f"target_{a.strength:g}sd", "target"), (f"random_{a.strength:g}sd", "random"), ("mean_ablate", "mean_abl")]
        if all((out / f"{job['id']}.{name}.json").exists() for name, _ in conds):  # resume: done in an earlier run
            log["calls"].append({"id": job["id"], "skipped_done": True})
            continue
        rows = [(name, kind, k) for name, kind in conds for k in range(a.samples)]
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        steer.set()
        if hasattr(model.model, "rope_deltas"):  # else the prefill keeps the last batch's [B, 1] offsets
            model.model.rope_deltas = None
        with torch.inference_mode():
            base = model(input_ids=torch.tensor([ids[:c]], device=a.device), use_cache=True).past_key_values
        prefill_s = time.time() - t0
        # transformers keeps the multimodal rope offset of the last batch; restore the prefill's per batch
        rope0 = getattr(model.model, "rope_deltas", None)
        rope0 = None if rope0 is None else rope0.clone()
        if cuda:
            base_gib = torch.cuda.memory_allocated() / 2**30
            # one row's cache ~ the prefilled cache; generation adds proportionally more
            cache_gib = sum(t.numel() * t.element_size() for t in _cache_tensors(base)) / 2**30
            row_gib = cache_gib * (len(ids) + a.expected_new_tokens) / max(c, 1) + 0.2
            bs = int(max(1, min(a.max_batch, len(rows), (a.mem_budget_gib - base_gib) // row_gib)))
        else:
            base_gib = cache_gib = row_gib = 0.0
            bs = min(a.max_batch, len(rows))
        call = {"id": job["id"], "n_prompt_tokens": len(ids), "prefill_s": round(prefill_s, 1),
                "base_gib": round(base_gib, 1), "row_gib_est": round(row_gib, 2), "batches": []}
        results = {name: [] for name, _ in conds}
        i = 0
        while i < len(rows):
            if a.deadline_s and time.time() - t_start > a.deadline_s:  # checked before every batch
                call["stopped_for_deadline"] = True
                break
            batch = rows[i:i + bs]
            offsets = torch.zeros(len(batch), u.numel(), device=a.device)
            abl = torch.zeros(len(batch), dtype=torch.bool, device=a.device)
            for r, (_, kind, _) in enumerate(batch):
                if kind == "target":
                    offsets[r] = alpha * u
                elif kind == "random":
                    offsets[r] = alpha * rnd
                elif kind == "mean_abl":
                    abl[r] = True
            try:
                cache = copy.deepcopy(base)
                cache.reorder_cache(torch.zeros(len(batch), dtype=torch.long, device=a.device))
                steer.set(offsets=offsets, abl_mask=abl, u=u, m=m)
                torch.manual_seed(a.seed + i)
                if hasattr(model.model, "rope_deltas"):
                    model.model.rope_deltas = None if rope0 is None else rope0.clone()
                t1 = time.time()
                inp = torch.tensor([ids] * len(batch), device=a.device)
                with torch.inference_mode():
                    seqs = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), past_key_values=cache, **gen)
                gen_s = time.time() - t1
            except torch.OutOfMemoryError:
                steer.set()
                cache = seqs = None
                if cuda:
                    torch.cuda.empty_cache()
                if bs == 1:
                    raise
                call["batches"].append({"oom_at_batch": bs})
                bs = max(1, bs // 2)
                continue
            steer.set()
            for (name, kind, k), row in zip(batch, seqs[:, len(ids):].tolist()):
                n_new = next((j + 1 for j, t in enumerate(row) if t in eos), len(row))
                capped = n_new >= a.max_new_tokens and not any(t in eos for t in row)
                text = tok.decode(row[:n_new], skip_special_tokens=False)
                mm = VERDICT_RE.search(text)
                results[name].append({"sample": k, "verdict": None if capped or not mm else mm.group(1).lower(),
                                      "capped": capped, "new_tokens": n_new, "text": text})
            call["batches"].append({"size": len(batch), "gen_s": round(gen_s, 1),
                                    "longest": max(r["new_tokens"] for rs in results.values() for r in rs)})
            del cache, seqs
            i += len(batch)
        call["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if cuda else None
        call["conditions"] = {}
        for name, rs in results.items():
            v = [r["verdict"] for r in rs if r["verdict"]]
            call["conditions"][name] = {"share_accept": None if not v else round(sum(x == "accept" for x in v) / len(v), 3),
                                        "valid": len(v), "invalid": len(rs) - len(v)}
            with open(out / f"{job['id']}.{name}.json", "w") as f:
                json.dump({"call": job["id"], "condition": name, "samples": rs}, f)
        log["calls"].append(call)
        (out / "_log.json").write_text(json.dumps(log, indent=1))
        print(json.dumps({"call": job["id"], "batches": call["batches"], "conditions": call["conditions"],
                          "peak_gpu_gib": call["peak_gpu_gib"]}), flush=True)
        del base
    handle.remove()
    log["total_s"] = round(time.time() - t_start, 1)
    (out / "_log.json").write_text(json.dumps(log, indent=1))


def _cache_tensors(cache):
    """All tensors held by a (hybrid) cache, for its memory size."""
    seen = []
    for layer in getattr(cache, "layers", []):
        for v in vars(layer).values():
            if torch.is_tensor(v):
                seen.append(v)
            elif isinstance(v, (list, tuple)):
                seen += [t for t in v if torch.is_tensor(t)]
    return seen


if __name__ == "__main__":
    main()
