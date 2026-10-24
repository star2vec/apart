"""Steering at one decoder layer with transformers. Runs inside the vLLM image (or on CPU with a
small random model for offline tests), launched by steer_qwen.py.

For each call: one prefill of the prompt up to (not including) C, cached. For each condition:
a deep copy of the cache, expanded to the batch of samples, then one batched generate() of all
samples from C on (C, the template's "<think>\\n", then free generation). A forward hook on the
layer adds alpha * u to the residual stream (or removes its component along u) at every position
from C on; the prefill runs without the hook.

Conditions (per call): unsteered; target direction at each strength (sign per call: toward
reject or toward accept); a random unit direction orthogonal to the target at the same
strengths; directional ablation of the target direction. Optional: the strongest strength at
C only.

Sampling: temperature 1.0, top_p 0.95, top_k 20 (the model's generation config, as served by
vLLM). The verdict is read from the generated <parameter=verdict>; a sample is invalid if it
hit max_new_tokens or has no verdict. Saves the generated text per sample.
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


class Steer:
    """Forward hook: h <- h + alpha * u, or h <- h - (h . u) u, at every position it sees."""

    def __init__(self):
        self.mode, self.vec, self.alpha, self.first_only, self.calls = None, None, 0.0, False, 0

    def set(self, mode, vec=None, alpha=0.0, first_only=False):
        self.mode, self.vec, self.alpha, self.first_only, self.calls = mode, vec, alpha, first_only, 0

    def __call__(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if self.mode is None or (self.first_only and self.calls > 0):
            self.calls += 1
            return output
        h = hidden
        if self.mode == "add":
            if self.first_only:  # only the first position of the first forward (C)
                h = h.clone()
                h[:, 0] = h[:, 0] + self.alpha * self.vec.to(h.dtype)
            else:
                h = h + self.alpha * self.vec.to(h.dtype)
        elif self.mode == "ablate":
            u = self.vec.to(h.dtype)
            h = h - (h @ u).unsqueeze(-1) * u
        self.calls += 1
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--jobs", required=True, help="json: list of calls")
    parser.add_argument("--direction", required=True, help="npz with unit, sd")
    parser.add_argument("--random-direction", required=True)
    parser.add_argument("--layer", type=int, default=47)
    parser.add_argument("--strengths", default="2,4", help="multiples of the projection SD at C")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--c-only", action="store_true", help="add the strongest-strength C-only condition")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deadline-s", type=float, default=0)
    args = parser.parse_args()
    t_start = time.time()
    cuda = args.device == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = transformers.AutoConfig.from_pretrained(args.model)
    model = getattr(transformers, config.architectures[0]).from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map=args.device, attn_implementation="sdpa")
    model.eval()
    lm = text_model(model)
    tok = transformers.AutoTokenizer.from_pretrained(args.model)
    d = np.load(args.direction)
    u = torch.tensor(d["unit"], dtype=torch.float32, device=args.device)
    sd = float(d["sd"])
    r = torch.tensor(np.load(args.random_direction)["unit"], dtype=torch.float32, device=args.device)
    strengths = [float(x) for x in args.strengths.split(",")]
    steer = Steer()
    handle = lm.layers[args.layer].register_forward_hook(steer)
    gen_cfg = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, max_new_tokens=args.max_new_tokens,
                   pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else config.get_text_config().eos_token_id)
    load_s = time.time() - t_start
    log = {"load_s": round(load_s, 1), "strengths_sd": strengths, "sd": sd, "layer": args.layer, "calls": []}
    jobs = json.loads(Path(args.jobs).read_text())
    for job in jobs:
        if args.deadline_s and time.time() - t_start > args.deadline_s:
            log["stopped_for_deadline"] = job["id"]
            break
        ids = job["token_ids"]  # prompt with the generation prompt: ... <|im_start|>assistant \n <think> \n
        c = job["c_token"]
        sign = job["target_sign"]
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        steer.set(None)
        with torch.inference_mode():
            prefill = model(input_ids=torch.tensor([ids[:c]], device=args.device), use_cache=True)
        base_cache = prefill.past_key_values
        del prefill
        prefill_s = time.time() - t0
        conditions = [("unsteered", None, None, 0.0, False)]
        for k in strengths:
            conditions.append((f"target_{k:g}sd", "add", u, sign * k * sd, False))
        for k in strengths:
            conditions.append((f"random_{k:g}sd", "add", r, sign * k * sd, False))
        conditions.append(("ablate", "ablate", u, 0.0, False))
        if args.c_only:
            conditions.append((f"target_{max(strengths):g}sd_C_only", "add", u, sign * max(strengths) * sd, True))
        call_log = {"id": job["id"], "n_prompt_tokens": len(ids), "prefill_s": round(prefill_s, 1), "conditions": []}
        for name, mode, vec, alpha, first_only in conditions:
            cache = copy.deepcopy(base_cache)
            cache.reorder_cache(torch.zeros(args.samples, dtype=torch.long, device=args.device))
            steer.set(mode, vec, alpha, first_only)
            torch.manual_seed(args.seed)
            t1 = time.time()
            with torch.inference_mode():
                inp = torch.tensor([ids] * args.samples, device=args.device)
                seqs = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp),
                                      past_key_values=cache, **gen_cfg)
            gen_s = time.time() - t1
            steer.set(None)
            samples = []
            for row in seqs[:, len(ids):].tolist():
                eos = config.get_text_config().eos_token_id
                eos = eos if isinstance(eos, list) else [eos]
                n_new = next((i + 1 for i, t in enumerate(row) if t in eos), len(row))
                text = tok.decode(row[:n_new], skip_special_tokens=False)
                capped = n_new >= args.max_new_tokens and not any(t in eos for t in row)
                m = VERDICT_RE.search(text)
                samples.append({"verdict": None if capped or not m else m.group(1).lower(), "capped": capped,
                                "new_tokens": n_new, "text": text})
            valid = [s["verdict"] for s in samples if s["verdict"]]
            call_log["conditions"].append({
                "name": name, "alpha": round(alpha, 3), "gen_s": round(gen_s, 1),
                "new_tokens": sum(s["new_tokens"] for s in samples),
                "share_accept": None if not valid else round(sum(v == "accept" for v in valid) / len(valid), 3),
                "valid": len(valid), "invalid": len(samples) - len(valid)})
            with open(out / f"{job['id']}.{name}.json", "w") as f:
                json.dump({"call": job["id"], "condition": name, "alpha": alpha, "samples": samples}, f)
            print(json.dumps({"call": job["id"], **call_log["conditions"][-1]}), flush=True)
            del cache, seqs
        call_log["peak_gpu_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1) if cuda else None
        log["calls"].append(call_log)
        (out / "_log.json").write_text(json.dumps(log, indent=1))
        del base_cache
    handle.remove()
    log["total_s"] = round(time.time() - t_start, 1)
    (out / "_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
