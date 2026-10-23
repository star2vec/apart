"""Measure prompt/output token sizes in the paper's released Qwen 3.6 traces."""

import gzip
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
files = sorted(root.glob("*.json.gz"))
first = json.load(gzip.open(files[0]))
cfg = first.get("run_config", {})
print("top-level keys:", sorted(first))
print("models:", cfg.get("models"))
print("endpoints:", cfg.get("endpoints"))
print("llm_request_parameters:", cfg.get("llm_request_parameters"))
print("litellm_version:", cfg.get("litellm_version"))

calls, per_episode_max, finish, prune_hint = [], {}, {}, 0
for f in files:
    run = json.load(gzip.open(f))
    for ep in run.get("results", []):
        idx = ep.get("episode_index")
        for u in ep.get("llm_usage", []):
            inp, out = u.get("input_tokens"), u.get("provider_output_tokens") or u.get("output_tokens")
            if inp is None:
                continue
            calls.append((inp, out or 0, f.name, idx, u.get("phase"), u.get("actor")))
            per_episode_max.setdefault(idx, []).append(inp)
            fr = u.get("finish_reason")
            finish[fr] = finish.get(fr, 0) + 1
        prune_hint += sum("context limit" in json.dumps(e) for e in ep.get("events", []))

inputs = [c[0] for c in calls]
outputs = [c[1] for c in calls]
totals = sorted(c[0] + c[1] for c in calls)
print(f"\n{len(files)} runs, {len(calls)} calls with usage")
print("input tokens: median", statistics.median(inputs), "p99", sorted(inputs)[int(.99 * len(inputs))], "max", max(inputs))
print("output tokens: median", statistics.median(outputs), "p99", sorted(outputs)[int(.99 * len(outputs))], "max", max(outputs))
print("input+output: max", totals[-1])
print("largest input + 64000:", max(inputs) + 64000)
print("calls with input > 67072 (would be rejected at max-model-len 131072):", sum(i > 67072 for i in inputs))
for limit in (131072, 196608, 262144):
    print(f"  calls with input > {limit}-64000={limit-64000}: {sum(i > limit - 64000 for i in inputs)}")
print("finish reasons:", finish)
print("events mentioning 'context limit':", prune_hint)
print("max input by episode index:", {k: max(v) for k, v in sorted(per_episode_max.items())})
top = sorted(calls, reverse=True)[:5]
print("largest-input calls:", [(c[0], c[1], c[2], c[3], c[4], c[5]) for c in top])
