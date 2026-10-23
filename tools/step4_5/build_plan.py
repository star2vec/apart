"""Steps 4-5 prep (Mac): call list, cut points and cost for the vLLM session.

    tools/.venv/bin/python tools/step4_5/build_plan.py [--samples 16] [--random-fixed-samples 16]

Chosen cases (out-of-fold C-probe scores from step 6):
  reverted_reject   all reverted calls ending in reject
  flip              onset / before-onset calls whose thinking first states the opposite verdict
  confident_wrong   onset / before-onset calls with P(accept) at C > 0.9 ending in reject, or
                    < 0.1 ending in accept
Random group (reported separately): 20 onset and 20 before-onset calls not among the chosen.

Per call:
  A  full verdict resampled from C (chat endpoint, the recorded request: named tool choice)
  B  the saved thinking cut at 30/50/70/90/100% and at two event cuts, continued through the
     raw completions endpoint (prompt + partial thinking as token ids):
       after_first_decision  right after the sentence of the first stated decision (any polarity)
       before_final          right before the sentence that first states the final verdict
C: step 5 variant B edited prompts resampled at C (chat endpoint).
Output cap 24k tokens; capped replies count as invalid.

Writes agent-collusion/results/step4_5/plan.json (calls, cuts as thinking-token counts, cost).
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))
import pins  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

S6 = ROOT / "agent-collusion" / "results" / "step6"
OUT = ROOT / "agent-collusion" / "results" / "step4_5"
FIXED = [0.3, 0.5, 0.7, 0.9, 1.0]
TOOL_CALL_TOKENS = 70  # "\n</think>\n\n<tool_call>...submit_verdict...</tool_call>"
HOURLY = 3.95


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--random-fixed-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    calls = {c["job"]: c for c in map(json.loads, open(S6 / "calls.jsonl")) if c["kind"] == "call"}
    per = {r["job"]: r for r in map(json.loads, open(S6 / "per_call.jsonl"))}
    vs = {r["job"]: r for r in map(json.loads, open(S6 / "verdict_statements.jsonl"))}
    edits = [c for c in map(json.loads, open(S6 / "calls.jsonl")) if c["kind"] == "editB"]

    groups = {}
    for j, c in calls.items():
        g = []
        if c["type"] == "reverted" and c["verdict"] == "reject":
            g.append("reverted_reject")
        if c["type"] in ("onset", "before_onset"):
            if vs[j]["changed_mind"]:
                g.append("flip")
            p = per[j]["p_C"]
            if (c["verdict"] == "reject" and p > 0.9) or (c["verdict"] == "accept" and p < 0.1):
                g.append("confident_wrong")
        if g:
            groups[j] = g
    rng = random.Random(args.seed)
    rand = {}
    for t in ("onset", "before_onset"):
        pool = sorted(j for j, c in calls.items() if c["type"] == t and j not in groups)
        for j in rng.sample(pool, 20):
            rand[j] = ["random_" + t]

    def sentence_tokens(c, stmt, where):
        """Thinking-token index right after (where='after') or right before ('before') the sentence."""
        text = c["reasoning"].strip()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        assert enc["input_ids"] == c["think_ids"]
        i = text.find(stmt["sentence"][:80])
        if i < 0:
            return None
        ch = i + len(stmt["sentence"]) if where == "after" else i
        if where == "after":
            return next((k for k, (s, e) in enumerate(enc["offset_mapping"]) if s >= ch), len(enc["input_ids"]))
        return next((k for k, (s, e) in enumerate(enc["offset_mapping"]) if e > ch), None)

    plan_calls = []
    for j, g in list(groups.items()) + list(rand.items()):
        c = calls[j]
        n = c["n_think"]
        is_random = g[0].startswith("random")
        k_fixed = args.random_fixed_samples if is_random else args.samples
        cuts = [{"name": f"fixed_{int(f * 100)}", "think_tokens": int(round(f * n)), "samples": k_fixed} for f in FIXED]
        v = vs[j]
        if v["first_any"]:
            k = sentence_tokens(c, v["first_any"], "after")
            if k is not None:
                cuts.append({"name": "after_first_decision", "think_tokens": k, "samples": args.samples,
                             "polarity": v["first_any"]["polarity"]})
        if v["first_final"]:
            k = sentence_tokens(c, v["first_final"], "before")
            if k is not None:
                cuts.append({"name": "before_final", "think_tokens": k, "samples": args.samples})
        plan_calls.append({"job": j, "groups": g, "run": c["run"], "agent": c["agent"], "episode": c["episode"],
                           "type": c["type"], "verdict": c["verdict"], "p_C": per[j]["p_C"], "n_think": n,
                           "n_prompt_tokens": c["n_prompt_tokens"], "lock_in_0.1": per[j]["lock_in_0.1"],
                           "lock_in_0.05": per[j]["lock_in_0.05"], "statement": per[j]["statement"],
                           "full_samples": args.samples, "cuts": cuts})

    # cost model
    prefill_unc = sum(c["n_prompt_tokens"] for c in plan_calls) + sum(e["n_prompt_tokens"] for e in edits)
    n_orig = {(c["run"], c["agent"], c["episode"]): c["n_think"] for c in calls.values()}
    dec_a = sum(c["full_samples"] * (c["n_think"] + TOOL_CALL_TOKENS) for c in plan_calls)
    dec_b = sum(k["samples"] * (c["n_think"] - k["think_tokens"] + TOOL_CALL_TOKENS) for c in plan_calls for k in c["cuts"])
    dec_c = sum(args.samples * (n_orig[(e["run"], e["agent"], e["episode"])] + TOOL_CALL_TOKENS) for e in edits)
    requests = len(plan_calls) * (1 + 7) + len(edits)
    samples = sum(c["full_samples"] + sum(k["samples"] for k in c["cuts"]) for c in plan_calls) + args.samples * len(edits)
    dec = dec_a + dec_b + dec_c
    est = {}
    for name, rate in (("optimistic", 2000), ("central", 1200), ("pessimistic", 650)):
        sec = prefill_unc / 14000 + dec / rate + 160 + 300
        est[name] = {"decode_tok_per_s": rate, "minutes": round(sec / 60, 1), "usd": round(sec / 3600 * HOURLY, 2)}
    counts = {}
    for c in plan_calls:
        for g in c["groups"]:
            counts[g] = counts.get(g, 0) + 1
    plan = {"settings": {"samples": args.samples, "random_fixed_samples": args.random_fixed_samples,
                         "max_tokens": 24000, "temperature": 1.0, "fixed_cuts": FIXED, "seed": args.seed},
            "group_counts": counts, "n_calls": len(plan_calls), "n_edits": len(edits),
            "n_requests": requests, "n_samples": samples,
            "prefill_uncached_tokens": prefill_unc, "decode_tokens": {"A": dec_a, "B": dec_b, "C": dec_c, "total": dec},
            "estimate": est, "calls": plan_calls,
            "edits": [{"job": e["job"], "run": e["run"], "agent": e["agent"], "episode": e["episode"]} for e in edits]}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: v for k, v in plan.items() if k not in ("calls", "edits")}, indent=1))
    print("event cuts available:", sum(any(k["name"] == "after_first_decision" for k in c["cuts"]) for c in plan_calls),
          sum(any(k["name"] == "before_final" for k in c["cuts"]) for c in plan_calls), "of", len(plan_calls))


if __name__ == "__main__":
    main()
