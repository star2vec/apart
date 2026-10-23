"""Step 6 (Mac): probes and replay jobs for the projections session.

    tools/.venv/bin/python tools/step6/prepare_step6.py [--limit-before-onset]

C probe (layer 47), one per trajectory fold: the first repeat of the step 3 splits (5 folds by
trajectory). Each fold's probe is trained on the other folds' onset and before-onset C vectors
(L2 strength from inner folds, as in step 3) and exported in raw-activation form
(logit = h . w + b), so every call is scored by a probe that never saw its trajectory.

Jobs (token ids for the worker):
  call   onset, before-onset and reverted verdict calls: the verdict prompt (with generation
         prompt) plus the call's own saved thinking, cut at </think>. Scored: every token
         from C to the last thinking token. Saved (layer 47 vector): C and the token ending
         each tenth of the thinking.
  edit   step 5's edited verdict prompts (variants A and B): scored and saved at C.
--limit-before-onset keeps only the before-onset episode right before onset (cost limit).

Writes agent-collusion/results/step6/{jobs.jsonl, probes.npz, calls.jsonl, plan.json}.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step2"))
sys.path.insert(0, str(ROOT / "tools" / "step3"))
sys.path.insert(0, str(ROOT / "tools" / "paper_prompts"))
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
import probe_lib as pl  # noqa: E402
import run_step3 as s3  # noqa: E402
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

RES = ROOT / "agent-collusion" / "results"
OUT = RES / "step6"
LAYER = 47


def export_probes(rows, splits):
    """Per fold (first repeat): raw-space weights and the runs it may score."""
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    y = np.array([r["y"] for r in probe])
    X = pl.rows_to_float(pl.load_layer(LAYER), np.array([r["pos"]["C"] for r in probe]))
    W, B, fold_of_run, checks = [], [], {}, []
    for k, sp in enumerate(splits):
        xtr, ytr = X[sp.train], y[sp.train]
        best = (-np.inf, None)
        bases = [(s3.Basis(xtr[a]), a, b) for a, b in sp.inner]
        for c in pl.C_GRID:
            pp = np.concatenate([s3.fit(bs.z, ytr[a], c).predict_proba(bs(xtr[b]))[:, 1] for bs, a, b in bases])
            score = pl.auc(np.concatenate([ytr[b] for _, _, b in bases]), pp)
            if score > best[0]:
                best = (score, c)
        basis = s3.Basis(xtr)
        clf = s3.fit(basis.z, ytr, best[1])
        w = (basis.vt.T @ clf.coef_[0]) / basis.sd
        b = float(clf.intercept_[0] - basis.mu.astype(np.float64) @ w)
        ref = clf.decision_function(basis(X[sp.test]))
        raw = X[sp.test].astype(np.float64) @ w + b
        checks.append({"fold": k, "c": float(best[1]), "max_abs_logit_diff": float(np.abs(ref - raw).max()),
                       "test_auc": round(float(pl.auc(y[sp.test], raw)), 4)})
        W.append(w.astype(np.float32))
        B.append(b)
        for j in sp.test:
            fold_of_run[probe[j]["run"]] = k
    return np.stack(W), np.array(B, np.float32), fold_of_run, checks


def render_call(tok, body, reply):
    conv, tools = vllm_conversation(body["messages"]), vllm_tools(body.get("tools"))
    prompt = tok.apply_chat_template(conv, tools=tools, add_generation_prompt=True, tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    if reply is None:
        return p_ids, None
    full = tok.apply_chat_template(conv + vllm_conversation([reply]), tools=tools, add_generation_prompt=False,
                                   tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("reply does not extend the prompt")
    close = full.index("</think>", len(prompt))
    f_ids = tok(full[:close], add_special_tokens=False)["input_ids"]
    if f_ids[: len(p_ids)] != p_ids:
        raise ValueError("prompt tokens are not a prefix")
    think = f_ids[len(p_ids):]
    while think and tok.decode(think[-1:]).strip() == "":  # the "\n" before </think>
        think = think[:-1]
    return p_ids, think


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-before-onset", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    nl = tok("\n", add_special_tokens=False)["input_ids"][0]
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = s3.build_rows(index, episodes)
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    splits = pl.make_splits([r["run"] for r in probe], repeats=5)[:5]
    W, B, fold_of_run, checks = export_probes(rows, splits)
    np.savez(OUT / "probes.npz", W=W, b=B, layer=LAYER)
    print("probe export checks:", checks)

    wire = {}
    for line in open(RES / "paper" / "paper_wire_verdict.jsonl"):
        r = json.loads(line)
        if "skip" not in r:
            wire[(r["run"].split(".")[0], r["ep"] + 1, r["agent"])] = r
    jobs, calls = [], []
    for (run, ep, agent), r in sorted(wire.items()):
        etype = episodes[(run, ep)]["episode_type"]
        if etype not in ("onset", "before_onset", "reverted"):
            continue
        if args.limit_before_onset and etype == "before_onset" and episodes[(run, ep + 1)]["episode_type"] != "onset":
            continue
        reply = r["reply"]
        reasoning = reply.get("reasoning") or reply.get("reasoning_content") or ""
        p_ids, think = render_call(tok, r["body"], reply)
        c = len(p_ids) - 3
        assert p_ids[c] == nl and tok.decode(p_ids[c - 1:c]) == "assistant", (run, ep, agent)
        n = len(think)
        ids = p_ids + think
        tenth_ends = [len(p_ids) + max(0, (j + 1) * n // 10 - 1) for j in range(10)]
        job_id = f"call.{run}.{agent}.ep{ep:02d}"
        jobs.append({"id": job_id, "token_ids": ids, "n_prompt_tokens": len(p_ids),
                     "positions": {"C": [c], "tenth_ends": tenth_ends},
                     "score": {"start": c, "end": len(ids), "probe": fold_of_run[run]},
                     "sha256_token_ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                     "model_revision": pins.MODEL_REVISION})
        e = episodes[(run, ep)]
        calls.append({"job": job_id, "kind": "call", "run": run, "agent": agent, "episode": ep, "type": etype,
                      "verdict": e[agent]["verdict"], "path": None, "fold": fold_of_run[run], "c_token": c,
                      "think_start": len(p_ids), "n_think": n, "tenth_ends": tenth_ends,
                      "logged_input_tokens": r["logged_input_tokens"], "n_prompt_tokens": len(p_ids),
                      "reasoning": reasoning, "think_ids": think})
    onset_path = {}
    for (run, ep), e in episodes.items():
        if e["episode_type"] == "onset":
            onset_path[run] = "agreement" if e["agreement"] else "no_agreement"
    for cinfo in calls:
        cinfo["path"] = onset_path[cinfo["run"]]
    for e in map(json.loads, open(RES / "step5" / "edits.jsonl")):
        p_ids, _ = render_call(tok, e["body"], None)
        c = len(p_ids) - 3
        assert p_ids[c] == nl
        job_id = f"edit{e['variant']}.{e['run']}.{e['agent']}.ep{e['episode']:02d}"
        jobs.append({"id": job_id, "token_ids": p_ids, "n_prompt_tokens": len(p_ids), "positions": {"C": [c]},
                     "score": {"start": c, "end": c + 1, "probe": fold_of_run[e["run"]]},
                     "sha256_token_ids": hashlib.sha256(json.dumps(p_ids).encode()).hexdigest(),
                     "model_revision": pins.MODEL_REVISION})
        calls.append({"job": job_id, "kind": f"edit{e['variant']}", "run": e["run"], "agent": e["agent"],
                      "episode": e["episode"], "type": "onset", "path": "agreement", "fold": fold_of_run[e["run"]],
                      "c_token": c, "n_prompt_tokens": len(p_ids),
                      "n_prompt_tokens_original": e["logged_input_tokens_original"]})
    with open(OUT / "jobs.jsonl", "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    with open(OUT / "calls.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    tokens = sum(len(j["token_ids"]) for j in jobs)
    plan = {"jobs": len(jobs), "calls": sum(c["kind"] == "call" for c in calls),
            "edits": sum(c["kind"] != "call" for c in calls), "tokens": tokens,
            "max_job_tokens": max(len(j["token_ids"]) for j in jobs),
            "thinking_tokens": sum(c.get("n_think", 0) for c in calls),
            "min_thinking": min(c["n_think"] for c in calls if c["kind"] == "call"),
            "by_type": {t: sum(c["type"] == t and c["kind"] == "call" for c in calls)
                        for t in ("before_onset", "onset", "reverted")},
            "probe_checks": checks, "limit_before_onset": args.limit_before_onset}
    (OUT / "plan.json").write_text(json.dumps(plan, indent=1))
    print(json.dumps({k: v for k, v in plan.items() if k != "probe_checks"}, indent=1))


if __name__ == "__main__":
    main()
