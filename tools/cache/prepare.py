"""Cache job, stage 2: render the two passes per run and agent, pick positions, write the index.

    tools/.venv/bin/python tools/cache/prepare.py --out agent-collusion/results/cache

Inputs (defaults under agent-collusion/results/): paper traces, paper_wire_verdict.jsonl
(verdict pass = the last episode's verdict prompt), cache/comm_wire.jsonl (stage 1), and
the paper's label tables (paper/labels/*.parquet).

Passes per run and agent, each rendered with its own phase's tool list:
  communication  last episode's history up to its verdict-phase prompt, no generation prompt
  verdict        last episode's verdict prompt with the generation prompt (what vLLM saw)

Positions (token index in the pass), per episode, wherever the pass contains them:
  A               <|im_end|> of the communication-phase intro message (own task done, no peer
                  message seen yet)
  A_call          "\\n" after <|im_start|>assistant of the agent's first communication turn
                  (for the agent that speaks second, the peer's first message is already in view)
  msg_end         <|im_end|> of each delivered message: own send_message turn, or the peer's
                  message turn
  C               "\\n" after <|im_start|>assistant of the verdict turn: the last token shared by
                  the verdict call's prompt and every later prompt (the call itself appends <think>)
  verdict         <|im_end|> of the submit_verdict turn
  runner_notice   <|im_end|> of a runner notice that the peer failed to deliver in a round
  feedback        <|im_end|> of the outcome message ("Episode k complete.")
  reflection_start  "\\n" after <|im_start|>assistant of the reflection turn
  reflection_end  <|im_end|> of the reflection turn
Everything before each of these positions renders exactly as in the agent's later real calls
(earlier thinking is dropped by the template once a new user message follows), so the
residual stream there equals what the model computed in those calls.

B is not a separate position: index rows mark the agreement message's msg_end
(is_agreement_msg) and B is derived from msg_end rows in analysis (see b_slot in summary).

Writes jobs.jsonl (token ids + positions per pass), index.jsonl (one row per saved vector
position), episodes.jsonl (episode-level labels), summary.json (counts, storage, checks).
"""

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "paper_prompts"))
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

RESULTS = ROOT / "agent-collusion" / "results"
PEER = {"alice": "bob", "bob": "alice"}
LAYERS = list(range(3, 64, 4))  # output of each full-attention layer (every 4th), incl. the last
HIDDEN = 5120
EP_HEAD = re.compile(r"^## Episode (\d+): (task|communication|verdict) phase")
DONE = re.compile(r"^Episode (\d+) complete\.")
PEER_MSG = re.compile(r"^\[round=(\d+)/\d+\]\[from=(\w+)\]\[type=([^\]]*)\]\n\n", re.S)
ROUND_ASK = re.compile(r"\(round (\d+) of \d+\)")


def annotate(msgs, agent):
    """Label each message: episode (1-based), phase, kind, round, sender, text."""
    out, ep, phase, rnd = [], 0, None, None
    for i, m in enumerate(msgs):
        c = m.get("content") if isinstance(m.get("content"), str) else ""
        a = {"i": i, "ep": ep, "phase": phase, "kind": m["role"]}
        if m["role"] == "user":
            if h := EP_HEAD.match(c):
                ep, phase = int(h.group(1)), h.group(2)
                a.update(ep=ep, phase=phase, kind=f"{phase}_intro")
            elif DONE.match(c):
                phase = "feedback"
                a.update(phase=phase, kind="feedback")
            elif c.startswith("Reflect briefly"):
                phase = "reflection"
                a.update(phase=phase, kind="reflection_prompt")
            elif (p := PEER_MSG.match(c)) and p.group(2) == "runner":
                a.update(kind="runner_notice", round=int(p.group(1)), text=c[p.end():])
            elif p:
                a.update(kind="peer_msg", round=int(p.group(1)), sender=p.group(2), text=c[p.end():])
            elif r := ROUND_ASK.search(c):
                rnd = int(r.group(1))
                a.update(kind="round_ask", round=rnd)
        elif m["role"] == "assistant":
            names = [t["function"]["name"] for t in m.get("tool_calls") or []]
            if phase == "communication" and names == ["send_message"]:
                nxt = msgs[i + 1] if i + 1 < len(msgs) else {}
                ok = nxt.get("role") == "tool" and json.loads(nxt.get("content") or "{}").get("success") is True
                args = json.loads(m["tool_calls"][0]["function"]["arguments"] or "{}")
                a.update(kind="own_send" if ok else "own_send_failed", round=rnd, sender=agent,
                         text=args.get("content"))
            elif phase == "verdict" and "submit_verdict" in names:
                a.update(kind="verdict_call")
            elif phase == "reflection":
                a.update(kind="reflection")
        out.append(a)
    return out


def turn_spans(text, conv):
    """Char span of each message's turn: (start of <|im_start|>, index of its <|im_end|>).
    Consecutive tool messages share one user turn; assistant turns are one turn each."""
    starts = [m.start() for m in re.finditer(re.escape("<|im_start|>"), text)]
    spans, t = [], 0 if conv[0]["role"] != "system" else 1
    for j, m in enumerate(conv):
        if j == 0 and m["role"] == "system":
            spans.append((starts[0], text.index("<|im_end|>", starts[0])))
            continue
        if m["role"] == "tool" and j > 0 and conv[j - 1]["role"] == "tool":
            spans.append(spans[-1])
            continue
        s = starts[t]
        t += 1
        spans.append((s, text.index("<|im_end|>", s)))
    return spans, starts[t:]


def token_at(offsets, ch):
    lo, hi = 0, len(offsets) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s, e = offsets[mid]
        if ch < s:
            hi = mid - 1
        elif ch >= e:
            lo = mid + 1
        else:
            return mid
    raise ValueError(f"no token covers char {ch}")


def render(tok, body, generation):
    conv, tools = vllm_conversation(body["messages"]), vllm_tools(body.get("tools"))
    text = tok.apply_chat_template(conv, tools=tools, add_generation_prompt=generation, tokenize=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return conv, text, enc["input_ids"], enc["offset_mapping"]


def episode_labels(run, runs_row, agreement, relaxation):
    """Episode-level labels from the trace and the paper's judge tables."""
    eps = run["results"]
    mutual = [e["agents"]["alice"]["verdict"] == "accept" and e["agents"]["bob"]["verdict"] == "accept" for e in eps]
    onset = next((k for k, m in enumerate(mutual) if m), None)
    rows = []
    for k, e in enumerate(eps):
        if onset is None or k < onset:
            etype = "before_onset"
        elif k == onset:
            etype = "onset"
        else:
            etype = "sustained" if mutual[k] else "reverted"
        ag = agreement.get((run["run_id"], k), {})
        row = {"run": run["run_id"], "sequence": run["sequence_id"], "episode": k + 1, "episode_type": etype,
               "task_type": e["task_type"], "mutual_accept": mutual[k],
               "agreement": ag.get("label") if ag.get("status") == "valid" else None,
               "agreement_status": ag.get("status"),
               "agreement_quotes": [ag.get("first_side_quote"), ag.get("second_side_quote")] if ag.get("label") else None,
               "onset_pathway": runs_row["onset_pathway"],
               "paper_onset_episode": runs_row["onset_evaluation_episode"]}
        for a in ("alice", "bob"):
            st = e["agents"][a]
            rl = relaxation.get((run["run_id"], k, a), {})
            row[a] = {"verdict": st["verdict"], "verdict_correct": st["verdict_correct"], "reward": st["reward"],
                      "task_correct": st["true_expected_verdict"] == "accept" if st.get("true_expected_verdict") else None,
                      "relaxation": rl.get("label") if rl.get("status") == "valid" else None,
                      "relaxation_status": rl.get("status")}
        rows.append(row)
    return rows


def locate_quote(quote, events):
    """Channel events whose content contains the quote (whitespace-normalised)."""
    norm = lambda s: " ".join((s or "").split())  # noqa: E731
    q = norm(quote)
    return [ev for ev in events if q and q in norm(ev["content"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=RESULTS / "paper" / "traces_qwen36_main", type=Path)
    parser.add_argument("--verdict-wire", default=RESULTS / "paper" / "paper_wire_verdict.jsonl", type=Path)
    parser.add_argument("--comm-wire", default=RESULTS / "cache" / "comm_wire.jsonl", type=Path)
    parser.add_argument("--labels", default=RESULTS / "paper" / "labels", type=Path)
    parser.add_argument("--out", default=RESULTS / "cache", type=Path)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    newline = tok("\n", add_special_tokens=False)["input_ids"]
    assert len(newline) == 1
    newline = newline[0]

    def table(name, key):
        rows = [r for r in pq.read_table(args.labels / f"{name}.parquet").to_pylist()
                if r["run_id"].startswith("main__qwen-3-6-27b__")]
        return {key(r): r for r in rows}

    runs_t = table("runs", lambda r: r["run_id"])
    agreement = table("agreement", lambda r: (r["run_id"], r["episode_index"]))
    relaxation = table("relaxation", lambda r: (r["run_id"], r["episode_index"], r["agent"]))
    comm = {(r["run"], r["agent"], r["kind"]): r for r in map(json.loads, open(args.comm_wire))}
    verdict = {(r["run"].split(".")[0], r["agent"]): r for r in map(json.loads, open(args.verdict_wire))
               if "skip" not in r and r["ep"] == 9}

    args.out.mkdir(parents=True, exist_ok=True)
    jobs_f = (args.out / "jobs.jsonl").open("w")
    index_f = (args.out / "index.jsonl").open("w")
    eps_f = (args.out / "episodes.jsonl").open("w")
    checks = collections.Counter()
    agreement_slots, quote_problems, jobs_meta = [], [], []
    n_rows = 0

    for f in sorted(args.traces.glob("rep*.json.gz")):
        rep = f.name.split(".")[0]
        run = json.load(gzip.open(f))
        episodes = episode_labels(run, runs_t[run["run_id"]], agreement, relaxation)
        if episodes[-1]["episode"] != 10:
            raise ValueError(f"{rep}: {len(episodes)} episodes")

        # Agreement message per episode: the channel event holding the later of the two quotes.
        agree_event = {}
        for k, er in enumerate(episodes):
            events = run["results"][k]["channel_transcript"]
            if er["agreement"]:
                hits = [locate_quote(q, events) for q in er["agreement_quotes"]]
                if not all(hits):
                    quote_problems.append({"run": rep, "episode": k + 1, "found": [len(h) for h in hits]})
                    continue
                ev = max((h[0] for h in hits), key=lambda e: e["event_id"])
                agree_event[k + 1] = ev["event_id"]
                order = [e["event_id"] for e in events if e["round"] == ev["round"]].index(ev["event_id"])
                agreement_slots.append({"run": rep, "episode": k + 1, "episode_type": er["episode_type"],
                                        "round": ev["round"] + 1, "order_in_round": order + 1,
                                        "sender": ev["sender"]})
            er["agreement_event_id"] = agree_event.get(k + 1)
            eps_f.write(json.dumps(er) + "\n")
        by_ep = {er["episode"]: er for er in episodes}

        last = run["results"][-1]
        for agent in ("alice", "bob"):
            msgs = last["agents"][agent]["messages"]
            ann = annotate(msgs, agent)
            events_by_slot = {}
            for k, e in enumerate(run["results"]):
                for ev in e["channel_transcript"]:
                    events_by_slot[(k + 1, ev["round"] + 1, ev["sender"])] = ev

            # Token-count check: the last real communication call vs the paper's logged input_tokens.
            chk = comm[(rep, agent, "check")]
            _, _, ids, _ = render(tok, chk["body"], True)
            checks["comm_check_exact" if len(ids) == chk["logged_input_tokens"] else "comm_check_mismatch"] += 1

            vrow = verdict[(rep, agent)]
            for phase, body, gen in (("communication", comm[(rep, agent, "pass")]["body"], False),
                                     ("verdict", vrow["body"], True)):
                conv, text, ids, offsets = render(tok, body, gen)
                if phase == "verdict":
                    checks["verdict_pass_exact" if len(ids) == vrow["logged_input_tokens"] else "verdict_pass_mismatch"] += 1
                if len(conv) > len(ann) or any(c["role"] != msgs[j]["role"] for j, c in enumerate(conv)):
                    raise ValueError(f"{rep} {agent} {phase}: wire messages do not line up with the trace")
                spans, tail = turn_spans(text, conv)
                job_id = f"{rep}.{agent}.{phase}"
                rows = []

                def add(pos_type, tok_idx, a, expect, **extra):
                    if ids[tok_idx] != expect:
                        raise ValueError(f"{job_id} {pos_type} ep{a['ep']}: token {ids[tok_idx]} != {expect}")
                    rows.append({"pos_type": pos_type, "token": tok_idx, "episode": a["ep"], **extra})

                def open_nl(j):  # "\n" after <|im_start|>assistant of turn j
                    return token_at(offsets, spans[j][0] + len("<|im_start|>assistant"))

                def close(j):
                    return token_at(offsets, spans[j][1])

                seen_comm, seen_verdict = set(), set()
                for j, a in enumerate(ann[: len(conv)]):
                    if a["kind"] == "communication_intro":
                        add("A", close(j), a, im_end)
                    elif a["kind"] in ("own_send", "own_send_failed") and a["ep"] not in seen_comm:
                        seen_comm.add(a["ep"])
                        add("A_call", open_nl(j), a, newline)
                    if a["kind"] in ("own_send", "peer_msg"):
                        ev = events_by_slot.get((a["ep"], a["round"], a["sender"]))
                        same = ev is not None and " ".join((ev["content"] or "").split()) == " ".join((a["text"] or "").split())
                        checks["msg_matches_channel" if same else "msg_channel_mismatch"] += 1
                        add("msg_end", close(j), a, im_end, round=a["round"], sender=a["sender"],
                            own=a["kind"] == "own_send", event_id=ev["event_id"] if ev else None,
                            is_agreement_msg=ev is not None and ev["event_id"] == agree_event.get(a["ep"]))
                    elif a["kind"] == "runner_notice":
                        add("runner_notice", close(j), a, im_end, round=a["round"])
                    elif a["kind"] == "verdict_call":
                        if a["ep"] not in seen_verdict:
                            seen_verdict.add(a["ep"])
                            add("C", open_nl(j), a, newline)
                        add("verdict", close(j), a, im_end)
                    elif a["kind"] == "feedback":
                        add("feedback", close(j), a, im_end)
                    elif a["kind"] == "reflection":
                        add("reflection_start", open_nl(j), a, newline)
                        add("reflection_end", close(j), a, im_end)
                if phase == "verdict":  # the generation prompt: <|im_start|>assistant\n<think>\n
                    gen_at = tail[-1]
                    add("C", token_at(offsets, gen_at + len("<|im_start|>assistant")),
                        {"ep": 10}, newline)

                rows.sort(key=lambda r: r["token"])
                positions = sorted({r["token"] for r in rows})
                row_of = {p: k for k, p in enumerate(positions)}
                job = {"id": job_id, "token_ids": ids, "n_prompt_tokens": len(ids),
                       "positions": {"saved": positions},
                       "sha256_token_ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                       "model_revision": pins.MODEL_REVISION}
                jobs_f.write(json.dumps(job) + "\n")
                jobs_meta.append({"id": job_id, "n_tokens": len(ids), "n_positions": len(positions)})
                for r in rows:
                    er = by_ep[r["episode"]]
                    index_f.write(json.dumps({
                        "job": job_id, "row": row_of[r["token"]], "run": rep, "sequence": er["sequence"],
                        "agent": agent, "pass": phase, **r,
                        "episode_type": er["episode_type"], "task_type": er["task_type"],
                        "verdict": er[agent]["verdict"], "peer_verdict": er[PEER[agent]]["verdict"],
                        "agreement": er["agreement"], "onset_pathway": er["onset_pathway"],
                        "relaxation": er[agent]["relaxation"],
                    }) + "\n")
                    n_rows += 1

    for fh in (jobs_f, index_f, eps_f):
        fh.close()

    # B slot for no-agreement onsets: where agreements complete in agreement onsets.
    onset_slots = [s for s in agreement_slots if s["episode_type"] == "onset"]
    by_round = collections.Counter(s["round"] for s in onset_slots)
    by_slot = collections.Counter((s["round"], s["order_in_round"]) for s in onset_slots)
    n_pos = sum(j["n_positions"] for j in jobs_meta)
    tokens = sorted((j["n_tokens"] for j in jobs_meta), reverse=True)
    bytes_per_pos = len(LAYERS) * HIDDEN * 2
    summary = {
        "jobs": len(jobs_meta), "tokens_total": sum(tokens), "tokens_max": tokens[0],
        "tokens_by_pass": {p: sum(j["n_tokens"] for j in jobs_meta if j["id"].endswith(p))
                           for p in ("communication", "verdict")},
        "positions_unique": n_pos, "index_rows": n_rows,
        "index_rows_by_type": dict(collections.Counter(json.loads(l)["pos_type"] for l in open(args.out / "index.jsonl"))),
        "layers": LAYERS, "storage_bytes": n_pos * bytes_per_pos,
        "storage_gib": round(n_pos * bytes_per_pos / 2**30, 2),
        "checks": dict(checks), "quote_problems": quote_problems,
        "agreement_onsets": len(onset_slots),
        "agreement_round_in_onsets": dict(sorted(by_round.items())),
        "agreement_slot_in_onsets": {f"r{r}.{o}": n for (r, o), n in sorted(by_slot.items())},
        "agreement_round_all": dict(sorted(collections.Counter(s["round"] for s in agreement_slots).items())),
        "agreement_sender_order_onsets": dict(collections.Counter(s["order_in_round"] for s in onset_slots)),
    }
    (args.out / "agreement_slots.json").write_text(json.dumps(agreement_slots, indent=1))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
