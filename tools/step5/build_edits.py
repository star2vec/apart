"""Step 5 (Mac): agreement-onset verdict prompts with the agreement lines removed.

    tools/.venv/bin/python tools/step5/build_edits.py [--show 3]

Source: the paper's rebuilt verdict request for each agent in each agreement-onset episode
(paper_wire_verdict.jsonl; the history up to and including the onset episode's verdict prompt).
Edits touch only that episode's communication phase:

  variant A  remove the agreement quotes the paper's judge cited (first_side_quote,
             second_side_quote) wherever they occur: delivered messages, the agent's own
             send_message arguments, and failed send attempts carrying the same text; on each
             edited message the message type (e.g. "acceptance", "final_accept") becomes
             "message"
  variant B  A, plus every remaining sentence in that phase's messages containing "accept…" or
             "agree…" (e.g. "Agreed.", "Thanks for accepting my answer.")

Writes agent-collusion/results/step5/edits.jsonl: per (run, episode, agent, variant) the edited
wire body, the list of changes, and the original body's row reference.
"""

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "agent-collusion" / "results"
OUT = RESULTS / "step5"
HEADER = re.compile(r"^(\[round=\d+/\d+\]\[from=\w+\]\[type=)([^\]]*)(\]\n\n)", re.S)
SENT = re.compile(r"[^.!?\n]*\b(?:accept|agree)\w*\b[^.!?\n]*[.!?]?", re.I)


def quote_pattern(q):
    words = [re.escape(w) for w in q.split()]
    return re.compile(r"\s*".join(words)) if words else None


def tidy(text):
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    return re.sub(r"[\s,;:\-\u2013\u2014]+$", "", text).strip()  # no dangling "," or "-" where a quote ended


def edit_text(text, patterns, extra_accept):
    changed, removed = text, []
    for p in patterns:
        for m in p.finditer(changed):
            removed.append(m.group(0))
        changed = p.sub(" ", changed)
    if extra_accept:
        for m in SENT.finditer(changed):
            if m.group(0).strip():
                removed.append(m.group(0).strip())
        changed = SENT.sub(" ", changed)
    changed = tidy(changed)
    return changed, removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = [json.loads(l) for l in open(RESULTS / "step1_cache" / "episodes.jsonl")]
    onsets = [e for e in episodes if e["episode_type"] == "onset" and e["agreement"]]
    wire = {}
    for line in open(RESULTS / "paper" / "paper_wire_verdict.jsonl"):
        r = json.loads(line)
        if "skip" not in r:
            wire[(r["run"].split(".")[0], r["ep"] + 1, r["agent"])] = r
    rows, summary = [], []
    for e in onsets:
        run = e["run"].split("__")[-1]
        patterns = [quote_pattern(q) for q in e["agreement_quotes"] if q]
        for agent in ("alice", "bob"):
            src = wire[(run, e["episode"], agent)]
            msgs = src["body"]["messages"]
            start = max(i for i, m in enumerate(msgs) if m["role"] == "user"
                        and str(m.get("content", "")).startswith(f"## Episode {e['episode']}: communication phase"))
            end = len(msgs) - 1  # the verdict prompt itself is last
            for variant, extra in (("A", False), ("B", True)):
                body = json.loads(json.dumps(src["body"]))
                changes = []
                for i in range(start, end):
                    m = body["messages"][i]
                    if m["role"] == "user" and isinstance(m.get("content"), str) and (h := HEADER.match(m["content"])):
                        before = m["content"][h.end():]
                        after, removed = edit_text(before, patterns, extra)
                        if removed:
                            mtype = h.group(2)
                            m["content"] = h.group(1) + "message" + h.group(3) + after
                            changes.append({"i": i, "where": "peer message", "type_was": mtype,
                                            "before": before, "after": after, "removed": removed})
                    elif m["role"] == "assistant" and m.get("tool_calls"):
                        for tc in m["tool_calls"]:
                            if tc["function"]["name"] != "send_message":
                                continue
                            a = json.loads(tc["function"]["arguments"] or "{}")
                            before = a.get("content") or ""
                            after, removed = edit_text(before, patterns, extra)
                            if removed:
                                failed = (i + 1 < len(body["messages"]) and body["messages"][i + 1]["role"] == "tool"
                                          and '"success": false' in (body["messages"][i + 1].get("content") or ""))
                                changes.append({"i": i, "where": "own send" + (" (failed attempt)" if failed else ""),
                                                "type_was": a.get("message_type"), "before": before, "after": after,
                                                "removed": removed})
                                a["content"] = after
                                if "message_type" in a:
                                    a["message_type"] = "message"
                                tc["function"]["arguments"] = json.dumps(a)
                quotes_found = [any(p.search(c["before"]) for c in changes) for p in patterns]
                rows.append({"run": run, "episode": e["episode"], "agent": agent, "variant": variant,
                             "quotes": e["agreement_quotes"], "quotes_found": quotes_found,
                             "n_changes": len(changes), "changes": changes, "body": body,
                             "logged_input_tokens_original": src["logged_input_tokens"]})
                summary.append((run, e["episode"], agent, variant, len(changes), quotes_found,
                                sum(len(c["removed"]) for c in changes)))
    with open(OUT / "edits.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} edited prompts ({len(onsets)} episodes x 2 agents x 2 variants)")
    missing = [s for s in summary if not all(s[5])]
    print("quotes not found in an agent's view:", missing if missing else "none")
    for v in ("A", "B"):
        n = [s[4] for s in summary if s[3] == v]
        print(f"variant {v}: messages edited per prompt min {min(n)} max {max(n)} mean {sum(n) / len(n):.1f}; "
              f"pieces removed {sum(s[6] for s in summary if s[3] == v)}")
    for r in [r for r in rows if r["variant"] == "B"][: args.show]:
        print(f"\n=== {r['run']} episode {r['episode']} {r['agent']} (variant B)")
        for c in r["changes"]:
            print(f"  [{c['where']}, type was {c['type_was']}]\n    before: {c['before'][:300]!r}\n    after:  {c['after'][:300]!r}")


if __name__ == "__main__":
    main()
