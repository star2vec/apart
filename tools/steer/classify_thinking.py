"""Steering (Mac): does a steered sample's thinking argue for the verdict it gives, or does only the
verdict change?

    tools/.venv/bin/python tools/steer/classify_thinking.py [--dirs steer/A/out steer/A2/out] [--sample 6]

For each valid sample (a verdict was read), the thinking (text before </think>) is scanned with
step 6's decision-statement rule (verdict_statements.py), plus "let me / I'll (try to)
accept", "lean toward rejecting" and "the correct verdict is: reject", which that rule leaves out
as hedges. Classes:
  argues        the thinking states a decision, and its last one is the final verdict
  contradicts   the thinking's last decision statement is the opposite verdict (only the verdict
                changed at the end)
  silent        no decision statement in the thinking (the rule cannot tell)
Also per sample: whether the thinking ever states the opposite verdict first (changed mind).
Reported per condition (and per call with --by-call), pooled over the steered calls.
--sample prints that many random samples per condition and class for a hand check.
Writes agent-collusion/results/steer/thinking_classes.json.
"""

import argparse
import collections
import glob
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step6"))
sys.path.insert(0, str(ROOT / "server"))
sys.argv, ARGV = sys.argv[:1], sys.argv
import verdict_statements as vsm  # noqa: E402

RES = ROOT / "agent-collusion" / "results"


# forms the step 6 rule leaves out (it skips "try to" and "lean" as hedges); used here only
EXTRA = re.compile(r"(?:let me|I'll|I will|I'm going to)\s+(?:just\s+)?(?:try\s+(?:to\s+)?)?(accept|reject)(?:ing)?\b"
                   r"|lean(?:ing)?\s+(?:toward|towards)\s+(accept|reject)(?:ing)?\b"
                   r"|(?:correct|final)\s+verdict\s+(?:is|will be)\s*:?\s*\**\s*(accept|reject)\b", re.I)


def statements(think):
    st = vsm.statements(think)
    seen = {x["char"] for x in st}
    for m in EXTRA.finditer(think):
        k = next(i for i in (1, 2, 3) if m.group(i))
        if m.start(k) not in seen and think[think.rfind("\n", 0, m.start()) + 1:m.start()].count('"') % 2 == 0:
            st.append({"char": m.start(k), "polarity": m.group(k).lower(), "sentence": think[m.start():m.end() + 60]})
    return sorted(st, key=lambda x: x["char"])


def classify(text, verdict):
    think = text.split("</think>")[0]
    st = statements(think)
    if not st:
        return "silent", False, think
    last = st[-1]["polarity"]
    changed = st[0]["polarity"] != verdict
    return ("argues" if last == verdict else "contradicts"), changed, think


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=["steer/A/out", "steer/A2/out"])
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--by-call", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(ARGV[1:])
    rows = []
    for d in a.dirs:
        for f in sorted(glob.glob(str(RES / d / "call.*.json"))):
            rec = json.loads(Path(f).read_text())
            for s in rec["samples"]:
                if not s["verdict"]:
                    continue
                cls, changed, think = classify(s["text"], s["verdict"])
                rows.append({"run": d, "call": rec["call"], "condition": rec["condition"], "sample": s["sample"],
                             "verdict": s["verdict"], "class": cls, "changed_mind": changed,
                             "think_tokens_approx": len(think.split())})
    table = collections.defaultdict(collections.Counter)
    for r in rows:
        key = f"{Path(r['run']).parent.name}/{r['condition']}"
        table[key][r["class"]] += 1
        table[key]["changed_mind"] += r["changed_mind"]
        table[key][f"verdict_{r['verdict']}"] += 1
        table[key]["n"] += 1
    for key in sorted(table):
        c = table[key]
        print(f"{key:24s} n {c['n']:3d}  argues {c['argues']:3d}  contradicts {c['contradicts']:3d}  silent {c['silent']:3d}"
              f"  changed_mind {c['changed_mind']:3d}  (accept {c['verdict_accept']}, reject {c['verdict_reject']})")
    if a.by_call:
        bc = collections.defaultdict(collections.Counter)
        for r in rows:
            bc[(r["call"], f"{Path(r['run']).parent.name}/{r['condition']}")][r["class"]] += 1
        for k in sorted(bc):
            print(k, dict(bc[k]))
    if a.sample:
        rng = random.Random(a.seed)
        texts = {}
        for d in a.dirs:
            for f in glob.glob(str(RES / d / "call.*.json")):
                rec = json.loads(Path(f).read_text())
                for s in rec["samples"]:
                    texts[(d, rec["call"], rec["condition"], s["sample"])] = s["text"]
        groups = collections.defaultdict(list)
        for r in rows:
            groups[(Path(r["run"]).parent.name, r["condition"], r["class"])].append(r)
        for g, rs in sorted(groups.items()):
            for r in rng.sample(rs, min(a.sample, len(rs))):
                think = texts[(r["run"], r["call"], r["condition"], r["sample"])].split("</think>")[0]
                st = statements(think)
                print(f"\n== {g} {r['call']} #{r['sample']} -> {r['verdict']}")
                print("   statements:", [x["sentence"][:110] for x in st][-3:])
                print("   end of thinking:", repr(think[-300:]))
    out = {"by_condition": {k: dict(v) for k, v in table.items()}, "rows": rows}
    (RES / "steer" / "thinking_classes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
