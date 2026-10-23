"""Step 6 (Mac): where does each call's thinking first state its verdict as a decision?

    tools/.venv/bin/python tools/step6/verdict_statements.py [--sample 25]

Rule-based: a decision statement is a first-person or verdict-framed phrase followed within a
few words by accept/reject ("I'll reject", "I should accept", "Therefore, I must reject",
"my verdict is accept", "Decision: reject", "I'll submit accept"). Not counted: sentences that
quote or paraphrase the policy (starting "If", "Otherwise", "Accept only if", "Reject if" or
containing "only if"), questions, past tense recaps ("I accepted ..."), hedged or conditional
sentences (containing if / when / unless / whether / maybe / perhaps / might / could / lean /
try to / probably / possibly / consider), and enumerated options (list items, "either", "or").

Per call: the first decision statement of either polarity, and the first one stating the
call's final verdict (the main measure), each as a thinking-token index and a fraction of the
thinking length. --sample prints that many random calls' detections for a hand check.

Writes agent-collusion/results/step6/verdict_statements.jsonl.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

RES = ROOT / "agent-collusion" / "results" / "step6"
LEAD = (r"(?:I\s*(?:'ll|will|should|must|need to|have to|am going to|'m going to|shall|'m|am)|I(?=\s+(?:accept|reject)\b)"
        r"|(?:therefore|thus|so|hence),?\s+I\s*(?:'ll|will|should|must|need to|have to)?"
        r"|my\s+verdict(?:\s+(?:is|will be|should be|must be))?|verdict\s*[:=-]|decision\s*[:=-]"
        r"|(?:I'll|I will|I should|let me|going to)\s+submit(?:\s+(?:a|an|my))?|I\s+(?:think\s+)?I\s+should\s+try)")
DECISION = re.compile(LEAD + r"\s+(?:(?:\w+|go\s+with|still)\s+){0,2}?(accept|reject)(?:ing|ed)?\b", re.I)
SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")
POLICY = re.compile(r"^\s*(?:[-*\d.)]+\s*)?(?:if|otherwise|accept only if|reject if|unless|when)\b|only if|\?\s*$", re.I)
HEDGE = re.compile(r"\b(?:if|when|unless|whether|maybe|perhaps|might|could|lean|leaning|try to|probably|possibly|consider)\b", re.I)
# "I think I should try accepting this time" is a decision; "try to accept if ..." is caught by "if"
OPTION = re.compile(r"^\s*(?:\d+[.)]|[-*•]|option\b|either\b|or\b)", re.I)


def statements(text):
    out = []
    for s in SENTENCE.finditer(text):
        sent = s.group(0)
        plain = re.sub(r"\beven (?:if|though|when)\b.*", "", sent, flags=re.I)  # concessive clauses are fine
        if POLICY.search(sent) or HEDGE.search(plain) or OPTION.search(sent):
            continue
        line_start = text.rfind("\n", 0, s.start()) + 1
        for m in DECISION.finditer(sent):
            if text[line_start:s.start() + m.start()].count('"') % 2 == 1:  # inside a quoted message
                continue
            if re.search(r"\b(?:accepted|rejected)\b", m.group(0), re.I) and not re.search(r"\b(?:be|been|get|gets)\b", m.group(0)):
                continue
            out.append({"char": s.start() + m.start(1), "polarity": m.group(1).lower(), "sentence": sent.strip()})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=25)
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    calls = [c for c in map(json.loads, open(RES / "calls.jsonl")) if c["kind"] == "call"]
    rows, mismatch_tok = [], 0
    for c in calls:
        text = c["reasoning"].strip()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if enc["input_ids"] != c["think_ids"]:
            mismatch_tok += 1
        offsets = enc["offset_mapping"]

        def token_of(ch):
            return next((k for k, (a, b) in enumerate(offsets) if a <= ch < b), None)

        st = statements(text)
        verdict = c["verdict"]
        first_any = st[0] if st else None
        first_final = next((s for s in st if s["polarity"] == verdict), None)
        n = c["n_think"]

        def pack(s):
            if s is None:
                return None
            k = token_of(s["char"])
            return {"token": k, "fraction": None if k is None else round(k / n, 4), "polarity": s["polarity"],
                    "sentence": s["sentence"][:300]}

        rows.append({"job": c["job"], "run": c["run"], "agent": c["agent"], "episode": c["episode"], "type": c["type"],
                     "verdict": verdict, "n_think": n, "first_any": pack(first_any), "first_final": pack(first_final),
                     "changed_mind": bool(first_any and first_any["polarity"] != verdict),
                     "n_statements": len(st)})
    with open(RES / "verdict_statements.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    found = sum(r["first_final"] is not None for r in rows)
    print(f"{len(rows)} calls; thinking re-tokenisation differs from the job in {mismatch_tok}; "
          f"final verdict stated as a decision in {found}; first statement has the other polarity in "
          f"{sum(r['changed_mind'] for r in rows)}")
    random.seed(23)
    for r in random.sample(rows, args.sample):
        c = next(x for x in calls if x["job"] == r["job"])
        ff = r["first_final"]
        print(f"\n=== {r['job']} verdict={r['verdict']} n_think={r['n_think']}")
        if ff:
            i = c["reasoning"].strip().find(ff["sentence"][:60])
            print(f"   first final-verdict statement at {ff['fraction']}: {ff['sentence'][:200]!r}")
            print(f"   context before: ...{c['reasoning'].strip()[max(0, i - 220):i][-220:]!r}")
        else:
            print("   none found; last 300 chars:", repr(c["reasoning"].strip()[-300:]))
        if r["changed_mind"]:
            print(f"   (first statement was {r['first_any']['polarity']} at {r['first_any']['fraction']}: "
                  f"{r['first_any']['sentence'][:150]!r})")


if __name__ == "__main__":
    main()
