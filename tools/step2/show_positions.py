"""Step 2, check before training: decode the token at saved positions and the text before it.

    tools/.venv/bin/python tools/step2/show_positions.py --types A C --n 3 [--context 300]

Rows are picked deterministically (spread over runs, agents, episodes and passes).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "server"))

import pins  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

CACHE = ROOT / "agent-collusion" / "results" / "step1_cache"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--types", nargs="+", default=["A", "C"])
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--context", type=int, default=300, help="characters of text before the token")
    parser.add_argument("--pass", dest="pass_", choices=["communication", "verdict"])
    parser.add_argument("--episode", type=int)
    parser.add_argument("--after", type=int, default=12, help="tokens shown after it")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    index = [json.loads(l) for l in open(CACHE / "index.jsonl")]
    picks = {}
    for t in args.types:
        rows = [r for r in index if r["pos_type"] == t and (args.pass_ is None or r["pass"] == args.pass_)
                and (args.episode is None or r["episode"] == args.episode)]
        step = max(1, len(rows) // args.n)
        picks[t] = [rows[(k * step + k * 37) % len(rows)] for k in range(args.n)]
    wanted = {r["job"] for rs in picks.values() for r in rs}
    ids = {}
    for line in open(CACHE / "jobs.jsonl"):
        j = json.loads(line)
        if j["id"] in wanted:
            ids[j["id"]] = j["token_ids"]
    for t, rows in picks.items():
        for r in rows:
            seq, p = ids[r["job"]], r["token"]
            before = tok.decode(seq[max(0, p - 400):p])[-args.context:]
            print(f"=== {t} | {r['run']} {r['agent']} episode {r['episode']} ({r['episode_type']}) | "
                  f"{r['pass']} pass | token {p} of {len(seq)} | verdict {r['verdict']}")
            print(f"...{before}", end="")
            print(f"\033[1m[[{tok.decode([seq[p]])!r} id={seq[p]}]]\033[0m", end="")
            print(tok.decode(seq[p + 1:p + 1 + args.after]).replace("\n", "\\n") if p + 1 < len(seq) else "<END>")
            print()


if __name__ == "__main__":
    main()
