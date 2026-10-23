"""Steps 4-5 prep (Mac): calls whose thinking first states the opposite verdict ("flips").

    tools/.venv/bin/python tools/step4_5/flips.py

Uses step 6's verdict_statements.jsonl (rule-based, hand-checked) and per-token C-probe scores.
Reports flips as a share of all accept-ending and all reject-ending calls, by episode type, and
draws each flip call's smoothed C-probe curve with its first (opposite) and first final-verdict
statements marked.

Writes agent-collusion/results/step4_5/flips.json and reports/figures/step4_5/flips.png.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step6"))
sys.argv = sys.argv[:1]
import analyse_step6 as a6  # noqa: E402

S6 = ROOT / "agent-collusion" / "results" / "step6"
OUT = ROOT / "agent-collusion" / "results" / "step4_5"
FIG = ROOT / "reports" / "figures" / "step4_5"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    calls = {c["job"]: c for c in map(json.loads, open(S6 / "calls.jsonl")) if c["kind"] == "call"}
    vs = [json.loads(l) for l in open(S6 / "verdict_statements.jsonl")]
    table = {}
    for t in ("before_onset", "onset", "reverted"):
        for v in ("accept", "reject"):
            rows = [r for r in vs if r["type"] == t and r["verdict"] == v]
            fl = [r for r in rows if r["changed_mind"]]
            table[f"{t}:{v}"] = {"calls": len(rows), "flips": len(fl),
                                 "share": round(len(fl) / len(rows), 3) if rows else None,
                                 "no_statement": sum(r["first_final"] is None for r in rows)}
    main_flips = [r for r in vs if r["changed_mind"] and r["type"] in ("onset", "before_onset")]
    rev_flips = [r for r in vs if r["changed_mind"] and r["type"] == "reverted"]
    out = {"shares": table, "n_main_flips": len(main_flips), "n_reverted_flips": len(rev_flips),
           "main_flips": [{"job": r["job"], "type": r["type"], "agent": r["agent"], "verdict": r["verdict"],
                           "first_any": r["first_any"]["fraction"], "first_any_polarity": r["first_any"]["polarity"],
                           "first_final": r["first_final"]["fraction"] if r["first_final"] else None,
                           "first_any_sentence": r["first_any"]["sentence"][:200],
                           "first_final_sentence": r["first_final"]["sentence"][:200] if r["first_final"] else None}
                          for r in main_flips],
           "reverted_flips": [r["job"] for r in rev_flips]}
    fa = [x["first_any"] for x in out["main_flips"]]
    ff = [x["first_final"] for x in out["main_flips"] if x["first_final"] is not None]
    out["positions"] = {"first_opposite_median": round(float(np.median(fa)), 3),
                        "first_final_median": round(float(np.median(ff)), 3)}
    (OUT / "flips.json").write_text(json.dumps(out, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    acts = ROOT / "agent-collusion" / "results" / "step6" / "acts"
    fl = out["main_flips"]
    cols = 7
    rows_n = int(np.ceil(len(fl) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(21, 2.6 * rows_n), sharey=True)
    for ax, x in zip(axes.flat, fl):
        c = calls[x["job"]]
        lg = a6.read_safetensors(acts / f"{x['job']}.safetensors")["probe_logits"].astype(float)
        th = lg[c["think_start"] - c["c_token"]:]
        sp = a6.smooth(a6.sigmoid(th))
        xs = np.linspace(0, 1, len(sp))
        ax.plot(xs, sp, color="C0", lw=1.2)
        ax.axhline(a6.sigmoid(lg[0]), color="grey", ls=":", lw=1)
        ax.axvline(x["first_any"], color="C3" if x["first_any_polarity"] == "reject" else "C2", lw=1.2)
        if x["first_final"] is not None:
            ax.axvline(x["first_final"], color="C2" if x["verdict"] == "accept" else "C3", lw=1.2, ls="--")
        ax.set_title(f"{x['job'].replace('call.', '')} ({x['type'][:6]}) → {x['verdict']}", fontsize=7)
        ax.set_ylim(0, 1)
    for ax in axes.flat[len(fl):]:
        ax.axis("off")
    fig.suptitle("Flip calls: smoothed C-probe P(accept) through the thinking; solid line = first decision stated "
                 "(red reject, green accept), dashed = first statement of the final verdict, dotted = score at C")
    fig.tight_layout()
    fig.savefig(FIG / "flips.png", dpi=110)
    plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k in ("shares", "n_main_flips", "n_reverted_flips", "positions")}, indent=1))


if __name__ == "__main__":
    main()
