"""Steps 4-5 follow-up (Mac): five predictors of the resampled ACCEPT share, on the same calls.

    tools/.venv/bin/python tools/step4_5/five_predictors.py

Target: the first session's full resample from C (part A share), the 102 calls of
baselines_vs_resampling.py. Predictors:
  probe_at_C       step 6's C-probe P(accept) at C (out of fold)
  feedback         previous-episode feedback baseline (out of fold, baselines_vs_resampling.py)
  stated_half      the latest decision the saved thinking has stated in its first half
                   (+1 accept, -1 reject, 0 none yet; verdict_statements.py rule)
  stated_full      the same over the full thinking: a ceiling, since the full thinking names the
                   verdict the saved call gave
  contrast         projection of the call's C activation (layer 47, step-1 cache) on the
                   approve-vs-reject direction from unrelated review tasks (steering item 3); not
                   fitted on the game. Its majority threshold is the midpoint of the class means
                   of that projection on the game's onset and before-onset rows of the other
                   trajectories (leave one trajectory out)
Reported: Spearman with the share, majority agreement (share > 0.5 vs predictor > 0.5, or > 0 for
the detectors; calls with no statement count as a miss), and how many calls the detectors cover,
random group and hard cases separately, and hard cases without the "confident_wrong" calls (they
were selected on probe errors, so the version without them is the fair one for the probe): without
the 6 whose only group is confident_wrong, and without all 10 tagged so (4 are also flips).
Writes agent-collusion/results/step4_5/five_predictors.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step6"))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools" / "step2"))
sys.path.insert(0, str(ROOT / "tools" / "step3"))
sys.argv = sys.argv[:1]
import verdict_statements as vsm  # noqa: E402
import probe_lib as pl  # noqa: E402
import run_step3 as s3  # noqa: E402


def contrast_scores():
    """Contrast-direction projection at C for every (run, agent, episode), and per-trajectory
    thresholds from the other trajectories' onset / before-onset rows."""
    uc = np.load(RES / "steer" / "I34" / "out" / "contrast_direction.npz")["unit"].astype(np.float64)
    index, episodes = pl.load_index(), pl.load_episodes()
    c_rows = {(r["run"], r["agent"], r["episode"]): i for i, r in enumerate(index)
              if r["pos_type"] == "C" and r["pass"] == "verdict"}
    keys = sorted(c_rows)
    X = pl.rows_to_float(pl.load_layer(47), np.array([c_rows[k] for k in keys])).astype(np.float64)
    proj = dict(zip(keys, X @ uc))
    lab = [(r["run"], proj[(r["run"], r["agent"], r["episode"])], r["y"]) for r in s3.build_rows(index, episodes)
           if r["type"] in ("onset", "before_onset")]
    thr = {}
    for run in {k[0] for k in keys}:
        p1 = [p for rr, p, y in lab if rr != run and y == 1]
        p0 = [p for rr, p, y in lab if rr != run and y == 0]
        thr[run] = (np.mean(p1) + np.mean(p0)) / 2
    return proj, thr

RES = ROOT / "agent-collusion" / "results"
PREDICTORS = ("probe_at_C", "feedback", "stated_half", "stated_full", "contrast")


def main() -> None:
    from scipy.stats import spearmanr
    import pins
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(pins.MODEL_ID, revision=pins.MODEL_REVISION)
    an = json.loads((RES / "step4_5" / "analysis.json").read_text())["calls"]
    calls6 = {c["job"]: c for c in map(json.loads, open(RES / "step6" / "calls.jsonl")) if c["kind"] == "call"}
    fb = {c["job"]: c["feedback"] for c in json.loads((RES / "step4_5" / "baselines_vs_resampling.json").read_text())["calls"]}
    proj, thr = contrast_scores()
    rows = []
    for c in an:
        if not c["A"] or c["A"]["share"] is None:
            continue
        j = c["job"]
        _, run, _, ep = j.split(".")
        text = calls6[j]["reasoning"].strip()
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = enc["offset_mapping"]
        n = len(offs)
        stmts = []
        for s in vsm.statements(text):
            k = next((i for i, (a, b) in enumerate(offs) if a <= s["char"] < b), None)
            if k is not None:
                stmts.append((k, s["polarity"]))

        def stated(upto):
            said = [p for t, p in stmts if t < upto]
            return 0 if not said else (1 if said[-1] == "accept" else -1)

        rows.append({"job": j, "random": c["groups"][0].startswith("random"), "share": c["A"]["share"],
                     "n_think": n, "probe_at_C": c["p_C"], "feedback": fb[j],
                     "stated_half": stated(n // 2), "stated_full": stated(n + 1),
                     "cw_only": c["groups"] == ["confident_wrong"], "cw_any": "confident_wrong" in c["groups"],
                     "contrast": float(proj[(run, c["agent"], int(ep[2:]))]), "contrast_thr": float(thr[run])})
    out = {}
    groups = {"random": lambda r: r["random"], "hard": lambda r: not r["random"],
              "hard_without_6_confident_wrong": lambda r: not r["random"] and not r["cw_only"],
              "hard_without_all_10_confident_wrong": lambda r: not r["random"] and not r["cw_any"]}
    for g, keep in groups.items():
        rs = [r for r in rows if keep(r)]
        share = np.array([r["share"] for r in rs])
        e = {"n": len(rs), "share_mean": round(float(share.mean()), 3)}
        for p in PREDICTORS:
            x = np.array([r[p] for r in rs], float)
            thr = 0.0 if p.startswith("stated") else 0.5
            if p == "contrast":  # out-of-fold threshold per call
                x_major = x > np.array([r["contrast_thr"] for r in rs])
                agree = x_major == (share > 0.5)
            else:
                agree = ((x > thr) == (share > 0.5)) & ((x != 0) | (thr == 0.5))
            e[p] = {"spearman": None if np.std(x) == 0 else round(float(spearmanr(x, share).statistic), 3),
                    "majority_agree": round(float(np.mean(agree)), 3)}
            if p.startswith("stated"):
                e[p]["covered"] = int(np.sum(x != 0))
        out[g] = e
        print(g, json.dumps(e), flush=True)
    (RES / "step4_5" / "five_predictors.json").write_text(json.dumps({"by_group": out, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
