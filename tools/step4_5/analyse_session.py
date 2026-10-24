"""Steps 4-5 (Mac): analyse the vLLM resampling session.

    tools/.venv/bin/python tools/step4_5/analyse_session.py

Inputs: plan.json, responses.jsonl (ok records), step 6 per_call.jsonl (C-probe score at C,
lock-in, first verdict statement) and step 6's edited-prompt C scores.

Per call: share of ACCEPT among valid samples for A (full verdict from C) and for each cut of B;
invalid = capped (24k tokens) or no verdict. Chosen cases (by group) and the random group are
reported separately.
  A vs C probe: Spearman correlation and agreement of the majority with P(accept) > 0.5
  B: share of samples ending in the call's saved verdict at each cut; the first fixed cut from
     which it stays >= 0.8 ("settled"), compared with the probe lock-in and the text statement
  event cuts: after the first stated decision vs right before the final statement
  C: share of ACCEPT on the variant B edited prompts (agreement onsets, saved verdict accept),
     next to the A share on the same unedited call where it was resampled
Writes agent-collusion/results/step4_5/analysis.json and figures in reports/figures/step4_5/.
"""

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
OUT = RES / "step4_5"
FIG = ROOT / "reports" / "figures" / "step4_5"
FIXED = ["fixed_30", "fixed_50", "fixed_70", "fixed_90", "fixed_100"]


def share(samples):
    valid = [s["verdict"] for s in samples if s["verdict"] in ("accept", "reject")]
    return (sum(v == "accept" for v in valid) / len(valid) if valid else None), len(valid), len(samples) - len(valid)


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~np.isnan(a) & ~np.isnan(b)
    return round(float(spearmanr(a[m], b[m]).statistic), 3) if m.sum() > 3 else None


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plan = json.loads((OUT / "plan.json").read_text())
    per = {r["job"]: r for r in map(json.loads, open(RES / "step6" / "per_call.jsonl"))}
    recs = {}
    for line in open(OUT / "responses.jsonl"):
        r = json.loads(line)
        if r.get("status") == "ok":
            recs[r["key"]] = r
    rows = []
    for c in plan["calls"]:
        row = {k: c[k] for k in ("job", "groups", "type", "agent", "verdict", "p_C", "n_think", "lock_in_0.1",
                                 "lock_in_0.05", "statement")}
        a = recs.get(f"{c['job']}|A")
        row["A"] = None if a is None else dict(zip(("share", "valid", "invalid"), share(a["samples"])))
        row["B"] = {}
        for k in c["cuts"]:
            r = recs.get(f"{c['job']}|B|{k['name']}")
            if r is not None:
                s, v, inv = share(r["samples"])
                row["B"][k["name"]] = {"share": s, "valid": v, "invalid": inv,
                                       "fraction": round(k["think_tokens"] / c["n_think"], 3)}
        rows.append(row)

    def group_of(row):
        g = row["groups"]
        return "random" if g[0].startswith("random") else "chosen"

    def match(s, verdict):  # share ending in the saved verdict
        return None if s is None else (s if verdict == "accept" else 1 - s)

    planned = sum(1 + len(c["cuts"]) for c in plan["calls"]) + len(plan["edits"])
    out = {"n_requests_ok": len(recs), "n_requests_planned": planned, "groups": {}}
    sets = {"reverted_reject": lambda r: "reverted_reject" in r["groups"], "flip": lambda r: "flip" in r["groups"],
            "confident_wrong": lambda r: "confident_wrong" in r["groups"],
            "random_onset": lambda r: "random_onset" in r["groups"],
            "random_before_onset": lambda r: "random_before_onset" in r["groups"],
            "random_all": lambda r: group_of(r) == "random"}
    for name, sel in sets.items():
        rs = [r for r in rows if sel(r) and r["A"] is not None and r["A"]["share"] is not None]
        if not rs:
            out["groups"][name] = {"n_calls_with_A": 0}
            continue
        e = {"n_calls_with_A": len(rs)}
        e["A_share_accept_mean"] = round(float(np.mean([r["A"]["share"] for r in rs])), 3)
        e["A_match_saved_mean"] = round(float(np.mean([match(r["A"]["share"], r["verdict"]) for r in rs])), 3)
        e["A_invalid"] = int(sum(r["A"]["invalid"] for r in rs))
        e["A_vs_probe_spearman"] = spearman([r["A"]["share"] for r in rs], [r["p_C"] for r in rs])
        e["A_majority_agrees_with_probe"] = int(sum((r["A"]["share"] > 0.5) == (r["p_C"] > 0.5) for r in rs))
        e["A_majority_agrees_with_saved"] = int(sum((r["A"]["share"] > 0.5) == (r["verdict"] == "accept") for r in rs))
        cut_m = {}
        for cut in FIXED + ["after_first_decision", "before_final"]:
            vals = [match(r["B"][cut]["share"], r["verdict"]) for r in rs if cut in r["B"] and r["B"][cut]["share"] is not None]
            if vals:
                cut_m[cut] = {"n": len(vals), "match_saved_mean": round(float(np.mean(vals)), 3),
                              "share_accept_mean": round(float(np.mean([r["B"][cut]["share"] for r in rs
                                                                         if cut in r["B"] and r["B"][cut]["share"] is not None])), 3),
                              "invalid": int(sum(r["B"][cut]["invalid"] for r in rs if cut in r["B"]))}
        e["B_by_cut"] = cut_m
        settled = collections.Counter()
        vs_lock = collections.Counter()
        for r in rs:
            seq = [(0.0, match(r["A"]["share"], r["verdict"]))] + [
                (int(c.split("_")[1]) / 100, match(r["B"][c]["share"], r["verdict"])) for c in FIXED if c in r["B"]]
            if len(seq) < 6 or any(v is None for _, v in seq):
                continue
            first = next((f for i, (f, _) in enumerate(seq) if all(v >= 0.8 for _, v in seq[i:])), None)
            r["settled_at"] = first
            settled[str(first)] += 1
            if first is not None:
                vs_lock["before lock-in (0.1)" if first < r["lock_in_0.1"] else "at/after lock-in (0.1)"] += 1
                if r["statement"] is not None:
                    vs_lock["before statement" if first < r["statement"] else "at/after statement"] += 1
        e["settled_at_fixed_cut"] = dict(settled)
        e["settled_vs_lock_in_and_statement"] = dict(vs_lock)
        out["groups"][name] = e

    # C: edited prompts
    c_rows = []
    for e in plan["edits"]:
        r = recs.get(f"{e['job']}|C")
        if r is None:
            continue
        s, v, inv = share(r["samples"])
        orig = next((x for x in rows if x["job"] == f"call.{e['run']}.{e['agent']}.ep{e['episode']:02d}"), None)
        c_rows.append({"job": e["job"], "share_accept_edited": s, "valid": v, "invalid": inv,
                       "share_accept_unedited_A": None if not orig or not orig["A"] else orig["A"]["share"]})
    if c_rows:
        out["C_edited_variant_B"] = {
            "n_prompts": len(c_rows), "share_accept_mean": round(float(np.mean([r["share_accept_edited"] for r in c_rows
                                                                               if r["share_accept_edited"] is not None])), 3),
            "invalid": int(sum(r["invalid"] for r in c_rows)),
            "paired_with_unedited_A": [r for r in c_rows if r["share_accept_unedited_A"] is not None],
            "rows": c_rows}
    (OUT / "analysis.json").write_text(json.dumps({"summary": out, "calls": rows}, indent=1, default=float))
    plots(rows)
    print(json.dumps(out, indent=1, default=float)[:5000])


def plots(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [0, 30, 50, 70, 90, 100]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    for name, sel, col in (("reverted rejects", lambda r: "reverted_reject" in r["groups"], "C3"),
                           ("flips", lambda r: "flip" in r["groups"], "C1"),
                           ("confidently wrong", lambda r: "confident_wrong" in r["groups"], "C4"),
                           ("random onset", lambda r: "random_onset" in r["groups"], "C2"),
                           ("random before-onset", lambda r: "random_before_onset" in r["groups"], "C0")):
        rs = [r for r in rows if sel(r) and r["A"] and r["A"]["share"] is not None]
        if not rs:
            continue
        ys = []
        for x, cut in zip(xs, ["A"] + FIXED):
            vals = []
            for r in rs:
                s = r["A"]["share"] if cut == "A" else (r["B"].get(cut) or {}).get("share")
                if s is not None:
                    vals.append(s if r["verdict"] == "accept" else 1 - s)
            ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(xs, ys, marker="o", color=col, label=f"{name} (n={len(rs)})")
    ax.axvspan(92, 96, color="grey", alpha=0.2, label="probe lock-in (median, bands 0.1-0.05)")
    ax.set_xlabel("cut: % of the saved thinking kept (0 = resample from C)")
    ax.set_ylabel("share of samples ending in the saved verdict")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Resampled verdicts by cut point")
    ax = axes[1]
    rs = [r for r in rows if r["A"] and r["A"]["share"] is not None]
    for name, sel, col in (("chosen", lambda r: not r["groups"][0].startswith("random"), "C3"),
                           ("random", lambda r: r["groups"][0].startswith("random"), "C0")):
        pts = [(r["p_C"], r["A"]["share"]) for r in rs if sel(r)]
        if pts:
            ax.scatter(*zip(*pts), s=18, color=col, alpha=0.7, label=name)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("C probe P(accept) at C (out-of-fold, step 6)")
    ax.set_ylabel("share ACCEPT, 16 full resamples from C")
    ax.legend(fontsize=8)
    ax.set_title("Full resample from C vs the C probe")
    fig.tight_layout()
    fig.savefig(FIG / "session_cuts_and_probe.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
