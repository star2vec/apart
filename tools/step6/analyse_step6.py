"""Step 6 (Mac): analyse the projections session.

    tools/.venv/bin/python tools/step6/analyse_step6.py --acts <downloaded step6 acts dir> [--boot 2000]

Inputs: the worker's per-job outputs (probe_logits from C to the last thinking token; layer 47
vectors at C and at the token ending each tenth), calls.jsonl, verdict_statements.jsonl, and the
step 1 cache (the C vector each call already has there, for a consistency check).

Main readout is separation (AUC, ACCEPT vs REJECT calls, onset and before-onset only), not
absolute scores: the C probe was trained on C's token type and its scores shift on others.
  - per tenth of the thinking: AUC of the C probe's mean logit in that tenth, and AUC of fresh
    probes fit on the saved tenth-end vectors (trajectory-grouped nested CV as in steps 2-3)
  - scores at C, at the first and at the last thinking token; first token vs C
  - smoothed curves per group (thinking position normalised to 0-1)
  - lock-in: the first thinking token after which the smoothed probability stays within a band
    (0.1, and 0.05 as a check) of that call's final smoothed probability, as a fraction of the
    thinking; compared with where the text first states the final verdict (before / at / after)
  - everything pooled and split by speaking order (Alice first, Bob second)
Reverted calls are out of distribution for the probe and are reported as case studies.
Step 5's edited prompts: C-probe score at C, edited vs original.

Writes agent-collusion/results/step6/analysis.json and figures in reports/figures/step6/.
"""

import argparse
import collections
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step2"))
sys.path.insert(0, str(ROOT / "tools" / "step3"))
import probe_lib as pl  # noqa: E402
import run_step3 as s3  # noqa: E402

RES = ROOT / "agent-collusion" / "results" / "step6"
FIG = ROOT / "reports" / "figures" / "step6"


def read_safetensors(path):
    raw = path.read_bytes()
    (n,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8:8 + n])
    out = {}
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        s, e = spec["data_offsets"]
        buf = raw[8 + n + s:8 + n + e]
        if spec["dtype"] == "BF16":
            a = (np.frombuffer(buf, "<u2").astype(np.uint32) << 16).view(np.float32)
        elif spec["dtype"] == "F32":
            a = np.frombuffer(buf, "<f4").copy()
        else:
            raise ValueError(spec["dtype"])
        out[name] = a.reshape(spec["shape"])
    return out


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def smooth(p, frac=0.05, min_w=5):
    """Centred moving average over max(min_w, frac * length) tokens (edges use what exists)."""
    w = max(min_w, int(round(frac * len(p))))
    k = np.ones(w)
    num = np.convolve(p, k, mode="same")
    den = np.convolve(np.ones_like(p), k, mode="same")
    return num / den


def lock_in(ps, band):
    """First index after which every value stays within `band` of the final value."""
    final = ps[-1]
    outside = np.where(np.abs(ps - final) > band)[0]
    return 0 if len(outside) == 0 else int(outside[-1]) + 1


def grouped_auc_cv(X, y, runs, repeats=5):
    """Fresh probe on X (n, hidden) with the steps 2-3 protocol; returns fold-mean AUC stats."""
    splits = pl.make_splits(runs, repeats=repeats)
    cache = pl.design_cache(X, splits, reduce=True)
    return pl.summarise(pl.evaluate(cache, splits, y))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts", required=True, type=Path)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    calls = [json.loads(l) for l in open(RES / "calls.jsonl")]
    vs = {r["job"]: r for r in map(json.loads, open(RES / "verdict_statements.jsonl"))}
    data, missing = {}, []
    for c in calls:
        f = args.acts / f"{c['job']}.safetensors"
        if not f.exists():
            missing.append(c["job"])
            continue
        data[c["job"]] = read_safetensors(f)
    print(f"{len(data)} jobs loaded, {len(missing)} missing", flush=True)

    # consistency: C vector here vs the step 1 cache (same prompt prefix, separate run)
    index, episodes = pl.load_index(), pl.load_episodes()
    c_rows = {(r["run"], r["agent"], r["episode"]): i for i, r in enumerate(index)
              if r["pos_type"] == "C" and r["pass"] == "verdict"}
    cache47 = pl.load_layer(47)
    cos = []
    for c in calls:
        if c["kind"] != "call" or c["job"] not in data:
            continue
        here = data[c["job"]]["resid_post.47"][0]
        there = pl.rows_to_float(cache47, np.array([c_rows[(c["run"], c["agent"], c["episode"])]]))[0]
        cos.append(float(here @ there / np.linalg.norm(here) / np.linalg.norm(there)))

    main_calls = [c for c in calls if c["kind"] == "call" and c["type"] in ("onset", "before_onset") and c["job"] in data]
    out = {"n_loaded": len(data), "missing": missing,
           "c_vector_vs_step1_cache_cosine": {"min": round(min(cos), 6), "median": round(float(np.median(cos)), 6)}}
    groups = {"pooled": lambda c: True, "alice (speaks first)": lambda c: c["agent"] == "alice",
              "bob (speaks second)": lambda c: c["agent"] == "bob"}

    def per_call(c):
        d = data[c["job"]]
        lg = d["probe_logits"]  # positions C .. last thinking token
        think0 = c["think_start"] - c["c_token"]
        th = lg[think0:]
        n = len(th)
        tenths = [th[(j * n) // 10:max((j * n) // 10 + 1, ((j + 1) * n) // 10)].mean() for j in range(10)]
        p = sigmoid(th.astype(np.float64))
        sp = smooth(p)
        v = vs[c["job"]]
        stmt = v["first_final"]["fraction"] if v["first_final"] else None
        li = {b: lock_in(sp, b) / n for b in (0.1, 0.05)}
        return {"c_logit": float(lg[0]), "first_logit": float(th[0]), "last_logit": float(th[-1]),
                "tenth_mean_logits": tenths, "smoothed": sp, "lock_in": li, "statement": stmt,
                "final_p": float(sp[-1]), "n": n, "tenth_vectors": d["resid_post.47"][1:11]}

    pc = {c["job"]: per_call(c) for c in calls if c["kind"] == "call" and c["job"] in data}

    for g, sel in groups.items():
        cs = [c for c in main_calls if sel(c)]
        y = np.array([c["verdict"] == "accept" for c in cs], int)
        runs = [c["run"] for c in cs]
        e = {"n": len(cs), "n_accept": int(y.sum())}
        e["auc_c_logit"] = round(float(pl.auc(y, [pc[c["job"]]["c_logit"] for c in cs])), 4)
        e["auc_first_token"] = round(float(pl.auc(y, [pc[c["job"]]["first_logit"] for c in cs])), 4)
        e["auc_last_token"] = round(float(pl.auc(y, [pc[c["job"]]["last_logit"] for c in cs])), 4)
        e["first_minus_c_logit"] = {
            "median": round(float(np.median([pc[c["job"]]["first_logit"] - pc[c["job"]]["c_logit"] for c in cs])), 3),
            "corr": round(float(np.corrcoef([pc[c["job"]]["first_logit"] for c in cs],
                                            [pc[c["job"]]["c_logit"] for c in cs])[0, 1]), 4)}
        e["c_probe_auc_by_tenth"] = [round(float(pl.auc(y, [pc[c["job"]]["tenth_mean_logits"][j] for c in cs])), 4)
                                     for j in range(10)]
        e["fresh_probe_auc_by_tenth"] = []
        for j in range(10):
            X = np.stack([pc[c["job"]]["tenth_vectors"][j] for c in cs])
            e["fresh_probe_auc_by_tenth"].append(grouped_auc_cv(X, y, runs))
            print(g, "tenth", j + 1, e["c_probe_auc_by_tenth"][j], e["fresh_probe_auc_by_tenth"][-1]["mean"], flush=True)
        # spread of C scores within each verdict group (to judge the lock-in bands)
        for lab, val in (("accept", 1), ("reject", 0)):
            ps = np.array([sigmoid(pc[c["job"]]["c_logit"]) for c, yy in zip(cs, y) if yy == val])
            e[f"c_prob_spread_{lab}"] = {"median": round(float(np.median(ps)), 3), "iqr": [round(float(np.percentile(ps, 25)), 3), round(float(np.percentile(ps, 75)), 3)], "sd": round(float(ps.std()), 3)}
        # lock-in vs the verdict statement
        lk = {}
        for b in (0.1, 0.05):
            rel = collections.Counter()
            fr = []
            for c in cs:
                x = pc[c["job"]]
                fr.append(x["lock_in"][b])
                if x["statement"] is None:
                    rel["no statement"] += 1
                else:
                    tol = 1 / max(x["n"], 1) * 3  # within 3 tokens counts as "at"
                    d = x["lock_in"][b] - x["statement"]
                    rel["before" if d < -tol else "after" if d > tol else "at"] += 1
            lk[str(b)] = {"median_fraction": round(float(np.median(fr)), 3), "iqr": [round(float(np.percentile(fr, 25)), 3), round(float(np.percentile(fr, 75)), 3)],
                          "vs_statement": dict(rel),
                          "by_verdict": {lab: dict(collections.Counter(
                              ("no statement" if pc[c["job"]]["statement"] is None else
                               "before" if pc[c["job"]]["lock_in"][b] - pc[c["job"]]["statement"] < -3 / pc[c["job"]]["n"] else
                               "after" if pc[c["job"]]["lock_in"][b] - pc[c["job"]]["statement"] > 3 / pc[c["job"]]["n"] else "at")
                              for c in cs if c["verdict"] == lab)) for lab in ("accept", "reject")}}
        e["lock_in"] = lk
        e["statement_fraction"] = {lab: round(float(np.median([pc[c["job"]]["statement"] for c in cs
                                                               if c["verdict"] == lab and pc[c["job"]]["statement"] is not None])), 3)
                                   for lab in ("accept", "reject")}
        out[g] = e

    # reverted calls: case studies
    rev = [c for c in calls if c["kind"] == "call" and c["type"] == "reverted" and c["job"] in data]
    out["reverted"] = [{"job": c["job"], "verdict": c["verdict"], "c_p": round(float(sigmoid(pc[c["job"]]["c_logit"])), 3),
                        "first_p": round(float(sigmoid(pc[c["job"]]["first_logit"])), 3),
                        "final_smoothed_p": round(pc[c["job"]]["final_p"], 3),
                        "lock_in_0.1": round(pc[c["job"]]["lock_in"][0.1], 3), "statement": pc[c["job"]]["statement"]}
                       for c in rev]

    # step 5 edits: score at C, edited vs original
    orig = {(c["run"], c["agent"]): pc[c["job"]]["c_logit"] for c in calls
            if c["kind"] == "call" and c["type"] == "onset" and c["path"] == "agreement" and c["job"] in pc}
    for v in ("A", "B"):
        rows = [(c, float(data[c["job"]]["probe_logits"][0])) for c in calls if c["kind"] == f"edit{v}" and c["job"] in data]
        d = [lg - orig[(c["run"], c["agent"])] for c, lg in rows if (c["run"], c["agent"]) in orig]
        out[f"edit_{v}"] = {"n": len(rows), "median_p_edited": round(float(np.median([sigmoid(lg) for _, lg in rows])), 3),
                            "median_p_original": round(float(np.median([sigmoid(orig[(c['run'], c['agent'])]) for c, _ in rows])), 3),
                            "median_logit_change": round(float(np.median(d)), 3),
                            "n_below_half_edited": int(sum(sigmoid(lg) < 0.5 for _, lg in rows)),
                            "n_below_half_original": int(sum(sigmoid(orig[(c['run'], c['agent'])]) < 0.5 for c, _ in rows)),
                            "rows": [{"run": c["run"], "agent": c["agent"], "p_edited": round(float(sigmoid(lg)), 3),
                                      "p_original": round(float(sigmoid(orig[(c['run'], c['agent'])])), 3)} for c, lg in rows]}

    (RES / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    plots(main_calls, rev, pc, out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("reverted",) and not k.startswith("edit_")}, indent=1, default=float)[:6000])


def plots(main_calls, rev, pc, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = np.linspace(0, 1, 101)

    def resample(sp):
        return np.interp(grid, np.linspace(0, 1, len(sp)), sp)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.5), sharey=True)
    for ax, (g, sel) in zip(axes, (("pooled", lambda c: True), ("alice (speaks first)", lambda c: c["agent"] == "alice"),
                                   ("bob (speaks second)", lambda c: c["agent"] == "bob"))):
        for (t, v), col in ((("onset", "accept"), "C3"), (("before_onset", "accept"), "C1"), (("before_onset", "reject"), "C0")):
            cs = [c for c in main_calls if sel(c) and c["type"] == t and c["verdict"] == v]
            if not cs:
                continue
            curves = np.stack([resample(pc[c["job"]]["smoothed"]) for c in cs])
            ax.plot(grid, np.median(curves, 0), color=col, lw=2, label=f"{t} → {v} (n={len(cs)})")
            ax.fill_between(grid, np.percentile(curves, 25, 0), np.percentile(curves, 75, 0), color=col, alpha=0.15)
        ax.set_title(g)
        ax.set_xlabel("position in the thinking (fraction)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("C probe P(accept), smoothed (median, IQR)")
    fig.suptitle("C probe through the verdict call's own thinking")
    fig.tight_layout()
    fig.savefig(FIG / "step6_thinking_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
    for ax, g in zip(axes, ("pooled", "alice (speaks first)", "bob (speaks second)")):
        e = out[g]
        xs = np.arange(1, 11)
        ax.plot(xs, e["c_probe_auc_by_tenth"], marker="o", label="C probe (mean logit in tenth)")
        m = [f["mean"] for f in e["fresh_probe_auc_by_tenth"]]
        sd = [f["sd"] for f in e["fresh_probe_auc_by_tenth"]]
        ax.errorbar(xs, m, yerr=sd, marker="s", capsize=3, label="fresh probe per tenth (fold mean ± SD)")
        ax.axhline(e["auc_c_logit"], color="grey", ls="--", label=f"C probe at C ({e['auc_c_logit']:.3f})")
        ax.set_title(f"{g} (n={e['n']})")
        ax.set_xlabel("tenth of the thinking")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("AUC, ACCEPT vs REJECT calls")
    fig.tight_layout()
    fig.savefig(FIG / "step6_auc_by_tenth.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
