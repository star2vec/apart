"""Step 3 (Mac): probe scores traced through the communication phase, the probe at C, A<->C
transfer, and the speaking-order split at A_call.

    tools/.venv/bin/python tools/step3/run_step3.py [--jobs 8] [--c-nulls 20] [--boot 2000]

Rows: one per agent and episode. Probe set = onset and before-onset episodes (label: the
agent's own verdict, accept = 1); sustained episodes are scored as a ceiling, never trained on.

Positions per row (step 1 cache; communication pass unless noted):
  A        end of the communication-phase intro
  A_call   opening of the agent's own first communication turn
  s1..s10  end of each message slot, s = 2*(round-1) + order in round; Alice always speaks
           first, so odd slots are Alice's messages and even slots Bob's (a runner notice
           stands in for an undelivered peer message; an own send that never went through
           leaves the slot empty)
  C        the verdict turn's opening "\\n", verdict pass

For each source position P and layer, grouped nested CV as in step 2 (5 folds x 5 repeats by
trajectory; L2 strength from inner folds) trains a probe at P and scores the test rows at
every position Q (Q = P: the position's own probe; Q != P: transfer) and the test
trajectories' sustained rows. The layer per (P, outer split) is the one with the best inner
AUC. Every row then has an out-of-fold score at every position (mean over the 5 repeats) and
an out-of-fold feedback-baseline probability (step 2's feedback features).

"Beats the feedback baseline" at a position: the trajectory bootstrap (resampling the 50
runs) 95% interval of AUC(probe) - AUC(feedback), on the out-of-fold scores, lies above 0.

Writes agent-collusion/results/step3/{results.json, scores.jsonl} and figures in
reports/figures/step3/.
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step2"))
import probe_lib as pl  # noqa: E402
import run_step2 as rs  # noqa: E402

OUT = pl.ROOT / "agent-collusion" / "results" / "step3"
FIG = pl.ROOT / "reports" / "figures" / "step3"
SLOTS = [f"s{k}" for k in range(1, 11)]
POSITIONS = ["A", "A_call"] + SLOTS + ["C"]
CURVE = ["A"] + SLOTS + ["C"]
LABEL = {"A": "A", "A_call": "A_call", "C": "C",
         **{f"s{2 * (r - 1) + 1}": f"R{r} Alice" for r in range(1, 6)},
         **{f"s{2 * r}": f"R{r} Bob" for r in range(1, 6)}}


# ----------------------------------------------------------------------------- rows

def build_rows(index, episodes):
    pos = collections.defaultdict(dict)
    events = collections.defaultdict(list)
    for i, r in enumerate(index):
        key = (r["run"], r["agent"], r["episode"])
        if r["pass"] == "communication":
            if r["pos_type"] in ("A", "A_call"):
                pos[key][r["pos_type"]] = i
            elif r["pos_type"] in ("msg_end", "runner_notice"):
                events[key].append((i, r))
        elif r["pass"] == "verdict" and r["pos_type"] == "C":
            pos[key]["C"] = i
    units = {(u["run"], u["agent"], u["episode"]): u for u in rs.units(index, episodes, "A")}
    rows = []
    for key in sorted(pos):
        run, agent, ep = key
        e = episodes[(run, ep)]
        if e["episode_type"] not in ("onset", "before_onset", "sustained"):
            continue
        a_call = index[pos[key]["A_call"]]["token"]
        peer_first = any((r["pos_type"] == "runner_notice" or not r.get("own")) and r["token"] < a_call
                         for _, r in events[key])
        agree_slot = None
        for i, r in events[key]:
            own = r["pos_type"] == "msg_end" and r["own"]
            order = (2 if own else 1) if peer_first else (1 if own else 2)
            slot = f"s{2 * (r['round'] - 1) + order}"
            pos[key][slot] = i
            if r.get("is_agreement_msg"):
                agree_slot = slot
        onset_ep = next(x for x in range(1, 11) if episodes[(run, x)]["episode_type"] == "onset")
        rows.append({"run": run, "agent": agent, "episode": ep, "type": e["episode_type"],
                     "y": int(e[agent]["verdict"] == "accept"), "agreement": e["agreement"],
                     "path": "agreement" if episodes[(run, onset_ep)]["agreement"] else "no_agreement",
                     "speaks_first": not peer_first, "agree_slot": agree_slot, "pos": pos[key],
                     "unit": units[key]})
    return rows


# ------------------------------------------------------------------------------ CV

class Basis:
    def __init__(self, x):
        self.mu = x.mean(0)
        self.sd = x.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        z = ((x - self.mu) / self.sd).astype(np.float64)
        u, s, vt = np.linalg.svd(z, full_matrices=False)
        keep = s > s[0] * 1e-6
        self.vt = vt[keep]
        self.z = u[:, keep] * s[keep]

    def __call__(self, x):
        return ((x - self.mu) / self.sd).astype(np.float64) @ self.vt.T


def fit(z, y, c):
    return LogisticRegression(C=c, max_iter=1000, tol=1e-4).fit(z, y)


def source_task(L, P, X, rowpos, probe, sust, y, splits, c_grid):
    """Train at position P (layer L) in every outer split; score test rows at every position.
    X: gathered vectors; rowpos[j][Q] -> row of X for probe row j (or -1); sust likewise."""
    has = np.array([rowpos[j].get(P, -1) >= 0 for j in range(len(probe))])
    out = []
    for sp in splits:
        tr = sp.train[has[sp.train]]
        loc = {j: k for k, j in enumerate(tr)}
        inner = []
        for itr, ite in sp.inner:
            a = np.array([loc[j] for j in sp.train[itr] if j in loc])
            b = np.array([loc[j] for j in sp.train[ite] if j in loc])
            inner.append((a, b))
        xtr = X[[rowpos[j][P] for j in tr]]
        ytr = y[tr]
        bases = [(Basis(xtr[a]), a, b) for a, b in inner]
        zs = [(bs.z, bs(xtr[b]), a, b) for bs, a, b in bases]
        best = (-np.inf, None)
        for c in c_grid:
            pp, yy = [], []
            for za, zb, a, b in zs:
                if len(np.unique(ytr[a])) < 2:
                    continue
                pp.append(fit(za, ytr[a], c).predict_proba(zb)[:, 1])
                yy.append(ytr[b])
            score = pl.auc(np.concatenate(yy), np.concatenate(pp))
            if score > best[0]:
                best = (score, c)
        if best[1] is None:
            best = (np.nan, float(c_grid[len(c_grid) // 2]))
        basis = Basis(xtr)
        clf = fit(basis.z, ytr, best[1])
        test_runs = {probe[j]["run"] for j in sp.test}
        pred, spred = {}, {}
        for Q in POSITIONS:
            ids = np.array([j for j in sp.test if rowpos[j].get(Q, -1) >= 0])
            if len(ids):
                pred[Q] = (ids, clf.predict_proba(basis(X[[rowpos[j][Q] for j in ids]]))[:, 1])
            sids = np.array([k for k, s in enumerate(sust) if s["run"] in test_runs and s["xpos"].get(Q, -1) >= 0])
            if len(sids):
                spred[Q] = (sids, clf.predict_proba(basis(X[[sust[k]["xpos"][Q] for k in sids]]))[:, 1])
        out.append({"inner_auc": float(best[0]), "c": float(best[1]), "pred": pred, "sust": spred})
    return out


def feedback_cv(feats, y, splits):
    """Out-of-fold feedback-baseline probabilities per split (C from inner folds, as in step 2)."""
    out = []
    for sp in splits:
        ftr, fte = pl.standardise(feats[sp.train], feats[sp.test])
        best = (-np.inf, None)
        for c in pl.C_GRID:
            pp, yy = [], []
            for itr, ite in sp.inner:
                a, b = pl.standardise(feats[sp.train][itr], feats[sp.train][ite])
                pp.append(fit(a, y[sp.train][itr], c).predict_proba(b)[:, 1])
                yy.append(y[sp.train][ite])
            score = pl.auc(np.concatenate(yy), np.concatenate(pp))
            if score > best[0]:
                best = (score, c)
        out.append((sp.test, fit(ftr, y[sp.train], best[1]).predict_proba(fte)[:, 1]))
    return out


# ------------------------------------------------------------------------ analysis

def oof_mean(n, pieces):
    acc, cnt = np.zeros(n), np.zeros(n)
    for ids, p in pieces:
        acc[ids] += p
        cnt[ids] += 1
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)


def rank_oof(n, pieces):
    """Within-fold percentile ranks, averaged over repeats: removes fold-to-fold differences in
    score scale (each fold picks its own L2 strength) before pooling rows across folds."""
    from scipy.stats import rankdata
    return oof_mean(n, [(ids, (rankdata(p) - 0.5) / len(p)) for ids, p in pieces])


def fold_aucs(y, pieces):
    return [pl.auc(y[ids], p) for ids, p in pieces]


def stats(v):
    v = np.asarray(v, float)
    return {"mean": round(float(np.nanmean(v)), 4), "sd": round(float(np.nanstd(v)), 4),
            "min": round(float(np.nanmin(v)), 4), "max": round(float(np.nanmax(v)), 4)}


def boot_diff(y, s, f, runs, mask, n_boot, seed=0):
    m = mask & ~np.isnan(s) & ~np.isnan(f)
    runs_m = np.array(runs)[m]
    y, s, f = y[m], s[m], f[m]
    uniq = np.unique(runs_m)
    by = {r: np.where(runs_m == r)[0] for r in uniq}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([by[r] for r in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(pl.auc(y[idx], s[idx]) - pl.auc(y[idx], f[idx]))
    d = np.array(diffs)
    return {"n_rows": int(m.sum()), "n_runs": int(len(uniq)), "auc_probe": round(float(pl.auc(y, s)), 4),
            "auc_feedback": round(float(pl.auc(y, f)), 4), "diff": round(float(pl.auc(y, s) - pl.auc(y, f)), 4),
            "ci95": [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)],
            "beats": bool(np.percentile(d, 2.5) > 0)}


def jump(curve, agree_idx):
    """First position above 0.5, and the largest one-step rise, relative to the agreement position."""
    cross = next((k for k, v in enumerate(curve) if v > 0.5), None)
    steps = np.diff(curve)
    rise = int(np.nanargmax(steps)) + 1
    rel = lambda k: None if k is None else ("earlier" if k < agree_idx else "at" if k == agree_idx else "later")  # noqa: E731
    return {"first_above_half": None if cross is None else CURVE[cross], "first_above_half_vs_agreement": rel(cross)
            if cross is not None else "never", "largest_rise_at": CURVE[rise], "largest_rise_vs_agreement": rel(rise),
            "largest_rise": round(float(steps[rise - 1]), 4)}


def compare_c_last(probe, y, runs, sel, own_oof, masks, n_split, n_boot, last="s10"):
    """Last message position vs C: C sees the same conversation plus the verdict prompt
    (and, for the agent who sent the last message, the send tool's result turn)."""
    both = ~np.isnan(own_oof[last]) & ~np.isnan(own_oof["C"])

    def restricted(pieces):
        return [(ids[both[ids]], p[both[ids]]) for ids, p in pieces]

    r_c = rank_oof(len(probe), restricted([s["pred"]["C"] for s in sel["C"]]))
    r_last = rank_oof(len(probe), restricted([s["pred"][last] for s in sel[last]]))
    r_c_at_last = rank_oof(len(probe), restricted([s["pred"][last] for s in sel["C"] if last in s["pred"]]))
    fold_c = np.array(fold_aucs(y, restricted([s["pred"]["C"] for s in sel["C"]])))
    fold_last = np.array(fold_aucs(y, restricted([s["pred"][last] for s in sel[last]])))
    speakers = {"alice (s10 = Bob's message)": np.array([r["agent"] == "alice" for r in probe]),
                "bob (s10 = his own message)": np.array([r["agent"] == "bob" for r in probe])}
    return {
        "last_position": last, "n_rows_with_both": int(both.sum()),
        "note": "boot_diff fields: auc_probe = C, auc_feedback = the other score",
        "own_probes_C_minus_last": {k: boot_diff(y, r_c, r_last, runs, m & both, n_boot)
                                    for k, m in {**masks, **speakers}.items()},
        "fold_auc_C": stats(fold_c), "fold_auc_last": stats(fold_last),
        "paired_folds_C_minus_last": {"mean_diff": round(float(np.nanmean(fold_c - fold_last)), 4),
                                      "folds_better": int((fold_c - fold_last > 0).sum()), "n_folds": n_split},
        "C_probe_at_C_minus_at_last": {k: boot_diff(y, r_c, r_c_at_last, runs, m & both, n_boot)
                                       for k, m in masks.items()},
    }


# --------------------------------------------------------------------------- plots

def plots(probe, own_oof, a_oof, fb_oof, sust, sust_oof, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(CURVE))
    xt = [LABEL[q] for q in CURVE]
    onset = [j for j, r in enumerate(probe) if r["type"] == "onset"]
    groups = {"agreement onsets": [j for j in onset if probe[j]["path"] == "agreement"],
              "no-agreement onsets": [j for j in onset if probe[j]["path"] == "no_agreement"]}

    def carry(v):
        v = v.copy()
        for k in range(1, len(v)):
            if np.isnan(v[k]):
                v[k] = v[k - 1]
        return v

    for name, oof in (("own", own_oof), ("A_probe", a_oof)):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        for ax, (title, js) in zip(axes, groups.items()):
            cmap = plt.get_cmap("tab20")
            for n, j in enumerate(js):
                col = cmap(n % 20)
                curve = carry(np.array([oof[q][j] for q in CURVE]))
                ax.plot(xs, curve, color=col, lw=1, alpha=0.8)
                ax.hlines(fb_oof[j], xs[0], xs[-1], color=col, lw=0.8, ls="--", alpha=0.5)
                if probe[j]["agree_slot"]:
                    k = CURVE.index(probe[j]["agree_slot"])
                    ax.plot(xs[k], curve[k], "o", color=col, ms=6, mec="black", mew=0.6)
            ax.axhline(0.5, color="grey", lw=0.5)
            ax.set_title(f"{title} (n={len(js)} agent-episodes; all verdicts accept)")
            ax.set_xticks(xs, xt, rotation=60, ha="right", fontsize=8)
            ax.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("P(accept), out-of-fold" + (" (each position's own probe)" if name == "own" else " (A probe)"))
        fig.suptitle("Probe score through the episode; dashed = the row's feedback-baseline probability; "
                     "dot = agreement message")
        fig.tight_layout()
        fig.savefig(FIG / f"step3_curves_{name}.png", dpi=130)
        plt.close(fig)

    # small multiples for agreement onsets
    js = groups["agreement onsets"]
    cols = 6
    fig, axes = plt.subplots(int(np.ceil(len(js) / cols)), cols, figsize=(18, 2.4 * np.ceil(len(js) / cols)),
                             sharex=True, sharey=True)
    for ax, j in zip(axes.flat, js):
        curve = carry(np.array([own_oof[q][j] for q in CURVE]))
        acurve = carry(np.array([a_oof[q][j] for q in CURVE]))
        ax.plot(xs, curve, color="C0", lw=1.4)
        ax.plot(xs, acurve, color="C2", lw=1, ls=":")
        ax.axhline(fb_oof[j], color="C1", ls="--", lw=1)
        if probe[j]["agree_slot"]:
            ax.axvline(CURVE.index(probe[j]["agree_slot"]), color="red", lw=1)
        ax.set_title(f"{probe[j]['run']} {probe[j]['agent']} ep{probe[j]['episode']}", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
    for ax in axes.flat[len(js):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xticks(xs, xt, rotation=90, fontsize=6)
    fig.suptitle("Agreement onsets: own-position probe (blue), A probe (green dotted), "
                 "feedback baseline (orange dashed), agreement message (red)")
    fig.tight_layout()
    fig.savefig(FIG / "step3_agreement_onsets_grid.png", dpi=120)
    plt.close(fig)

    # AUC per position vs flat feedback AUC, overall and by path; group medians
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    for key, col, lab in (("overall", "black", "all"), ("agreement", "C3", "agreement-onset trajectories"),
                          ("no_agreement", "C0", "no-agreement-onset trajectories")):
        a = [summary["per_position"][q]["bootstrap"][key]["auc_probe"] for q in CURVE]
        f = summary["per_position"]["A"]["bootstrap"][key]["auc_feedback"]
        ax.plot(xs, a, color=col, marker="o", ms=3, label=f"probe, {lab}")
        ax.axhline(f, color=col, ls="--", lw=1, label=f"feedback, {lab}")
    m = [summary["per_position"][q]["fold_auc"]["mean"] for q in CURVE]
    sd = [summary["per_position"][q]["fold_auc"]["sd"] for q in CURVE]
    ax.fill_between(xs, np.array(m) - sd, np.array(m) + sd, color="grey", alpha=0.2, label="fold mean ± SD (all)")
    ax.set_xticks(xs, xt, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("AUC, own verdict (out-of-fold)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("AUC at each position vs the flat feedback baseline")
    ax = axes[1]
    for title, js_, col in (("before onset (floor)", [j for j, r in enumerate(probe) if r["type"] == "before_onset"], "grey"),
                            ("agreement onsets", groups["agreement onsets"], "C3"),
                            ("no-agreement onsets", groups["no-agreement onsets"], "C0")):
        med = [np.nanmedian([own_oof[q][j] for j in js_]) for q in CURVE]
        ax.plot(xs, med, color=col, marker="o", ms=3, label=f"{title}, n={len(js_)}")
    med = [np.nanmedian(sust_oof[q]) for q in CURVE]
    ax.plot(xs, med, color="C2", marker="o", ms=3, label=f"sustained (ceiling), n={len(sust)}")
    ax.set_xticks(xs, xt, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("median P(accept), own-position probe")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("Median probe score by episode type")
    fig.tight_layout()
    fig.savefig(FIG / "step3_auc_by_position.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--c-nulls", type=int, default=20)
    parser.add_argument("--boot", type=int, default=2000)
    parser.add_argument("--layers", default=None, help="comma-separated subset, for tests")
    parser.add_argument("--out", default=None, help="results dir (default results/step3)")
    parser.add_argument("--fig", default=None, help="figures dir (default reports/figures/step3)")
    parser.add_argument("--rows", choices=["all", "prev_reject"], default="all",
                        help="prev_reject: probe rows only where the agent's previous verdict was reject")
    args = parser.parse_args()
    global OUT, FIG
    if args.layers:
        pl.LAYERS = [int(x) for x in args.layers.split(",")]
    if args.out:
        OUT = Path(args.out)
        FIG = OUT / "figures"
    if args.fig:
        FIG = Path(args.fig)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = build_rows(index, episodes)
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    if args.rows == "prev_reject":
        probe = [r for r in probe if r["unit"]["prev"] is not None and not r["unit"]["prev"]["own_accept"]]
    sust = [r for r in rows if r["type"] == "sustained"]
    y = np.array([r["y"] for r in probe])
    runs = [r["run"] for r in probe]
    splits = pl.make_splits(runs, repeats=args.repeats)
    feats = rs.baseline_features([r["unit"] for r in probe])["feedback"]

    # gather every needed index row once
    needed = sorted({i for r in probe + sust for i in r["pos"].values()})
    where = {i: k for k, i in enumerate(needed)}
    rowpos = [{q: where[i] for q, i in r["pos"].items()} for r in probe]
    for s in sust:
        s["xpos"] = {q: where[i] for q, i in s["pos"].items()}
    sust_light = [{"run": s["run"], "xpos": s["xpos"]} for s in sust]
    print(f"{len(probe)} probe rows, {len(sust)} sustained rows, {len(needed)} vectors per layer", flush=True)

    res = {}
    for L in pl.LAYERS:
        t = time.time()
        X = pl.rows_to_float(pl.load_layer(L), np.array(needed))
        parts = Parallel(n_jobs=args.jobs)(
            delayed(source_task)(L, P, X, rowpos, [{"run": r["run"]} for r in probe], sust_light, y, splits, pl.C_GRID)
            for P in POSITIONS)
        res[L] = dict(zip(POSITIONS, parts))
        print(f"layer {L:2d}: " + " ".join(f"{P}={np.nanmean(fold_aucs(y, [s['pred'][P] for s in res[L][P]])):.3f}"
                                           for P in ("A", "s4", "s8", "s10", "C")) + f" ({time.time() - t:.0f}s)", flush=True)

    n_split = len(splits)
    chosen = {P: [max(pl.LAYERS, key=lambda L: res[L][P][k]["inner_auc"]) for k in range(n_split)] for P in POSITIONS}
    sel = {P: [res[chosen[P][k]][P][k] for k in range(n_split)] for P in POSITIONS}
    fb = feedback_cv(feats, y, splits)
    fb_oof = oof_mean(len(probe), fb)
    own_oof = {Q: oof_mean(len(probe), [s["pred"][Q] for s in sel[Q] if Q in s["pred"]]) for Q in POSITIONS}
    a_oof = {Q: oof_mean(len(probe), [s["pred"][Q] for s in sel["A"] if Q in s["pred"]]) for Q in POSITIONS}
    sust_oof = {Q: oof_mean(len(sust), [s["sust"][Q] for s in sel[Q] if Q in s["sust"]]) for Q in POSITIONS}

    masks = {"overall": np.ones(len(probe), bool),
             "agreement": np.array([r["path"] == "agreement" for r in probe]),
             "no_agreement": np.array([r["path"] == "no_agreement" for r in probe])}
    summary = {"settings": {"repeats": args.repeats, "layers": pl.LAYERS, "c_grid": pl.C_GRID.tolist(),
                            "boot": args.boot, "n_probe_rows": len(probe), "n_sustained_rows": len(sust),
                            "rows_by_path_type": dict(collections.Counter(f"{r['path']}:{r['type']}" for r in probe))},
               "feedback_fold_auc": stats(fold_aucs(y, fb)), "per_position": {},
               "scoring_note": "bootstrap AUCs use within-fold percentile ranks of out-of-fold scores, "
                               "averaged over repeats, for probe and feedback alike"}
    ranks = {}
    for Q in POSITIONS:
        has = ~np.isnan(own_oof[Q])
        fb_q = [(ids[has[ids]], p[has[ids]]) for ids, p in fb]
        entry = {
            "n_rows": int(has.sum()),
            "fold_auc": stats(fold_aucs(y, [s["pred"][Q] for s in sel[Q]])),
            "feedback_fold_auc_same_rows": stats(fold_aucs(y, fb_q)),
            "per_layer_fold_auc": {L: stats(fold_aucs(y, [res[L][Q][k]["pred"][Q] for k in range(n_split)]))["mean"]
                                   for L in pl.LAYERS},
            "layers_chosen": dict(collections.Counter(int(x) for x in chosen[Q])),
            "a_probe_here_fold_auc": stats(fold_aucs(y, [s["pred"][Q] for s in sel["A"] if Q in s["pred"]])),
        }
        own_r = rank_oof(len(probe), [s["pred"][Q] for s in sel[Q]])
        a_r = rank_oof(len(probe), [s["pred"][Q] for s in sel["A"] if Q in s["pred"]])
        fb_r = rank_oof(len(probe), fb_q)
        ranks[Q] = (own_r, a_r, fb_r)
        entry["bootstrap"] = {k: boot_diff(y, own_r, fb_r, runs, m, args.boot) for k, m in masks.items()}
        entry["bootstrap_a_probe"] = {k: boot_diff(y, a_r, fb_r, runs, m, args.boot) for k, m in masks.items()}
        fold_d = np.array(fold_aucs(y, [s["pred"][Q] for s in sel[Q]])) - np.array(fold_aucs(y, fb_q))
        entry["paired_folds_vs_feedback"] = {"mean_diff": round(float(np.nanmean(fold_d)), 4),
                                             "folds_better": int((fold_d > 0).sum()), "n_folds": n_split}
        summary["per_position"][Q] = entry
        b = entry["bootstrap"]
        print(f"{Q:7s} AUC {b['overall']['auc_probe']:.3f} vs fb {b['overall']['auc_feedback']:.3f} "
              f"CI {b['overall']['ci95']} | agr {b['agreement']['diff']:+.3f} {b['agreement']['ci95']} "
              f"| no-agr {b['no_agreement']['diff']:+.3f} {b['no_agreement']['ci95']}", flush=True)

    def first_beat(key, which="bootstrap"):
        return next((Q for Q in CURVE if summary["per_position"][Q][which][key]["beats"]), None)

    summary["first_beats_feedback"] = {k: first_beat(k) for k in masks}

    summary["C_vs_last_message"] = compare_c_last(probe, y, runs, sel, own_oof, masks, n_split, args.boot)
    summary["first_beats_feedback_a_probe"] = {k: first_beat(k, "bootstrap_a_probe") for k in masks}

    # A <-> C transfer: selected (source's layer from its training folds) and per layer
    transfer = {}
    for P, Q in (("A", "C"), ("C", "A"), ("A", "A"), ("C", "C")):
        transfer[f"{P}->{Q}"] = {
            "selected": stats(fold_aucs(y, [s["pred"][Q] for s in sel[P]])),
            "per_layer": {L: stats(fold_aucs(y, [res[L][P][k]["pred"][Q] for k in range(n_split)]))["mean"]
                          for L in pl.LAYERS}}
    summary["transfer_A_C"] = transfer

    # C probe with a shuffled null (step 2 pipeline, same splits)
    us_c = sorted((u for u in rs.units(index, episodes, "C", "verdict") if u["type"] in ("onset", "before_onset")
                   and (args.rows == "all" or (u["prev"] is not None and not u["prev"]["own_accept"]))),
                  key=lambda u: (u["run"], u["agent"], u["episode"]))
    assert [(u["run"], u["agent"], u["episode"]) for u in us_c] == [(r["run"], r["agent"], r["episode"]) for r in probe]
    rng = np.random.default_rng(0)
    labels = [y] + [rng.permutation(y) for _ in range(args.c_nulls)]
    c_layers = rs.run_probe(us_c, labels, splits, pl.LAYERS, args.jobs, "probe_C")
    c_rep, _ = rs.report(c_layers, args.c_nulls)
    summary["probe_C"] = c_rep

    # Agreement position vs the jump, agreement onsets
    jumps = []
    for j, r in enumerate(probe):
        if r["type"] == "onset" and r["path"] == "agreement" and r["agree_slot"]:
            k = CURVE.index(r["agree_slot"])
            def carry(v):
                v = v.copy()
                for i in range(1, len(v)):
                    if np.isnan(v[i]):
                        v[i] = v[i - 1]
                return v
            own_c = carry(np.array([own_oof[q][j] for q in CURVE]))
            a_c = carry(np.array([a_oof[q][j] for q in CURVE]))
            jumps.append({"run": r["run"], "agent": r["agent"], "episode": r["episode"], "agree_slot": r["agree_slot"],
                          "agree_label": LABEL[r["agree_slot"]], "feedback_p": round(float(fb_oof[j]), 3),
                          "own": jump(own_c, k), "a_probe": jump(a_c, k),
                          "own_curve": [round(float(v), 3) for v in own_c]})
    summary["agreement_jumps"] = {
        "n": len(jumps),
        "own_first_above_half": dict(collections.Counter(x["own"]["first_above_half_vs_agreement"] for x in jumps)),
        "own_largest_rise": dict(collections.Counter(x["own"]["largest_rise_vs_agreement"] for x in jumps)),
        "a_probe_first_above_half": dict(collections.Counter(x["a_probe"]["first_above_half_vs_agreement"] for x in jumps)),
        "a_probe_largest_rise": dict(collections.Counter(x["a_probe"]["largest_rise_vs_agreement"] for x in jumps)),
        "rows": jumps}

    # Speaking order at A_call (Alice always speaks first)
    so = {}
    for name, m in (("speaks_first", np.array([r["speaks_first"] for r in probe])),
                    ("speaks_second", ~np.array([r["speaks_first"] for r in probe]))):
        so[name] = {"agents": sorted({r["agent"] for r, mm in zip(probe, m) if mm}),
                    "A_call_vs_feedback": boot_diff(y, ranks["A_call"][0], ranks["A_call"][2], runs, m, args.boot),
                    "A_vs_feedback": boot_diff(y, ranks["A"][0], ranks["A"][2], runs, m, args.boot),
                    "A_call_vs_A": boot_diff(y, ranks["A_call"][0], ranks["A"][0], runs, m, args.boot)}
    summary["speaking_order_A_call"] = so

    (OUT / "results.json").write_text(json.dumps(summary, indent=1))
    with open(OUT / "scores.jsonl", "w") as f:
        for j, r in enumerate(probe):
            f.write(json.dumps({k: r[k] for k in ("run", "agent", "episode", "type", "y", "path", "agree_slot", "speaks_first")}
                               | {"feedback_p": None if np.isnan(fb_oof[j]) else round(float(fb_oof[j]), 4),
                                  "own": {q: None if np.isnan(own_oof[q][j]) else round(float(own_oof[q][j]), 4) for q in POSITIONS},
                                  "a_probe": {q: None if np.isnan(a_oof[q][j]) else round(float(a_oof[q][j]), 4) for q in POSITIONS}}) + "\n")
    plots(probe, own_oof, a_oof, fb_oof, sust, sust_oof, summary)
    print(json.dumps({k: summary[k] for k in ("first_beats_feedback", "first_beats_feedback_a_probe")}))
    print(json.dumps({k: v for k, v in summary["agreement_jumps"].items() if k != "rows"}))
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
