"""Steps 4-5 follow-up (Mac): do the feedback and word baselines predict the resampled shares as
well as the C probe does?

    tools/.venv/bin/python tools/step4_5/baselines_vs_resampling.py

Same calls as the probe's correlation (the session's 102 calls with a full resample from C).
Every predictor is out of fold on the same 5 trajectory folds as step 6's C probe: trained on
onset and before-onset rows of the other folds, applied to every call (reverted included)
whose trajectory is in the held-out fold.
  probe        step 6's C-probe P(accept) at C (layer 47)
  feedback     previous episode's outcome (steps 2-3 feedback features), L2 from inner folds
  words_own    word counts of the agent's own messages in the episode
  words_all    word counts of all the episode's messages
Reported: Spearman with the resampled ACCEPT share, and the share of calls whose majority
(share > 0.5) matches the predictor (P > 0.5), random group and hard cases separately.

Writes agent-collusion/results/step4_5/baselines_vs_resampling.json.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step2"))
sys.path.insert(0, str(ROOT / "tools" / "step3"))
import probe_lib as pl  # noqa: E402
import run_step2 as rs  # noqa: E402
import run_step3 as s3  # noqa: E402
import text_baselines as tb  # noqa: E402

RES = ROOT / "agent-collusion" / "results"


def main() -> None:
    from scipy.stats import spearmanr
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = s3.build_rows(index, episodes)
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    splits = pl.make_splits([r["run"] for r in probe], repeats=5)[:5]  # step 6's probe folds
    fold_of_run = {probe[j]["run"]: k for k, sp in enumerate(splits) for j in sp.test}
    units = {(u["run"], u["agent"], u["episode"]): u for u in rs.units(index, episodes, "A")}
    msgs = tb.episode_messages()

    def features(keys):
        us = [units[k] for k in keys]
        fb = rs.baseline_features(us)["feedback"]
        own = [" ".join(t for s, t in msgs[(k[0], k[2])] if s == k[1]) for k in keys]
        allm = [" ".join(t for _, t in msgs[(k[0], k[2])]) for k in keys]
        return fb, own, allm

    analysis = json.loads((RES / "step4_5" / "analysis.json").read_text())
    calls = [c for c in analysis["calls"] if c["A"] and c["A"]["share"] is not None]
    keys = [(c["job"].split(".")[1], c["job"].split(".")[2], int(c["job"].split(".ep")[1])) for c in calls]
    train_keys = [(r["run"], r["agent"], r["episode"]) for r in probe]
    y_train = np.array([r["y"] for r in probe])
    fb_tr, own_tr, all_tr = features(train_keys)
    fb_te, own_te, all_te = features(keys)
    preds = {"feedback": np.full(len(calls), np.nan), "words_own": np.full(len(calls), np.nan),
             "words_all": np.full(len(calls), np.nan)}
    for k, sp in enumerate(splits):
        test_idx = [i for i, kk in enumerate(keys) if fold_of_run[kk[0]] == k]
        if not test_idx:
            continue
        tr = sp.train
        # inner splits re-indexed for a one-split "evaluate"-style fit
        single = [pl.Split(np.arange(len(tr)), np.arange(len(test_idx)) + len(tr),
                           [(a, b) for a, b in sp.inner])]
        # feedback: standardise on train, L2 from inner folds
        X = np.vstack([fb_tr[tr], fb_te[test_idx]])
        Y = np.concatenate([y_train[tr], np.zeros(len(test_idx), int)])
        cache = pl.design_cache(X, single, reduce=False)
        ztr, zte, inner = cache[0]
        best = (-np.inf, None)
        for c in pl.C_GRID:
            pp = np.concatenate([pl.fit_predict(a, Y[:len(tr)][ia], b, c)[0] for (ia, ib), (a, b) in zip(sp.inner, inner)])
            yy = np.concatenate([Y[:len(tr)][ib] for ia, ib in sp.inner])
            sc = pl.auc(yy, pp)
            if sc > best[0]:
                best = (sc, c)
        preds["feedback"][test_idx] = pl.fit_predict(ztr, Y[:len(tr)], zte, best[1])[0]
        for name, tr_txt, te_txt in (("words_own", own_tr, own_te), ("words_all", all_tr, all_te)):
            texts = [tr_txt[j] for j in tr] + [te_txt[i] for i in test_idx]
            pieces = tb.text_cv(texts, Y, single)
            preds[name][test_idx] = pieces[0][1]
    share = np.array([c["A"]["share"] for c in calls])
    preds["probe"] = np.array([c["p_C"] for c in calls])
    groups = {"random": np.array([c["groups"][0].startswith("random") for c in calls])}
    groups["hard"] = ~groups["random"]
    for g in ("reverted_reject", "flip", "confident_wrong"):
        groups[g] = np.array([g in c["groups"] for c in calls])
    out = {"n_calls": len(calls), "per_group": {}}
    for g, m in groups.items():
        e = {"n": int(m.sum())}
        for name, p in preds.items():
            ok = m & ~np.isnan(p)
            e[name] = {"spearman": round(float(spearmanr(p[ok], share[ok]).statistic), 3),
                       "majority_agree": f"{int(((p[ok] > 0.5) == (share[ok] > 0.5)).sum())}/{int(ok.sum())}"}
        out["per_group"][g] = e
        print(g, e, flush=True)
    out["calls"] = [{"job": c["job"], "groups": c["groups"], "share_A": c["A"]["share"],
                     **{n: None if np.isnan(p[i]) else round(float(p[i]), 4) for n, p in preds.items()}}
                    for i, c in enumerate(calls)]
    (RES / "step4_5" / "baselines_vs_resampling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
