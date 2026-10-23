"""Step 2 follow-ups (Mac): does the probe add anything to the feedback features, and the
same probe at A_call.

    tools/.venv/bin/python tools/step2/followups.py [--nulls-a-call 20]

Rows are sorted by (run, agent, episode) so A and A_call use identical splits and can be
compared fold by fold.

Added value, per outer split, using training data only:
  1. layer and L2 strength: the inner-CV choice (as in the selected-layer probe)
  2. probe scores for the training rows: out-of-fold over the inner splits (logit)
  3. stacked model: logistic regression (C=1) on [probe logit, standardised feedback features]
  4. compared with the same model on the feedback features alone (C=1)
Test rows get the probe trained on the whole outer training set.

Writes agent-collusion/results/step2/followups.json.
"""

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

import probe_lib as pl
import run_step2 as rs

OUT = rs.OUT


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stacked(us, y, splits, per_layer_real, feats, layers):
    """Per outer split: AUC of [probe logit + feedback] and of feedback alone."""
    rows = np.array([u["i"] for u in us])
    chosen = [max(layers, key=lambda L: per_layer_real[L][k]["inner_auc"]) for k in range(len(splits))]
    caches = {}
    for L in sorted(set(chosen)):
        caches[L] = pl.design_cache(pl.rows_to_float(pl.load_layer(L), rows), splits, reduce=True)
    out = []
    for k, sp in enumerate(splits):
        L = chosen[k]
        c = per_layer_real[L][k]["c"]
        ztr, zte, inner = caches[L][k]
        ytr = y[sp.train]
        oof = np.zeros(len(sp.train))
        for (itr, ite), (iztr, izte) in zip(sp.inner, inner):
            p, _ = pl.fit_predict(iztr, ytr[itr], izte, c, np.array([0, 1]))
            oof[ite] = logit(p)
        p_test, _ = pl.fit_predict(ztr, ytr, zte, c, np.array([0, 1]))
        ftr, fte = pl.standardise(feats[sp.train], feats[sp.test])
        both_tr = np.column_stack([oof, ftr])
        both_te = np.column_stack([logit(p_test), fte])
        m_both = LogisticRegression(C=1.0, max_iter=2000).fit(both_tr, ytr)
        m_fb = LogisticRegression(C=1.0, max_iter=2000).fit(ftr, ytr)
        yte = y[sp.test]
        out.append({"layer": L, "c": c,
                    "probe_auc": pl.auc(yte, p_test), "feedback_auc": pl.auc(yte, m_fb.predict_proba(fte)[:, 1]),
                    "both_auc": pl.auc(yte, m_both.predict_proba(both_te)[:, 1]),
                    "probe_coef": float(m_both.coef_[0, 0])})
    return out


def fold_stats(v):
    v = np.asarray(v, float)
    return {"mean": round(float(np.nanmean(v)), 4), "sd": round(float(np.nanstd(v)), 4),
            "min": round(float(np.nanmin(v)), 4), "max": round(float(np.nanmax(v)), 4)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nulls-a-call", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    layers = pl.LAYERS
    index, episodes = pl.load_index(), pl.load_episodes()
    results, real_by_pos, splits = {}, {}, None
    for pos, n_null in (("A", 0), ("A_call", args.nulls_a_call)):
        us = sorted((u for u in rs.units(index, episodes, pos_type=pos) if u["type"] in ("onset", "before_onset")),
                    key=lambda u: (u["run"], u["agent"], u["episode"]))
        keys = [(u["run"], u["agent"], u["episode"]) for u in us]
        if splits is None:
            first_keys = keys
            splits = pl.make_splits([u["run"] for u in us], repeats=args.repeats)
        elif keys != first_keys:
            raise ValueError(f"{pos} rows do not line up with A rows")
        y = np.array([u["y"] for u in us])
        rng = np.random.default_rng(0)
        labels = [y] + [rng.permutation(y) for _ in range(n_null)]
        per_layer = rs.run_probe(us, labels, splits, layers, args.jobs, f"probe_{pos}")
        rep, sel = rs.report(per_layer, n_null)
        feats = rs.baseline_features(us)["feedback"]
        st = stacked(us, y, splits, {L: per_layer[L][0] for L in layers}, feats, layers)
        d = np.array([s["both_auc"] - s["feedback_auc"] for s in st])
        rep["added_value"] = {
            "probe_only": fold_stats([s["probe_auc"] for s in st]),
            "feedback_only_C1": fold_stats([s["feedback_auc"] for s in st]),
            "probe_plus_feedback": fold_stats([s["both_auc"] for s in st]),
            "gain_over_feedback": {**fold_stats(d), "folds_better": int((d > 0).sum()), "n_folds": len(d)},
            "probe_coef_in_stack": fold_stats([s["probe_coef"] for s in st]),
            "layers_chosen": {int(L): sum(s["layer"] == L for s in st) for L in sorted({s["layer"] for s in st})},
        }
        print(pos, json.dumps(rep["added_value"]), flush=True)
        results[pos] = rep
        real_by_pos[pos] = sel
        (OUT / "followups.json").write_text(json.dumps(results, indent=1))
    d = np.array([a["test_auc"] - b["test_auc"] for a, b in zip(real_by_pos["A_call"], real_by_pos["A"])])
    results["A_call_minus_A_selected"] = {**fold_stats(d), "folds_better": int((d > 0).sum()), "n_folds": len(d)}
    (OUT / "followups.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results["A_call_minus_A_selected"]))


if __name__ == "__main__":
    main()
