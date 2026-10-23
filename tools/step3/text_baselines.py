"""Step 3 addition (Mac): text baselines for the last-message comparison.

    tools/.venv/bin/python tools/step3/text_baselines.py [--boot 2000]

Same rows (onset and before-onset, with a last message slot s10), labels and trajectory folds
as run_step3.py. Next to each position's own probe at s10 and at C:
  words_own   word-count logistic regression on the agent's own messages in that episode
  words_all   the same on all delivered messages of the episode (what s10 and C have seen)
  agreement   the paper's agreement label for the episode (one binary feature)
Vocabulary is fitted on the training fold only; L2 strength from inner folds. Reported pooled
and for Alice (last message = Bob's) and Bob (last message = his own) separately.

Writes agent-collusion/results/step3/text_baselines.json.
"""

import argparse
import gzip
import json

import numpy as np
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import run_step3 as s3
from run_step3 import pl

TRACES = pl.ROOT / "agent-collusion" / "results" / "paper" / "traces_qwen36_main"
C_GRID_TEXT = np.logspace(-3, 2, 6)


def episode_messages():
    out = {}
    for f in sorted(TRACES.glob("rep*.json.gz")):
        run = json.load(gzip.open(f))
        for k, e in enumerate(run["results"]):
            out[(f.name.split(".")[0], k + 1)] = [(ev["sender"], ev["content"] or "") for ev in e["channel_transcript"]]
    return out


def text_cv(texts, y, splits, c_grid=C_GRID_TEXT, binary_feature=None):
    pieces = []
    for sp in splits:
        if binary_feature is not None:
            xtr, xte = binary_feature[sp.train], binary_feature[sp.test]
            inner_x = [(binary_feature[sp.train][a], binary_feature[sp.train][b]) for a, b in sp.inner]
        else:
            tr_texts = [texts[j] for j in sp.train]
            vec = CountVectorizer(lowercase=True, min_df=2, token_pattern=r"(?u)\b\w+\b").fit(tr_texts)
            xtr, xte = vec.transform(tr_texts), vec.transform([texts[j] for j in sp.test])
            inner_x = []
            for a, b in sp.inner:
                v = CountVectorizer(lowercase=True, min_df=2, token_pattern=r"(?u)\b\w+\b").fit([tr_texts[i] for i in a])
                inner_x.append((v.transform([tr_texts[i] for i in a]), v.transform([tr_texts[i] for i in b])))
        ytr = y[sp.train]
        best = (-np.inf, c_grid[len(c_grid) // 2])
        for c in c_grid:
            pp, yy = [], []
            for (a, b), (xa, xb) in zip(sp.inner, inner_x):
                pp.append(LogisticRegression(C=c, max_iter=3000).fit(xa, ytr[a]).predict_proba(xb)[:, 1])
                yy.append(ytr[b])
            score = pl.auc(np.concatenate(yy), np.concatenate(pp))
            if score > best[0]:
                best = (score, c)
        clf = LogisticRegression(C=best[1], max_iter=3000).fit(xtr, ytr)
        pieces.append((sp.test, clf.predict_proba(xte)[:, 1]))
    return pieces


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = s3.build_rows(index, episodes)
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    sust = [r for r in rows if r["type"] == "sustained"]
    y = np.array([r["y"] for r in probe])
    runs = [r["run"] for r in probe]
    splits = pl.make_splits(runs, repeats=5)
    msgs = episode_messages()
    own = [" ".join(t for s, t in msgs[(r["run"], r["episode"])] if s == r["agent"]) for r in probe]
    allm = [" ".join(t for _, t in msgs[(r["run"], r["episode"])]) for r in probe]
    agree = np.array([[1.0 if r["agreement"] else 0.0] for r in probe])

    # the two position probes, as in run_step3.py / c_vs_last.py
    needed = sorted({i for r in probe + sust for i in r["pos"].values()})
    where = {i: k for k, i in enumerate(needed)}
    rowpos = [{q: where[i] for q, i in r["pos"].items()} for r in probe]
    sust_light = [{"run": s["run"], "xpos": {q: where[i] for q, i in s["pos"].items()}} for s in sust]
    res = {}
    for L in pl.LAYERS:
        X = pl.rows_to_float(pl.load_layer(L), np.array(needed))
        chunks = np.array_split(np.arange(len(splits)), 4)
        parts = Parallel(n_jobs=8)(
            delayed(s3.source_task)(L, P, X, rowpos, [{"run": r["run"]} for r in probe], sust_light, y,
                                    [splits[k] for k in ch], pl.C_GRID) for P in ("s10", "C") for ch in chunks)
        res[L] = {P: [x for part in parts[i * 4:(i + 1) * 4] for x in part] for i, P in enumerate(("s10", "C"))}
        print(f"layer {L}", flush=True)
    n = len(splits)
    sel = {P: [res[max(pl.LAYERS, key=lambda L: res[L][P][k]["inner_auc"])][P][k] for k in range(n)] for P in ("s10", "C")}

    has = np.array(["s10" in r["pos"] for r in probe])

    def restrict(pieces):
        return [(ids[has[ids]], p[has[ids]]) for ids, p in pieces]

    scores = {"probe_s10": restrict([s["pred"]["s10"] for s in sel["s10"]]),
              "probe_C": restrict([s["pred"]["C"] for s in sel["C"]]),
              "words_own": restrict(text_cv(own, y, splits)),
              "words_all": restrict(text_cv(allm, y, splits)),
              "agreement_label": restrict(text_cv(None, y, splits, binary_feature=agree))}
    ranks = {k: s3.rank_oof(len(probe), v) for k, v in scores.items()}
    groups = {"pooled": has, "alice (last message = Bob's)": has & np.array([r["agent"] == "alice" for r in probe]),
              "bob (last message = his own)": has & np.array([r["agent"] == "bob" for r in probe])}
    out = {"n_rows": int(has.sum()), "per_group": {}}
    for g, m in groups.items():
        entry = {}
        for k in scores:
            fold = [pl.auc(y[ids[m[ids]]], p[m[ids]]) for ids, p in scores[k]]
            entry[k] = {"pooled_auc": round(float(pl.auc(y[m], ranks[k][m])), 4), "fold_auc": s3.stats(fold)}
        for k in ("words_own", "words_all", "agreement_label"):
            entry[f"probe_s10_minus_{k}"] = s3.boot_diff(y, ranks["probe_s10"], ranks[k], runs, m, args.boot)
            entry[f"probe_C_minus_{k}"] = s3.boot_diff(y, ranks["probe_C"], ranks[k], runs, m, args.boot)
        entry["corr_probe_s10_words_own"] = round(float(np.corrcoef(ranks["probe_s10"][m], ranks["words_own"][m])[0, 1]), 3)
        out["per_group"][g] = entry
        print(g, {k: entry[k]["pooled_auc"] for k in scores}, flush=True)
    (s3.OUT / "text_baselines.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
