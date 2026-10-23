"""Grouped, nested cross-validation for linear probes on the step 1 cache (Mac, CPU).

Outer splits: GroupKFold by trajectory (all episodes and both agents of a run on one side),
repeated with different shuffles. Inside each outer training set: GroupKFold again to pick
the L2 strength (and, for the selected-layer probe, the layer) from pooled out-of-fold AUC.
Test folds are used only for the reported score.

High-dimensional fits are exact in the training rows' SVD basis: with features standardised
on the training fold, an L2-penalised logistic regression has its weights in the row space,
so fitting on U*S (n_train x rank) gives the same predictions as fitting on all 5,120
features, much faster. Baseline feature sets are small and fit directly.
"""

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "agent-collusion" / "results" / "step1_cache"
LAYERS = list(range(3, 64, 4))
C_GRID = np.logspace(-5, 1, 7)

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_index():
    return [json.loads(l) for l in open(CACHE / "index.jsonl")]


def load_episodes():
    return {(e["run"].split("__")[-1], e["episode"]): e for e in map(json.loads, open(CACHE / "episodes.jsonl"))}


def load_layer(layer):
    bits = np.load(CACHE / "packed" / f"layer_{layer:02d}.npy", mmap_mode="r")
    return bits


def rows_to_float(bits, rows):
    return (np.asarray(bits[rows]).astype(np.uint32) << 16).view(np.float32)


@dataclass
class Split:
    train: np.ndarray
    test: np.ndarray
    inner: list  # [(train, test)] as indices into self.train


def make_splits(groups, n_outer=5, n_inner=4, repeats=5, seed=0):
    groups = np.asarray(groups)
    out = []
    for rep in range(repeats):
        outer = GroupKFold(n_splits=n_outer, shuffle=True, random_state=seed + rep)
        for k, (tr, te) in enumerate(outer.split(np.zeros(len(groups)), groups=groups)):
            inner = GroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1000 + 10 * rep + k)
            out.append(Split(tr, te, list(inner.split(np.zeros(len(tr)), groups=groups[tr]))))
    return out


def standardise(xtr, xte):
    mu = xtr.mean(0)
    sd = xtr.std(0)
    sd[sd < 1e-8] = 1.0
    return (xtr - mu) / sd, (xte - mu) / sd


def svd_basis(xtr, xte):
    """Standardise on train, then express both in the train rows' SVD basis."""
    ztr, zte = standardise(xtr, xte)
    u, s, vt = np.linalg.svd(ztr.astype(np.float64), full_matrices=False)
    keep = s > s[0] * 1e-6
    return u[:, keep] * s[keep], zte.astype(np.float64) @ vt[keep].T


def design_cache(x, splits, reduce=True):
    """Per outer split: (train Z, test Z, [(inner train Z, inner test Z)])."""
    f = svd_basis if reduce else (lambda a, b: standardise(a.astype(np.float64), b.astype(np.float64)))
    cache = []
    for sp in splits:
        xtr = x[sp.train]
        inner = [f(xtr[itr], xtr[ite]) for itr, ite in sp.inner]
        cache.append((*f(xtr, x[sp.test]), inner))
    return cache


def fit_predict(ztr, ytr, zte, c, classes=None):
    """Probabilities for zte: a vector for binary labels, else columns aligned to `classes`
    (a class absent from ytr gets probability 0)."""
    clf = LogisticRegression(C=c, max_iter=1000, tol=1e-4)
    clf.fit(ztr, ytr)
    p = clf.predict_proba(zte)
    classes = clf.classes_ if classes is None else np.asarray(classes)
    if len(classes) == 2:
        return p[:, list(clf.classes_).index(classes[1])], classes
    full = np.zeros((len(zte), len(classes)))
    full[:, np.searchsorted(classes, clf.classes_)] = p
    return full, classes


def auc(y, p, classes=None):
    if len(np.unique(y)) < 2:
        return np.nan
    if np.ndim(p) == 2 and len(np.unique(y)) < len(classes):  # a class absent from this test fold
        present = np.isin(classes, np.unique(y))
        q = p[:, present] / np.clip(p[:, present].sum(1, keepdims=True), 1e-12, None)
        return roc_auc_score(y, q, multi_class="ovr", average="macro", labels=classes[present]) if present.sum() > 2 \
            else roc_auc_score(y == classes[present][1], q[:, 1])
    if np.ndim(p) == 1:
        return roc_auc_score(y, p)
    return roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes)


def evaluate(cache, splits, y, c_grid=C_GRID):
    """For each outer split: inner-chosen C, its pooled inner AUC, test AUC and accuracy.
    y: labels for all rows (binary or multi-class)."""
    y = np.asarray(y)
    classes = np.unique(y)
    res = []
    for sp, (ztr, zte, inner) in zip(splits, cache):
        ytr = y[sp.train]
        best = (-np.inf, None)
        for c in c_grid:
            pooled_p, pooled_y = [], []
            for (itr, ite), (iztr, izte) in zip(sp.inner, inner):
                if len(np.unique(ytr[itr])) < 2:
                    continue
                p, _ = fit_predict(iztr, ytr[itr], izte, c, classes)
                pooled_p.append(p)
                pooled_y.append(ytr[ite])
            score = auc(np.concatenate(pooled_y), np.concatenate(pooled_p), classes) if pooled_y else np.nan
            if score > best[0]:
                best = (score, c)
        if best[1] is None:  # no inner fold had two classes
            best = (np.nan, float(c_grid[len(c_grid) // 2]))
        p, cls = fit_predict(ztr, ytr, zte, best[1], classes)
        yte = y[sp.test]
        pred = cls[p.argmax(1)] if np.ndim(p) == 2 else (p >= 0.5).astype(int)
        majority = np.bincount(np.searchsorted(classes, ytr)).argmax()
        res.append({"c": float(best[1]), "inner_auc": float(best[0]), "test_auc": float(auc(yte, p, classes)),
                    "test_acc": float((pred == yte).mean()),
                    "majority_acc": float((yte == classes[majority]).mean())})
    return res


def summarise(res, key="test_auc"):
    v = np.array([r[key] for r in res], dtype=float)
    return {"mean": round(float(np.nanmean(v)), 4), "sd": round(float(np.nanstd(v)), 4),
            "min": round(float(np.nanmin(v)), 4), "max": round(float(np.nanmax(v)), 4), "n_folds": int(len(v))}


def select_layer(per_layer):
    """per_layer: {layer: evaluate() result}. Per outer split, take the layer with the best
    inner AUC (training data only) and report its test scores."""
    layers = sorted(per_layer)
    n = len(per_layer[layers[0]])
    out = []
    for k in range(n):
        best = max(layers, key=lambda L: per_layer[L][k]["inner_auc"])
        out.append({**per_layer[best][k], "layer": best})
    return out
