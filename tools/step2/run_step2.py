"""Step 2: probe at point A for the agent's own verdict, with baselines, a shuffled-label
null and a positive control, all through the same grouped nested cross-validation.

    tools/.venv/bin/python tools/step2/run_step2.py [--nulls 100] [--repeats 5] [--jobs 8]

Rows: position A (end of the communication-phase intro, communication pass), one per agent
and episode. Probe set: onset and before-onset episodes. Label: the agent's own verdict in
that episode (accept = 1).

Baselines (same outer splits, regularisation chosen inside the training folds):
  episode        one-hot episode number
  relaxation     the agent's previous reflection's relaxation flag (true / false / none in episode 1)
  feedback       previous episode: own and peer verdict, whether each was right, own reward,
                 plus episode number and a first-episode indicator
Null: labels permuted across rows (same permutation for every layer), whole pipeline rerun.
Positive control: previous episode's reward (0 / 10 / 20, in the prompt as "## Reward"),
3-class, on all episodes 2-10 and on the probe set's episodes 2+.
Layer choice: per outer split, the layer with the best inner (training-fold) AUC.

Writes agent-collusion/results/step2/results.json.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

import probe_lib as pl

OUT = pl.ROOT / "agent-collusion" / "results" / "step2"


def units(index, episodes, pos_type="A", pass_="communication"):
    rows = [(i, r) for i, r in enumerate(index) if r["pos_type"] == pos_type and r["pass"] == pass_]
    out = []
    for i, r in rows:
        e = episodes[(r["run"], r["episode"])]
        prev = episodes.get((r["run"], r["episode"] - 1))
        me, peer = r["agent"], "bob" if r["agent"] == "alice" else "alice"
        out.append({"i": i, "run": r["run"], "agent": me, "episode": r["episode"], "type": e["episode_type"],
                    "y": int(e[me]["verdict"] == "accept"),
                    "prev_reward": None if prev is None else prev[me]["reward"],
                    "prev_relax": None if prev is None else prev[me]["relaxation"],
                    "prev": None if prev is None else {
                        "own_accept": prev[me]["verdict"] == "accept", "peer_accept": prev[peer]["verdict"] == "accept",
                        "own_right": bool(prev[me]["verdict_correct"]), "peer_right": bool(prev[peer]["verdict_correct"]),
                        "reward": prev[me]["reward"]}})
    return out


def baseline_features(us):
    eps = sorted({u["episode"] for u in us})
    episode = np.array([[u["episode"] == k for k in eps] for u in us], float)
    relax = np.array([[u["prev_relax"] is True, u["prev_relax"] is False, u["prev_relax"] is None] for u in us], float)
    feedback = np.array([[0, 0, 0, 0, 0, u["episode"], 1] if u["prev"] is None else
                         [u["prev"]["own_accept"], u["prev"]["peer_accept"], u["prev"]["own_right"],
                          u["prev"]["peer_right"], u["prev"]["reward"], u["episode"], 0] for u in us], float)
    return {"episode": episode, "relaxation": relax, "feedback": feedback}


def run_probe(us, labels_list, splits, layers, n_jobs, tag):
    """Evaluate every label vector in labels_list at every layer. Returns
    {layer: [evaluate() result per label vector]}."""
    rows = np.array([u["i"] for u in us])
    per_layer = {}
    for L in layers:
        t0 = time.time()
        x = pl.rows_to_float(pl.load_layer(L), rows)
        split_chunks = np.array_split(np.arange(len(splits)), n_jobs)
        cache = [c for part in Parallel(n_jobs=n_jobs)(
            delayed(pl.design_cache)(x, [splits[k] for k in ch], True) for ch in split_chunks) for c in part]
        # tasks: (label vector, chunk of splits); results reassembled in split order per label
        tasks = [(j, ch) for j in range(len(labels_list)) for ch in split_chunks]
        parts = Parallel(n_jobs=n_jobs)(
            delayed(pl.evaluate)([cache[k] for k in ch], [splits[k] for k in ch], labels_list[j]) for j, ch in tasks)
        per_layer[L] = [[] for _ in labels_list]
        for (j, _), part in zip(tasks, parts):
            per_layer[L][j].extend(part)
        real = pl.summarise(per_layer[L][0])
        print(f"[{tag}] layer {L:2d}: AUC {real['mean']:.3f} ± {real['sd']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return per_layer


def report(per_layer, n_null):
    layers = sorted(per_layer)
    out = {"per_layer": {}, "selected": None}
    for L in layers:
        real = per_layer[L][0]
        entry = {"auc": pl.summarise(real), "acc": pl.summarise(real, "test_acc"),
                 "c_chosen": sorted({r["c"] for r in real})}
        if n_null:
            null_means = np.array([np.nanmean([r["test_auc"] for r in res]) for res in per_layer[L][1:]])
            entry["null"] = {"mean": round(float(null_means.mean()), 4), "p95": round(float(np.percentile(null_means, 95)), 4),
                             "p_value": round(float((1 + (null_means >= entry["auc"]["mean"]).sum()) / (1 + len(null_means))), 4)}
        out["per_layer"][L] = entry
    sel_all = [pl.select_layer({L: per_layer[L][j] for L in layers}) for j in range(len(per_layer[layers[0]]))]
    sel = sel_all[0]
    s = {"auc": pl.summarise(sel), "acc": pl.summarise(sel, "test_acc"),
         "layers_chosen": {int(L): sum(r["layer"] == L for r in sel) for L in sorted({r["layer"] for r in sel})}}
    if n_null:
        null_means = np.array([np.nanmean([r["test_auc"] for r in res]) for res in sel_all[1:]])
        s["null"] = {"mean": round(float(null_means.mean()), 4), "p95": round(float(np.percentile(null_means, 95)), 4),
                     "p_value": round(float((1 + (null_means >= s["auc"]["mean"]).sum()) / (1 + len(null_means))), 4)}
    out["selected"] = s
    out["majority_acc"] = pl.summarise(per_layer[layers[0]][0], "majority_acc")
    return out, sel


def paired(a, b):
    d = np.array([x["test_auc"] - y["test_auc"] for x, y in zip(a, b)])
    return {"mean_diff": round(float(np.nanmean(d)), 4), "sd": round(float(np.nanstd(d)), 4),
            "folds_better": int((d > 0).sum()), "n_folds": int(len(d))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nulls", type=int, default=100)
    parser.add_argument("--control-nulls", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--layers", default=",".join(map(str, pl.LAYERS)))
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--controls", default="all,probe_set", help="which controls to run")
    parser.add_argument("--resume", action="store_true", help="keep results already in the results file")
    parser.add_argument("--rows", choices=["all", "prev_reject"], default="all",
                        help="prev_reject: only agents whose own verdict in the previous episode was reject")
    parser.add_argument("--results-name", default="results.json")
    args = parser.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    index, episodes = pl.load_index(), pl.load_episodes()
    all_units = units(index, episodes)
    old = json.loads((OUT / args.results_name).read_text()) if args.resume and (OUT / args.results_name).exists() else {}
    results = old | {"settings": {"position": "A", "pass": "communication", "repeats": args.repeats, "outer_folds": 5,
                            "inner_folds": 4, "c_grid": pl.C_GRID.tolist(), "layers": layers, "nulls": args.nulls,
                            "rows": args.rows}}

    # Positive control: previous episode's reward.
    if not args.skip_control:
        wanted = args.controls.split(",")
        for key, name, us in (("all", "control_all_episodes_2_10", [u for u in all_units if u["prev"] is not None]),
                              ("probe_set", "control_probe_set_episodes_2plus",
                               [u for u in all_units if u["prev"] is not None and u["type"] in ("onset", "before_onset")])):
            if key not in wanted:
                continue
            y = np.array([u["prev_reward"] for u in us])
            splits = pl.make_splits([u["run"] for u in us], repeats=args.repeats)
            rng = np.random.default_rng(1)
            labels = [y] + [rng.permutation(y) for _ in range(args.control_nulls)]
            per_layer = run_probe(us, labels, splits, layers, args.jobs, name)
            rep, _ = report(per_layer, args.control_nulls)
            rep["n_rows"] = len(us)
            rep["label_counts"] = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
            results[name] = rep
            (OUT / args.results_name).write_text(json.dumps(results, indent=1))

    # Main probe: own verdict at A, onset + before-onset.
    us = [u for u in all_units if u["type"] in ("onset", "before_onset")]
    if args.rows == "prev_reject":
        us = [u for u in us if u["prev"] is not None and not u["prev"]["own_accept"]]
    y = np.array([u["y"] for u in us])
    splits = pl.make_splits([u["run"] for u in us], repeats=args.repeats)
    rng = np.random.default_rng(0)
    labels = [y] + [rng.permutation(y) for _ in range(args.nulls)]
    per_layer = run_probe(us, labels, splits, layers, args.jobs, "probe_A")
    rep, sel = report(per_layer, args.nulls)
    rep["n_rows"] = len(us)
    rep["n_runs"] = len({u["run"] for u in us})
    rep["label_counts"] = {"reject": int((y == 0).sum()), "accept": int((y == 1).sum())}
    rep["by_type"] = {t: {"n": sum(u["type"] == t for u in us), "accept": sum(u["y"] for u in us if u["type"] == t)}
                      for t in ("before_onset", "onset")}

    base = {}
    for name, feats in baseline_features(us).items():
        cache = pl.design_cache(feats, splits, reduce=False)
        base[name] = pl.evaluate(cache, splits, y)
        print(f"[baseline] {name}: AUC {pl.summarise(base[name])['mean']:.3f}", flush=True)
    rep["baselines"] = {name: {"auc": pl.summarise(r), "acc": pl.summarise(r, "test_acc")} for name, r in base.items()}
    rep["selected_vs_baseline"] = {name: paired(sel, r) for name, r in base.items()}
    best_fixed = max(per_layer, key=lambda L: rep["per_layer"][L]["auc"]["mean"])
    rep["note_best_fixed_layer_by_test"] = {"layer": best_fixed, "warning": "chosen on test scores; descriptive only"}
    results["probe_A"] = rep
    (OUT / args.results_name).write_text(json.dumps(results, indent=1))
    print(json.dumps({k: rep[k] for k in ("selected", "baselines", "selected_vs_baseline", "majority_acc")}, indent=1))


if __name__ == "__main__":
    main()
