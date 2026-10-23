"""Step 3 addition (Mac): the last message position (s10) vs C on the full probe set, with the
same per-position probes as run_step3.py (only those two positions are trained).

    tools/.venv/bin/python tools/step3/c_vs_last.py

Merges a "C_vs_last_message" block into agent-collusion/results/step3/results.json.
"""

import json

import numpy as np
from joblib import Parallel, delayed

import run_step3 as s3
from run_step3 import pl, rs


def main() -> None:
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = s3.build_rows(index, episodes)
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    sust = [r for r in rows if r["type"] == "sustained"]
    y = np.array([r["y"] for r in probe])
    runs = [r["run"] for r in probe]
    splits = pl.make_splits(runs, repeats=5)
    needed = sorted({i for r in probe + sust for i in r["pos"].values()})
    where = {i: k for k, i in enumerate(needed)}
    rowpos = [{q: where[i] for q, i in r["pos"].items()} for r in probe]
    sust_light = [{"run": s["run"], "xpos": {q: where[i] for q, i in s["pos"].items()}} for s in sust]
    sources = ["s10", "C"]
    res = {}
    for L in pl.LAYERS:
        X = pl.rows_to_float(pl.load_layer(L), np.array(needed))
        chunks = np.array_split(np.arange(len(splits)), 4)
        parts = Parallel(n_jobs=8)(
            delayed(s3.source_task)(L, P, X, rowpos, [{"run": r["run"]} for r in probe], sust_light, y,
                                    [splits[k] for k in ch], pl.C_GRID) for P in sources for ch in chunks)
        res[L] = {P: [x for part in parts[i * len(chunks):(i + 1) * len(chunks)] for x in part]
                  for i, P in enumerate(sources)}
        print(f"layer {L}", flush=True)
    n = len(splits)
    sel = {P: [res[max(pl.LAYERS, key=lambda L: res[L][P][k]["inner_auc"])][P][k] for k in range(n)] for P in sources}
    own_oof = {Q: s3.oof_mean(len(probe), [s["pred"][Q] for s in sel[Q]]) for Q in sources}
    masks = {"overall": np.ones(len(probe), bool),
             "agreement": np.array([r["path"] == "agreement" for r in probe]),
             "no_agreement": np.array([r["path"] == "no_agreement" for r in probe])}
    block = s3.compare_c_last(probe, y, runs, sel, own_oof, masks, n, 2000)
    path = s3.OUT / "results.json"
    summary = json.loads(path.read_text())
    summary["C_vs_last_message"] = block
    path.write_text(json.dumps(summary, indent=1))
    print(json.dumps(block, indent=1))


if __name__ == "__main__":
    main()
