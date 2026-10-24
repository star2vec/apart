"""Steering prep (Mac) for items 3-4.

    tools/.venv/bin/python tools/steer/layer_directions.py

Item 4: a separate decision direction at each of layers 31, 35, ..., 63: difference of means
between ACCEPT and REJECT at C (verdict pass, onset and before-onset rows, as direction.py), on
trajectories that contain none of the steered calls (run A, the 2 SD rerun and session 3). Per
layer: the unit direction, the projection SD on the training rows and the mean-ablation target
(the midpoint of the class means); held-out AUC on the excluded trajectories.
Item 3: the layer-47 C activations of the same training rows (float16), so the GPU session can
express the contrast direction's strength in SD units of the projection at C, as for the C
direction.
Writes agent-collusion/results/steer/{layer_directions.npz, layer_directions.json, c_acts_l47.npz}.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step2"))
sys.path.insert(0, str(ROOT / "tools" / "step3"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_lib as pl  # noqa: E402
import run_step3 as s3  # noqa: E402

RES = ROOT / "agent-collusion" / "results"
OUT = RES / "steer"
LAYERS = list(range(31, 64, 4))


def steered_runs():
    runs = set()
    for f in ("jobs_reduced.json", "jobs_session3.json"):
        runs |= {j["id"].split(".")[1] for j in json.loads((OUT / f).read_text())}
    calls = json.loads((OUT / "steer_calls.json").read_text())
    return sorted(runs | {j.split(".")[1] for j in calls})


def main() -> None:
    excluded = steered_runs()
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = [r for r in s3.build_rows(index, episodes) if r["type"] in ("onset", "before_onset")]
    pos = np.array([r["pos"]["C"] for r in rows])
    y = np.array([r["y"] for r in rows])
    train = np.array([r["run"] not in excluded for r in rows])
    units, sds, abl, info = [], [], [], {}
    for layer in LAYERS:
        X = pl.rows_to_float(pl.load_layer(layer), pos).astype(np.float64)
        d = X[train & (y == 1)].mean(0) - X[train & (y == 0)].mean(0)
        u = d / np.linalg.norm(d)
        proj = X @ u
        m1, m0 = proj[train & (y == 1)].mean(), proj[train & (y == 0)].mean()
        units.append(u.astype(np.float32))
        sds.append(float(proj[train].std()))
        abl.append(float((m1 + m0) / 2))
        info[layer] = {"sd": round(sds[-1], 3), "abl_mean": round(abl[-1], 3), "gap_sd_units": round(float((m1 - m0) / sds[-1]), 3),
                       "held_out_auc": round(float(pl.auc(y[~train], proj[~train])), 4)}
        if layer == 47:
            np.savez(OUT / "c_acts_l47.npz", acts=X[train].astype(np.float16), y=y[train])
            ref = np.load(OUT / "direction.npz")["unit"]
            info[layer]["cosine_with_run_A_direction"] = round(float(u @ ref), 4)
        print(layer, info[layer], flush=True)
    np.savez(OUT / "layer_directions.npz", layers=np.array(LAYERS), units=np.stack(units), sd=np.array(sds),
             abl_mean=np.array(abl))
    out = {"layers": LAYERS, "excluded_runs": excluded, "n_train_rows": int(train.sum()),
           "n_held_out_rows": int((~train).sum()), "per_layer": info}
    (OUT / "layer_directions.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
