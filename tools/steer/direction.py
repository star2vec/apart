"""Steering prep (Mac): the steered calls and the C direction they are steered along.

    tools/.venv/bin/python tools/steer/direction.py

Steered calls (pilot and main run) are fixed here. The direction is the difference of means
between ACCEPT and REJECT at C (layer 47, verdict pass, onset and before-onset rows), computed
only on trajectories that contain none of the steered calls.

Separation is reported two ways:
  cv_within   5-fold cross-validation by trajectory inside the training trajectories
              (difference of means on the training folds, AUC of the projection on the test fold)
  held_out    the final direction applied to the steered calls' trajectories (never used)
Strength units: the spread (SD) of the projection onto the unit direction at C, pooled over the
training rows; the gap between the class means is given in the same units.

Writes agent-collusion/results/steer/{direction.npz, steer_calls.json, direction.json}.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step2"))
sys.path.insert(0, str(ROOT / "tools" / "step3"))
import probe_lib as pl  # noqa: E402
import run_step3 as s3  # noqa: E402

RES = ROOT / "agent-collusion" / "results"
OUT = RES / "steer"
LAYER = 47
PILOT = ["call.rep013.bob.ep02", "call.rep015.alice.ep02", "call.rep043.alice.ep02"]
HARD = PILOT + ["call.rep013.bob.ep03", "call.rep048.alice.ep03", "call.rep036.bob.ep03", "call.rep011.alice.ep02",
                "call.rep010.alice.ep03", "call.rep049.alice.ep03", "call.rep012.alice.ep04"]
SETTLED_ACCEPT = ["call.rep024.alice.ep03", "call.rep042.alice.ep03", "call.rep025.bob.ep03", "call.rep034.bob.ep04",
                  "call.rep011.bob.ep03"]
SETTLED_REJECT = ["call.rep048.alice.ep01", "call.rep035.alice.ep01", "call.rep034.bob.ep02"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    analysis = {c["job"]: c for c in json.loads((RES / "step4_5" / "analysis.json").read_text())["calls"]}
    steered = HARD + SETTLED_ACCEPT + SETTLED_REJECT
    excluded = sorted({j.split(".")[1] for j in steered})
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = [r for r in s3.build_rows(index, episodes) if r["type"] in ("onset", "before_onset")]
    X = pl.rows_to_float(pl.load_layer(LAYER), np.array([r["pos"]["C"] for r in rows])).astype(np.float64)
    y = np.array([r["y"] for r in rows])
    train = np.array([r["run"] not in excluded for r in rows])

    def diff_dir(idx):
        return X[idx & (y == 1)].mean(0) - X[idx & (y == 0)].mean(0)

    # cross-validated separation inside the training trajectories
    tr_rows = [r for r, t in zip(rows, train) if t]
    tr_idx = np.where(train)[0]
    splits = pl.make_splits([r["run"] for r in tr_rows], repeats=5)
    aucs = []
    for sp in splits:
        m = np.zeros(len(rows), bool)
        m[tr_idx[sp.train]] = True
        d = diff_dir(m)
        te = tr_idx[sp.test]
        aucs.append(pl.auc(y[te], X[te] @ d))
    d = diff_dir(train)
    u = d / np.linalg.norm(d)
    proj = X @ u
    sd = float(proj[train].std())
    gap = float(proj[train & (y == 1)].mean() - proj[train & (y == 0)].mean())
    held = ~train
    calls = {}
    for name, group in (("hard", HARD), ("settled_accept", SETTLED_ACCEPT), ("settled_reject", SETTLED_REJECT)):
        for j in group:
            _, run, agent, ep = j.split(".")
            ci = next(i for i, r in enumerate(index) if r["pos_type"] == "C" and r["pass"] == "verdict"
                      and r["run"] == run and r["agent"] == agent and r["episode"] == int(ep[2:]))
            pk = float(pl.rows_to_float(pl.load_layer(LAYER), np.array([ci]))[0].astype(np.float64) @ u)
            a = analysis[j]
            calls[j] = {"group": name, "pilot": j in PILOT, "share_A_vllm": a["A"]["share"], "p_C": a["p_C"],
                        "verdict_saved": a["verdict"], "n_think_saved": a["n_think"],
                        "projection_at_C_sd_units": round(float((pk - proj[train].mean()) / sd), 3),
                        # sign of the push: toward reject for accepts and hard cases, toward accept for rejects
                        "target_sign": +1 if name == "settled_reject" else -1}
    # mean-ablation target: the midpoint of the class means (the direction's decision boundary)
    abl_mean = float((proj[train & (y == 1)].mean() + proj[train & (y == 0)].mean()) / 2)
    np.savez(OUT / "direction.npz", unit=u.astype(np.float32), sd=sd, gap=gap, layer=LAYER,
             train_mean=float(proj[train].mean()), abl_mean=abl_mean)
    rng = np.random.default_rng(0)
    r = rng.normal(size=u.shape)
    r -= (r @ u) * u
    np.savez(OUT / "random_direction.npz", unit=(r / np.linalg.norm(r)).astype(np.float32), seed=0)
    out = {"layer": LAYER, "excluded_runs": excluded, "n_train_rows": int(train.sum()),
           "n_train_runs": len({r["run"] for r in tr_rows}), "n_held_out_rows": int(held.sum()),
           "cv_within_auc": pl.summarise([{"test_auc": v} for v in aucs]),
           "held_out_auc": round(float(pl.auc(y[held], proj[held])), 4),
           "projection_sd": round(sd, 3), "mean_ablation_target": round(abl_mean, 3),
           "class_means": [round(float(proj[train & (y == 1)].mean()), 3), round(float(proj[train & (y == 0)].mean()), 3)], "class_gap": round(gap, 3), "class_gap_sd_units": round(gap / sd, 3),
           "within_class_sd_units": {"accept": round(float(proj[train & (y == 1)].std() / sd), 3),
                                     "reject": round(float(proj[train & (y == 0)].std() / sd), 3)},
           "norm_of_mean_difference": round(float(np.linalg.norm(d)), 3),
           "typical_residual_norm_at_C": round(float(np.linalg.norm(X[train], axis=1).mean()), 1),
           "cosine_with_probe_direction": None}
    probes = np.load(RES / "step6" / "probes.npz")
    out["cosine_with_probe_direction"] = [round(float(u @ (w / np.linalg.norm(w))), 3) for w in probes["W"]]
    (OUT / "steer_calls.json").write_text(json.dumps(calls, indent=1))
    (OUT / "direction.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    for j, c in calls.items():
        print(j, c["group"], "share", round(c["share_A_vllm"], 3), "proj(sd)", c["projection_at_C_sd_units"])


if __name__ == "__main__":
    main()
