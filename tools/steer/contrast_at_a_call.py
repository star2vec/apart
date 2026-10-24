"""Final round (Mac): the approve-vs-reject direction from unrelated review tasks (steering item 3),
read at A_call from the step-1 cache (layer 47, communication pass: the opening of the agent's
first communication turn, the same token type as C).

    tools/.venv/bin/python tools/steer/contrast_at_a_call.py [--repeats 5]

Rows as in step 2: onset and before-onset episodes, label = the agent's own verdict; and the
previous-REJECT subset (the agent rejected in the previous episode: who switches to accept).
The contrast direction is fixed (built from unrelated prompts, never fitted on the game), so its
score is the projection; the feedback baseline is fitted out of fold (steps 2-3's feedback
features, L2 strength from inner folds). Both are scored on the same trajectory-grouped test folds
(5 folds x repeats): mean and SD of the fold AUCs, folds where the contrast direction beats the
baseline, and the pooled AUC of the projection. For reference, the C-direction projection (run A's
steering direction, also at A_call) on the same rows.
Writes agent-collusion/results/steer/contrast_at_a_call.json.
"""

import argparse
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

RES = ROOT / "agent-collusion" / "results"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    uc = np.load(RES / "steer" / "I34" / "out" / "contrast_direction.npz")["unit"].astype(np.float64)
    ud = np.load(RES / "steer" / "direction.npz")["unit"].astype(np.float64)
    index, episodes = pl.load_index(), pl.load_episodes()
    bits = pl.load_layer(47)
    out = {}
    for subset in ("onset_and_before_onset", "previous_reject"):
        us = sorted((u for u in rs.units(index, episodes, pos_type="A_call") if u["type"] in ("onset", "before_onset")
                     and (subset != "previous_reject" or (u["prev"] is not None and not u["prev"]["own_accept"]))),
                    key=lambda u: (u["run"], u["agent"], u["episode"]))
        y = np.array([u["y"] for u in us])
        X = pl.rows_to_float(bits, np.array([u["i"] for u in us])).astype(np.float64)
        proj_c, proj_d = X @ uc, X @ ud
        splits = pl.make_splits([u["run"] for u in us], repeats=a.repeats)
        fb = s3.feedback_cv(rs.baseline_features(us)["feedback"], y, splits)
        f_auc = np.array([pl.auc(y[ids], pr) for ids, pr in fb])
        c_auc = np.array([pl.auc(y[sp.test], proj_c[sp.test]) for sp in splits])
        d_auc = np.array([pl.auc(y[sp.test], proj_d[sp.test]) for sp in splits])
        e = {"rows": len(us), "trajectories": len({u["run"] for u in us}), "accept_rate": round(float(y.mean()), 3),
             "contrast_fold_auc": s3.stats(c_auc), "feedback_fold_auc": s3.stats(f_auc),
             "C_direction_fold_auc": s3.stats(d_auc),
             "contrast_minus_feedback": s3.stats(c_auc - f_auc),
             "folds_contrast_better": int(np.sum(c_auc > f_auc)), "n_folds": len(splits),
             "contrast_pooled_auc": round(float(pl.auc(y, proj_c)), 4),
             "C_direction_pooled_auc": round(float(pl.auc(y, proj_d)), 4)}
        out[subset] = e
        print(subset, json.dumps(e), flush=True)
    (RES / "steer" / "contrast_at_a_call.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
