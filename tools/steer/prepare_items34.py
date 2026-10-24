"""Steering prep (Mac): job file for items 3-4 (6 samples per condition, 5 hard cases).

    tools/.venv/bin/python tools/steer/prepare_items34.py

Item 3  contrast_reject_2sd: the approve-vs-reject direction from unrelated review tasks
        (contrast_prompts.py, computed in the worker), pushed toward reject at 2 SD of its own
        projection at C, from C on.
Item 4  layers_mean_abl: at layers 31, 35, ..., 63 at once, the projection on each layer's own
        decision direction (layer_directions.py, trajectories without steered calls) set to that
        layer's class midpoint, from C on.
Calls: the 5 hard cases with the highest unsteered shares (the C-only set of session 3), so a push
toward reject has room to show. Unsteered samples come from run A.
Writes agent-collusion/results/steer/jobs_items34.json.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"
CALLS = ["call.rep012.alice.ep04", "call.rep049.alice.ep03", "call.rep036.bob.ep03", "call.rep048.alice.ep03",
         "call.rep015.alice.ep02"]
CONDS = [["contrast_reject_2sd", "add", "contrast", 2, "from_c"],
         ["layers_mean_abl", "mean_abl_layers", None, 0, "from_c"]]


def main() -> None:
    reduced = {j["id"]: j for j in json.loads((RES / "steer" / "jobs_reduced.json").read_text())}
    jobs = [dict(reduced[c], conds=CONDS, samples=6) for c in CALLS]
    jobs.sort(key=lambda j: -len(j["token_ids"]))
    (RES / "steer" / "jobs_items34.json").write_text(json.dumps(jobs))
    print([(j["id"], len(j["token_ids"])) for j in jobs])


if __name__ == "__main__":
    main()
