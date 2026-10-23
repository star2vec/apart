"""Step 3 addition (Mac): trace the C probe (layer 47) through every saved position.

    tools/.venv/bin/python tools/step3/trace_c_probe.py [--layer 47] [--boot 2000]

One readout for every position, unlike run_step3.py's per-position probes. In each outer
split (trajectory-grouped, 5 folds x 5 repeats, as in steps 2-3) a probe is trained on the
training trajectories' C vectors (verdict pass, onset and before-onset rows, label = own
verdict, L2 strength from inner folds) and applied, unchanged, to the test trajectories'
vectors at A, A_call, every message slot s1-s10 and C, in the communication pass and in the
verdict pass (same positions, verdict-phase tool list), and to their sustained rows.

Writes agent-collusion/results/step3/trace_c_probe.json, trace_c_scores.jsonl and figures
in reports/figures/step3/ (step3_c_probe_*.png).
"""

import argparse
import collections
import json
import time

import numpy as np

import run_step3 as s3
from run_step3 import CURVE, LABEL, SLOTS, pl, rs

POSITIONS = ["A", "A_call"] + SLOTS + ["C"]


def verdict_pass_positions(index):
    """Same slot rule as run_step3.build_rows, on the verdict pass."""
    pos = collections.defaultdict(dict)
    events = collections.defaultdict(list)
    for i, r in enumerate(index):
        if r["pass"] != "verdict":
            continue
        key = (r["run"], r["agent"], r["episode"])
        if r["pos_type"] in ("A", "A_call", "C"):
            pos[key][r["pos_type"]] = i
        elif r["pos_type"] in ("msg_end", "runner_notice"):
            events[key].append(r | {"i": i})
    for key, evs in events.items():
        if "A_call" not in pos[key]:
            continue
        a_call = index[pos[key]["A_call"]]["token"]
        peer_first = any((e["pos_type"] == "runner_notice" or not e.get("own")) and e["token"] < a_call for e in evs)
        for e in evs:
            own = e["pos_type"] == "msg_end" and e["own"]
            order = (2 if own else 1) if peer_first else (1 if own else 2)
            pos[key][f"s{2 * (e['round'] - 1) + order}"] = e["i"]
    return pos


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=47)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()
    t0 = time.time()
    index, episodes = pl.load_index(), pl.load_episodes()
    rows = s3.build_rows(index, episodes)
    vpos = verdict_pass_positions(index)
    probe = [r for r in rows if r["type"] in ("onset", "before_onset")]
    sust = [r for r in rows if r["type"] == "sustained"]
    y = np.array([r["y"] for r in probe])
    runs = [r["run"] for r in probe]
    splits = pl.make_splits(runs, repeats=args.repeats)
    feats = rs.baseline_features([r["unit"] for r in probe])["feedback"]

    def key(r):
        return (r["run"], r["agent"], r["episode"])

    # vectors needed: comm-pass positions (+ C from the verdict pass) and verdict-pass positions
    need = set()
    for r in probe + sust:
        need |= set(r["pos"].values()) | set(vpos[key(r)].values())
    need = sorted(need)
    where = {i: k for k, i in enumerate(need)}
    X = pl.rows_to_float(pl.load_layer(args.layer), np.array(need))
    comm = [{q: where[i] for q, i in r["pos"].items()} for r in probe]
    verd = [{q: where[i] for q, i in vpos[key(r)].items()} for r in probe]
    comm_s = [{q: where[i] for q, i in r["pos"].items()} for r in sust]
    print(f"{len(probe)} probe rows, {len(sust)} sustained rows, layer {args.layer}", flush=True)

    pieces = {(p, q): [] for p in ("comm", "verdict", "sust") for q in POSITIONS}
    chosen_c = []
    for sp in splits:
        tr = sp.train
        xtr, ytr = X[[comm[j]["C"] for j in tr]], y[tr]
        best = (-np.inf, None)
        bases = [(s3.Basis(xtr[a]), a, b) for a, b in sp.inner]
        for c in pl.C_GRID:
            pp, yy = [], []
            for bs, a, b in bases:
                pp.append(s3.fit(bs.z, ytr[a], c).predict_proba(bs(xtr[b]))[:, 1])
                yy.append(ytr[b])
            score = pl.auc(np.concatenate(yy), np.concatenate(pp))
            if score > best[0]:
                best = (score, c)
        chosen_c.append(best[1])
        basis = s3.Basis(xtr)
        clf = s3.fit(basis.z, ytr, best[1])
        test_runs = {probe[j]["run"] for j in sp.test}
        for q in POSITIONS:
            for name, table, ids_all in (("comm", comm, sp.test), ("verdict", verd, sp.test)):
                ids = np.array([j for j in ids_all if q in table[j]])
                if len(ids):
                    pieces[(name, q)].append((ids, clf.predict_proba(basis(X[[table[j][q] for j in ids]]))[:, 1]))
            sids = np.array([k for k, s in enumerate(sust) if s["run"] in test_runs and q in comm_s[k]])
            if len(sids):
                pieces[("sust", q)].append((sids, clf.predict_proba(basis(X[[comm_s[k][q] for k in sids]]))[:, 1]))

    fb = s3.feedback_cv(feats, y, splits)
    fb_oof = s3.oof_mean(len(probe), fb)
    oof = {q: s3.oof_mean(len(probe), pieces[("comm", q)]) for q in POSITIONS}
    oof_v = {q: s3.oof_mean(len(probe), pieces[("verdict", q)]) for q in POSITIONS}
    sust_oof = {q: s3.oof_mean(len(sust), pieces[("sust", q)]) for q in POSITIONS}
    masks = {"overall": np.ones(len(probe), bool),
             "agreement": np.array([r["path"] == "agreement" for r in probe]),
             "no_agreement": np.array([r["path"] == "no_agreement" for r in probe])}

    out = {"settings": {"layer": args.layer, "repeats": args.repeats, "boot": args.boot,
                        "c_chosen": dict(collections.Counter(float(c) for c in chosen_c))},
           "per_position": {}}
    for q in POSITIONS:
        has = ~np.isnan(oof[q])
        fb_q = [(ids[has[ids]], p[has[ids]]) for ids, p in fb]
        r_c = s3.rank_oof(len(probe), pieces[("comm", q)])
        r_f = s3.rank_oof(len(probe), fb_q)
        onset = np.array([r["type"] == "onset" for r in probe])
        out["per_position"][q] = {
            "fold_auc": s3.stats(s3.fold_aucs(y, pieces[("comm", q)])),
            "vs_feedback": {k: s3.boot_diff(y, r_c, r_f, runs, m, args.boot) for k, m in masks.items()},
            "median_score": {
                "before_onset": round(float(np.nanmedian(oof[q][~onset])), 3),
                "agreement_onset": round(float(np.nanmedian(oof[q][onset & masks["agreement"]])), 3),
                "no_agreement_onset": round(float(np.nanmedian(oof[q][onset & masks["no_agreement"]])), 3),
                "sustained": round(float(np.nanmedian(sust_oof[q])), 3)},
            "verdict_pass_minus_comm_pass": None if q == "C" else round(float(np.nanmedian(oof_v[q] - oof[q])), 4),
            "verdict_pass_fold_auc": None if q == "C" else s3.stats(s3.fold_aucs(y, pieces[("verdict", q)]))["mean"],
        }
        e = out["per_position"][q]
        print(f"{q:7s} fold AUC {e['fold_auc']['mean']:.3f} | vs fb {e['vs_feedback']['overall']['diff']:+.3f} "
              f"{e['vs_feedback']['overall']['ci95']} | medians {e['median_score']}", flush=True)
    out["first_beats_feedback"] = {k: next((q for q in CURVE if out["per_position"][q]["vs_feedback"][k]["beats"]), None)
                                   for k in masks}

    # where each onset curve rises, relative to the agreement message (agreement onsets) or slot 10
    def carry(v):
        v = v.copy()
        for k in range(1, len(v)):
            if np.isnan(v[k]):
                v[k] = v[k - 1]
        return v

    jumps = []
    for j, r in enumerate(probe):
        if r["type"] != "onset":
            continue
        curve = carry(np.array([oof[q][j] for q in CURVE]))
        ref = CURVE.index(r["agree_slot"]) if r["agree_slot"] else CURVE.index("s10")
        jj = s3.jump(curve, ref)
        jumps.append({"run": r["run"], "agent": r["agent"], "episode": r["episode"], "path": r["path"],
                      "ref": LABEL[CURVE[ref]], "feedback_p": round(float(fb_oof[j]), 3), **jj,
                      "score_at_A": round(float(curve[0]), 3), "score_at_ref": round(float(curve[ref]), 3),
                      "score_at_C": round(float(curve[-1]), 3), "curve": [round(float(v), 3) for v in curve]})
    out["onset_jumps"] = {}
    for path in ("agreement", "no_agreement"):
        js = [x for x in jumps if x["path"] == path]
        out["onset_jumps"][path] = {
            "n": len(js), "reference": "agreement message" if path == "agreement" else "R5 Bob (slot 10)",
            "first_above_half": dict(collections.Counter(x["first_above_half_vs_agreement"] for x in js)),
            "largest_rise": dict(collections.Counter(x["largest_rise_vs_agreement"] for x in js)),
            "largest_rise_at": dict(collections.Counter(LABEL[x["largest_rise_at"]] for x in js)),
            "median_A_ref_C": [round(float(np.median([x[k] for x in js])), 3) for k in ("score_at_A", "score_at_ref", "score_at_C")],
            "rows": js}

    (s3.OUT).mkdir(parents=True, exist_ok=True)
    (s3.OUT / "trace_c_probe.json").write_text(json.dumps(out, indent=1))
    with open(s3.OUT / "trace_c_scores.jsonl", "w") as f:
        for j, r in enumerate(probe):
            f.write(json.dumps({k: r[k] for k in ("run", "agent", "episode", "type", "y", "path", "agree_slot")}
                               | {"feedback_p": round(float(fb_oof[j]), 4),
                                  "comm": {q: None if np.isnan(oof[q][j]) else round(float(oof[q][j]), 4) for q in POSITIONS},
                                  "verdict_pass": {q: None if np.isnan(oof_v[q][j]) else round(float(oof_v[q][j]), 4) for q in POSITIONS}}) + "\n")
    plots(probe, oof, fb_oof, sust_oof, out, args.layer)
    print(json.dumps(out["first_beats_feedback"]))
    print(json.dumps({p: {k: v for k, v in d.items() if k != "rows"} for p, d in out["onset_jumps"].items()}))
    print(f"done in {(time.time() - t0) / 60:.1f} min")


def plots(probe, oof, fb_oof, sust_oof, out, layer):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s3.FIG.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(CURVE))
    xt = [LABEL[q] for q in CURVE]
    onset = [j for j, r in enumerate(probe) if r["type"] == "onset"]
    groups = {"agreement onsets": [j for j in onset if probe[j]["path"] == "agreement"],
              "no-agreement onsets": [j for j in onset if probe[j]["path"] == "no_agreement"]}

    def carry(v):
        v = v.copy()
        for k in range(1, len(v)):
            if np.isnan(v[k]):
                v[k] = v[k - 1]
        return v

    fig, axes = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, (title, js) in zip(axes[:2], groups.items()):
        cmap = plt.get_cmap("tab20")
        for n, j in enumerate(js):
            col = cmap(n % 20)
            curve = carry(np.array([oof[q][j] for q in CURVE]))
            ax.plot(xs, curve, color=col, lw=1, alpha=0.8)
            ax.hlines(fb_oof[j], xs[0], xs[-1], color=col, lw=0.8, ls="--", alpha=0.5)
            if probe[j]["agree_slot"]:
                k = CURVE.index(probe[j]["agree_slot"])
                ax.plot(xs[k], curve[k], "o", color=col, ms=6, mec="black", mew=0.6)
        med = [np.nanmedian([oof[q][j] for j in js]) for q in CURVE]
        ax.plot(xs, med, color="black", lw=2.5, label="median")
        ax.axhline(0.5, color="grey", lw=0.5)
        ax.set_title(f"{title} (n={len(js)}; all accept)")
        ax.set_xticks(xs, xt, rotation=60, ha="right", fontsize=8)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel(f"P(accept) from the C probe (layer {layer}), out-of-fold")
    ax = axes[2]
    for name, key, col in (("before onset (floor)", "before_onset", "grey"), ("agreement onsets", "agreement_onset", "C3"),
                           ("no-agreement onsets", "no_agreement_onset", "C0"), ("sustained (ceiling)", "sustained", "C2")):
        ax.plot(xs, [out["per_position"][q]["median_score"][key] for q in CURVE], color=col, marker="o", ms=3, label=name)
    ax.set_xticks(xs, xt, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    ax.set_title("median C-probe score by episode type")
    fig.suptitle(f"One readout everywhere: the C probe (layer {layer}) traced through the episode; "
                 "dashed = row's feedback-baseline probability; dot = agreement message")
    fig.tight_layout()
    fig.savefig(s3.FIG / "step3_c_probe_curves.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, col, lab in (("overall", "black", "all"), ("agreement", "C3", "agreement-onset trajectories"),
                        ("no_agreement", "C0", "no-agreement-onset trajectories")):
        ax.plot(xs, [out["per_position"][q]["vs_feedback"][k]["auc_probe"] for q in CURVE], color=col, marker="o", ms=3,
                label=f"C probe, {lab}")
        ax.axhline(out["per_position"]["C"]["vs_feedback"][k]["auc_feedback"], color=col, ls="--", lw=1,
                   label=f"feedback, {lab}")
    ax.set_xticks(xs, xt, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("AUC, own verdict (out-of-fold ranks)")
    ax.set_title(f"C probe (layer {layer}) read at each position vs the feedback baseline")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(s3.FIG / "step3_c_probe_auc.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
