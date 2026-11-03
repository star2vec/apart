"""LessWrong versions of the report figures, redrawn from saved results (nothing is refit).

    tools/.venv/bin/python tools/lw/make_lw_figures.py

Same data and transforms as the original plotting code (run_step3.plots, trace_c_probe.plots,
analyse_session.plots, flips.py, analyse_step6.plots); only the styling changes: plain position
names, one ACCEPT and one REJECT colour throughout, random/control groups in grey, fonts sized for
a ~700 px post column, no titles. Writes *_lw.png beside the originals in reports/figures/.

Not reproducible from saved files: the "sustained" median line of step3_auc_by_position's right
panel (the own-position probe's sustained-row scores were never written out), so it is omitted.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "step6"))
sys.argv = sys.argv[:1]
import analyse_step6 as a6  # noqa: E402

RES = ROOT / "agent-collusion" / "results"
FIG = ROOT / "reports" / "figures"

ACCEPT, REJECT = "#1a9641", "#d7191c"
CONTROL = "#8c8c8c"
BLUE, PURPLE, ORANGE, BLACK = "#2c7bb6", "#7b3294", "#e66101", "#222222"
BROWN, TEAL = "#8c510a", "#35978f"  # Fig 3 hard groups, kept off the Fig 1 path colours and ACCEPT/REJECT

SLOTS = [f"s{k}" for k in range(1, 11)]
CURVE = ["A"] + SLOTS + ["C"]
LABEL = {"A": "talk start", "C": "decision prompt", **{f"s{k}": f"message {k}" for k in range(1, 11)}}
XT = [LABEL[q] for q in CURVE]
XS = np.arange(len(CURVE))
TYPE = {"before_onset": "before collusion", "onset": "collusion start", "sustained": "sustained",
        "reverted": "reverted"}
PATH = {"agreement": "explicit agreement", "no_agreement": "no agreement"}
W = 7.0  # inches; saved at 200 dpi, shown at ~700 px, so 1 pt ~ 1.4 px on the page

plt.rcParams.update({"font.size": 10, "axes.labelsize": 10.5, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                     "legend.fontsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})


def positions_axis(ax):
    ax.set_xticks(XS, XT, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.4, len(CURVE) - 0.6)


def carry(v):
    v = v.copy()
    for k in range(1, len(v)):
        if np.isnan(v[k]):
            v[k] = v[k - 1]
    return v


def nan(x):
    return np.nan if x is None else x


def save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    print(path.relative_to(ROOT))


# ---------------------------------------------------------------- Fig 1: run_step3 AUC by position

def fig1():
    summary = json.load(open(RES / "step3" / "results.json"))
    rows = [json.loads(l) for l in open(RES / "step3" / "scores.jsonl")]
    pp = summary["per_position"]

    fig, ax = plt.subplots(figsize=(W, 4.6))
    m = np.array([pp[q]["fold_auc"]["mean"] for q in CURVE])
    sd = np.array([pp[q]["fold_auc"]["sd"] for q in CURVE])
    ax.fill_between(XS, m - sd, m + sd, color=CONTROL, alpha=0.18, lw=0)
    sets = (("overall", BLACK, "all episodes"), ("agreement", BLUE, PATH["agreement"]),
            ("no_agreement", PURPLE, PATH["no_agreement"]))
    for key, col, _ in sets:
        ax.plot(XS, [pp[q]["bootstrap"][key]["auc_probe"] for q in CURVE], color=col, marker="o", ms=4, lw=1.8)
        ax.axhline(pp["A"]["bootstrap"][key]["auc_feedback"], color=col, ls="--", lw=1.2)
    positions_axis(ax)
    ax.set_ylabel("AUC for own verdict (out-of-fold)")
    handles = [Line2D([], [], color=col, lw=2.5, label=lab) for _, col, lab in sets]
    handles += [Line2D([], [], color=BLACK, marker="o", ms=4, lw=1.8, label="probe"),
                Line2D([], [], color=BLACK, ls="--", lw=1.2, label="feedback baseline"),
                Patch(color=CONTROL, alpha=0.3, label="probe, fold mean ± SD (all)")]
    ax.legend(handles=handles, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.3), borderaxespad=0)
    save(fig, FIG / "step3" / "step3_auc_by_position_auc_lw.png")

    fig, ax = plt.subplots(figsize=(W, 4.4))
    for lab, sel, col in (("before collusion", lambda r: r["type"] == "before_onset", ORANGE),
                          (f"collusion start, {PATH['agreement']}",
                           lambda r: r["type"] == "onset" and r["path"] == "agreement", BLUE),
                          (f"collusion start, {PATH['no_agreement']}",
                           lambda r: r["type"] == "onset" and r["path"] == "no_agreement", PURPLE)):
        js = [r for r in rows if sel(r)]
        med = [np.nanmedian([nan(r["own"][q]) for r in js]) for q in CURVE]
        ax.plot(XS, med, color=col, marker="o", ms=4, lw=1.8, label=f"{lab} (n={len(js)})")
    positions_axis(ax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("median P(accept), probe")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.3), borderaxespad=0)
    save(fig, FIG / "step3" / "step3_auc_by_position_median_lw.png")


# ------------------------------------------------------------- Fig 2: trace_c_probe (both figures)

def fig2():
    out = json.load(open(RES / "step3" / "trace_c_probe.json"))
    rows = [json.loads(l) for l in open(RES / "step3" / "trace_c_scores.jsonl")]
    pp = out["per_position"]
    probe_name = "decision-prompt probe"

    # AUC by position
    fig, ax = plt.subplots(figsize=(W, 4.4))
    sets = (("overall", BLACK, "all episodes"), ("agreement", BLUE, PATH["agreement"]),
            ("no_agreement", PURPLE, PATH["no_agreement"]))
    for k, col, _ in sets:
        ax.plot(XS, [pp[q]["vs_feedback"][k]["auc_probe"] for q in CURVE], color=col, marker="o", ms=4, lw=1.8)
        ax.axhline(pp["C"]["vs_feedback"][k]["auc_feedback"], color=col, ls="--", lw=1.2)
    positions_axis(ax)
    ax.set_ylabel("AUC for own verdict (out-of-fold)")
    handles = [Line2D([], [], color=col, lw=2.5, label=lab) for _, col, lab in sets]
    handles += [Line2D([], [], color=BLACK, marker="o", ms=4, lw=1.8, label=probe_name),
                Line2D([], [], color=BLACK, ls="--", lw=1.2, label="feedback baseline")]
    ax.legend(handles=handles, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.3), borderaxespad=0)
    save(fig, FIG / "step3" / "step3_c_probe_auc_lw.png")

    # curves: two path panels and the medians by episode type, stacked
    fig, axes = plt.subplots(3, 1, figsize=(W, 11), sharex=True)
    for ax, path in zip(axes[:2], ("agreement", "no_agreement")):
        js = [r for r in rows if r["type"] == "onset" and r["path"] == path]
        for r in js:
            curve = carry(np.array([nan(r["comm"][q]) for q in CURVE]))
            ax.plot(XS, curve, color=ACCEPT, lw=0.8, alpha=0.35)
            ax.hlines(r["feedback_p"], XS[0], XS[-1], color=CONTROL, lw=0.6, ls="--", alpha=0.45)
            if r["agree_slot"]:
                k = CURVE.index(r["agree_slot"])
                ax.plot(XS[k], curve[k], "o", color=ACCEPT, ms=5, mec="black", mew=0.6, alpha=0.9)
        med = [np.nanmedian([nan(r["comm"][q]) for r in js]) for q in CURVE]
        ax.plot(XS, med, color=BLACK, lw=2.5)
        ax.axhline(0.5, color=CONTROL, lw=0.6)
        ax.set_ylim(-0.02, 1.02)
        ax.set_ylabel("P(accept)")
        ax.text(0.01, 0.03, f"collusion start, {PATH[path]} (n={len(js)}; all ACCEPT)", transform=ax.transAxes,
                fontsize=10, va="bottom")
    axes[0].legend(handles=[Line2D([], [], color=ACCEPT, lw=1, alpha=0.6, label="one agent-episode"),
                            Line2D([], [], color=BLACK, lw=2.5, label="median"),
                            Line2D([], [], color=CONTROL, lw=1, ls="--", label="its feedback-baseline P(accept)"),
                            Line2D([], [], color=ACCEPT, marker="o", mec="black", ls="", label="agreement message")],
                   ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    ax = axes[2]
    for lab, key, col in (("before collusion", "before_onset", ORANGE),
                          (f"collusion start, {PATH['agreement']}", "agreement_onset", BLUE),
                          (f"collusion start, {PATH['no_agreement']}", "no_agreement_onset", PURPLE),
                          ("sustained", "sustained", BLACK)):
        ax.plot(XS, [pp[q]["median_score"][key] for q in CURVE], color=col, marker="o", ms=4, lw=1.8, label=lab)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("median P(accept)")
    ax.legend(loc="upper center", ncol=2)
    positions_axis(ax)
    fig.supylabel(f"{probe_name} score (out-of-fold)", fontsize=10.5)
    fig.tight_layout()
    save(fig, FIG / "step3" / "step3_c_probe_curves_lw.png")


# ---------------------------------------------------------- Fig 3: analyse_session (two figures)

def fig3():
    rows = json.load(open(RES / "step4_5" / "analysis.json"))["calls"]
    fixed = ["fixed_30", "fixed_50", "fixed_70", "fixed_90", "fixed_100"]
    xs = [0, 30, 50, 70, 90, 100]

    fig, ax = plt.subplots(figsize=(W, 4.4))
    ax.axvspan(92, 96, color=CONTROL, alpha=0.2, lw=0, label="decision-prompt probe lock-in (median)")
    for name, g, col, style in (("reverted, REJECT verdicts", "reverted_reject", BROWN, dict(marker="o")),
                                ("flips", "flip", ORANGE, dict(marker="o")),
                                ("confidently wrong", "confident_wrong", TEAL, dict(marker="D")),
                                ("random, collusion start", "random_onset", CONTROL, dict(marker="s")),
                                ("random, before collusion", "random_before_onset", CONTROL,
                                 dict(marker="^", ls="--"))):
        rs = [r for r in rows if g in r["groups"] and r["A"] and r["A"]["share"] is not None]
        ys = []
        for cut in ["A"] + fixed:
            vals = []
            for r in rs:
                s = r["A"]["share"] if cut == "A" else (r["B"].get(cut) or {}).get("share")
                if s is not None:
                    vals.append(s if r["verdict"] == "accept" else 1 - s)
            ys.append(np.mean(vals) if vals else np.nan)
        ax.plot(xs, ys, color=col, lw=1.8, ms=5, label=f"{name} (n={len(rs)})", **style)
    ax.set_xlabel("% of the saved thinking kept before resampling\n(0 = resample from the decision prompt)")
    ax.set_ylabel("share of resamples ending\nin the saved verdict")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    save(fig, FIG / "step4_5" / "session_cuts_and_probe_cuts_lw.png")

    fig, ax = plt.subplots(figsize=(W, 5))
    rs = [r for r in rows if r["A"] and r["A"]["share"] is not None]
    ax.plot([0, 1], [0, 1], color=CONTROL, lw=0.9, ls="--")
    for name, sel, col in (("chosen cases", lambda r: not r["groups"][0].startswith("random"), BLUE),
                           ("random cases", lambda r: r["groups"][0].startswith("random"), CONTROL)):
        pts = [(r["p_C"], r["A"]["share"]) for r in rs if sel(r)]
        ax.scatter(*zip(*pts), s=26, color=col, alpha=0.75, lw=0, label=f"{name} (n={len(pts)})")
    ax.set_xlabel("decision-prompt probe P(accept) at the decision prompt (out-of-fold)")
    ax.set_ylabel("share ACCEPT in 16 full resamples\nfrom the decision prompt")
    ax.legend(loc="upper left")
    save(fig, FIG / "step4_5" / "session_cuts_and_probe_probe_lw.png")


# ----------------------------------------------------------------- Fig 4: analyse_step6 (both)

def fig4():
    out = json.load(open(RES / "step6" / "analysis.json"))
    calls = [json.loads(l) for l in open(RES / "step6" / "calls.jsonl")]
    acts = RES / "step6" / "acts"
    groups = (("pooled", lambda c: True), ("alice (speaks first)", lambda c: c["agent"] == "alice"),
              ("bob (speaks second)", lambda c: c["agent"] == "bob"))
    panel = {"pooled": "both agents", "alice (speaks first)": "Alice (speaks first)",
             "bob (speaks second)": "Bob (speaks second)"}

    # thinking curves (smoothed C-probe P(accept) from the saved per-token probe logits, as in step 6)
    main_calls = [c for c in calls if c["kind"] == "call" and c["type"] in ("onset", "before_onset")
                  and (acts / f"{c['job']}.safetensors").exists()]
    grid = np.linspace(0, 1, 101)
    smoothed = {}
    for c in main_calls:
        lg = a6.read_safetensors(acts / f"{c['job']}.safetensors")["probe_logits"]
        sp = a6.smooth(a6.sigmoid(lg[c["think_start"] - c["c_token"]:].astype(np.float64)))
        smoothed[c["job"]] = np.interp(grid, np.linspace(0, 1, len(sp)), sp)
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.4), sharey=True)
    sets = ((("onset", "accept"), ACCEPT, "-"), (("before_onset", "accept"), ACCEPT, "--"),
            (("before_onset", "reject"), REJECT, "-"))
    for ax, (g, sel) in zip(axes, groups):
        for (t, v), col, ls in sets:
            cs = [c for c in main_calls if sel(c) and c["type"] == t and c["verdict"] == v]
            curves = np.stack([smoothed[c["job"]] for c in cs])
            ax.plot(grid, np.median(curves, 0), color=col, ls=ls, lw=1.8)
            ax.fill_between(grid, np.percentile(curves, 25, 0), np.percentile(curves, 75, 0), color=col,
                            alpha=0.08 if ls == "--" else 0.15, lw=0)
        ax.set_title(panel[g], fontsize=10)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    axes[1].set_xlabel("position in the decision call's thinking (fraction)")
    axes[0].set_ylabel("decision-prompt probe\nP(accept), smoothed")
    n = {k: sum(c["type"] == k[0] and c["verdict"] == k[1] for c in main_calls) for k, _, _ in sets}
    fig.legend(handles=[Line2D([], [], color=col, ls=ls, lw=1.8,
                               label=f"{TYPE[t]} → {v.upper()} (n={n[(t, v)]})") for (t, v), col, ls in sets]
               + [Patch(color=CONTROL, alpha=0.3, label="interquartile range")],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    fig.tight_layout()
    save(fig, FIG / "step6" / "step6_thinking_curves_lw.png")

    # AUC by tenth
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.4), sharey=True)
    xs = np.arange(1, 11)
    for ax, (g, _) in zip(axes, groups):
        e = out[g]
        ax.plot(xs, e["c_probe_auc_by_tenth"], color=BLUE, marker="o", ms=4, lw=1.6)
        ax.errorbar(xs, [f["mean"] for f in e["fresh_probe_auc_by_tenth"]],
                    yerr=[f["sd"] for f in e["fresh_probe_auc_by_tenth"]], color=ORANGE, marker="s", ms=4,
                    lw=1.6, capsize=2)
        ax.axhline(e["auc_c_logit"], color=BLACK, ls="--", lw=1.1)
        ax.set_title(f"{panel[g]} (n={e['n']})", fontsize=10)
        ax.set_xticks([1, 5, 10])
    axes[1].set_xlabel("tenth of the thinking")
    axes[0].set_ylabel("AUC, ACCEPT vs REJECT calls")
    fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=4, lw=1.6,
                               label="decision-prompt probe (mean logit in the tenth)"),
                        Line2D([], [], color=ORANGE, marker="s", ms=4, lw=1.6,
                               label="fresh probe per tenth (fold mean ± SD)"),
                        Line2D([], [], color=BLACK, ls="--", lw=1.1,
                               label="decision-prompt probe at the decision prompt")],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=1)
    fig.tight_layout()
    save(fig, FIG / "step6" / "step6_auc_by_tenth_lw.png")


# ------------------------------------------------------------------------ Fig 5: flips

SELECTED_FLIPS = ["call.rep003.bob.ep04", "call.rep011.alice.ep02", "call.rep024.bob.ep02",
                  "call.rep049.alice.ep03", "call.rep048.alice.ep03"]


def plain_title(job):
    run, agent, ep = job.replace("call.", "").split(".")
    return (f"trajectory {int(run[3:])}\n{'first' if agent == 'alice' else 'second'} agent\n"
            f"episode {int(ep[2:])}")


def fig5(selected=False):
    fl = json.load(open(RES / "step4_5" / "flips.json"))["main_flips"]
    if selected:
        by_job = {x["job"]: x for x in fl}
        fl = [by_job[j] for j in SELECTED_FLIPS]
    calls = {c["job"]: c for c in map(json.loads, open(RES / "step6" / "calls.jsonl")) if c["kind"] == "call"}
    acts = RES / "step6" / "acts"
    colour = {"accept": ACCEPT, "reject": REJECT}
    cols = len(fl) if selected else 4
    rows_n = int(np.ceil(len(fl) / cols))
    fig, axes = plt.subplots(rows_n, cols, figsize=(W, 1.35 * rows_n + (1.5 if selected else 0.6)),
                             sharex=True, sharey=True, squeeze=False)
    for ax, x in zip(axes.flat, fl):
        c = calls[x["job"]]
        lg = a6.read_safetensors(acts / f"{x['job']}.safetensors")["probe_logits"].astype(float)
        sp = a6.smooth(a6.sigmoid(lg[c["think_start"] - c["c_token"]:]))
        ax.plot(np.linspace(0, 1, len(sp)), sp, color=BLUE, lw=0.9)
        ax.axhline(a6.sigmoid(lg[0]), color=CONTROL, ls=":", lw=1)
        ax.axvline(x["first_any"], color=colour[x["first_any_polarity"]], lw=1.3)
        if x["first_final"] is not None:
            ax.axvline(x["first_final"], color=colour[x["verdict"]], lw=1.3, ls="--")
        if selected:
            title = f"{plain_title(x['job'])}\n{TYPE[x['type']]}\n→ {x['verdict'].upper()}"
        else:
            run, agent, ep = x["job"].replace("call.", "").split(".")
            title = f"{run} {agent.capitalize()} ep{int(ep[2:])}\n{TYPE[x['type']]} → {x['verdict'].upper()}"
        ax.set_title(title, fontsize=7.5, pad=3)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1], ["0", "0.5", "1"])
        ax.tick_params(labelsize=8)
    for ax in axes.flat[len(fl):]:
        ax.axis("off")
    for ax in axes.flat[len(fl) - cols:len(fl)]:
        ax.tick_params(labelbottom=True)
    fig.supxlabel("position in the decision call's thinking (fraction)", fontsize=10)
    fig.supylabel("decision-prompt probe\nP(accept), smoothed" if selected
                  else "decision-prompt probe P(accept), smoothed", fontsize=10)
    fig.legend(handles=[Line2D([], [], color=BLACK, lw=1.3, label="first stated decision"),
                        Line2D([], [], color=BLACK, lw=1.3, ls="--", label="first statement of the final verdict"),
                        Patch(color=ACCEPT, label="ACCEPT"), Patch(color=REJECT, label="REJECT"),
                        Line2D([], [], color=BLUE, lw=1, label="probe score"),
                        Line2D([], [], color=CONTROL, ls=":", lw=1, label="score at decision prompt")],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9, columnspacing=1.2)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    save(fig, FIG / "step4_5" / ("flips_selected_lw.png" if selected else "flips_lw.png"))


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig5(selected=True)
