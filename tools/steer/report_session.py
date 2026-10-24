"""Steering readout (Mac): per call and condition, ACCEPT share among valid and among all samples,
invalid count (of which hit the token cap), with the call's unsteered share (this session's own
unsteered condition if it has one, else run A's) and the vLLM share from the first session (16
samples from C), and a Fisher test of unsteered transformers vs vLLM.

    tools/.venv/bin/python tools/steer/report_session.py --out steer/S3/out [--base steer/A/out steer/A2/out]

Pooled rows per condition over the calls that have it. Writes <out>/../report.json.
"""

import argparse
import collections
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "agent-collusion" / "results"


def tally(samples):
    v = [s["verdict"] for s in samples if s["verdict"]]
    return {"n": len(samples), "valid": len(v), "accept": sum(x == "accept" for x in v),
            "capped": sum(bool(s.get("capped")) for s in samples)}


def fmt(t):
    if not t:
        return "-"
    sv = f"{t['accept'] / t['valid']:.2f}" if t["valid"] else " -  "
    inv = t["n"] - t["valid"]
    return f"{t['accept'] / t['n']:.2f} / {sv}" + (f" ({inv} inv{', ' + str(t['capped']) + ' cap' if t['capped'] else ''})" if inv else "")


def main() -> None:
    from scipy.stats import fisher_exact
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--base", nargs="*", default=["steer/A/out"])
    a = p.parse_args()
    an = {c["job"]: c for c in json.loads((RES / "step4_5" / "analysis.json").read_text())["calls"]}
    res = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(RES / a.out / "call.*.json"))):
        r = json.loads(Path(f).read_text())
        res[r["call"]][r["condition"]] = tally(r["samples"])
    base = {}
    for d in a.base:
        for f in glob.glob(str(RES / d / "call.*.unsteered.json")):
            r = json.loads(Path(f).read_text())
            base.setdefault(r["call"], tally(r["samples"]))
    conds = sorted({c for v in res.values() for c in v if c != "unsteered"})
    pooled = collections.defaultdict(collections.Counter)
    rows = []
    for call, cs in res.items():
        uns = cs.get("unsteered") or base.get(call)
        A = an[call]["A"]
        vn = A.get("valid", 16)
        va = round(A["share"] * vn)
        fp = fisher_exact([[uns["accept"], uns["valid"] - uns["accept"]], [va, vn - va]]).pvalue if uns else None
        src = "new" if "unsteered" in cs else "run A"
        print(f"\n{call[5:]}  vLLM {va}/{vn} = {va / vn:.2f}   unsteered ({src}) {fmt(uns)}   Fisher p {fp:.2f}")
        for c in conds:
            if c in cs:
                d = (cs[c]["accept"] / cs[c]["valid"] - uns["accept"] / uns["valid"]) if cs[c]["valid"] and uns["valid"] else None
                print(f"   {c:20s} {fmt(cs[c]):28s} shift (valid) {'' if d is None else f'{d:+.2f}'}")
                pooled[c].update(cs[c])
                pooled[c + " | unsteered same calls"].update(uns)
        rows.append({"call": call, "vllm": [va, vn], "unsteered": uns, "unsteered_source": src, "fisher_p": fp,
                     "conditions": cs})
    print("\npooled (all / valid, invalid):")
    for c in sorted(pooled):
        print(f"   {c:42s} {fmt(dict(pooled[c]))}")
    (RES / a.out).parent.joinpath("report.json").write_text(json.dumps({"calls": rows, "pooled": {k: dict(v) for k, v in pooled.items()}}, indent=1))


if __name__ == "__main__":
    main()
