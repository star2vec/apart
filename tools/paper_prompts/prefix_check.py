"""Is each agent's rendered verdict prompt in episode k a strict text prefix of episode k+1's?"""
import json, sys
from collections import Counter, defaultdict
sys.path.insert(0, sys.argv[3])
from render_wire import vllm_conversation, vllm_tools  # noqa: E402
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[2])
by = defaultdict(dict)
for line in open(sys.argv[1]):
    r = json.loads(line)
    if "skip" in r: continue
    b = r["body"]
    p = tok.apply_chat_template(vllm_conversation(b["messages"]), tools=vllm_tools(b.get("tools")), add_generation_prompt=False, tokenize=False)
    by[(r["run"], r["agent"])][r["ep"]] = (p, b["messages"])
res, where = Counter(), Counter()
for key, eps in by.items():
    for k in sorted(eps):
        if k + 1 not in eps: continue
        a, b = eps[k][0], eps[k + 1][0]
        if b.startswith(a): res["strict prefix"] += 1; continue
        res["NOT prefix"] += 1
        i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y) if any(x != y for x, y in zip(a, b)) else min(len(a), len(b))
        where[a[max(0, i-60):i+40].replace("\n", "\\n")[:100]] += 1
        ma, mb = eps[k][1], eps[k + 1][1]
        res["messages list is prefix" if mb[:len(ma)] == ma else "messages list NOT prefix"] += 1
print(dict(res))
for w, n in where.most_common(3): print(n, "x first divergence near:", w)
