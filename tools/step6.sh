#!/usr/bin/env bash
# Step 6 in one GPU start: boot at 262k, teacher-forcing check, 2-episode timing run,
# prefix check on the run's call log. Starts the Modal GPU container; run only when approved.
#   tools/step6.sh <paper_wire_verdict.jsonl>
set -euo pipefail
cd "$(dirname "$0")/.."
WIRE="$1"
FROM="${2:-1}"  # first stage to run (1-4), to resume after a failure
OUT=agent-collusion/results/step6
mkdir -p "$OUT"
stamp() { date '+%F %T %z'; }

echo "[$(stamp)] 1/4 boot: waking server"
[ "$FROM" -le 1 ] && curl -sL --max-redirs 40 -m 1200 -o /dev/null \
  -w "health http=%{http_code} total_s=%{time_total}\n" \
  https://star2vec--qwen36-vllm-serve.us-east.modal.run/health

echo "[$(stamp)] 2/4 teacher forcing"
[ "$FROM" -le 2 ] && tools/.venv/bin/python tools/teacher_force.py --wire "$WIRE" --samples 3 \
  --pick rep049:1:alice rep023:1:alice rep024:1:bob rep001:2:alice rep001:4:alice \
  --out "$OUT/teacher_force.json"

echo "[$(stamp)] 3/4 timing run: sequence 1, 2 episodes, 45 min hard timeout"
agent-collusion/.venv/bin/python tools/run_trajectory.py --sequence 1 --episodes 2 \
  --output-dir results/step6/timing --timeout-min 45

echo "[$(stamp)] 4/4 prefix check"
RUN_DIR=$(ls -td agent-collusion/results/step6/timing/run_* | head -1)
tools/.venv/bin/python tools/log_prefix_check.py "$RUN_DIR"
echo "[$(stamp)] done; server idles 5 min, then scales down"
