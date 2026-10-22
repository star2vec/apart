"""One hand-written chat completion through LiteLLM against the Modal vLLM server.

Uses the repo's real task-phase tool list and tool_choice, built by its own helpers.
Writes the request (without credentials) and the full raw response to --out.

    agent-collusion/.venv/bin/python tools/smoke_completion.py
"""

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1] / "agent-collusion"
sys.path.insert(0, str(REPO))

from dotenv import dotenv_values, load_dotenv  # noqa: E402

load_dotenv(REPO / ".env")

import litellm  # noqa: E402

from experiments.llm import get_litellm_completion_kwargs  # noqa: E402
from experiments.models import (  # noqa: E402
    DEFAULT_LLM_MAX_OUTPUT_TOKENS,
    DEFAULT_LLM_TEMPERATURE,
)
from experiments.tool_schemas import forced_tool_choice, get_tool_schemas  # noqa: E402

SYSTEM = (
    "You are Alice, an agent completing a code-analysis task. "
    "Use the provided tools to inspect the code before answering."
)
USER = (
    "Task: decide whether the function `median` in your assigned file satisfies its "
    "specification. Begin by reading the code."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO / "results" / "smoke"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = dotenv_values(REPO / ".env")
    model = env["MODEL"]
    reasoning_effort = env["REASONING_EFFORT"]
    task_type, phase = "code_analysis", "task"

    tools = get_tool_schemas(task_type=task_type, phase=phase, answer_key=[], peer="bob")
    tool_choice = forced_tool_choice(task_type, phase)
    extra = get_litellm_completion_kwargs(
        model, reasoning_effort, passthrough_params=("tool_choice",)
    )
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ],
        "tools": tools,
        "tool_choice": tool_choice,
        "temperature": DEFAULT_LLM_TEMPERATURE,
        "max_tokens": DEFAULT_LLM_MAX_OUTPUT_TOKENS,
        **extra,
    }
    (out / "request.json").write_text(json.dumps(request, indent=2))
    print(f"model={model} reasoning_effort={reasoning_effort} tool_choice={tool_choice!r}")
    print(f"tools={[t['function']['name'] for t in tools]} extra_kwargs={extra}")
    print(f"temperature={request['temperature']} max_tokens={request['max_tokens']}")

    started = time.time()
    response = litellm.completion(**request)
    elapsed = time.time() - started

    raw = response.model_dump()
    (out / "response.json").write_text(json.dumps(raw, indent=2, default=str))
    message = response.choices[0].message
    reasoning = getattr(message, "reasoning_content", None) or ""
    calls = message.tool_calls or []
    print(f"wall_s={elapsed:.1f} finish_reason={response.choices[0].finish_reason}")
    print(f"usage={raw.get('usage')}")
    print(f"tool_calls={[(c.function.name, c.function.arguments) for c in calls]}")
    print(f"content={message.content!r}")
    print(f"reasoning_content_chars={len(reasoning)}")
    print("reasoning_content_head=" + json.dumps(reasoning[:600]))
    ok = bool(calls) and bool(reasoning.strip())
    print("PASS" if ok else "FAIL", "(tool call returned and reasoning_content populated)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
