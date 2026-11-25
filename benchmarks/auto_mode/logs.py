"""Cases into eval logs, once: each case replayed by a scripted model, so its log holds the transcript with the call under test.

Each case is a task of one sample. Its model is `mockllm` playing the case's tool calls in order, and its tools return the recorded results, so no command runs and the transcript is the case's exactly. The solver sends any later user message where the case has it. No sentinel is attached: the monitors run later, in replay (`score.py`), so the logs are made once and every monitor is scored on the same ones.

The sample's input is the case's task, and its metadata is empty, so nothing a judge is shown carries the label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver, use_tools
from inspect_ai.tool import Tool, ToolCall, tool

from .cases import CALL_UNDER_TEST, Call, Case

MANIFEST = "cases.json"
"""The file in the log directory recording each case's digest, so a scan can check the logs match the cases."""


def case_digest(case: Case) -> str:
    """A digest of everything in `case` that reaches the log."""
    shown = case.model_dump(mode="json", include={"task", "steps", "call"})
    return hashlib.sha256(json.dumps(shown, sort_keys=True).encode()).hexdigest()[:16]


def case_task(case: Case) -> Task:
    """A task of one sample that replays `case`: a scripted model makes its calls, and its tools return its recorded results."""
    calls = [step.as_call() for step in case.steps if step.user is None] + [case.call]
    results = [step.result or "" for step in case.steps if step.user is None] + [""]
    ids = [f"call-{i + 1}" for i in range(len(calls) - 1)] + [CALL_UNDER_TEST]
    outputs = [_turn(call, id) for call, id in zip(calls, ids, strict=True)]
    played = iter(results)
    return Task(
        name="auto_mode_case",
        dataset=[Sample(id=case.id, input=case.task)],
        solver=[
            use_tools(bash(played), write_file(played)),
            replay_case([step.user for step in case.steps]),
        ],
        model=get_model("mockllm/model", custom_outputs=outputs, memoize=False),
    )


@solver
def replay_case(users: list[str | None]) -> Solver:
    """Play a case's earlier steps, then its call under test: one generate per tool call, each running its call, and a user message sent where `users` has one.

    Args:
        users: Each earlier step in order: its user message, or `None` for a tool call. The solver's arguments are recorded in the log, so they hold nothing the transcript doesn't.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        for user in users:
            if user is not None:
                state.messages.append(ChatMessageUser(content=user))
            else:
                state = await generate(state, tool_calls="single")
        return await generate(state, tool_calls="single")

    return solve


@tool
def bash(results: Iterator[str]) -> Tool:
    async def execute(command: str) -> str:
        """Run a bash command in the agent's container and return its output.

        Args:
            command: The command to run.
        """
        return next(results)

    return execute


@tool
def write_file(results: Iterator[str]) -> Tool:
    async def execute(path: str, content: str) -> str:
        """Write `content` to the file at `path`, replacing it if it exists.

        Args:
            path: The file's path.
            content: What to write.
        """
        return next(results)

    return execute


def generate_logs(
    cases: Sequence[Case], log_dir: Path, *, max_tasks: int = 32, force: bool = False
) -> Path:
    """Write a log for each case under `log_dir`, with the manifest naming each case's digest, and return `log_dir`.

    Raises:
        FileExistsError: If `log_dir` already holds logs and `force` is not set; logs are made once and reused.
        RuntimeError: If a case's log did not succeed.
    """
    if (log_dir / MANIFEST).exists() and not force:
        raise FileExistsError(
            f"{log_dir} already holds logs; they are made once and reused. Pass --force to make them again."
        )
    log_dir.mkdir(parents=True, exist_ok=True)
    for old in log_dir.glob("*.eval"):
        old.unlink()
    logs = eval(
        [case_task(case) for case in cases],
        log_dir=str(log_dir),
        display="none",
        max_tasks=max_tasks,
        fail_on_error=True,
        score=False,
    )
    failed = [log.location for log in logs if log.status != "success"]
    if failed:
        raise RuntimeError(f"{len(failed)} case logs did not succeed.")
    (log_dir / MANIFEST).write_text(
        json.dumps({case.id: case_digest(case) for case in cases}, indent=1)
    )
    return log_dir


def check_manifest(cases: Sequence[Case], log_dir: Path) -> None:
    """Check that `log_dir`'s logs were made from `cases` as they are now.

    Raises:
        FileNotFoundError: If `log_dir` has no manifest.
        ValueError: If a case is missing from the logs or changed since they were made.
    """
    manifest: dict[str, str] = json.loads((log_dir / MANIFEST).read_text())
    missing = [c.id for c in cases if c.id not in manifest]
    changed = [
        c.id for c in cases if c.id in manifest and manifest[c.id] != case_digest(c)
    ]
    if missing or changed:
        raise ValueError(
            f"The logs in {log_dir} don't match the cases (missing: {missing[:5]}, changed: {changed[:5]}); make them again with `logs --force`."
        )


def _turn(call: Call, id: str) -> ModelOutput:
    return ModelOutput.from_message(
        ChatMessageAssistant(
            content=call.text,
            tool_calls=[ToolCall(id=id, function=call.tool, arguments=dict(call.args))],
            model="mockllm/model",
        ),
        stop_reason="tool_calls",
    )
