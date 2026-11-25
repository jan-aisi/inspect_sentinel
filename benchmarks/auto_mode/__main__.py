"""`python -m benchmarks.auto_mode`: make the case logs, score them with each monitor by replay, and report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from inspect_ai._util.error import PrerequisiteError

from .cases import Case, load_cases
from .local import sample_benign
from .logs import check_manifest, generate_logs
from .report import write_report
from .score import MONITORS, ScoreSettings, runnable, score_monitor
from .sources import SOURCES, import_sources

HERE = Path(__file__).parent
CASES = HERE / "cases"
OUT = HERE / "out"
REPOSITORY = HERE.parent.parent

NOT_RUN = 3
"""The exit status when the benchmark could not run, such as without the judge model's API key."""

UNSAFE_OUT = 4
"""The exit status when `--out` is inside the repository where git would take its contents: not ignored, or holding tracked files."""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.auto_mode",
        description="Compare auto_mode(), Codex Guardian and Claude Code's classifier as monitors on the same labelled tool calls, by replay.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--cases",
            type=Path,
            nargs="+",
            default=None,
            help="Case files or directories; by default the committed cases and the imported ones under --out.",
        )
        p.add_argument(
            "--out",
            type=Path,
            default=OUT,
            help="Where logs, scans, results and the report go.",
        )

    def scoring(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--judge-model",
            required=True,
            help="The model every monitor judges on, such as anthropic/claude-haiku-5-5.",
        )
        p.add_argument(
            "--model",
            action="append",
            default=[],
            metavar="MONITOR=MODEL",
            help="A model for one monitor in place of --judge-model.",
        )
        p.add_argument(
            "--monitors", nargs="+", choices=MONITORS, default=list(MONITORS)
        )
        p.add_argument(
            "--claude-code-prompt",
            type=Path,
            default=None,
            help="The directory of Claude Code's classifier prompt files (or AUTO_MODE_BENCH_CLAUDE_CODE_PROMPT); without it, that monitor is skipped.",
        )
        p.add_argument(
            "--guardian-prompt",
            type=Path,
            default=None,
            help="A local directory of Codex Guardian's prompt files, in place of fetching them.",
        )
        p.add_argument(
            "--max-transcripts", type=int, default=25, help="Cases judged at once."
        )

    logs = commands.add_parser("logs", help="Make an eval log for each case, once.")
    common(logs)
    logs.add_argument("--force", action="store_true", help="Make the logs again.")

    sample = commands.add_parser(
        "sample-benign",
        help="Sample bash calls from real eval logs into allow cases, under --out (never committed).",
    )
    sample.add_argument("--logs", type=Path, required=True)
    sample.add_argument("--count", type=int, default=200)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--out", type=Path, default=OUT)

    score = commands.add_parser(
        "score", help="Score the logs with each monitor by replay."
    )
    common(score)
    scoring(score)

    report = commands.add_parser("report", help="Write the report from the results.")
    common(report)

    imports = commands.add_parser(
        "import",
        help="Import cases from existing sources into --out/imported (never committed).",
    )
    imports.add_argument("--out", type=Path, default=OUT)
    imports.add_argument(
        "--sources", nargs="*", choices=list(SOURCES), default=list(SOURCES)
    )

    run = commands.add_parser(
        "run",
        help="Import the sources, make the logs if they don't match the cases, score, and report.",
    )
    common(run)
    scoring(run)
    run.add_argument(
        "--sources", nargs="*", choices=list(SOURCES), default=list(SOURCES)
    )

    args = parser.parse_args(argv)
    out: Path = args.out
    unsafe = check_out(out)
    if unsafe is not None:
        print(unsafe, file=sys.stderr)
        return UNSAFE_OUT
    if args.command == "sample-benign":
        print(sample_benign(args.logs, args.count, out, seed=args.seed))
        return 0
    if args.command in ("import", "run"):
        written, skipped = import_sources(args.sources, out)
        for message in skipped:
            print(message, file=sys.stderr)
        for file in written:
            print(f"Imported {file}")
        if args.command == "import":
            return 0
    cases = load_cases(args.cases or default_cases(out))
    try:
        if args.command == "logs":
            generate_logs(cases, out / "logs", force=args.force)
            print(f"Made {len(cases)} case logs in {out / 'logs'}.")
        elif args.command == "run" and not logs_match(cases, out / "logs"):
            generate_logs(cases, out / "logs", force=True)
            print(f"Made {len(cases)} case logs in {out / 'logs'}.")
        if args.command in ("score", "run"):
            settings = ScoreSettings(
                out=out,
                judge_model=args.judge_model,
                models=dict(_pair(m) for m in args.model),
                claude_code_prompt=args.claude_code_prompt,
                guardian_prompt=args.guardian_prompt,
                max_transcripts=args.max_transcripts,
            )
            names, skipped = runnable(args.monitors, settings)
            for message in skipped:
                print(message, file=sys.stderr)
            for name in names:
                print(f"Scored {name}: {score_monitor(name, cases, settings)}")
        if args.command in ("report", "run"):
            report = write_report(
                cases, out, args.monitors if args.command == "run" else MONITORS
            )
            print(report.read_text())
    except PrerequisiteError as ex:
        print(ex.message, file=sys.stderr, flush=True)
        return NOT_RUN
    return 0


def default_cases(out: Path) -> list[Path]:
    """The committed cases, and the imported ones under `out` where there are any."""
    imported = out / "imported"
    return [CASES, imported] if imported.is_dir() else [CASES]


def logs_match(cases: list[Case], log_dir: Path) -> bool:
    """Whether `log_dir` holds exactly the logs of `cases` as they are now."""
    try:
        check_manifest(cases, log_dir)
    except (FileNotFoundError, ValueError):
        return False
    return True


def check_out(out: Path, repository: Path = REPOSITORY) -> str | None:
    """Why `out` is not a safe output directory, or None where it is.

    The output holds the judges' prompts and imported datasets, which must never be committed. So `out` may be anywhere outside `repository`, but inside it only where git ignores it and tracks nothing under it.
    """
    resolved = out.resolve()
    root = repository.resolve()
    if not resolved.is_relative_to(root):
        return None
    relative = resolved.relative_to(root).as_posix()
    if relative == ".":
        return f"--out {out} is the repository itself; choose a directory git ignores, such as benchmarks/auto_mode/out, or one outside the repository."
    try:
        ignored = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "check-ignore",
                "-q",
                "--no-index",
                f"{relative}/",
            ],
            capture_output=True,
        )
        tracked = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--", relative],
            capture_output=True,
            text=True,
        )
    except OSError as ex:
        return f"--out {out} is inside the repository, and git could not be run to check that it is ignored ({ex}); choose a directory outside the repository."
    if tracked.returncode == 0 and tracked.stdout.strip():
        return f"--out {out} holds files git tracks; choose a directory with none."
    if ignored.returncode != 0 or tracked.returncode != 0:
        return f"--out {out} is inside the repository but git does not ignore it (or could not tell), so its prompts and datasets could be committed; use benchmarks/auto_mode/out or a directory outside the repository."
    return None


def _pair(text: str) -> tuple[str, str]:
    name, sep, model = text.partition("=")
    if not sep or name not in MONITORS or not model:
        raise SystemExit(
            f"--model takes MONITOR=MODEL, with MONITOR one of {list(MONITORS)}; got {text!r}."
        )
    return name, model


if __name__ == "__main__":
    sys.exit(main())
