from __future__ import annotations

import argparse
import logging
import sys

from .config import AgentConfig, add_common_arguments, config_from_args
from .orchestrator import Orchestrator
from .providers import MockProvider, ProviderError, build_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="Local-first autonomous AI software engineer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="inspect a local project")
    add_common_arguments(analyze)
    run = subparsers.add_parser("run", help="plan, implement, validate, repair, and review a task")
    add_common_arguments(run)
    run.add_argument("task", help="implementation task")
    return parser


def _print_context(context) -> None:
    print(f"Project: {context.root}")
    print(f"Stack: {', '.join(context.metadata.get('stacks', ['Unknown']))}")
    print(f"Files: {context.metadata.get('file_count', 0)} total, {len(context.source_files)} source, {len(context.test_files)} tests")
    print(f"Validation: {', '.join(command.display() for command in context.validation_commands) or 'none detected'}")
    print(f"Git status: {context.git_status or 'clean / not a Git repository'}")


def _print_report(report) -> None:
    print("\nPlan:")
    print(f"  Objective: {report.plan.objective if report.plan else 'none'}")
    if report.plan:
        for index, step in enumerate(report.plan.steps, 1):
            print(f"  {index}. {step}")
    print(f"\nIterations: {report.iterations}")
    print(f"Changed files: {', '.join(sorted(set(report.changed_files))) or 'none'}")
    for result in report.executions:
        status = "PASS" if result.succeeded else "FAIL"
        print(f"{status}: {result.command} (exit {result.exit_code}, {result.duration_seconds:.2f}s)")
        if not result.succeeded and result.stderr:
            print(result.stderr[-2000:].rstrip())
    if report.failures:
        for failure in report.failures:
            print(f"Failure analysis: {failure.probable_root_cause}")
            print(f"Recommended fix: {failure.recommended_fix}")
    if report.review:
        print(f"Review: {report.review.verdict} - {report.review.summary}")
        for finding in report.review.findings:
            print(f"  - {finding}")
    print(f"Result: {'COMPLETE' if report.completed else 'INCOMPLETE'}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        logging.basicConfig(level=getattr(logging, config.log_level), format="%(levelname)s %(message)s")
        if args.command == "analyze":
            _print_context(Orchestrator(config, MockProvider()).analyze())
            return 0
        provider = build_provider(config)
        report = Orchestrator(config, provider).run(args.task, progress=print)
        _print_report(report)
        return 0 if report.completed else 2
    except (ValueError, ProviderError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
