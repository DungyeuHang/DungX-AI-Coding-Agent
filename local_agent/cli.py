from __future__ import annotations

import argparse
import logging
import sys

from .config import AgentConfig, add_common_arguments, config_from_args
from .context import ContextSelector
from .orchestrator import Orchestrator
from .providers import MockProvider, ProviderError, build_provider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="Local-first autonomous AI software engineer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="inspect a local project")
    add_common_arguments(analyze)
    context = subparsers.add_parser("context", help="retrieve task-relevant project context")
    add_common_arguments(context)
    context.add_argument("task", help="task used to select relevant context")
    run = subparsers.add_parser("run", help="plan, implement, validate, repair, and review a task")
    add_common_arguments(run, include_provider_args=True)
    run.add_argument("task", help="implementation task objective or ID to resume")
    run.add_argument("--resume", action="store_true", help="treat 'task' argument as a task ID to resume") # Added resume flag

    scheduler = subparsers.add_parser("scheduler", help="manage and run the persistent task scheduler")
    add_common_arguments(scheduler)
    scheduler_sub = scheduler.add_subparsers(dest="scheduler_command", required=True)
    scheduler_run = scheduler_sub.add_parser("run-once", help="run one cycle of the scheduler loop")
    scheduler_status = scheduler_sub.add_parser("status", help="show the status of registered providers")

    providers = subparsers.add_parser("providers", help="manage provider configurations")
    providers_sub = providers.add_subparsers(dest="providers_command", required=True)
    providers_list = providers_sub.add_parser("list", help="list configured providers")
    providers_set = providers_sub.add_parser("set", help="add or update a provider configuration")
    providers_set.add_argument("provider_id", help="The ID of the provider (e.g., 'gemini')")
    providers_set.add_argument("--model", help="The model to use for this provider")
    providers_set.add_argument("--priority", type=int, help="The selection priority (lower is higher)")
    providers_set.add_argument("--enabled", type=lambda x: str(x).lower() in ('true', '1', 'yes'), help="Enable or disable the provider (true/false)")

    # New commands for Phase 3.12
    approve_plan = subparsers.add_parser("approve-plan", help="approve a generated task plan for execution")
    add_common_arguments(approve_plan)
    approve_plan.add_argument("task_id", help="ID of the task to approve")

    reject_plan = subparsers.add_parser("reject-plan", help="reject a generated task plan")
    add_common_arguments(reject_plan)
    reject_plan.add_argument("task_id", help="ID of the task to reject")

    edit_plan = subparsers.add_parser("edit-plan", help="edit a subtask within a generated task plan")
    add_common_arguments(edit_plan)
    edit_plan.add_argument("task_id", help="ID of the task to edit")
    edit_plan.add_argument("--subtask", required=True, help="ID of the subtask to edit")
    edit_plan.add_argument("--title", help="New title for the subtask")
    edit_plan.add_argument("--goal", help="New goal for the subtask")
    edit_plan.add_argument("--acceptance-criteria", nargs='*', help="New acceptance criteria (space-separated)")
    edit_plan.add_argument("--dependencies", nargs='*', help="New dependencies (space-separated subtask IDs)")

    # New commands for Phase 3.14
    approve_proposal = subparsers.add_parser("approve-proposal", help="approve an AI-generated plan modification proposal")
    add_common_arguments(approve_proposal)
    approve_proposal.add_argument("task_id", help="ID of the task whose proposal to approve")

    reject_proposal = subparsers.add_parser("reject-proposal", help="reject an AI-generated plan modification proposal")
    add_common_arguments(reject_proposal)
    reject_proposal.add_argument("task_id", help="ID of the task whose proposal to reject")

    # New commands for Phase 3.9
    create_task = subparsers.add_parser("create-task", help="create a new persistent task")
    add_common_arguments(create_task)
    create_task.add_argument("objective", help="the objective of the new task")

    list_tasks = subparsers.add_parser("list-tasks", help="list all persistent tasks")
    add_common_arguments(list_tasks)

    show_task = subparsers.add_parser("show-task", help="show details of a specific task")
    add_common_arguments(show_task)
    show_task.add_argument("task_id", help="ID of the task to show")

    credentials = subparsers.add_parser("credentials", help="securely manage provider API keys")
    credentials_sub = credentials.add_subparsers(dest="credentials_command", required=True)
    credentials_set = credentials_sub.add_parser("set", help="set the API key for a provider")
    credentials_set.add_argument("provider_id", help="The ID of the provider to set credentials for")
    credentials_delete = credentials_sub.add_parser("delete", help="delete the API key for a provider")
    credentials_delete.add_argument("provider_id", help="The ID of the provider to delete credentials for")

    return parser


def _print_context(context) -> None:
    print(f"Project: {context.root}")
    print(f"Stack: {', '.join(context.metadata.get('stacks', ['Unknown']))}")
    print(f"Files: {context.metadata.get('file_count', 0)} total, {len(context.source_files)} source, {len(context.test_files)} tests")
    print(f"Validation: {', '.join(command.display() for command in context.validation_commands) or 'none detected'}")
    print(f"Git status: {context.git_status or 'clean / not a Git repository'}")


def _print_selected_context(context) -> None:
    print("Selected context:")
    for path in context.metadata.get("selected_files", []):
        print(f"  - {path}")


def _print_report(report) -> None:
    print("\nPlan:")
    print(f"  Objective: {report.plan.objective if report.plan else 'none'}")
    if report.plan:
        for index, step in enumerate(report.plan.steps, 1):
            print(f"  {index}. {step}")
    if report.impact:
        print("\nImpact Analysis:")
        print(f"  Summary: {report.impact.summary}")
        sorted_targets = sorted(report.impact.targets, key=lambda t: (-t.confidence, t.path))
        for target in sorted_targets:
            if target.confidence > 0.3 and target.role != "unrelated":
                print(f"  - {target.path} ({target.role}, {target.risk} risk, {target.confidence:.0%} confidence)")
                if target.reason:
                    print(f"    Reason: {target.reason}")
    print(f"\nIterations: {report.iterations}")
    print(f"Changed files: {', '.join(sorted(set(report.changed_files))) or 'none'}")
    if report.validation_plan: # Added for ValidationIntelligence
        print("\nValidation Plan:")
        print(f"  Overall Risk: {report.validation_plan.risk_level.upper()}")
        print(f"  Reasons: {', '.join(report.validation_plan.reasons)}")
        print("  Primary Commands:")
        for cmd in report.validation_plan.primary_commands:
            print(f"    - {cmd.display()} (Category: {cmd.category}, Risk: {cmd.risk})")
        if report.validation_plan.secondary_commands:
            print("  Secondary Commands:")
            for cmd in report.validation_plan.secondary_commands:
                print(f"    - {cmd.display()} (Category: {cmd.category}, Risk: {cmd.risk})")
        if report.validation_plan.skipped_commands:
            print("  Skipped Commands (High Risk/Destructive):")
            for cmd in report.validation_plan.skipped_commands:
                print(f"    - {cmd.display()} (Category: {cmd.category}, Risk: {cmd.risk})")
    if report.dry_run or report.approval_required:
        print(f"Proposed diff:\n{report.proposed_diff or '(no changes proposed)'}")
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


def _approval_prompt(changes) -> bool:
    added = sum(line.startswith("+") and not line.startswith("+++") for change in changes for line in change.diff.splitlines())
    removed = sum(line.startswith("-") and not line.startswith("---") for change in changes for line in change.diff.splitlines())
    print(f"\nAI proposes:\n{len(changes)} files modified/created\n{added} lines added\n{removed} lines removed")
    try:
        return input("Apply these changes? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        logging.basicConfig(level=getattr(logging, config.log_level), format="%(levelname)s %(message)s")
        if args.command == "analyze":
            _print_context(Orchestrator(config, MockProvider()).analyze())
            return 0
        if args.command == "context":
            context = Orchestrator(config, MockProvider()).analyze()
            ContextSelector(
                config.project,
                max_files=config.max_context_files,
                max_chars=config.planning_context_bytes,
                max_file_chars=config.max_context_file_bytes,
                max_tokens=config.max_context_tokens,
                dependency_depth=config.dependency_depth,
            ).select(args.task, context)
            _print_selected_context(context)
            return 0
        provider = build_provider(config)
        print(f"Provider: {config.provider}")
        print(f"Model: {config.model}")
        print(f"Project: {config.project}")
        report = Orchestrator(config, provider).run(args.task, progress=print, approval_callback=_approval_prompt)
        _print_report(report)
        return 0 if report.completed else 2
    except (ValueError, ProviderError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
