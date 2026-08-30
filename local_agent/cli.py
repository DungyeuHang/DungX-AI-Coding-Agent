from __future__ import annotations

import argparse
import datetime
import logging
import json
import sys
from pathlib import Path

from .config import AgentConfig, add_common_arguments, config_from_args
from .context import ContextSelector
from .orchestrator import Orchestrator
from .providers import MockProvider, ProviderError, build_provider
from .repository import RepositoryIntelligence
from .storage import JsonFileStorage
from .models import TaskStatus
from .planner import GraphValidator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent", description="Local-first autonomous AI software engineer")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="inspect a local project")
    add_common_arguments(analyze)
    context = subparsers.add_parser("context", help="retrieve task-relevant project context")
    add_common_arguments(context)
    context.add_argument("task", help="task used to select relevant context")
    context.add_argument("--verbose", action="store_true", help="display detailed selection metadata")
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

    doctor = subparsers.add_parser("doctor", help="run a health check on the agent and project environment")
    add_common_arguments(doctor)

    # New for Phase 3.23
    ci_repair = subparsers.add_parser("ci-repair", help="create an autonomous task to repair a CI failure")
    add_common_arguments(ci_repair)
    ci_repair.add_argument("--failure-file", required=True, help="path to a JSON file with CI failure details")

    # New for Phase 3.24
    commit_task = subparsers.add_parser("commit-task", help="commit the changes from a completed task to a new branch")
    add_common_arguments(commit_task)
    commit_task.add_argument("task_id", help="ID of the completed task to commit")
    commit_task.add_argument("--branch-name", help="optional name for the new branch")

    push_task = subparsers.add_parser("push-task", help="push a committed task branch to the remote")
    add_common_arguments(push_task)
    push_task.add_argument("task_id", help="ID of the committed task to push")

    create_pr = subparsers.add_parser("create-pr", help="create a pull request for a pushed task branch")
    add_common_arguments(create_pr)
    create_pr.add_argument("task_id", help="ID of the pushed task to create a PR for")

    show_config = subparsers.add_parser("show-config", help="show the current agent configuration")
    add_common_arguments(show_config)

    credentials = subparsers.add_parser("credentials", help="securely manage provider API keys")
    credentials_sub = credentials.add_subparsers(dest="credentials_command", required=True)
    credentials_set = credentials_sub.add_parser("set", help="set the API key for a provider")
    credentials_set.add_argument("provider_id", help="The ID of the provider to set credentials for")
    credentials_delete = credentials_sub.add_parser("delete", help="delete the API key for a provider")
    credentials_delete.add_argument("provider_id", help="The ID of the provider to delete credentials for")

    # Phase 4.20: diagnostic surface over the validation intelligence stores.
    # Read-only by construction: every subcommand loads a store and prints it.
    # None of them can record, finalize, calibrate or otherwise mutate anything.
    validation = subparsers.add_parser(
        "validation", help="inspect validation intelligence telemetry and lifecycle history"
    )
    validation_sub = validation.add_subparsers(dest="validation_command", required=True)
    for name, description in (
        ("health", "show validation intelligence health diagnostics"),
        ("history", "list recorded validation lifecycles"),
        ("defects", "show recurring defect signatures and repair effectiveness"),
        ("calibration", "show shadow-mode calibration and evidence-type reliability"),
        ("recommendations", "show the advisory scope recommendation and its safety floor"),
    ):
        sub = validation_sub.add_parser(name, help=description)
        add_common_arguments(sub)
        sub.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    validation_lifecycle = validation_sub.add_parser(
        "lifecycle", help="show one lifecycle trace, including its repair lineage"
    )
    add_common_arguments(validation_lifecycle)
    validation_lifecycle.add_argument("lifecycle_id", help="ID of the lifecycle to show")
    validation_lifecycle.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    # Phase 4.21: the maintenance operator surface.
    #
    # ``scan``/``dry-run`` are read-only by construction. ``run`` is the only
    # subcommand that can reach an executing tier, and even then only when the
    # configured autonomy ceiling allows it *and* the policy grants it for the
    # individual candidate.
    maintenance = subparsers.add_parser(
        "maintenance", help="discover, rank and act on repository maintenance opportunities"
    )
    maintenance_sub = maintenance.add_subparsers(dest="maintenance_command", required=True)
    for name, description in (
        ("scan", "discover maintenance candidates from real repository intelligence"),
        ("health", "summarise maintenance state and advisory actionability statistics"),
        ("history", "list previous maintenance runs and what they accomplished"),
        ("candidates", "list known maintenance candidates and their lifecycle state"),
        ("recommendations", "show ranked recommendations with their priority explanations"),
        ("status", "show the most recent maintenance run and what remains unresolved"),
        ("dry-run", "score, select and plan candidates without executing anything"),
    ):
        sub = maintenance_sub.add_parser(name, help=description)
        add_common_arguments(sub)
        sub.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    maintenance_run = maintenance_sub.add_parser(
        "run", help="run the full maintenance lifecycle, subject to the configured autonomy tier"
    )
    add_common_arguments(maintenance_run)
    maintenance_run.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    # ``--dry-run`` is already supplied by ``add_common_arguments`` with
    # compatible semantics ("produce the work, write nothing"), so it is reused
    # here rather than redefined.
    maintenance_run.add_argument(
        "--enqueue",
        action="store_true",
        help=(
            "create an ordinary pending Task for each planned work order so the "
            "existing scheduler executes it through the normal pipeline"
        ),
    )
    # Phase 4.22. Two separate opt-ins, deliberately: --execute wires the narrow
    # executor so a supported candidate can be implemented and prospectively
    # validated, and --apply additionally permits the authoritative write. Neither
    # is implied by maintenance discovery being enabled, and --apply without
    # --execute does nothing.
    maintenance_run.add_argument(
        "--execute",
        action="store_true",
        help=(
            "wire the narrow maintenance executor so supported candidates are "
            "implemented and prospectively validated in an isolated workspace "
            "(nothing is written to the repository without --apply)"
        ),
    )
    maintenance_run.add_argument(
        "--apply",
        action="store_true",
        help=(
            "permit an executed, prospectively-validated maintenance change to be "
            "written to the repository through the existing approval and apply "
            "pipeline; requires --execute"
        ),
    )
    maintenance_candidate = maintenance_sub.add_parser(
        "candidate", help="show one maintenance candidate in full, including its evidence"
    )
    add_common_arguments(maintenance_candidate)
    maintenance_candidate.add_argument("candidate_id", help="ID of the candidate to show")
    maintenance_candidate.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    return parser


def _print_context(context) -> None:
    print(f"Project: {context.root}")
    print(f"Stack: {', '.join(context.metadata.get('stacks', ['Unknown']))}")
    print(f"Files: {context.metadata.get('file_count', 0)} total, {len(context.source_files)} source, {len(context.test_files)} tests")
    print(f"Validation: {', '.join(command.display() for command in context.validation_commands) or 'none detected'}")
    print(f"Git status: {context.git_status or 'clean / not a Git repository'}")


def _print_selected_context(context, args) -> None:
    print("Selected context:")
    selection_metadata = context.metadata.get("context_selection", {})
    if not selection_metadata:
        print("  No files selected.")
        return

    print(f"  - Keyword Count: {selection_metadata.get('keyword_count', 'N/A')}")
    print(f"  - Candidate Files: {selection_metadata.get('candidate_count', 'N/A')}")
    print(f"  - Selected Files: {selection_metadata.get('selected_count', 'N/A')}")
    print(f"  - Estimated Tokens: {selection_metadata.get('estimated_tokens', 'N/A')}")

    selected_files = context.metadata.get("selected_files", [item["path"] for item in selection_metadata.get("selected_items", [])])
    if selected_files:
        print("  Files:")
        for path in selected_files:
            print(f"    - {path}")

    if getattr(args, "verbose", False):
        print("\n  Selected Items (ranked):")
        for item in selection_metadata.get('selected_items', []):
            print(f"    - Path: {item['path']} (Score: {item['score']:.3f}, Depth: {item['dependency_depth']})")
            print(f"      Reasons: {', '.join(item['reason'])}")

        print("\n  Excluded Items:")
        for item in context.metadata.get('context_excluded', []):
            print(f"    - Path: {item['path']} (Reason: {item['reason']})")


def _print_task_details(task):
    """Prints a detailed view of a single task."""
    print(f"Task ID: {task.task_id}")
    print(f"Status: {task.status.value}")
    print(f"Objective: {task.objective}")
    print(f"Created: {task.created_at.isoformat()}")
    print(f"Updated: {task.updated_at.isoformat()}")
    if task.outcome:
        print(f"Outcome: {task.outcome}")

    if task.plan:
        print("\nTask Plan:")
        print(f"  Objective: {task.plan.objective}")
        if task.plan.subtasks:
            print("  Subtasks:")
            for subtask in sorted(task.plan.subtasks, key=lambda s: s.created_at or datetime.datetime.min):
                deps = f" (depends on: {', '.join(subtask.dependencies)})" if subtask.dependencies else ""
                print(f"    - {subtask.title} (ID: {subtask.subtask_id}, Status: {subtask.status.value}){deps}")
                print(f"      Goal: {subtask.goal}")
                if subtask.acceptance_criteria:
                    print(f"      Acceptance Criteria: {', '.join(subtask.acceptance_criteria)}")
    
    if task.plan_proposal:
        print("\nPlan Proposal (awaiting approval):")
        print(f"  Reason: {task.plan_proposal.reason}")
        if task.plan_proposal.modifications:
            print("  Modifications:")
            # In a real CLI, you'd format this better
            print(f"    {json.dumps([m.to_dict() for m in task.plan_proposal.modifications], indent=2)}")
        if task.plan_proposal.additions:
            print("  Additions:")
            print(f"    {json.dumps([a.to_dict() for a in task.plan_proposal.additions], indent=2)}")

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


def _handle_validation_command(args, config, storage) -> int:
    """Phase 4.20 diagnostics. Read-only over the two bounded stores.

    Every path here must survive an empty or brand-new repository: the stores
    both return a well-formed empty instance when their file does not exist, so
    the only extra work is saying "no data yet" instead of printing zeros as if
    they were measurements.
    """
    from .validation_lifecycle import (
        AdaptiveValidationRecommender,
        compute_repair_effectiveness,
    )
    from .validation_telemetry import compute_health

    command = getattr(args, "validation_command", "")
    as_json = bool(getattr(args, "json", False))
    min_samples = getattr(config, "validation_calibration_min_samples", 20)
    lifecycle_min_samples = getattr(config, "validation_lifecycle_min_samples", 10)
    telemetry = storage.load_validation_telemetry()
    lifecycles = storage.load_validation_lifecycle()

    def _emit(payload: dict) -> None:
        print(json.dumps(payload, indent=2, default=str))

    if command == "health":
        health = compute_health(telemetry, min_samples=min_samples)
        effectiveness = compute_repair_effectiveness(
            lifecycles, min_samples=lifecycle_min_samples
        )
        if as_json:
            _emit({
                "telemetry": health.to_dict(),
                "lifecycle": effectiveness.to_dict(),
            })
            return 0
        print("Validation Intelligence Health")
        print("=" * 30)
        print(f"  Decisions recorded:      {health.total_decisions}")
        print(f"  Observations:            {health.total_observations} "
              f"({health.resolved_observations} resolved)")
        print(f"  Calibration status:      {health.calibration_status}")
        print(f"  Scope counts:            {health.scope_counts or 'none'}")
        print(f"  Broad validation rate:   {health.broad_validation_rate:.2%}")
        print(f"  Reuse hit rate:          {health.reuse_hit_rate:.2%}")
        print(f"  False-confidence events: {health.false_confidence_incidents}")
        print(f"  Analysis degradation:    {health.analysis_degradation_rate:.2%}")
        print(f"  Corrupted records:       {health.corrupted_records_skipped}")
        print(f"\n  Lifecycles recorded:     {effectiveness.lifecycles} "
              f"({effectiveness.resolved} resolved)")
        if effectiveness.insufficient_data:
            print(f"  NOTE: fewer than {effectiveness.min_samples} resolved lifecycle(s); "
                  "rates below are not established.")
        print(f"  First-pass success:      {effectiveness.first_pass_success_rate:.2%} "
              f"(conservative lower bound {effectiveness.first_pass_success_lower_bound:.2f})")
        print(f"  Repair success:          {effectiveness.repair_success_rate:.2%} "
              f"(conservative lower bound {effectiveness.repair_success_lower_bound:.2f})")
        if not effectiveness.history_trustworthy:
            print("  WARNING: lifecycle history contains records that failed to load; "
                  "treated as no history.")
        if health.total_decisions == 0 and effectiveness.lifecycles == 0:
            print("\n  No validation history recorded yet for this repository.")
        return 0

    if command == "history":
        records = lifecycles.lifecycles
        if as_json:
            _emit({
                "lifecycles": [r.to_dict() for r in records],
                "corrupted_records_skipped": lifecycles.corrupted_records_skipped,
            })
            return 0
        if not records:
            print("No validation lifecycles recorded yet.")
            return 0
        print(f"{len(records)} validation lifecycle(s), oldest first:")
        for record in records:
            print(f"- {record.lifecycle_id} [{record.state}] task={record.task_id or '-'} "
                  f"subtask={record.subtask_id or '-'} iterations={len(record.iterations)} "
                  f"repairs={record.repair_count} updated={record.updated_at}")
        if lifecycles.corrupted_records_skipped:
            print(f"\nWARNING: {lifecycles.corrupted_records_skipped} record(s) could not be "
                  "loaded and were skipped.")
        return 0

    if command == "lifecycle":
        record = lifecycles.find(args.lifecycle_id)
        if record is None:
            print(f"error: no lifecycle with ID {args.lifecycle_id}", file=sys.stderr)
            return 1
        if as_json:
            _emit(record.to_dict())
            return 0
        print(f"Lifecycle: {record.lifecycle_id}")
        print(f"  State:     {record.state}")
        print(f"  Task:      {record.task_id or '-'} / subtask {record.subtask_id or '-'}")
        print(f"  Created:   {record.created_at}")
        print(f"  Updated:   {record.updated_at}")
        print(f"  Outcome:   {record.terminal_outcome or 'in progress'} "
              f"({record.failure_category or 'n/a'})")
        print("\n  State history:")
        for entry in record.state_history:
            print(f"    - {entry.get('state', '?')} at {entry.get('at', '?')}"
                  f"{': ' + entry['reason'] if entry.get('reason') else ''}")
        print(f"\n  Iterations ({len(record.iterations)}):")
        for iteration in record.iterations:
            parent = iteration.parent_iteration_id or "-"
            print(f"    #{iteration.iteration_number} [{iteration.kind}] "
                  f"{iteration.iteration_id} parent={parent}")
            print(f"        scope={iteration.scope or '-'} "
                  f"stage={iteration.validation_stage} "
                  f"result={iteration.validation_result} "
                  f"duration={iteration.duration_seconds:.2f}s")
            if iteration.defect_signature is not None:
                print(f"        defect={iteration.defect_signature.fingerprint} "
                      f"{iteration.defect_signature.describe()[:100]}")
        recurring = record.recurring_defects()
        if recurring:
            print("\n  Recurring defects within this lifecycle:")
            for fingerprint, count in sorted(recurring.items()):
                print(f"    - {fingerprint}: {count} occurrence(s)")
        return 0

    if command == "defects":
        effectiveness = compute_repair_effectiveness(
            lifecycles, min_samples=lifecycle_min_samples
        )
        if as_json:
            _emit(effectiveness.to_dict())
            return 0
        if effectiveness.lifecycles == 0:
            print("No validation lifecycles recorded yet; no defect history to report.")
            return 0
        print("Repair Effectiveness and Defect Recurrence")
        print("=" * 42)
        if effectiveness.insufficient_data:
            print(f"NOTE: only {effectiveness.resolved} resolved lifecycle(s), below the "
                  f"minimum of {effectiveness.min_samples}. Rates are reported but are "
                  "NOT established.")
        print(f"  Lifecycles:              {effectiveness.lifecycles} "
              f"({effectiveness.resolved} resolved, {effectiveness.in_progress} in progress)")
        print(f"  Completed/abandoned/failed: {effectiveness.completed}/"
              f"{effectiveness.abandoned}/{effectiveness.failed}")
        print(f"  Repairs (total):         {effectiveness.total_repair_iterations}")
        print(f"  Median repairs (when needed): {effectiveness.median_repair_iterations}")
        print(f"  Abandonment rate:        {effectiveness.abandonment_rate:.2%} "
              f"(pessimistic upper bound {effectiveness.abandonment_rate_upper_bound:.2f})")
        print(f"  Repeated-defect rate:    {effectiveness.repeated_defect_rate:.2%} "
              f"(pessimistic upper bound {effectiveness.repeated_defect_rate_upper_bound:.2f})")
        print(f"  Defects by stage:        {effectiveness.stage_distribution or 'none'}")
        print(f"  Candidate vs post-apply: {effectiveness.candidate_stage_defects} vs "
              f"{effectiveness.post_apply_defects}")
        if effectiveness.measured_duration_samples:
            print(f"  Validation duration:     mean "
                  f"{effectiveness.mean_validation_seconds:.2f}s / median "
                  f"{effectiveness.median_validation_seconds:.2f}s over "
                  f"{effectiveness.measured_duration_samples} measured sample(s)")
        else:
            print("  Validation duration:     NOT MEASURED (no iteration recorded a duration)")
        if effectiveness.top_recurring_defects:
            print("\n  Top recurring defect signatures:")
            for defect in effectiveness.top_recurring_defects:
                print(f"    - {defect.fingerprint} x{defect.occurrences} "
                      f"across {defect.lifecycles} lifecycle(s)")
                if defect.description:
                    print(f"      {defect.description[:140]}")
        else:
            print("\n  No defect signatures recorded.")
        return 0

    if command == "calibration":
        health = compute_health(telemetry, min_samples=min_samples)
        if as_json:
            _emit({
                "calibration_status": health.calibration_status,
                "shadow_comparisons": health.shadow_comparisons,
                "shadow_would_narrow": health.shadow_would_narrow,
                "shadow_would_broaden": health.shadow_would_broaden,
                "shadow_safety_overrides": health.shadow_safety_overrides,
                "calibration_drift": health.calibration_drift,
                "evidence_type_reliability": {
                    k: v.to_dict() for k, v in health.evidence_type_reliability.items()
                },
                "quality": health.quality.to_dict(),
                "min_samples": min_samples,
                "calibration_enabled": getattr(config, "validation_calibration_enabled", False),
                "live_calibration_implemented": False,
            })
            return 0
        print("Validation Calibration (SHADOW MODE ONLY)")
        print("=" * 40)
        print("  No calibrated confidence is ever applied to a real validation decision;")
        print("  there is no live-calibration code path in this build.")
        print(f"\n  Status:                {health.calibration_status}")
        print(f"  Minimum samples:       {min_samples}")
        print(f"  Shadow comparisons:    {health.shadow_comparisons}")
        print(f"  Would narrow/broaden:  {health.shadow_would_narrow}/"
              f"{health.shadow_would_broaden}")
        print(f"  Safety-floor overrides:{health.shadow_safety_overrides}")
        print(f"  Mean |confidence drift|: {health.calibration_drift:.4f}")
        if health.evidence_type_reliability:
            print("\n  Evidence-type reliability (Wilson lower bound, conservative):")
            for name in sorted(health.evidence_type_reliability):
                item = health.evidence_type_reliability[name]
                flag = "" if item.sufficient_data else "  [INSUFFICIENT DATA]"
                print(f"    - {name}: {item.successes}/{item.trials} "
                      f"lower_bound={item.lower_bound:.3f}{flag}")
        else:
            print("\n  No reliability data recorded yet.")
        print(f"\n  Observed targeted escape rate: {health.quality.observed_escape_rate:.2%} "
              f"(pessimistic upper bound "
              f"{health.quality.observed_escape_rate_upper_bound:.2f})")
        print("  Recall is not reported: the data cannot support it.")
        return 0

    if command == "recommendations":
        # No impact analysis has been run in this process, so there is no
        # change-specific floor to hand the recommender. BROAD is the only
        # honest floor for "we have not analysed anything": a diagnostic
        # command must never display a narrower floor than a real run would
        # compute, or the output would read as permission the analysis never
        # actually gave.
        recommendation = AdaptiveValidationRecommender(
            min_samples=lifecycle_min_samples
        ).recommend(safety_floor="broad", store=lifecycles)
        if as_json:
            _emit(recommendation.to_dict())
            return 0
        print("Adaptive Validation Recommendation (ADVISORY)")
        print("=" * 45)
        print("  This is not a decision. ValidationDecisionEngine remains the only")
        print("  authority; the effective scope below is never narrower than the floor.")
        print(f"\n  Safety floor:      {recommendation.safety_floor}")
        print(f"  Recommended scope: {recommendation.recommended_scope}")
        print(f"  Effective scope:   {recommendation.effective_scope}")
        print(f"  Conflicts w/ floor:{recommendation.conflicts_with_floor}")
        print(f"  Data sufficient:   {recommendation.data_sufficient} "
              f"({recommendation.samples} resolved lifecycle(s))")
        print(f"  History trusted:   {recommendation.history_trustworthy}")
        print("\n  Safety reasons:")
        for reason in recommendation.safety_reasons:
            print(f"    - {reason}")
        print("\n  Recommendation reasons:")
        for reason in recommendation.reasons:
            print(f"    - {reason}")
        return 0

    print(f"error: unknown validation subcommand '{command}'", file=sys.stderr)
    return 1


def _build_maintenance_executor(config, storage, *, apply_enabled=False, progress=None):
    """Wire the Phase 4.22 narrow maintenance executor.

    Everything it needs is a live object: the real provider factory, the real
    execution policy, the real lifecycle/telemetry managers and a real
    repository context. The executor itself does no implementation - it hands
    the work to :class:`~local_agent.coding_agent.InteractiveCodingAgent` and
    reads back what the existing validation authorities decided.

    ``apply_enabled`` is the operator's explicit ``--apply``. Without it the
    executor still runs the full implement -> prospective-validate chain, and
    then refuses the authoritative write with ``approval_required``.
    """
    from .approval import ApprovalPolicyEngine
    from .maintenance import MaintenanceBudget
    from .maintenance_execution import (
        ExecutionJournal,
        MaintenanceApprovalGate,
        MaintenanceExecutor,
    )
    from .maintenance_policy import MaintenanceExecutionPolicy
    from .models import ApprovalPolicy
    from .providers import build_provider
    from .validation_lifecycle import ValidationLifecycleManager
    from .validation_telemetry import ValidationTelemetryManager

    data_dir = Path(getattr(config, "data_dir", None) or (config.project / ".agent_data"))
    approval_engine = ApprovalPolicyEngine(
        [ApprovalPolicy.from_dict(entry) for entry in getattr(config, "approval_policies", [])]
    )
    return MaintenanceExecutor(
        root=config.project,
        provider_factory=lambda: build_provider(config),
        policy=MaintenanceExecutionPolicy(repository_root=config.project),
        budget=MaintenanceBudget.from_config(config),
        configured_tier=getattr(config, "maintenance_autonomy_tier", "observe_only"),
        journal=ExecutionJournal(data_dir / "maintenance_execution"),
        approval_gate=MaintenanceApprovalGate(
            approval_mode=getattr(config, "approval", "never"),
            policy_engine=approval_engine,
            approver=None,
            apply_enabled=bool(apply_enabled),
        ),
        context_provider=lambda: RepositoryIntelligence(config.project).scan(),
        lifecycle_manager=ValidationLifecycleManager(storage, config.project),
        telemetry_manager=ValidationTelemetryManager(storage, config.project),
        workspace_parent=data_dir / "maintenance_workspaces",
        progress=progress,
    )


def _build_maintenance_runner(
    config, storage, *, progress=None, executor=None
):
    """Wire a :class:`MaintenanceRunner` against this repository's real state.

    Every source is a live one: the persisted lifecycle and telemetry stores,
    a freshly-built semantic graph, the knowledge graph, and real ``git log``
    churn. Nothing is stubbed, and each provider is resolved lazily on every
    scan so that reassessment genuinely re-reads the repository rather than
    re-reporting a cached view.

    ``executor`` defaults to ``None``, which is still the shipped default: no
    maintenance run modifies anything unless the operator explicitly asks for
    the Phase 4.22 executor with ``maintenance run --execute``.
    """
    from .evidence import compute_state_fingerprint
    from .git import GitIntegration
    from .maintenance import MaintenanceBudget
    from .maintenance_analysis import MaintenanceAnalyzer, collect_churn
    from .maintenance_policy import MaintenanceExecutionPolicy, MaintenancePriorityEngine
    from .maintenance_runner import MaintenanceManager, MaintenanceRunner, build_scan_function
    from .semantic_impact import SemanticGraph

    analyzer = MaintenanceAnalyzer(config.project)
    git = GitIntegration(config.project)
    scan = build_scan_function(
        analyzer,
        lifecycle_provider=storage.load_validation_lifecycle,
        telemetry_provider=storage.load_validation_telemetry,
        graph_provider=lambda: SemanticGraph.build(config.project),
        knowledge_provider=storage.load_knowledge_graph,
        churn_provider=lambda: collect_churn(git),
    )
    manager = MaintenanceManager(
        storage,
        config.project,
        max_candidates=getattr(config, "maintenance_retention", 300),
        max_runs=getattr(config, "maintenance_run_retention", 50),
    )
    return MaintenanceRunner(
        analyzer=analyzer,
        manager=manager,
        scan=scan,
        budget=MaintenanceBudget.from_config(config),
        policy=MaintenanceExecutionPolicy(repository_root=config.project),
        priority_engine=MaintenancePriorityEngine(),
        # Phase 4.22: this is the executor seam, and it stays empty unless the
        # operator asked for it. `maintenance run --enqueue` remains the other
        # real integration - ordinary tasks for the existing scheduler.
        executor=executor,
        configured_tier=getattr(config, "maintenance_autonomy_tier", "observe_only"),
        progress=progress,
        fingerprint_fn=lambda paths: compute_state_fingerprint(config.project, paths),
    ), manager


def _candidates_with_live_tasks(storage) -> set[str]:
    """Maintenance candidate ids that already have a non-terminal task.

    Reads the provenance stamp ``_enqueue_work_orders`` writes into the task's
    execution history. A storage backend that cannot list tasks, or a task
    whose history is malformed, contributes nothing: failing to detect a
    duplicate costs an extra queued task, whereas failing the whole run over an
    unreadable record would cost the operator the scan.
    """
    from .models import TaskStatus as _TaskStatus

    live_states = {
        _TaskStatus.PENDING,
        _TaskStatus.RUNNING,
        _TaskStatus.PAUSED,
        _TaskStatus.PLAN_REVIEW,
        _TaskStatus.PLAN_PROPOSED,
    }
    outstanding: set[str] = set()
    try:
        tasks = storage.list_tasks()
    except Exception:
        return outstanding
    for task in tasks or []:
        if getattr(task, "status", None) not in live_states:
            continue
        for entry in getattr(task, "execution_history", None) or []:
            if not isinstance(entry, dict) or entry.get("source") != "maintenance":
                continue
            candidate_id = entry.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                outstanding.add(candidate_id)
    return outstanding


def _enqueue_work_orders(result, storage) -> list[str]:
    """Turn planned work orders into ordinary pending tasks.

    This is the integration point with the existing execution architecture:
    the created tasks are indistinguishable from ones a human created with
    ``agent create-task``, and are planned, routed, implemented, validated,
    approved and applied by exactly the same machinery. The maintenance layer
    contributes an objective and a scope statement; it contributes no
    execution, no validation and no approval behaviour of its own.
    """
    import uuid as _uuid

    from .models import Task, TaskStatus as _TaskStatus

    # A maintenance signal persists until something fixes it, so every run
    # re-discovers it and re-plans the same work order. Without this check a
    # scheduled `maintenance run --enqueue` - which is the whole point of a
    # *continuous* maintenance loop - would add another identical pending task
    # on every tick, and the queue would grow without bound while nothing new
    # was ever proposed. A candidate that already has live work outstanding is
    # skipped; once that task reaches a terminal state the candidate becomes
    # enqueueable again, which is what makes a genuinely recurring problem
    # recur.
    outstanding = _candidates_with_live_tasks(storage)

    created: list[str] = []
    for candidate_id in sorted(result.work_orders):
        if candidate_id in outstanding:
            continue
        order = result.work_orders[candidate_id]
        now = datetime.datetime.now(datetime.timezone.utc)
        objective = order.objective
        if order.scope_files:
            objective += " (scope: " + ", ".join(order.scope_files[:5]) + ")"
        task = Task(
            task_id=str(_uuid.uuid4()),
            objective=objective,
            status=_TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        task.execution_history.append(
            {
                "source": "maintenance",
                "candidate_id": candidate_id,
                "work_order": order.to_dict(),
                "at": now.isoformat(),
            }
        )
        storage.save_task(task)
        created.append(task.task_id)
    return created


def _execution_payload(executor, want_execute: bool, want_apply: bool) -> dict:
    """Machine-readable account of what the executor actually did."""
    return {
        "executor_wired": bool(executor is not None),
        "execute_requested": bool(want_execute),
        "apply_permitted": bool(want_apply),
        "results": [
            entry.to_dict() for entry in getattr(executor, "results", []) or []
        ],
    }


def _print_execution_report(executor, want_execute: bool, want_apply: bool, result) -> None:
    """Report execution honestly, never conflating the seven distinct states.

    DISCOVERED, ELIGIBLE, PLANNED, ATTEMPTED, APPLIED, VALIDATED and RESOLVED
    are separate facts and are printed separately. In particular ``resolved`` is
    reported from the runner's own reassessment - a fresh scan that no longer
    produces the signal - and never inferred from a successful apply.
    """
    if executor is None:
        if want_execute:
            print("\n  Execution was requested but no executor could be wired.")
            return
        print("\n  Nothing was executed. Add --execute to wire the narrow maintenance "
              "executor (parse-failure signals only), or --enqueue to create ordinary "
              "pending tasks that the existing scheduler will execute through the "
              "normal pipeline.")
        return

    results = list(getattr(executor, "results", []) or [])
    print("\n  Narrow maintenance execution (Phase 4.22)")
    print(f"    authoritative apply permitted: {'yes' if want_apply else 'no (--apply not given)'}")
    if not results:
        print("    ATTEMPTED: 0 - no candidate reached the executor. A candidate must be")
        print("      a supported signal, eligible under the policy at an executing tier,")
        print("      and planned into a work order before it is attempted.")
        return
    applied = sum(1 for entry in results if entry.applied and not entry.rolled_back)
    validated = sum(1 for entry in results if entry.validation_passed is True)
    resolved = sum(
        1
        for verdict in result.reassessments.values()
        if verdict.outcome == "resolved"
    )
    print(f"    ATTEMPTED: {len(results)}   APPLIED: {applied}   "
          f"VALIDATED: {validated}   RESOLVED (by rescan): {resolved}")
    for entry in results:
        print(f"    - {entry.candidate_id} [{entry.signal_kind}] -> {entry.status}")
        print(f"        applied={entry.applied} rolled_back={entry.rolled_back} "
              f"post-apply validation={_verdict_text(entry.validation_passed)}")
        if entry.changed_files:
            print(f"        changed: {', '.join(entry.changed_files)}")
        if entry.post_apply_commands:
            print("        validation commands actually executed: "
                  + "; ".join(" ".join(command) for command in entry.post_apply_commands))
        for reason in entry.reasons[:6]:
            print(f"        note: {reason}")
        for error in entry.errors[:4]:
            print(f"        error: {error}")


def _verdict_text(value) -> str:
    if value is True:
        return "PASSED"
    if value is False:
        return "FAILED"
    return "no verdict (never assumed to be a pass)"


def _handle_maintenance_command(args, config, storage) -> int:
    """Phase 4.21 operator surface.

    ``scan``, ``dry-run`` and every read-only subcommand can be executed
    safely on any repository at any time: they read, rank and report. ``run``
    is the only one that consults the autonomy ceiling, and this build has no
    executor wired, so the strongest thing it can do is create pending tasks
    for the existing scheduler when ``--enqueue`` is given.
    """
    from .maintenance import RUN_MODE_DRY_RUN, RUN_MODE_EXECUTE, RUN_MODE_SCAN, summarize_candidates
    from .maintenance_policy import describe_tier
    from .maintenance_runner import MaintenanceManager, compute_actionability

    command = getattr(args, "maintenance_command", "")
    as_json = bool(getattr(args, "json", False))
    enabled = bool(getattr(config, "maintenance_enabled", False))
    min_samples = getattr(config, "maintenance_min_samples", 5)

    def _emit(payload: dict) -> None:
        print(json.dumps(payload, indent=2, default=str))

    manager = MaintenanceManager(
        storage,
        config.project,
        max_candidates=getattr(config, "maintenance_retention", 300),
        max_runs=getattr(config, "maintenance_run_retention", 50),
    )

    # -- read-only reporting over persisted state --------------------------

    if command in {"health", "history", "candidates", "status", "candidate"}:
        store = manager.load()
        if command == "health":
            actionability = compute_actionability(store, min_samples=min_samples)
            summary = summarize_candidates(store.candidates)
            if as_json:
                _emit({
                    "enabled": enabled,
                    "autonomy_tier": getattr(config, "maintenance_autonomy_tier", ""),
                    "candidates": summary,
                    "runs_recorded": len(store.runs),
                    "history_trustworthy": store.history_trustworthy(),
                    "corrupted_records_skipped": store.corrupted_records_skipped,
                    "actionability": actionability,
                })
                return 0
            print("Maintenance Health")
            print("=" * 18)
            print(f"  Subsystem enabled:     {enabled}")
            print(f"  Autonomy ceiling:      {getattr(config, 'maintenance_autonomy_tier', '')} "
                  f"({describe_tier(getattr(config, 'maintenance_autonomy_tier', ''))})")
            print(f"  Known candidates:      {summary['total']}")
            print(f"  By severity:           {summary['by_severity'] or 'none'}")
            print(f"  By state:              {summary['by_state'] or 'none'}")
            print(f"  By outcome:            {summary['by_outcome'] or 'none'}")
            print(f"  Runs recorded:         {len(store.runs)}")
            if store.corrupted_records_skipped:
                print(f"  WARNING: {store.corrupted_records_skipped} persisted record(s) "
                      "could not be loaded and were skipped.")
            print("\n  Advisory actionability (ADVISORY ONLY - nothing consumes this):")
            if not actionability["data_sufficient"]:
                print(f"    NOTE: {actionability['total_attempts']} executed attempt(s), below "
                      f"the minimum of {min_samples}. Rates are NOT established.")
            if not actionability["by_kind"]:
                print("    No maintenance work has been executed yet.")
            for kind in sorted(actionability["by_kind"]):
                stats = actionability["by_kind"][kind]
                print(f"    - {kind}: {stats['resolved']}/{stats['attempts']} resolved "
                      f"({stats['resolution_rate']:.0%}), {stats['persisting']} persisting, "
                      f"{stats['regressed']} regressed, {stats['inconclusive']} inconclusive")
            return 0

        if command == "history":
            if as_json:
                _emit({"runs": [record.to_dict() for record in store.runs]})
                return 0
            if not store.runs:
                print("No maintenance runs recorded yet.")
                return 0
            print(f"{len(store.runs)} maintenance run(s), oldest first:")
            for record in store.runs:
                print(f"- {record.run_id} [{record.status}] mode={record.mode} "
                      f"tier={record.configured_tier} discovered={record.candidates_discovered} "
                      f"selected={record.candidates_selected} "
                      f"executed={record.execution_attempts} "
                      f"({record.executions_succeeded} ok / {record.executions_failed} failed) "
                      f"{record.elapsed_seconds:.2f}s")
                if record.outcome_counts:
                    print(f"    outcomes: {record.outcome_counts}")
                for error in record.errors[:3]:
                    print(f"    error: {error}")
            return 0

        if command == "candidates":
            records = store.candidates
            if as_json:
                _emit({
                    "candidates": [candidate.to_dict() for candidate in records],
                    "summary": summarize_candidates(records),
                })
                return 0
            if not records:
                print("No maintenance candidates recorded yet. Run 'agent maintenance scan'.")
                return 0
            print(f"{len(records)} known maintenance candidate(s):")
            for candidate in records:
                print(f"- {candidate.candidate_id} [{candidate.severity}] {candidate.kind}")
                print(f"    {candidate.title}")
                print(f"    state={candidate.state} outcome={candidate.outcome} "
                      f"seen={candidate.occurrence_count}x confidence={candidate.confidence:.2f} "
                      f"({candidate.sample_size} sample(s))")
                if candidate.affected_files:
                    print(f"    files: {', '.join(candidate.affected_files[:5])}")
            return 0

        if command == "candidate":
            candidate = store.find(args.candidate_id)
            if candidate is None:
                print(f"error: no maintenance candidate with ID {args.candidate_id}",
                      file=sys.stderr)
                return 1
            if as_json:
                _emit(candidate.to_dict())
                return 0
            print(f"Candidate: {candidate.candidate_id}")
            print(f"  Kind:        {candidate.kind}")
            print(f"  Title:       {candidate.title}")
            print(f"  Detail:      {candidate.detail or '-'}")
            print(f"  Provenance:  {candidate.provenance}")
            print(f"  Severity:    {candidate.severity}")
            print(f"  Confidence:  {candidate.confidence:.2f} from {candidate.sample_size} sample(s)")
            print(f"  Occurrences: {candidate.occurrence_count} "
                  f"(first {candidate.first_seen_at}, last {candidate.last_seen_at})")
            print(f"  State:       {candidate.state}")
            print(f"  Outcome:     {candidate.outcome}")
            print(f"  Attempts:    {candidate.attempt_count} ({candidate.failure_count} failed)")
            print(f"  Action:      {candidate.recommended_action or '-'}")
            print(f"  Files:       {', '.join(candidate.affected_files) or '-'}")
            print(f"  Evidence:    {', '.join(candidate.evidence_refs) or '-'}")
            if candidate.uncertainty:
                print("  Uncertainty:")
                for note in candidate.uncertainty:
                    print(f"    - {note}")
            if candidate.metrics:
                print("  Metrics:")
                for name in sorted(candidate.metrics):
                    print(f"    - {name}: {candidate.metrics[name]:g}")
            if candidate.blocked_reasons:
                print("  Policy notes:")
                for reason in candidate.blocked_reasons:
                    print(f"    - {reason}")
            if candidate.history:
                print("  History:")
                for entry in candidate.history[-10:]:
                    print(f"    - {entry.get('at', '?')} {entry.get('event', '?')}"
                          f"{': ' + entry['reason'] if entry.get('reason') else ''}")
            return 0

        if command == "status":
            latest = store.latest_run()
            unresolved = [
                candidate
                for candidate in store.candidates
                if candidate.outcome not in {"resolved"}
            ]
            if as_json:
                _emit({
                    "enabled": enabled,
                    "latest_run": latest.to_dict() if latest else None,
                    "unresolved_candidates": len(unresolved),
                    "unresolved": [c.to_dict() for c in unresolved[:20]],
                })
                return 0
            print("Maintenance Status")
            print("=" * 18)
            print(f"  Subsystem enabled: {enabled}")
            if latest is None:
                print("  No maintenance run has been recorded for this repository yet.")
            else:
                print(f"  Last run:          {latest.run_id} [{latest.status}] "
                      f"mode={latest.mode} at {latest.started_at}")
                print(f"  Discovered/selected/executed: {latest.candidates_discovered}/"
                      f"{latest.candidates_selected}/{latest.execution_attempts}")
                print(f"  Outcomes:          {latest.outcome_counts or 'none'}")
                for note in latest.notes:
                    print(f"    note: {note}")
            print(f"  Unresolved candidates: {len(unresolved)}")
            for candidate in unresolved[:10]:
                print(f"    - [{candidate.severity}] {candidate.kind}: {candidate.title}")
            return 0

    # -- live scanning / running ------------------------------------------

    if command in {"scan", "dry-run", "recommendations", "run"}:
        if command == "run" and not enabled:
            print("error: the maintenance subsystem is disabled. Enable it explicitly with "
                  "--maintenance true (or AGENT_MAINTENANCE_ENABLED=1) before running it.",
                  file=sys.stderr)
            return 1
        executor = None
        want_execute = command == "run" and getattr(args, "execute", False)
        want_apply = command == "run" and getattr(args, "apply", False)
        if want_apply and not want_execute:
            print("error: --apply requires --execute; refusing to apply changes that "
                  "were never executed or prospectively validated.", file=sys.stderr)
            return 1
        if want_execute and getattr(args, "dry_run", False):
            print("error: --execute and --dry-run are mutually exclusive.", file=sys.stderr)
            return 1
        if want_execute:
            executor = _build_maintenance_executor(
                config, storage, apply_enabled=want_apply
            )
        runner, _ = _build_maintenance_runner(config, storage, executor=executor)
        if command == "scan":
            mode = RUN_MODE_SCAN
        elif command == "run" and not getattr(args, "dry_run", False):
            mode = RUN_MODE_EXECUTE
        else:
            mode = RUN_MODE_DRY_RUN
        result = runner.run(mode=mode)

        enqueued: list[str] = []
        if command == "run" and getattr(args, "enqueue", False) and result.work_orders:
            enqueued = _enqueue_work_orders(result, storage)

        if as_json:
            payload = result.to_dict()
            payload["enqueued_task_ids"] = enqueued
            payload["maintenance_enabled"] = enabled
            payload["execution"] = _execution_payload(executor, want_execute, want_apply)
            _emit(payload)
            return 0

        if command == "recommendations":
            print("Maintenance Recommendations (ranked)")
            print("=" * 36)
            if not result.ranked:
                print("  No maintenance candidates detected in this repository.")
                return 0
            for candidate, explanation in result.ranked:
                verdict = result.verdicts.get(candidate.candidate_id)
                print(f"\n  [{explanation.severity}] {candidate.title}")
                print(f"    id:         {candidate.candidate_id}")
                print(f"    kind:       {candidate.kind} (from {candidate.provenance})")
                print(f"    priority:   {explanation.score:.3f} "
                      f"(severity band {explanation.severity_rank} decides ordering first)")
                print(f"    action:     {candidate.recommended_action or '-'}")
                print(f"    files:      {', '.join(candidate.affected_files) or '-'}")
                print("    why ranked here:")
                for reason in explanation.reasons:
                    print(f"      - {reason}")
                if verdict is not None:
                    print(f"    autonomy:   {verdict.granted_tier} "
                          f"({describe_tier(verdict.granted_tier)})")
                    for reason in verdict.blocking_reasons:
                        print(f"      BLOCKED: {reason}")
                    for reason in verdict.cap_reasons:
                        print(f"      capped: {reason}")
            return 0

        record = result.record
        title = {
            "scan": "Maintenance Scan",
            "dry-run": "Maintenance Dry Run",
            "run": "Maintenance Run",
        }[command]
        print(title)
        print("=" * len(title))
        print(f"  Run id:            {record.run_id} [{record.status}] mode={record.mode}")
        print(f"  Autonomy ceiling:  {record.configured_tier}")
        print(f"  Discovered:        {record.candidates_discovered}")
        print(f"  Rejected/capped:   {record.candidates_rejected}")
        print(f"  Selected:          {record.candidates_selected}")
        print(f"  Execution attempts:{record.execution_attempts} "
              f"({record.executions_succeeded} ok / {record.executions_failed} failed)")
        print(f"  Reassessments:     {record.reassessments}")
        print(f"  Elapsed:           {record.elapsed_seconds:.2f}s")
        if result.analysis.degraded:
            print("  NOTE: this scan was degraded (a source was unavailable or an "
                  "extractor failed); absence of a signal is not evidence it is gone.")
        for name, error in sorted(result.analysis.extractor_errors.items()):
            print(f"    extractor {name} failed: {error}")
        for name, reason in sorted(result.analysis.skipped.items()):
            print(f"    extractor {name}: {reason}")
        if result.batches:
            print(f"  Parallel batches:  {len(result.batches)} "
                  f"(widths {[len(b) for b in result.batches]}; overlapping "
                  "candidates are serialised)")
        if result.work_orders:
            print(f"\n  Planned work orders ({len(result.work_orders)}):")
            for candidate_id in sorted(result.work_orders):
                order = result.work_orders[candidate_id]
                print(f"    - {candidate_id}: {order.objective}")
                print(f"      scope: {', '.join(order.scope_files) or '-'}")
                print(f"      tier:  {order.granted_tier}")
        if command == "run":
            _print_execution_report(executor, want_execute, want_apply, result)
        if enqueued:
            print(f"\n  Enqueued {len(enqueued)} task(s) for the existing scheduler:")
            for task_id in enqueued:
                print(f"    - {task_id}")
        for note in record.notes:
            print(f"  note: {note}")
        for error in record.errors:
            print(f"  error: {error}")
        return 0

    print(f"error: unknown maintenance subcommand '{command}'", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        logging.basicConfig(level=getattr(logging, config.log_level), format="%(levelname)s %(message)s")
        storage = JsonFileStorage(getattr(config, "data_dir", None) or (config.project / ".agent_data"))

        if args.command == "validation":
            return _handle_validation_command(args, config, storage)

        if args.command == "maintenance":
            return _handle_maintenance_command(args, config, storage)

        if args.command == "analyze":
            _print_context(RepositoryIntelligence(config.project).scan())
            return 0

        if args.command == "context":
            context = RepositoryIntelligence(config.project).scan()
            ContextSelector(
                config.project,
                max_files=config.max_context_files,
                max_chars=config.planning_context_bytes,
                max_file_chars=config.max_context_file_bytes,
                max_tokens=config.max_context_tokens,
                dependency_depth=config.dependency_depth,
            ).select(args.task, context)
            _print_selected_context(context, args)
            return 0

        if args.command == "approve-plan":
            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.PLAN_REVIEW:
                print(f"error: Task {args.task_id} is not in PLAN_REVIEW status.")
                raise SystemExit(1)
            if task.plan:
                errors = GraphValidator(task.plan.subtasks).validate()
                if errors:
                    print(f"error: Plan for task {args.task_id} is invalid and cannot be approved: {'; '.join(errors)}")
                    raise SystemExit(1)
            task.status = TaskStatus.PENDING
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            storage.save_task(task)
            print(f"Plan for task {args.task_id} approved. Task status set to PENDING.")
            return 0

        if args.command == "reject-plan":
            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.PLAN_REVIEW:
                print(f"error: Task {args.task_id} is not in PLAN_REVIEW status.")
                raise SystemExit(1)
            task.status = TaskStatus.REJECTED
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            storage.save_task(task)
            print(f"Plan for task {args.task_id} rejected. Task status set to REJECTED.")
            return 0

        if args.command == "edit-plan":
            task = storage.load_task(args.task_id)
            if task.status not in {TaskStatus.PLAN_REVIEW, TaskStatus.PENDING, TaskStatus.PAUSED}:
                print(f"error: Task {args.task_id} is in status {task.status.value} and cannot be edited.")
                raise SystemExit(1)
            if not task.plan:
                print(f"error: Task {args.task_id} has no plan to edit.")
                raise SystemExit(1)

            subtask_map = {s.subtask_id: s for s in task.plan.subtasks}
            if args.subtask not in subtask_map:
                print(f"error: Subtask {args.subtask} not found in plan for task {args.task_id}.")
                raise SystemExit(1)

            subtask_to_edit = subtask_map[args.subtask]
            if args.title is not None: subtask_to_edit.title = args.title
            if args.goal is not None: subtask_to_edit.goal = args.goal
            if args.acceptance_criteria is not None: subtask_to_edit.acceptance_criteria = args.acceptance_criteria
            if args.dependencies is not None: subtask_to_edit.dependencies = args.dependencies

            errors = GraphValidator(task.plan.subtasks).validate()
            if errors:
                print(f"error: Edited plan is invalid: {'; '.join(errors)}")
                raise SystemExit(1)

            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.PLAN_REVIEW

            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            storage.save_task(task)
            print(f"Subtask {args.subtask} in task {args.task_id} updated. Task status is now {task.status.value}.")
            return 0

        if args.command == "approve-proposal":
            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.PLAN_PROPOSED or not task.plan_proposal:
                print(f"error: Task {args.task_id} is not in 'plan_proposed' state or has no proposal.", file=sys.stderr)
                return 1

            if task.plan:
                proposal = task.plan_proposal
                if hasattr(task.plan, "apply_amendment"):
                    task.plan.apply_amendment(proposal, approved_by="user_approval")
                elif getattr(proposal, "additions", None):
                    for addition in proposal.additions:
                        task.plan.subtasks.append(addition.subtask if hasattr(addition, "subtask") else addition)

                if hasattr(proposal, "modifications") and proposal.modifications:
                    subtask_map = {s.subtask_id: s for s in task.plan.subtasks}
                    for mod in proposal.modifications:
                        if mod.subtask_id in subtask_map:
                            subtask = subtask_map[mod.subtask_id]
                            if getattr(mod, "title", None) is not None: subtask.title = mod.title
                            if getattr(mod, "goal", None) is not None: subtask.goal = mod.goal
                            if getattr(mod, "acceptance_criteria", None) is not None: subtask.acceptance_criteria = mod.acceptance_criteria
                            if getattr(mod, "dependencies", None) is not None: subtask.dependencies = mod.dependencies

                active_subs = getattr(task.plan, "active_subtasks", task.plan.subtasks)
                validator = GraphValidator(active_subs)
                errors = validator.validate()
                if errors:
                    print(f"error: Approving proposal for task {args.task_id} results in an invalid plan:", file=sys.stderr)
                    for error in errors:
                        print(f"  - {error}", file=sys.stderr)
                    return 1

            task.plan_proposal = None
            task.status = TaskStatus.PENDING
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            storage.save_task(task)
            print(f"Proposal for task {args.task_id} approved. Task status set to 'pending'.")
            return 0

        if args.command == "reject-proposal":
            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.PLAN_PROPOSED:
                print(f"error: Task {args.task_id} is not in 'plan_proposed' state.", file=sys.stderr)
                return 1

            task.plan_proposal = None
            task.status = TaskStatus.PAUSED
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            task.next_retry_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)
            storage.save_task(task)
            print(f"Proposal for task {args.task_id} rejected. Task status set to 'paused'.")
            return 0

        if args.command == "scheduler" and args.scheduler_command == "run-once":
            from .scheduler import Scheduler
            from .credentials import MockCredentialStore
            scheduler = Scheduler(config, storage, MockCredentialStore())
            scheduler.run_once(progress=print)
            return 0

        if args.command == "create-task":
            from .models import Task
            import uuid
            now = datetime.datetime.now(datetime.timezone.utc)
            new_task = Task(
                task_id=str(uuid.uuid4()),
                objective=args.objective,
                status=TaskStatus.PENDING,
                created_at=now,
                updated_at=now,
                autonomous=args.autonomous,
            )
            storage.save_task(new_task)
            print(f"Created task: {new_task.task_id}")
            return 0

        if args.command == "list-tasks":
            tasks = storage.list_tasks()
            if not tasks:
                print("No tasks found.")
            for task in sorted(tasks, key=lambda t: t.created_at):
                print(f"- {task.task_id}: [{task.status.value}] {task.objective}")
            return 0

        if args.command == "show-task":
            task = storage.load_task(args.task_id)
            _print_task_details(task)
            return 0

        if args.command == "ci-repair":
            from .models import Task, CIFailureContext
            import uuid

            try:
                failure_path = Path(args.failure_file)
                if not failure_path.is_file():
                    raise FileNotFoundError(f"Failure file not found: {failure_path}")
                with failure_path.open('r', encoding='utf-8') as f:
                    failure_data = json.load(f)
                ci_context = CIFailureContext.from_dict(failure_data)
            except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"error: Invalid or missing failure file: {e}", file=sys.stderr)
                return 1

            now = datetime.datetime.now(datetime.timezone.utc)
            objective = f"Autonomously repair CI failure from command: {ci_context.failed_command}"
            new_task = Task(
                task_id=str(uuid.uuid4()), objective=objective, status=TaskStatus.PENDING,
                created_at=now, updated_at=now, autonomous=True,
                initial_failure_context=ci_context
            )
            storage.save_task(new_task)
            print(f"Created CI repair task: {new_task.task_id}")
            return 0

        if args.command == "commit-task":
            from .git import GitIntegration
            git = GitIntegration(config.project)
            if not git.is_repository():
                print("error: Project is not a Git repository.", file=sys.stderr)
                return 1

            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.COMPLETED:
                print(f"error: Task {task.task_id} is not in COMPLETED state.", file=sys.stderr)
                return 1
            if not task.changed_files:
                print(f"error: Task {task.task_id} has no recorded changes to commit.", file=sys.stderr)
                return 1

            if git.is_dirty(expected_changes=task.changed_files):
                print("error: Working tree contains unexpected changes. Please stash or commit them before proceeding.", file=sys.stderr)
                return 1

            branch_name = args.branch_name or f"agent-task/{task.task_id[:8]}"
            if not git.create_branch(branch_name):
                print(f"error: Failed to create branch '{branch_name}'. It may already exist.", file=sys.stderr)
                return 1
            
            if not git.add(task.changed_files):
                print("error: Failed to stage changed files.", file=sys.stderr)
                return 1

            commit_message = f"feat: Complete task '{task.objective}'\n\nTask-ID: {task.task_id}"
            if not git.commit(commit_message):
                print("error: Failed to create commit.", file=sys.stderr)
                return 1
            
            print(f"Successfully committed changes for task {task.task_id} to branch '{branch_name}'.")
            return 0

        if args.command == "push-task":
            from .git import GitIntegration
            git = GitIntegration(config.project)
            branch_name = f"agent-task/{args.task_id[:8]}"
            
            # A simple check: does the current branch match the expected task branch?
            if git.get_current_branch() != branch_name:
                print(f"error: You are not on the task branch. Please check out '{branch_name}' first.", file=sys.stderr)
                return 1

            if not git.push(config.git_default_remote, branch_name, set_upstream=True):
                print(f"error: Failed to push branch '{branch_name}' to remote '{config.git_default_remote}'.", file=sys.stderr)
                return 1
            
            print(f"Successfully pushed branch '{branch_name}' to remote '{config.git_default_remote}'.")
            return 0

        if args.command == "create-pr":
            from .git import GitIntegration
            from .remotes import build_remote_provider, RemoteError

            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.COMPLETED:
                print(f"error: Task {task.task_id} is not in COMPLETED state.", file=sys.stderr)
                return 1
            
            if task.pull_request:
                print(f"Pull request for task {task.task_id} already exists: {task.pull_request.url}")
                return 0

            try:
                remote_provider = build_remote_provider(config)
                if not remote_provider:
                    print("error: No remote provider configured (e.g., set AGENT_GIT_HOSTING_PROVIDER='github' and GITHUB_TOKEN).", file=sys.stderr)
                    return 1

                branch_name = f"agent-task/{task.task_id[:8]}"
                existing_pr = remote_provider.find_pull_request_for_branch(branch_name)
                if existing_pr:
                    print(f"Pull request for branch '{branch_name}' already exists: {existing_pr.url}")
                    task.pull_request = existing_pr
                else:
                    new_pr = remote_provider.create_pull_request(task, branch_name)
                    print(f"Successfully created pull request: {new_pr.url}")
                    task.pull_request = new_pr
                storage.save_task(task)
                return 0
            except RemoteError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1

        if args.command == "doctor":
            print("DungX AI Coding Agent Health Check")
            print("=" * 35)

            # 1. Python/runtime
            print(f"\n[1/5] Runtime Environment")
            print(f"  - Python Version: {sys.version.split()[0]}")
            print(f"  - Python Executable: {sys.executable}")

            # 2. Project and Storage
            print(f"\n[2/5] Project and Storage")
            print(f"  - Project Root: {config.project}")
            storage_dir = getattr(storage, "base_dir", getattr(storage, "data_dir", config.project / ".agent_data"))
            try:
                storage_dir.mkdir(exist_ok=True)
                (storage_dir / "write_test.tmp").touch()
                (storage_dir / "write_test.tmp").unlink()
                print(f"  - Storage Directory: {storage_dir} (Writable)")
            except OSError as e:
                print(f"  - Storage Directory: {storage_dir} (NOT WRITABLE: {e})")

            # 3. Git Repository
            print(f"\n[3/5] Git Repository")
            from .git import GitIntegration
            git = GitIntegration(config.project)
            if git.is_repository():
                print(f"  - Status: Detected Git repository")
                print(f"  - Branch: {git.branch()}")
                status = git.status()
                print(f"  - Working Tree: {'Clean' if 'nothing to commit, working tree clean' in status else 'Contains uncommitted changes'}")
            else:
                print("  - Status: Not a Git repository")

            # 4. Provider Status
            print(f"\n[4/5] Provider Status")
            from .scheduler import Scheduler
            from .credentials import MockCredentialStore
            try:
                scheduler = Scheduler(config, storage, MockCredentialStore())
                provider_configs = storage.load_provider_configs()
                print(f"  - Configured Providers: {len(provider_configs)}")
            except Exception as e:
                print(f"  - Could not determine provider status: {e}")
            return 0

        if args.command == "show-config":
            # Exclude sensitive fields like api_key
            config_dict = {k: v for k, v in config.__dict__.items() if k != "api_key"}
            print(json.dumps(config_dict, indent=2, default=str))
            return 0

        if args.command == "run":
            print("error: 'run' command is deprecated. Use 'create-task' and 'scheduler run-once'.", file=sys.stderr)
            return 1

        # Fallback for other commands not yet implemented in this version of the CLI main function
        print(f"error: command '{args.command}' not fully implemented in this CLI version.", file=sys.stderr)
        return 1

    except (ValueError, ProviderError, OSError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
