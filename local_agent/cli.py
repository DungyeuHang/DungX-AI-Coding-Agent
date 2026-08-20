from __future__ import annotations

import argparse
import datetime
import logging
import json
import sys

from .config import AgentConfig, add_common_arguments, config_from_args
from .context import ContextSelector
from .orchestrator import Orchestrator
from .providers import MockProvider, ProviderError, build_provider
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
    create_task.add_argument("--autonomous", action="store_true", help="Enable autonomous mode for this task")
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

    if args.verbose:
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
        logging.basicConfig(level=getattr(logging, config.log_level), format="%(levelname)s %(message)s")
        storage = JsonFileStorage(config.data_dir)

        if args.command == "analyze":
            _print_context(Orchestrator(config, MockProvider(), storage).analyze())
            return 0

        if args.command == "context":
            context = Orchestrator(config, MockProvider(), storage).analyze()
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
                print(f"error: Task {args.task_id} is not in PLAN_REVIEW status.", file=sys.stderr)
                return 1
            if task.plan:
                errors = GraphValidator(task.plan.subtasks).validate()
                if errors:
                    print(f"error: Plan for task {args.task_id} is invalid and cannot be approved: {'; '.join(errors)}", file=sys.stderr)
                    return 1
            task.status = TaskStatus.PENDING
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            storage.save_task(task)
            print(f"Plan for task {args.task_id} approved. Task status set to PENDING.")
            return 0

        if args.command == "reject-plan":
            task = storage.load_task(args.task_id)
            if task.status != TaskStatus.PLAN_REVIEW:
                print(f"error: Task {args.task_id} is not in PLAN_REVIEW status.", file=sys.stderr)
                return 1
            task.status = TaskStatus.REJECTED
            task.updated_at = datetime.datetime.now(datetime.timezone.utc)
            storage.save_task(task)
            print(f"Plan for task {args.task_id} rejected. Task status set to REJECTED.")
            return 0

        if args.command == "edit-plan":
            task = storage.load_task(args.task_id)
            if task.status not in {TaskStatus.PLAN_REVIEW, TaskStatus.PENDING, TaskStatus.PAUSED}:
                print(f"error: Task {args.task_id} is in status {task.status.value} and cannot be edited.", file=sys.stderr)
                return 1
            if not task.plan:
                print(f"error: Task {args.task_id} has no plan to edit.", file=sys.stderr)
                return 1

            subtask_map = {s.subtask_id: s for s in task.plan.subtasks}
            if args.subtask not in subtask_map:
                print(f"error: Subtask {args.subtask} not found in plan for task {args.task_id}.", file=sys.stderr)
                return 1

            subtask_to_edit = subtask_map[args.subtask]
            if args.title is not None: subtask_to_edit.title = args.title
            if args.goal is not None: subtask_to_edit.goal = args.goal
            if args.acceptance_criteria is not None: subtask_to_edit.acceptance_criteria = args.acceptance_criteria
            if args.dependencies is not None: subtask_to_edit.dependencies = args.dependencies

            errors = GraphValidator(task.plan.subtasks).validate()
            if errors:
                print(f"error: Edited plan is invalid: {'; '.join(errors)}", file=sys.stderr)
                return 1

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
                # Apply modifications
                if task.plan_proposal.modifications:
                    subtask_map = {s.subtask_id: s for s in task.plan.subtasks}
                    for mod in task.plan_proposal.modifications:
                        if mod.subtask_id in subtask_map:
                            subtask = subtask_map[mod.subtask_id]
                            if mod.title is not None: subtask.title = mod.title
                            if mod.goal is not None: subtask.goal = mod.goal
                            if mod.acceptance_criteria is not None: subtask.acceptance_criteria = mod.acceptance_criteria
                            if mod.dependencies is not None: subtask.dependencies = mod.dependencies

                # Apply additions
                if task.plan_proposal.additions:
                    for addition in task.plan_proposal.additions:
                        task.plan.subtasks.append(addition.subtask)

                # Validate graph after changes
                validator = GraphValidator(task.plan.subtasks)
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
            try:
                storage.data_dir.mkdir(exist_ok=True)
                (storage.data_dir / "write_test.tmp").touch()
                (storage.data_dir / "write_test.tmp").unlink()
                print(f"  - Storage Directory: {storage.data_dir} (Writable)")
            except OSError as e:
                print(f"  - Storage Directory: {storage.data_dir} (NOT WRITABLE: {e})")

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
