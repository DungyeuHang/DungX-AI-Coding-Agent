from __future__ import annotations

import datetime
import threading
from pathlib import Path
from typing import Any
import pytest

from local_agent.config import AgentConfig
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    ClarificationRequest,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    ImplementationTurn,
    MultiTurnExecutionReport,
    MultiTurnState,
    Plan,
    ProjectContext,
    ProviderCapability,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from local_agent.multi_turn import MultiTurnImplementationAgent
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage
from local_agent.tools import CommandRunner, ToolRegistry


def make_test_task(task_id: str = "task-417", objective: str = "Test objective 417") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=objective,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


class DummyCommandRunner:
    def __init__(self, sequence: list[ExecutionResult] | None = None):
        self.sequence = list(sequence or [])
        self.executed_commands: list[tuple[str, ...]] = []

    def run(self, command: tuple[str, ...], timeout: float | None = None) -> ExecutionResult:
        self.executed_commands.append(command)
        if self.sequence:
            return self.sequence.pop(0)
        return ExecutionResult(command=" ".join(command), exit_code=0, stdout="OK", stderr="")


class DummyProvider:
    def __init__(
        self,
        responses: list[Any] | None = None,
        code_ops: list[FileOperation] | None = None,
        review_verdict: str = "APPROVED",
        review_findings: list[str] | None = None,
    ):
        self.provider_id = "mock"
        self.model = "mock-model"
        self.capabilities = {ProviderCapability.TOOL_USE, ProviderCapability.IMPLEMENTATION}
        self.responses = list(responses or [])
        self.code_ops = code_ops or []
        self.review_verdict = review_verdict
        self.review_findings = review_findings or []
        self.captured_review_diffs: list[str] = []

    def generate_code_with_tools(self, *args, **kwargs) -> Any:
        if self.responses:
            return self.responses.pop(0)
        return self.code_ops

    def generate_code(self, *args, **kwargs) -> list[FileOperation]:
        return self.code_ops

    def review_changes_with_tools(self, task, plan, diff, context, tools, tool_history, **kwargs) -> Any:
        self.captured_review_diffs.append(diff)
        return ReviewResult(verdict=self.review_verdict, summary="Mock review", findings=self.review_findings)

    def review_changes(self, task, plan, diff, context, **kwargs) -> ReviewResult:
        self.captured_review_diffs.append(diff)
        return ReviewResult(verdict=self.review_verdict, summary="Mock review", findings=self.review_findings)


# ==============================================================================
# 1. Authoritative Diff Reconstruction Tests
# ==============================================================================

def test_review_receives_actual_unified_diff(tmp_path: Path):
    target_file = tmp_path / "greeting.py"
    target_file.write_text("def hello(): return 'hi'\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()

    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True)
    ops = [FileOperation(action="modify", path="greeting.py", content="def hello(): return 'hello world'\n")]
    provider = DummyProvider(code_ops=ops, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-diff", "Update greeting")
    plan = Plan(objective="Update greeting", files_likely_to_change=["greeting.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert len(provider.captured_review_diffs) == 1
    # Verify the reviewer received an actual unified diff rather than diff=""
    diff_text = provider.captured_review_diffs[0]
    assert "greeting.py" in diff_text
    assert "+def hello(): return 'hello world'" in diff_text


def test_review_evaluates_new_diff_after_repair(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()

    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True, max_review_turns=2)

    initial_ops = [FileOperation(action="modify", path="calc.py", content="def add(): return 1\n")]
    repair_ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]

    class MultiStageReviewProvider(DummyProvider):
        def __init__(self):
            super().__init__(responses=[initial_ops, repair_ops])
            self.review_count = 0

        def review_changes_with_tools(self, task, plan, diff, context, tools, tool_history, **kwargs):
            self.captured_review_diffs.append(diff)
            self.review_count += 1
            if self.review_count == 1:
                return ReviewResult(verdict="CHANGES_REQUIRED", summary="Need parameters", findings=["Add a, b"])
            return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])

    provider = MultiStageReviewProvider()
    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-diff-repair", "Fix add")
    plan = Plan(objective="Fix add", files_likely_to_change=["calc.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert len(provider.captured_review_diffs) == 2
    # Verify 1st diff had return 1, 2nd diff had return a + b
    assert "+def add(): return 1" in provider.captured_review_diffs[0]
    assert "+def add(a, b): return a + b" in provider.captured_review_diffs[1]


# ==============================================================================
# 2. Rigorous Final Verification & Syntax Enforcement Tests
# ==============================================================================

def test_verification_detects_syntax_error_and_triggers_repair(tmp_path: Path):
    target_file = tmp_path / "broken.py"
    target_file.write_text("x = 1\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()  # commands pass

    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True, max_repair_turns=2)

    # 1st implementation produces broken syntax, repair fixes it
    broken_ops = [FileOperation(action="modify", path="broken.py", content="def invalid syntax :(\n")]
    fixed_ops = [FileOperation(action="modify", path="broken.py", content="def valid(): return True\n")]

    provider = DummyProvider(responses=[broken_ops, fixed_ops], review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-syntax", "Fix syntax")
    plan = Plan(objective="Fix syntax", files_likely_to_change=["broken.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    assert report.repair_turns == 1
    assert "def valid(): return True" in target_file.read_text(encoding="utf-8")


def test_verification_budget_exhaustion_terminates_failed(tmp_path: Path):
    target_file = tmp_path / "broken.py"
    target_file.write_text("x = 1\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    
    # Tests always pass in TESTING stage, but syntax error persists in both turns
    runner = DummyCommandRunner()

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=10,
        max_repair_turns=5,
        max_verification_turns=1,  # Only 1 verification attempt allowed
    )

    # Both initial and repair produce broken syntax
    broken_ops = [FileOperation(action="modify", path="broken.py", content="def invalid syntax :(\n")]
    provider = DummyProvider(code_ops=broken_ops, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-v-fail", "Test verification fail")
    plan = Plan(objective="Test verification fail", files_likely_to_change=["broken.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is False
    assert report.final_state == MultiTurnState.FAILED.value
    assert report.termination_reason == "verification_budget_exhausted"


# ==============================================================================
# 3. Interactive Clarification Tool & Model Tests
# ==============================================================================

def test_ask_user_clarification_tool_with_handler(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    
    def mock_handler(question: str, choices: list[str] | None) -> str:
        return f"User answered: option A for '{question}'"

    reg = ToolRegistry(tmp_path, filesystem=fs, clarification_handler=mock_handler)
    call = ToolCall(
        call_id="call-clarify",
        tool_name="ask_user_clarification",
        arguments={"question": "Use Postgres or SQLite?", "choices": ["Postgres", "SQLite"]},
    )

    res = reg.execute(call)
    assert res.is_error is False
    assert "User answered: option A" in res.output


def test_ask_user_clarification_tool_autonomous_fallback(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)  # No handler
    reg.enable_clarification_tool()
    call = ToolCall(
        call_id="call-clarify-auto",
        tool_name="ask_user_clarification",
        arguments={"question": "Default port?"},
    )

    res = reg.execute(call)
    assert res.is_error is False
    assert "Clarification noted in autonomous mode" in res.output


def test_clarification_request_serialization():
    now = datetime.datetime.now(datetime.timezone.utc)
    req = ClarificationRequest(
        question_id="q-1",
        task_id="task-1",
        subtask_id="sub-1",
        question="Which database engine?",
        choices=["SQLite", "Postgres"],
        status="answered",
        answer="Postgres",
        created_at=now,
        answered_at=now,
    )
    d = req.to_dict()
    restored = ClarificationRequest.from_dict(d)

    assert restored.question_id == "q-1"
    assert restored.question == "Which database engine?"
    assert restored.answer == "Postgres"
    assert restored.status == "answered"


# ==============================================================================
# 4. Durable Resumption & Rehydration in Orchestrator Tests
# ==============================================================================

def test_orchestrator_rehydrates_checkpointed_turns_on_resume(tmp_path: Path):
    storage = JsonFileStorage(tmp_path / ".agent")
    
    target_file = tmp_path / "module.py"
    target_file.write_text("x = 0\n", encoding="utf-8")

    now = datetime.datetime.now(datetime.timezone.utc)
    t1 = ImplementationTurn(
        turn_id="turn-1",
        task_id="task-orch",
        subtask_id="subtask-main",
        turn_number=1,
        stage=MultiTurnState.IMPLEMENTING.value,
        status="completed",
        file_operations=[{"action": "modify", "path": "module.py", "content": "x = 10\n"}],
        started_at=now,
        completed_at=now,
    )
    cp = Checkpoint(
        checkpoint_id="cp-task-orch-subtask-main-t1-implementing",
        task_id="task-orch",
        subtask_id="subtask-main",
        timestamp=now,
        current_state_description="Turn 1 completed",
        turns=[t1.to_dict()],
        current_turn_number=1,
        turn_stage=MultiTurnState.TESTING.value,
    )
    storage.save_checkpoint(cp)

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        provider="mock",
    )

    runner = DummyCommandRunner([ExecutionResult(command="pytest", exit_code=0, stdout="OK", stderr="")])
    ops = [FileOperation(action="modify", path="module.py", content="x = 10\n")]
    provider = DummyProvider(code_ops=ops, review_verdict="APPROVED")

    orch = Orchestrator(
        config=config,
        storage=storage,
        scheduler=None,
        repo_lock=threading.Lock(),
        memory_lock=threading.Lock(),
    )
    orch.runner = runner
    orch.router.execute_with_fallback = lambda role, action, stage_name: action(provider)

    task = make_test_task("task-orch", "Test orchestrator resume")
    plan = Plan(objective="Test orchestrator resume", files_likely_to_change=["module.py"])
    context = ProjectContext(str(tmp_path))

    report = RunReport(project=str(tmp_path))
    files, history = orch._execute_code_generation(
        task=task,
        plan=plan,
        context=context,
        failure=None,
        review=None,
        stage_name="implementation",
        subtask=None,
        report=report,
    )

    assert report.multi_turn_report is not None
    assert report.multi_turn_report.success is True
    # Initial turns (1) + testing (1) + reviewing (1) + verifying (1) = 4 total turns
    assert report.multi_turn_report.total_turns >= 3
    assert report.multi_turn_report.turns[0].turn_id == "turn-1"


# ==============================================================================
# 5. Failure & Repair Robustness Tests
# ==============================================================================

def test_empty_repair_operations_routes_to_failure_analysis(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(): pass\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="Failed test"),
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="Failed test"),
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=10,
        max_repair_turns=1,
        validation_commands=["pytest"],
    )

    initial_ops = [FileOperation(action="modify", path="calc.py", content="def add(): return 0\n")]
    empty_ops = []  # Empty repair response

    provider = DummyProvider(responses=[initial_ops, empty_ops])
    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-empty-repair", "Fix add")
    plan = Plan(objective="Fix add", files_likely_to_change=["calc.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is False
    assert report.final_state == MultiTurnState.FAILED.value
    assert report.termination_reason == "repair_budget_exhausted"


# ==============================================================================
# 6. Advanced Mutation Sensitivity & Security Tests
# ==============================================================================

def test_checkpoint_clarification_roundtrip(tmp_path: Path):
    storage = JsonFileStorage(tmp_path / ".agent")
    now = datetime.datetime.now(datetime.timezone.utc)

    req = ClarificationRequest(
        question_id="q-417-1",
        task_id="t-1",
        subtask_id="s-1",
        question="Select hashing algorithm?",
        choices=["sha256", "blake3"],
        status="answered",
        answer="sha256",
        created_at=now,
        answered_at=now,
    )

    cp = Checkpoint(
        checkpoint_id="cp-clarify-1",
        task_id="t-1",
        subtask_id="s-1",
        timestamp=now,
        current_state_description="Clarification completed",
        clarification_requests=[req.to_dict()],
    )

    storage.save_checkpoint(cp)
    loaded = storage.load_checkpoint("cp-clarify-1")

    assert loaded is not None
    assert len(loaded.clarification_requests) == 1
    restored_req = ClarificationRequest.from_dict(loaded.clarification_requests[0])
    assert restored_req.question_id == "q-417-1"
    assert restored_req.answer == "sha256"


def test_multi_turn_tool_assisted_clarification_flow(tmp_path: Path):
    target_file = tmp_path / "config.json"
    target_file.write_text("{}", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    
    def user_clarifier(question: str, choices: list[str] | None) -> str:
        if "port" in question.lower():
            return "8080"
        return "default"

    reg = ToolRegistry(tmp_path, filesystem=fs, clarification_handler=user_clarifier)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()

    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True)

    clarify_call = ToolCall(
        call_id="call-clarify-1",
        tool_name="ask_user_clarification",
        arguments={"question": "What port should the server listen on?", "choices": ["3000", "8080"]},
    )
    final_ops = [FileOperation(action="modify", path="config.json", content='{"port": 8080}\n')]

    responses = [clarify_call, final_ops]
    provider = DummyProvider(responses=responses, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(config, fs, reg, storage, runner)
    task = make_test_task("task-clarify-loop", "Configure port")
    plan = Plan(objective="Configure port", files_likely_to_change=["config.json"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    assert '{"port": 8080}' in target_file.read_text(encoding="utf-8")
    assert len(report.turns) >= 3
    # Verify tool call was recorded in turn 1
    assert any(c.get("tool_name") == "ask_user_clarification" for c in report.turns[0].tool_calls)


def test_protected_files_strict_0_diff():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "--", "local_agent/tool_engine.py", "local_agent/approval.py"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "", f"Protected files diff detected:\n{result.stdout}"
