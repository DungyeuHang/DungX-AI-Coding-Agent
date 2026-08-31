from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
import pytest

from local_agent.config import AgentConfig
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    ImplementationTurn,
    MultiTurnExecutionReport,
    MultiTurnState,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    QuotaExceededError,
    RateLimitError,
    ReviewResult,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskStatus,
    ToolCall,
    ToolDefinition,
    ToolExecutionMetrics,
    ToolExecutionPolicy,
    ToolResult,
)
from local_agent.multi_turn import MultiTurnImplementationAgent
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage
from local_agent.tool_engine import ToolEngineResult
from local_agent.tools import CommandRunner, ToolRegistry


def make_test_task(task_id: str = "task-1", objective: str = "Test objective") -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    return Task(
        task_id=task_id,
        objective=objective,
        status=TaskStatus.PENDING,
        created_at=now,
        updated_at=now,
    )


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
        self.generate_code_call_count = 0
        self.review_call_count = 0

    def generate_code_with_tools(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]],
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> Any:
        self.generate_code_call_count += 1
        if self.responses:
            return self.responses.pop(0)
        return self.code_ops

    def generate_code(
        self,
        task: str,
        plan: Plan,
        context: ProjectContext,
        failure: FailureAnalysis | None = None,
        review: ReviewResult | None = None,
    ) -> list[FileOperation]:
        self.generate_code_call_count += 1
        return self.code_ops

    def review_changes_with_tools(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        tools: list[ToolDefinition],
        tool_history: list[tuple[ToolCall, ToolResult]],
        report: RunReport | None = None,
    ) -> Any:
        self.review_call_count += 1
        return ReviewResult(verdict=self.review_verdict, summary="Mock review", findings=self.review_findings)

    def review_changes(
        self,
        task: str,
        plan: Plan,
        diff: str,
        context: ProjectContext,
        report: RunReport | None = None,
    ) -> ReviewResult:
        self.review_call_count += 1
        return ReviewResult(verdict=self.review_verdict, summary="Mock review", findings=self.review_findings)


class DummyCommandRunner:
    def __init__(self, sequence: list[ExecutionResult] | None = None):
        self.sequence = list(sequence or [])
        self.executed_commands: list[tuple[str, ...]] = []

    def run(self, command: tuple[str, ...], timeout: float | None = None) -> ExecutionResult:
        self.executed_commands.append(command)
        if self.sequence:
            return self.sequence.pop(0)
        return ExecutionResult(command=" ".join(command), exit_code=0, stdout="OK", stderr="")


# ==============================================================================
# 1. State Machine & Lifecycle Transitions
# ==============================================================================

def test_state_machine_valid_transitions():
    assert MultiTurnState.can_transition(MultiTurnState.IDLE, MultiTurnState.PLANNING)
    assert MultiTurnState.can_transition(MultiTurnState.IDLE, MultiTurnState.IMPLEMENTING)
    assert MultiTurnState.can_transition(MultiTurnState.IMPLEMENTING, MultiTurnState.TESTING)
    assert MultiTurnState.can_transition(MultiTurnState.TESTING, MultiTurnState.ANALYZING_FAILURE)
    assert MultiTurnState.can_transition(MultiTurnState.ANALYZING_FAILURE, MultiTurnState.REPAIRING)
    assert MultiTurnState.can_transition(MultiTurnState.REPAIRING, MultiTurnState.TESTING)
    assert MultiTurnState.can_transition(MultiTurnState.TESTING, MultiTurnState.REVIEWING)
    assert MultiTurnState.can_transition(MultiTurnState.REVIEWING, MultiTurnState.VERIFYING)
    assert MultiTurnState.can_transition(MultiTurnState.VERIFYING, MultiTurnState.COMPLETED)


def test_state_machine_illegal_transitions_rejected():
    assert not MultiTurnState.can_transition(MultiTurnState.COMPLETED, MultiTurnState.PLANNING)
    assert not MultiTurnState.can_transition(MultiTurnState.COMPLETED, MultiTurnState.IMPLEMENTING)
    assert not MultiTurnState.can_transition(MultiTurnState.FAILED, MultiTurnState.TESTING)
    assert not MultiTurnState.can_transition(MultiTurnState.IDLE, MultiTurnState.COMPLETED)
    assert not MultiTurnState.can_transition(MultiTurnState.TESTING, MultiTurnState.PLANNING)
    assert not MultiTurnState.can_transition("invalid_state", MultiTurnState.IMPLEMENTING)


def test_turn_record_and_report_serialization():
    now = datetime.datetime.now(datetime.timezone.utc)
    turn = ImplementationTurn(
        turn_id="turn-1",
        task_id="task-1",
        subtask_id="sub-1",
        turn_number=1,
        stage=MultiTurnState.IMPLEMENTING.value,
        provider="mock",
        model="mock-v1",
        prompt_summary="Objective prompt",
        tool_calls=[{"name": "read_file", "arguments": {"path": "main.py"}}],
        tool_results=[{"output": "print('hello')", "is_error": False}],
        tests_executed=[{"command": "pytest", "exit_code": 0, "success": True}],
        failures_detected=[],
        repair_reason=None,
        started_at=now,
        completed_at=now,
        status="completed",
        file_operations=[{"action": "modify", "path": "main.py", "content": "print('world')"}],
        metadata={"key": "value"},
    )
    data = turn.to_dict()
    restored = ImplementationTurn.from_dict(data)

    assert restored.turn_id == "turn-1"
    assert restored.turn_number == 1
    assert restored.stage == MultiTurnState.IMPLEMENTING.value
    assert len(restored.tool_calls) == 1
    assert restored.file_operations[0]["path"] == "main.py"

    report = MultiTurnExecutionReport(
        task_id="task-1",
        subtask_id="sub-1",
        success=True,
        turns=[turn],
        total_turns=1,
        repair_turns=0,
        review_turns=0,
        final_state=MultiTurnState.COMPLETED.value,
        termination_reason="completed",
        elapsed_time_seconds=1.23,
    )
    rep_data = report.to_dict()
    rep_restored = MultiTurnExecutionReport.from_dict(rep_data)

    assert rep_restored.task_id == "task-1"
    assert rep_restored.success is True
    assert len(rep_restored.turns) == 1
    assert rep_restored.final_state == MultiTurnState.COMPLETED.value


def test_checkpoint_turn_fields_roundtrip():
    now = datetime.datetime.now(datetime.timezone.utc)
    cp = Checkpoint(
        checkpoint_id="cp-123",
        task_id="task-1",
        subtask_id="sub-1",
        timestamp=now,
        current_state_description="Turn 1 completed",
        turns=[{"turn_id": "turn-1", "stage": "implementing"}],
        current_turn_number=1,
        turn_stage="implementing",
    )
    data = cp.to_dict()
    assert data["current_turn_number"] == 1
    assert data["turn_stage"] == "implementing"
    assert len(data["turns"]) == 1

    restored = Checkpoint.from_dict(data)
    assert restored.current_turn_number == 1
    assert restored.turn_stage == "implementing"
    assert len(restored.turns) == 1


# ==============================================================================
# 2. Autonomous Multi-Turn Execution Loop Tests
# ==============================================================================

def test_multi_turn_successful_tool_assisted_implementation(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return a - b\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=0, stdout="1 passed", stderr="")
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=5,
        max_repair_turns=2,
        validation_commands=["pytest"],
    )

    ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]
    responses = [
        ToolCall(call_id="call-1", tool_name="find_files", arguments={"pattern": "*.py"}),
        ops,
    ]
    provider = DummyProvider(responses=responses, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-calc", "Fix add function in calc.py")
    subtask = Subtask(subtask_id="sub-1", title="Fix add", goal="Fix add function")
    plan = Plan(objective="Fix add function", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=subtask,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    assert report.termination_reason == "completed"
    assert target_file.read_text(encoding="utf-8") == "def add(a, b): return a + b\n"
    assert len(report.turns) >= 3


def test_multi_turn_test_failure_triggers_repair_and_passes(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    
    # Sequence: 1st test fails, 2nd test (after repair) passes
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="AssertionError: 0 != 5"),
        ExecutionResult(command="pytest", exit_code=0, stdout="1 passed", stderr=""),
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=8,
        max_repair_turns=3,
        validation_commands=["pytest"],
    )

    initial_ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return 1\n")]
    repair_ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]

    responses = [
        initial_ops,  # Turn 1 (Implementing)
        repair_ops,   # Turn 4 (Repairing)
    ]
    provider = DummyProvider(responses=responses, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-calc-repair", "Fix add in calc.py")
    subtask = Subtask(subtask_id="sub-1", title="Fix add", goal="Fix add")
    plan = Plan(objective="Fix add", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=subtask,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    assert report.repair_turns == 1
    assert target_file.read_text(encoding="utf-8") == "def add(a, b): return a + b\n"


def test_multi_turn_repair_budget_exhaustion_terminates_failed(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    
    # All tests fail
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="Error 1"),
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="Error 2"),
        ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="Error 3"),
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=10,
        max_repair_turns=1,  # Only 1 repair attempt allowed
        validation_commands=["pytest"],
    )

    ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return -1\n")]
    responses = [ops, ops, ops]
    provider = DummyProvider(responses=responses)

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-fail", "Fix add")
    subtask = Subtask(subtask_id="sub-1", title="Fix add", goal="Fix add")
    plan = Plan(objective="Fix add", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=subtask,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is False
    assert report.final_state == MultiTurnState.FAILED.value
    assert report.termination_reason == "repair_budget_exhausted"
    assert "Repair turn budget" in (report.error_message or "")


def test_multi_turn_review_rejection_triggers_repair(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()  # tests always pass

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=10,
        max_repair_turns=2,
        max_review_turns=2,
        validation_commands=["pytest"],
    )

    initial_ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b # missing docstring\n")]
    fixed_ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    return a + b\n")]

    # 1st review fails, 2nd review passes
    class DynamicReviewProvider(DummyProvider):
        def review_changes_with_tools(self, *args, **kwargs):
            self.review_call_count += 1
            if self.review_call_count == 1:
                return ReviewResult(verdict="CHANGES_REQUIRED", summary="Needs docstring", findings=["Add docstring"])
            return ReviewResult(verdict="APPROVED", summary="Looks good", findings=[])

    responses = [initial_ops, fixed_ops]
    provider = DynamicReviewProvider(responses=responses)

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-review", "Fix add with docstring")
    subtask = Subtask(subtask_id="sub-1", title="Fix add", goal="Fix add")
    plan = Plan(objective="Fix add with docstring", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=subtask,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    assert report.review_turns == 1


def test_multi_turn_global_turn_budget_exhaustion(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(): pass\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner([ExecutionResult(command="pytest", exit_code=1, stdout="", stderr="err")] * 10)

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=2,  # Very tight budget
        max_repair_turns=5,
    )

    ops = [FileOperation(action="modify", path="calc.py", content="def add(): return 1\n")]
    provider = DummyProvider(code_ops=ops)

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-budget", "Test turn budget")
    plan = Plan(objective="Test turn budget", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=None,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is False
    assert report.final_state == MultiTurnState.FAILED.value
    assert report.termination_reason == "max_turns_exceeded"


def test_multi_turn_rate_limit_pauses_agent(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=5,
    )

    class RateLimitingProvider(DummyProvider):
        def generate_code_with_tools(self, *args, **kwargs):
            raise RateLimitError("Rate limit reached", retry_after_seconds=30.0)

    provider = RateLimitingProvider()

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-rate-limit", "Test rate limit pause")
    plan = Plan(objective="Test rate limit")
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=None,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is False
    assert report.final_state == MultiTurnState.PAUSED.value
    assert report.termination_reason == "provider_quota_paused"


# ==============================================================================
# 3. Integration & Configuration Tests
# ==============================================================================

def test_orchestrator_multi_turn_disabled_preserves_single_shot(tmp_path: Path):
    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=False,
        interactive_implementation=False,
        provider="mock",
    )
    assert config.multi_turn_implementation is False
    assert config.max_implementation_turns == 10


def test_orchestrator_multi_turn_enabled_configuration(tmp_path: Path):
    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=12,
        max_repair_turns=4,
        max_review_turns=3,
    )
    assert config.multi_turn_implementation is True
    assert config.max_implementation_turns == 12
    assert config.max_repair_turns == 4
    assert config.max_review_turns == 3


# ==============================================================================
# 4. Resumption, Checkpoint Durability, & State Continuity Tests
# ==============================================================================

def test_multi_turn_resumption_from_existing_turns(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return 0\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=0, stdout="1 passed", stderr="")
    ])

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=10,
        validation_commands=["pytest"],
    )

    # Simulate existing turn history: Turn 1 (Implementing) and Turn 2 (Testing) previously completed
    now = datetime.datetime.now(datetime.timezone.utc)
    t1 = ImplementationTurn(
        turn_id="turn-1",
        task_id="task-resume",
        subtask_id="sub-1",
        turn_number=1,
        stage=MultiTurnState.IMPLEMENTING.value,
        status="completed",
        started_at=now,
        completed_at=now,
    )
    t2 = ImplementationTurn(
        turn_id="turn-2",
        task_id="task-resume",
        subtask_id="sub-1",
        turn_number=2,
        stage=MultiTurnState.TESTING.value,
        status="completed",
        failures_detected=[{"command": "pytest", "exit_code": 1}],
        started_at=now,
        completed_at=now,
    )

    fixed_ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]
    provider = DummyProvider(responses=[fixed_ops], review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-resume", "Resume add function")
    subtask = Subtask(subtask_id="sub-1", title="Fix add", goal="Fix add")
    plan = Plan(objective="Resume add function", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=subtask,
        plan=plan,
        context=context,
        provider=provider,
        existing_turns=[t1, t2],
        initial_state=MultiTurnState.REPAIRING,
        failure=FailureAnalysis(probable_root_cause="Test failure on turn 2"),
    )

    assert report.success is True
    assert report.final_state == MultiTurnState.COMPLETED.value
    # Initial turns (2) + repair (1) + test (1) + review (1) + verify (1) = 6 turns
    assert len(report.turns) == 6
    assert report.turns[0].turn_id == "turn-1"
    assert report.turns[1].turn_id == "turn-2"
    assert report.turns[2].stage == MultiTurnState.REPAIRING.value


def test_multi_turn_persists_checkpoints_at_every_stage_transition(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(a, b): return a - b\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    reg = ToolRegistry(tmp_path, filesystem=fs)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner([
        ExecutionResult(command="pytest", exit_code=0, stdout="1 passed", stderr="")
    ])

    saved_stages: list[MultiTurnState] = []
    def checkpoint_hook(turn: ImplementationTurn, stage: MultiTurnState):
        saved_stages.append(stage)

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        validation_commands=["pytest"],
    )

    ops = [FileOperation(action="modify", path="calc.py", content="def add(a, b): return a + b\n")]
    provider = DummyProvider(code_ops=ops, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=reg,
        storage=storage,
        runner=runner,
        checkpoint_callback=checkpoint_hook,
    )

    task = make_test_task("task-cp", "Checkpoint test")
    subtask = Subtask(subtask_id="sub-1", title="Fix add", goal="Fix add")
    plan = Plan(objective="Checkpoint test", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=subtask,
        plan=plan,
        context=context,
        provider=provider,
    )

    assert report.success is True
    # Verify each stage transition triggered a checkpoint callback
    assert MultiTurnState.IMPLEMENTING in saved_stages
    assert MultiTurnState.TESTING in saved_stages
    assert MultiTurnState.REVIEWING in saved_stages
    assert MultiTurnState.VERIFYING in saved_stages


# ==============================================================================
# 5. Parallel Worktree Isolation & Security Invariance Tests
# ==============================================================================

def test_multi_turn_isolated_worktree_execution(tmp_path: Path):
    wt1 = tmp_path / "wt1"
    wt2 = tmp_path / "wt2"
    wt1.mkdir()
    wt2.mkdir()

    (wt1 / "mod1.py").write_text("x = 1\n", encoding="utf-8")
    (wt2 / "mod2.py").write_text("y = 2\n", encoding="utf-8")

    fs1 = ProjectFilesystem(wt1)
    fs2 = ProjectFilesystem(wt2)

    config1 = AgentConfig(project=str(wt1), multi_turn_implementation=True)
    config2 = AgentConfig(project=str(wt2), multi_turn_implementation=True)

    agent1 = MultiTurnImplementationAgent(config1, fs1, ToolRegistry(wt1, filesystem=fs1), JsonFileStorage(wt1 / ".agent"), DummyCommandRunner())
    agent2 = MultiTurnImplementationAgent(config2, fs2, ToolRegistry(wt2, filesystem=fs2), JsonFileStorage(wt2 / ".agent"), DummyCommandRunner())

    ops1 = [FileOperation(action="modify", path="mod1.py", content="x = 10\n")]
    ops2 = [FileOperation(action="modify", path="mod2.py", content="y = 20\n")]

    rep1 = agent1.execute(make_test_task("t1"), Subtask("s1", "s1", "g1"), Plan("t1", files_likely_to_change=["mod1.py"]), ProjectContext(str(wt1)), DummyProvider(code_ops=ops1))
    rep2 = agent2.execute(make_test_task("t2"), Subtask("s2", "s2", "g2"), Plan("t2", files_likely_to_change=["mod2.py"]), ProjectContext(str(wt2)), DummyProvider(code_ops=ops2))

    assert rep1.success is True
    assert rep2.success is True

    # Ensure complete isolation: wt1 only modified mod1.py, wt2 only modified mod2.py
    assert (wt1 / "mod1.py").read_text(encoding="utf-8") == "x = 10\n"
    assert not (wt1 / "mod2.py").exists()
    assert (wt2 / "mod2.py").read_text(encoding="utf-8") == "y = 20\n"
    assert not (wt2 / "mod1.py").exists()


def test_multi_turn_protected_file_modification_rejected(tmp_path: Path):
    target = tmp_path / "local_agent" / "tool_engine.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# protected\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True, max_repair_turns=0)
    agent = MultiTurnImplementationAgent(config, fs, ToolRegistry(tmp_path, filesystem=fs), JsonFileStorage(tmp_path / ".agent"), DummyCommandRunner())

    # Attempt to modify protected file
    ops = [FileOperation(action="modify", path="local_agent/tool_engine.py", content="# hacked\n")]
    provider = DummyProvider(code_ops=ops)

    task = make_test_task("task-hack", "Hacking attempt")
    plan = Plan(objective="Hacking attempt", files_likely_to_change=["local_agent/tool_engine.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    # Modification must be rejected
    assert report.success is False
    assert target.read_text(encoding="utf-8") == "# protected\n"


def test_multi_turn_policy_step_budget_enforced(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_tool_steps=3,
    )
    agent = MultiTurnImplementationAgent(config, fs, ToolRegistry(tmp_path, filesystem=fs), JsonFileStorage(tmp_path / ".agent"))
    policy = agent.build_policy()

    assert policy.max_tool_steps == 3


# ==============================================================================
# 6. Mutation Sensitivity & Contract Guard Tests
# ==============================================================================

def test_mutation_upstream_contract_injection_in_prompt(tmp_path: Path):
    fs = ProjectFilesystem(tmp_path)
    config = AgentConfig(project=str(tmp_path), multi_turn_implementation=True)
    agent = MultiTurnImplementationAgent(config, fs, ToolRegistry(tmp_path, filesystem=fs), JsonFileStorage(tmp_path / ".agent"), DummyCommandRunner())

    contract = SubtaskContract(
        subtask_id="sub-auth",
        title="Auth Implementation",
        created_files=["auth.py"],
        architectural_notes=["User must be authenticated before token issue"],
    )

    captured_prompts: list[str] = []
    class PromptCapturingProvider(DummyProvider):
        def generate_code_with_tools(self, task, plan, context, tools, tool_history, **kwargs):
            captured_prompts.append(task)
            return [FileOperation(action="create", path="auth.py", content="# auth\n")]

    task = make_test_task("task-contract", "Implement auth")
    plan = Plan(objective="Implement auth", files_likely_to_create=["auth.py"])
    context = ProjectContext(str(tmp_path))

    report = agent.execute(
        task=task,
        subtask=Subtask("sub-auth", "Auth", "Auth goal"),
        plan=plan,
        context=context,
        provider=PromptCapturingProvider(),
        upstream_contracts=[contract],
    )

    assert report.success is True
    assert len(captured_prompts) > 0
    assert "UPSTREAM INTERFACE CONSTRAINTS" in captured_prompts[0]
    assert "User must be authenticated before token issue" in captured_prompts[0]


def test_mutation_review_budget_exhaustion_terminates_failed(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(): pass\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()  # tests pass

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
        max_implementation_turns=10,
        max_repair_turns=5,
        max_review_turns=1,  # Only 1 review retry allowed
    )

    class AlwaysRejectReviewProvider(DummyProvider):
        def review_changes_with_tools(self, *args, **kwargs):
            return ReviewResult(verdict="CHANGES_REQUIRED", summary="Never passes", findings=["Flaw"])

    ops = [FileOperation(action="modify", path="calc.py", content="def add(): return 1\n")]
    provider = AlwaysRejectReviewProvider(code_ops=ops)

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=ToolRegistry(tmp_path, filesystem=fs),
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-rev-fail", "Review fail")
    plan = Plan(objective="Review fail", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is False
    assert report.final_state == MultiTurnState.FAILED.value
    assert report.termination_reason == "review_budget_exhausted"


def test_mutation_tool_metrics_aggregated(tmp_path: Path):
    target_file = tmp_path / "calc.py"
    target_file.write_text("def add(): pass\n", encoding="utf-8")

    fs = ProjectFilesystem(tmp_path)
    storage = JsonFileStorage(tmp_path / ".agent")
    runner = DummyCommandRunner()

    config = AgentConfig(
        project=str(tmp_path),
        multi_turn_implementation=True,
    )

    ops = [FileOperation(action="modify", path="calc.py", content="def add(): return 1\n")]
    responses = [
        ToolCall(call_id="call-1", tool_name="find_files", arguments={"pattern": "*.py"}),
        ops,
    ]
    provider = DummyProvider(responses=responses, review_verdict="APPROVED")

    agent = MultiTurnImplementationAgent(
        config=config,
        filesystem=fs,
        registry=ToolRegistry(tmp_path, filesystem=fs),
        storage=storage,
        runner=runner,
    )

    task = make_test_task("task-metrics", "Metrics test")
    plan = Plan(objective="Metrics test", files_likely_to_change=["calc.py"])
    context = ProjectContext(root=str(tmp_path))

    report = agent.execute(task, None, plan, context, provider)

    assert report.success is True
    assert len(report.tool_metrics) > 0
    assert report.tool_metrics[0].total_calls >= 1
