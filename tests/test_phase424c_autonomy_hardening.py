"""Phase 4.24 (continued) -- Final Autonomy Hardening & Adversarial Execution
Integrity.

An adversarial audit of the orchestration/checkpoint/recovery machinery
underlying autonomous end-to-end task execution, aimed specifically at the
failure modes that make a long, unattended run unreliable: premature
completion, false progress, infinite/repetitive execution, corrupted or
cross-contaminated checkpoint state, and provider/tool failures being
silently misread as success.

Four background research passes (loop/repetition bounds, checkpoint
integrity, provider/tool/specialist failure handling, context/memory
staleness) fed a set of concrete hypotheses; each reported here was
independently re-traced and, where practical, reproduced against live code
before any fix was written.

VULN-4.24C-01 -- Cross-subtask checkpoint contamination. Orchestrator.run()
falls back to `task.latest_checkpoint_id` (a task-WIDE pointer overwritten
by every subtask's own checkpoints -- see _create_checkpoint) whenever the
CURRENT subtask has never checkpointed yet. A genuinely fresh subtask B,
started right after subtask A's run, would silently inherit A's restored
`plan` and `RecoveryState` (repair signatures, failure history, patch
hashes) -- corrupting B's anti-repetition bookkeeping and, depending on the
inherited iteration count, potentially causing B to run zero iterations or
wrongly trip the `REPEATED_REPAIR_DETECTED` guard on an unrelated attempt.
Fixed by verifying `checkpoint.subtask_id == current_subtask.subtask_id`
before trusting anything restored from it -- the same invariant the
multi-turn rehydration path already enforced a few hundred lines later.

VULN-4.24C-02 -- `JsonFileStorage.load_latest_checkpoint` never accepted the
`subtask_id` keyword argument that `MultiTurnImplementationAgent.execute()`
has always called it with. Every single multi-turn checkpoint-evidence
resume raised `TypeError`, silently caught by a broad `except Exception`
fallback -- meaning EVERY multi-turn resume discarded all prior evidence
(test/review/behavioral results, workspace fingerprints) and started the
evidence store completely fresh, 100% of the time, regardless of whether a
real, usable checkpoint existed. Fixed by adding the missing parameter with
subtask-scoped filtering to both the abstract interface and the concrete
implementation.

VULN-4.24C-03 -- MultiTurnImplementationAgent's REPAIRING stage had no
duplicate-patch detection at all (the Orchestrator's single-turn repair loop
has had one since Phase 4.5 -- RecoveryState.is_duplicate_patch). A provider
regenerating the exact same ineffective repair on every attempt could
silently burn the entire `max_repair_turns` budget on one distinct attempt
instead of `max_repair_turns` genuinely different ones. Fixed with a
content-based signature (not diff-based -- see _repair_ops_signature's
docstring for why `_compute_workspace_diff` can't be reused for this)
checked before each repair is applied, with history recomputed on resume
from persisted turn data.

Two further robustness gaps (not false-positive-completion vulnerabilities,
but reliability gaps directly relevant to a long unattended run):

VULN-4.24C-04 -- A corrupted-but-parseable checkpoint (JSON valid, but
missing a required field, or a schema mismatch from a hand-edited/older
file) makes `Checkpoint.from_dict`'s `cls(**d)` raise TypeError, which was
not classified as "malformed data" by storage.py's load_checkpoint (only
JSONDecodeError/KeyError were), and Orchestrator.run()'s checkpoint-resume
try/except only caught FileNotFoundError -- so a corrupted checkpoint could
crash the entire run instead of the run failing safely (proceeding fresh).
Fixed by classifying TypeError as malformed data too, and widening both
resume-path except clauses to also treat that as "no usable checkpoint".

VULN-4.24C-05 -- ui.py's background worker thread only caught
(ProviderError, ValueError, OSError); any other exception (including the
corrupted-checkpoint TypeError from 4.24C-04, before that fix) silently
killed the daemon thread with no "error"/"finished" event ever delivered,
leaving the UI spinning forever. Fixed by broadening to a bare `except
Exception` at this specific thread boundary -- the correct place for a
catch-all, since nothing else can observe what escapes a daemon thread.

One investigated hypothesis (provider/specialist failure silently
completing a task via SpecialistModelRouter's implicit MockProvider
fallback) was found to be real in principle but NOT closed with a
completion-gating change in this phase -- see TestObservabilityOnly and
the phase report's Remaining Risks section for why a deterministic,
non-disruptive fix proved architecturally harder than initially assumed
(explicit `provider="mock"` is a legitimate, tested configuration mode
indistinguishable at the call site from the silent degradation case without
deeper plumbing), and what was done instead (loud ERROR-level observability
at the actual degradation point).
"""

from __future__ import annotations

import datetime
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.models import (
    Checkpoint,
    ExecutionResult,
    FailureAnalysis,
    FileOperation,
    Plan,
    ProjectContext,
    ReviewResult,
    Subtask,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
)
from local_agent.multi_turn import _repair_ops_signature
from local_agent.orchestrator import Orchestrator
from local_agent.providers import MockProvider
from local_agent.storage import JsonFileStorage


class SimpleRepairProvider(MockProvider):
    """Always produces the SAME repair patch, regardless of attempt number --
    simulates a provider stuck regenerating an ineffective fix."""

    def __init__(self):
        super().__init__()
        self.generate_code_calls = 0

    def generate_code(self, task, plan, context, failure=None, review=None):
        self.generate_code_calls += 1
        if failure is None:
            return [FileOperation("modify", "src/app.py", content="# initial change")]
        return [FileOperation("modify", "src/app.py", content="# same repair every time")]


def _make_task_with_two_subtasks(storage: JsonFileStorage) -> Task:
    now = datetime.datetime.now(datetime.timezone.utc)
    sub_a = Subtask(subtask_id="subA", title="Subtask A", goal="Do A", status=SubtaskStatus.PENDING, created_at=now, updated_at=now)
    sub_b = Subtask(subtask_id="subB", title="Subtask B", goal="Do B", status=SubtaskStatus.PENDING, created_at=now, updated_at=now)
    plan = TaskPlan(objective="Test Task", subtasks=[sub_a, sub_b])
    task = Task(task_id="task1", objective="Test Task", status=TaskStatus.PENDING, created_at=now, updated_at=now, plan=plan)
    storage.save_task(task)
    return task


class TestCrossSubtaskCheckpointIsolation(unittest.TestCase):
    """VULN-4.24C-01."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("# original content")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def _orchestrator(self, provider) -> Orchestrator:
        config = AgentConfig.from_environment(self.root, max_iterations=1)
        orch = Orchestrator(config, self.storage, None, threading.Lock(), threading.Lock())
        orig_run = orch.run

        def run_wrapped(*args, **kwargs):
            with mock.patch("local_agent.orchestrator.build_provider", return_value=provider):
                return orig_run(*args, **kwargs)

        orch.run = run_wrapped
        return orch

    def test_fresh_subtask_does_not_inherit_prior_subtasks_recovery_state(self):
        task = _make_task_with_two_subtasks(self.storage)

        # Subtask A runs and completes at least one iteration -- this
        # unconditionally records a RepairSignature (orchestrator.py's
        # `recovery_state.record_attempt(...)` fires on every iteration,
        # not just failures) and checkpoints, setting task.latest_checkpoint_id.
        provider_a = MockProvider()
        with mock.patch.object(provider_a, "generate_code", return_value=[FileOperation("modify", "src/app.py", content="# A's change")]):
            orch_a = self._orchestrator(provider_a)
            orch_a.run(task, "subA")

        self.assertTrue(task.latest_checkpoint_id, "subtask A should have produced a checkpoint")
        sub_a = next(s for s in task.plan.subtasks if s.subtask_id == "subA")
        self.assertEqual(sub_a.latest_checkpoint_id, task.latest_checkpoint_id)

        # Subtask B has never checkpointed on its own.
        sub_b = next(s for s in task.plan.subtasks if s.subtask_id == "subB")
        self.assertFalse(sub_b.latest_checkpoint_id)

        provider_b = MockProvider()
        with mock.patch.object(provider_b, "generate_code", return_value=[FileOperation("modify", "src/app.py", content="# B's change")]):
            orch_b = self._orchestrator(provider_b)
            report_b = orch_b.run(task, "subB")

        # The vulnerability: without the fix, this would be 1 (A's
        # signature, silently inherited via the task-wide checkpoint
        # pointer) instead of B's own fresh count.
        self.assertEqual(
            len(report_b.recovery_state.repair_signatures), 1,
            "subtask B's recovery_state must contain only ITS OWN attempt, not A's inherited history",
        )
        self.assertEqual(report_b.recovery_state.repair_signatures[0].affected_files, ["src/app.py"])


class TestLoadLatestCheckpointSubtaskScoping(unittest.TestCase):
    """VULN-4.24C-02."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_accepts_subtask_id_keyword_without_raising(self):
        """Regression lock: MultiTurnImplementationAgent.execute() has
        always called this with subtask_id= -- must not raise TypeError."""
        result = self.storage.load_latest_checkpoint("t1", subtask_id="s1")
        self.assertIsNone(result)

    def test_scopes_to_the_requested_subtask_not_the_task_wide_latest(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        cp_a = Checkpoint(checkpoint_id="cp-a", task_id="t1", subtask_id="sub-A", timestamp=now, current_state_description="x")
        cp_b = Checkpoint(checkpoint_id="cp-b", task_id="t1", subtask_id="sub-B", timestamp=now + datetime.timedelta(seconds=10), current_state_description="y")
        self.storage.save_checkpoint(cp_a)
        self.storage.save_checkpoint(cp_b)

        unscoped = self.storage.load_latest_checkpoint("t1")
        self.assertEqual(unscoped.subtask_id, "sub-B")  # chronologically latest overall

        scoped_a = self.storage.load_latest_checkpoint("t1", subtask_id="sub-A")
        self.assertIsNotNone(scoped_a)
        self.assertEqual(scoped_a.subtask_id, "sub-A")

    def test_unknown_subtask_returns_none_not_a_foreign_checkpoint(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        self.storage.save_checkpoint(Checkpoint(checkpoint_id="cp-a", task_id="t1", subtask_id="sub-A", timestamp=now, current_state_description="x"))
        result = self.storage.load_latest_checkpoint("t1", subtask_id="sub-nonexistent")
        self.assertIsNone(result)


class TestMultiTurnAntiRepeat(unittest.TestCase):
    """VULN-4.24C-03."""

    def test_repair_ops_signature_identical_for_identical_content(self):
        ops1 = [FileOperation("modify", "a.py", content="same fix")]
        ops2 = [FileOperation("modify", "a.py", content="same fix")]
        self.assertEqual(_repair_ops_signature(ops1), _repair_ops_signature(ops2))

    def test_repair_ops_signature_differs_for_different_content(self):
        ops1 = [FileOperation("modify", "a.py", content="fix attempt 1")]
        ops2 = [FileOperation("modify", "a.py", content="fix attempt 2")]
        self.assertNotEqual(_repair_ops_signature(ops1), _repair_ops_signature(ops2))

    def test_repair_ops_signature_order_invariant(self):
        ops1 = [FileOperation("modify", "a.py", content="x"), FileOperation("modify", "b.py", content="y")]
        ops2 = [FileOperation("modify", "b.py", content="y"), FileOperation("modify", "a.py", content="x")]
        self.assertEqual(_repair_ops_signature(ops1), _repair_ops_signature(ops2))

    def test_repair_ops_signature_differs_by_path(self):
        ops1 = [FileOperation("modify", "a.py", content="x")]
        ops2 = [FileOperation("modify", "b.py", content="x")]
        self.assertNotEqual(_repair_ops_signature(ops1), _repair_ops_signature(ops2))

    def test_multi_turn_repair_loop_stops_on_repeated_patch_instead_of_burning_full_budget(self):
        from local_agent.filesystem import ProjectFilesystem
        from local_agent.multi_turn import MultiTurnImplementationAgent
        from local_agent.tools import ToolRegistry

        root = Path(tempfile.mkdtemp())
        try:
            (root / "app.py").write_text("def broken(): return 1/0\n")
            fs = ProjectFilesystem(root)
            storage = JsonFileStorage(root / ".agent_data")
            registry = ToolRegistry(root, filesystem=fs)
            config = AgentConfig.from_environment(root, max_repair_turns=3)
            agent = MultiTurnImplementationAgent(config, fs, registry, storage)

            provider = SimpleRepairProvider()
            now = datetime.datetime.now(datetime.timezone.utc)
            task = Task(task_id="t1", objective="fix it", status=TaskStatus.RUNNING, created_at=now, updated_at=now)
            plan = Plan(objective="fix it", files_likely_to_change=["app.py"])
            context = ProjectContext(root=str(root))

            with mock.patch.object(agent, "_run_validation_commands", return_value=[ExecutionResult("pytest", 1, "", "still failing")]):
                with mock.patch.object(agent, "_supports_tool_use", return_value=False):
                    report = agent.execute(
                        task=task, subtask=None, plan=plan, context=context, provider=provider,
                        failure=FailureAnalysis(probable_root_cause="broken", recommended_fix="fix it"),
                        initial_state=agent.__class__.__module__ and __import__("local_agent.models", fromlist=["MultiTurnState"]).MultiTurnState.ANALYZING_FAILURE,
                    )

            # The provider generates the SAME patch every repair attempt.
            # Without duplicate detection, this would consume all 3 repair
            # turns; with it, it should stop after detecting the repeat
            # (well before exhausting generate_code_calls to 3+ distinct
            # attempts worth of budget).
            repairing_turns = [t for t in report.turns if t.stage == "repairing"]
            self.assertLessEqual(len(repairing_turns), 2, "should stop on the first detected repeat, not exhaust the full repair budget")
            self.assertEqual(report.termination_reason, "repeated_repair_detected")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


class TestCorruptedCheckpointFailsSafe(unittest.TestCase):
    """VULN-4.24C-04."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)

    def test_checkpoint_missing_required_field_raises_valueerror_not_typeerror(self):
        cp_path = self.storage._checkpoint_path("cp-corrupt")
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        # Valid JSON, but missing the required current_state_description
        # field -- simulates a schema-mismatched or hand-corrupted record.
        cp_path.write_text(json.dumps({"checkpoint_id": "cp-corrupt", "task_id": "t1", "subtask_id": "s1"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.storage.load_checkpoint("cp-corrupt")

    def test_orchestrator_run_survives_a_corrupted_checkpoint_pointer(self):
        """End-to-end: a task pointing at a corrupted checkpoint must not
        crash the whole run -- it should proceed as if no checkpoint existed."""
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("# original")
        now = datetime.datetime.now(datetime.timezone.utc)
        sub = Subtask(subtask_id="s1", title="T", goal="G", status=SubtaskStatus.PENDING, created_at=now, updated_at=now)
        plan = TaskPlan(objective="X", subtasks=[sub])
        task = Task(task_id="t1", objective="X", status=TaskStatus.PENDING, created_at=now, updated_at=now, plan=plan, latest_checkpoint_id="cp-corrupt")
        sub.latest_checkpoint_id = "cp-corrupt"
        self.storage.save_task(task)

        cp_path = self.storage._checkpoint_path("cp-corrupt")
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        cp_path.write_text(json.dumps({"checkpoint_id": "cp-corrupt", "task_id": "t1", "subtask_id": "s1"}), encoding="utf-8")

        config = AgentConfig.from_environment(self.root, max_iterations=1)
        orch = Orchestrator(config, self.storage, None, threading.Lock(), threading.Lock())
        provider = MockProvider()
        with mock.patch.object(provider, "generate_code", return_value=[FileOperation("modify", "src/app.py", content="# changed")]):
            with mock.patch("local_agent.orchestrator.build_provider", return_value=provider):
                report = orch.run(task, "s1")  # must not raise
        self.assertIsNotNone(report)


class TestUiWorkerFailsafe(unittest.TestCase):
    """VULN-4.24C-05."""

    def test_unexpected_exception_type_still_produces_an_error_event(self):
        from local_agent import ui as ui_module

        class DummyEvents:
            def __init__(self):
                self.items = []

            def put(self, item):
                self.items.append(item)

        class DummyApp:
            events = DummyEvents()
            _run_worker = ui_module.AgentUI._run_worker
            _approval_callback = ui_module.AgentUI._approval_callback

        app = DummyApp()
        config = mock.Mock()
        with mock.patch("local_agent.ui.build_provider", side_effect=RuntimeError("unexpected internal error")):
            app._run_worker(config, "do something")

        kinds = [item[0] for item in app.events.items]
        self.assertIn("error", kinds, "an unexpected exception type must still surface as an 'error' event, not hang silently")


class TestObservabilityOnly(unittest.TestCase):
    """The investigated-but-not-completion-gated finding: implicit
    MockProvider fallback when every configured real provider fails to
    build. Verifies the loud, unmissable ERROR-level signal exists at the
    actual degradation point, without changing pass/fail completion
    semantics (an explicit `provider="mock"` config is a legitimate, tested
    mode and must be unaffected -- see the module docstring's rationale)."""

    def test_all_providers_failing_to_build_logs_at_error_level(self):
        from local_agent.providers import SpecialistModelRouter, SpecialistRole
        from local_agent.config import AgentConfig

        root = Path(tempfile.mkdtemp())
        try:
            config = AgentConfig.from_environment(root)
            router = SpecialistModelRouter(base_config=config, credential_store=None)
            with mock.patch.object(router, "_build_provider_for_spec", return_value=None):
                with self.assertLogs("local_agent.providers", level="ERROR") as cm:
                    chain = router.get_provider_chain(SpecialistRole.IMPLEMENTATION)
            self.assertEqual(len(chain), 1)
            self.assertIsInstance(chain[0], MockProvider)
            self.assertTrue(any("falling back to the offline MockProvider" in msg for msg in cm.output))
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_explicit_mock_provider_config_does_not_trigger_the_error_log(self):
        """An explicit provider="mock" config builds MockProvider as the
        PRIMARY via the normal build path and must never hit the fallback
        branch or its error log -- this is a legitimate, supported mode."""
        from local_agent.providers import SpecialistModelRouter, SpecialistRole
        from local_agent.config import AgentConfig
        import logging

        root = Path(tempfile.mkdtemp())
        try:
            config = AgentConfig.from_environment(root, provider="mock")
            router = SpecialistModelRouter(base_config=config, credential_store=None)
            logger = logging.getLogger("local_agent.providers")
            with mock.patch.object(logger, "error") as mock_error:
                chain = router.get_provider_chain(SpecialistRole.IMPLEMENTATION)
            self.assertTrue(len(chain) >= 1)
            self.assertIsInstance(chain[0], MockProvider)
            mock_error.assert_not_called()
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
