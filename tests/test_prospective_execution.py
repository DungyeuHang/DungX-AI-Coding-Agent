"""Phase 4.16 - staged overlay execution sandbox & prospective validation engine.

These tests prove the PROPOSE -> APPLY TO ISOLATED CANDIDATE TREE -> RUN REAL
VALIDATION -> OBSERVE -> REFINE -> REBUILD -> REVALIDATE -> FINALIZE loop is
genuinely real:

* real ``pytest`` / ``compileall`` subprocesses execute against candidate
  contents (not the unchanged authoritative tree),
* a candidate patch that does not fix a bug genuinely fails,
* a refined candidate genuinely passes,
* the authoritative tree is byte-for-byte untouched throughout.

No test depends on a real LLM API; providers are deterministic mocks.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any

from local_agent.coding_agent import (
    DEFAULT_MAX_CANDIDATE_ITERATIONS,
    CodingAgent,
    InteractiveCodingAgent,
    UnsafeModificationError,
)
from local_agent.commands import CommandRunner
from local_agent.config import AgentConfig, add_common_arguments, config_from_args
from local_agent.filesystem import ProjectFilesystem, SandboxViolation
from local_agent.models import (
    CommandSpec,
    FailureAnalysis,
    FileOperation,
    ImplementationResult,
    ImplementationTerminationReason,
    Plan,
    ProjectContext,
    ProviderCapability,
    ProviderError,
    ReviewResult,
    RunReport,
    Task,
    TaskStatus,
    ToolCall,
    ToolResult,
)
from local_agent.providers import AIProvider
from local_agent.sandbox import (
    EXCLUDED_DIRECTORY_NAMES,
    CandidateCommandResult,
    CandidateValidationReport,
    CandidateWorkspace,
    CandidateWorkspaceError,
    ProspectiveValidator,
)
from local_agent.tools import ToolRegistry

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

BUGGY_MODULE = "def add(a, b):\n    return a - b\n"
FIXED_MODULE = "def add(a, b):\n    return a + b\n"
STILL_BROKEN_MODULE = "def add(a, b):\n    return a * b\n"
MODULE_TEST = (
    "from module import add\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
)


def make_tiny_project(root: Path) -> None:
    """A minimal but genuinely executable Python project with a real bug."""
    (root / "module.py").write_text(BUGGY_MODULE, encoding="utf-8")
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_module.py").write_text(MODULE_TEST, encoding="utf-8")
    (root / "README.md").write_text("tiny\n", encoding="utf-8")


def make_plan(*change: str, create: list[str] | None = None) -> Plan:
    return Plan(
        objective="fix add()",
        files_to_inspect=list(change),
        files_likely_to_change=list(change),
        files_likely_to_create=list(create or []),
        steps=["edit"],
        validation_strategy=["pytest"],
        risks=[],
    )


def snapshot_tree(root: Path) -> dict[str, bytes]:
    """Byte-for-byte snapshot of a tree, skipping caches so it is stable."""
    snapshot: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part.lower() in EXCLUDED_DIRECTORY_NAMES for part in rel.parts):
            continue
        snapshot[rel.as_posix()] = path.read_bytes()
    return snapshot


class ScriptedProvider(AIProvider):
    """Deterministic tool-use provider driven by a scripted response list."""

    def __init__(
        self,
        responses: list[Any] | None = None,
        single_shot_ops: list[FileOperation] | None = None,
        supports_tools: bool = True,
        raise_on_step: int | None = None,
        raise_exc: Exception | None = None,
    ):
        super().__init__()
        self.provider_id = "scripted"
        self.model = "scripted-v1"
        self.responses = list(responses or [])
        self.single_shot_ops = single_shot_ops if single_shot_ops is not None else []
        self.histories: list[list[tuple[ToolCall, ToolResult]]] = []
        self.step_count = 0
        self.raise_on_step = raise_on_step
        self.raise_exc = raise_exc or ProviderError("scripted provider failure")
        caps = {
            ProviderCapability.PLANNING,
            ProviderCapability.IMPLEMENTATION,
            ProviderCapability.REPAIR,
            ProviderCapability.REVIEW,
        }
        if supports_tools:
            caps.add(ProviderCapability.TOOL_USE)
        self._capabilities = caps

    @property
    def capabilities(self) -> set[ProviderCapability]:
        return self._capabilities

    def generate_plan(self, task, context) -> Plan:
        return make_plan("module.py")

    def generate_code(self, task, plan, context, failure=None, review=None):
        return self.single_shot_ops

    def generate_code_with_tools(
        self, task, plan, context, tools, tool_history=None, failure=None, review=None
    ):
        self.step_count += 1
        self.histories.append(list(tool_history or []))
        if self.raise_on_step is not None and self.step_count == self.raise_on_step:
            raise self.raise_exc
        if self.responses:
            return self.responses.pop(0)
        return self.single_shot_ops

    def review_changes(self, task, plan, diff, context) -> ReviewResult:
        return ReviewResult("APPROVED", "ok", [])

    def analyze_failure(self, execution, diff, context, plan) -> FailureAnalysis:
        return FailureAnalysis("boom", ["module.py"], "fix it")


class TempProjectCase(unittest.TestCase):
    """Base case owning a disposable authoritative project tree."""

    def setUp(self) -> None:
        self.base_dir = Path(tempfile.mkdtemp(prefix="agentbase_")).resolve()
        make_tiny_project(self.base_dir)
        self.addCleanup(shutil.rmtree, self.base_dir, ignore_errors=True)
        self.workspaces: list[CandidateWorkspace] = []

    def tearDown(self) -> None:
        for workspace in self.workspaces:
            workspace.cleanup()

    def make_workspace(self, **kwargs) -> CandidateWorkspace:
        workspace = CandidateWorkspace(self.base_dir, **kwargs)
        self.workspaces.append(workspace)
        return workspace


# ---------------------------------------------------------------------------
# 1. Candidate lifecycle
# ---------------------------------------------------------------------------


class TestCandidateWorkspaceLifecycle(TempProjectCase):
    def test_setup_creates_isolated_root_outside_base(self):
        ws = self.make_workspace().setup()
        self.assertTrue(ws.root.is_dir())
        self.assertNotEqual(ws.root, self.base_dir)
        self.assertNotIn(self.base_dir, ws.root.parents)
        self.assertNotIn(ws.root, self.base_dir.parents)

    def test_setup_mirrors_base_contents(self):
        ws = self.make_workspace().setup()
        self.assertEqual((ws.root / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)
        self.assertEqual(
            (ws.root / "tests" / "test_module.py").read_text(encoding="utf-8"), MODULE_TEST
        )
        self.assertGreaterEqual(ws.files_mirrored, 3)

    def test_setup_is_idempotent(self):
        ws = self.make_workspace()
        first = ws.setup().root
        second = ws.setup().root
        self.assertEqual(first, second)

    def test_excluded_directories_are_not_mirrored(self):
        for name in (".git", "__pycache__", "node_modules", ".pytest_cache", ".agent_data"):
            directory = self.base_dir / name
            directory.mkdir()
            (directory / "junk.txt").write_text("junk", encoding="utf-8")
        ws = self.make_workspace().setup()
        for name in (".git", "__pycache__", "node_modules", ".pytest_cache", ".agent_data"):
            self.assertFalse((ws.root / name).exists(), f"{name} should not be mirrored")

    def test_git_worktree_pointer_file_is_not_mirrored(self):
        """Inside a Git worktree ``.git`` is a FILE; it must never be copied."""
        (self.base_dir / ".git").write_text(
            "gitdir: D:/real/repo/.git/worktrees/sub\n", encoding="utf-8"
        )
        ws = self.make_workspace().setup()
        self.assertFalse((ws.root / ".git").exists())
        self.assertTrue((ws.root / "module.py").exists())

    def test_candidate_of_real_git_worktree_is_not_a_git_repo(self):
        """A candidate built from a worktree cannot reach the real repository."""
        (self.base_dir / ".git").write_text("gitdir: /somewhere/.git\n", encoding="utf-8")
        ws = self.make_workspace().setup()
        result = ws.run(["git", "rev-parse", "--is-inside-work-tree"])
        # Either git is absent (127) or it reports this is not a work tree.
        self.assertNotEqual(result.exit_code, 0)

    def test_oversized_blobs_are_skipped(self):
        (self.base_dir / "huge.bin").write_bytes(b"0" * 4096)
        ws = self.make_workspace(max_file_bytes=1024).setup()
        self.assertFalse((ws.root / "huge.bin").exists())
        self.assertTrue((ws.root / "module.py").exists())

    def test_cleanup_removes_root(self):
        ws = self.make_workspace().setup()
        root = ws.root
        self.assertTrue(ws.cleanup())
        self.assertFalse(root.exists())

    def test_cleanup_is_idempotent(self):
        ws = self.make_workspace().setup()
        self.assertTrue(ws.cleanup())
        self.assertTrue(ws.cleanup())
        self.assertTrue(ws.cleanup())
        self.assertFalse(ws.is_active)

    def test_cleanup_after_exception_inside_context_manager(self):
        root_holder: list[Path] = []
        with self.assertRaises(RuntimeError):
            with CandidateWorkspace(self.base_dir) as ws:
                root_holder.append(ws.root)
                raise RuntimeError("boom")
        self.assertFalse(root_holder[0].exists())
        self.assertTrue(self.base_dir.exists())

    def test_cleanup_survives_locked_pycache(self):
        """A candidate that ran pytest has caches; cleanup must still succeed."""
        ws = self.make_workspace().setup()
        cache = ws.root / "__pycache__"
        cache.mkdir()
        (cache / "x.pyc").write_bytes(b"\x00\x01")
        os.chmod(cache / "x.pyc", 0o444)
        root = ws.root
        self.assertTrue(ws.cleanup())
        self.assertFalse(root.exists())

    def test_accessors_raise_before_setup(self):
        ws = self.make_workspace()
        for accessor in ("root", "filesystem", "runner", "registry"):
            with self.assertRaises(CandidateWorkspaceError):
                getattr(ws, accessor)

    def test_rebuild_before_setup_raises(self):
        ws = self.make_workspace()
        with self.assertRaises(CandidateWorkspaceError):
            ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)])

    def test_missing_base_tree_raises(self):
        with self.assertRaises(CandidateWorkspaceError):
            CandidateWorkspace(self.base_dir / "does_not_exist")

    def test_cleanup_refuses_to_delete_base_tree(self):
        ws = self.make_workspace().setup()
        ws._root = self.base_dir  # simulate catastrophic misconfiguration
        self.assertFalse(ws.cleanup())
        self.assertTrue(self.base_dir.exists())
        self.assertTrue((self.base_dir / "module.py").exists())
        self.assertEqual(ws.cleanup_failures, 1)

    def test_symlinks_are_not_followed(self):
        target = Path(tempfile.mkdtemp(prefix="agentlink_")).resolve()
        self.addCleanup(shutil.rmtree, target, ignore_errors=True)
        (target / "outside.txt").write_text("secret", encoding="utf-8")
        try:
            (self.base_dir / "link").symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation not permitted in this environment")
        ws = self.make_workspace().setup()
        self.assertFalse((ws.root / "link").exists())


# ---------------------------------------------------------------------------
# 2. File operations against the candidate
# ---------------------------------------------------------------------------


class TestCandidateFileOperations(TempProjectCase):
    def test_modify_with_content(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            make_plan("module.py"),
        )
        self.assertEqual(changed, ["module.py"])
        self.assertEqual(ws.read_candidate("module.py"), FIXED_MODULE)
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_create_file(self):
        ws = self.make_workspace().setup()
        ws.rebuild(
            [FileOperation("create", "helper.py", content="X = 1\n")],
            make_plan(create=["helper.py"]),
        )
        self.assertTrue((ws.root / "helper.py").exists())
        self.assertFalse((self.base_dir / "helper.py").exists())

    def test_delete_file(self):
        ws = self.make_workspace().setup()
        ws.rebuild(
            [FileOperation("delete", "README.md")],
            make_plan("README.md"),
        )
        self.assertFalse((ws.root / "README.md").exists())
        self.assertTrue((self.base_dir / "README.md").exists())

    def test_multi_file_operations(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [
                FileOperation("modify", "module.py", content=FIXED_MODULE),
                FileOperation("create", "extra.py", content="Y = 2\n"),
                FileOperation("delete", "README.md"),
            ],
            make_plan("module.py", "README.md", create=["extra.py"]),
        )
        self.assertEqual(sorted(changed), ["README.md", "extra.py", "module.py"])
        self.assertEqual(ws.read_candidate("module.py"), FIXED_MODULE)
        self.assertTrue((ws.root / "extra.py").exists())
        self.assertFalse((ws.root / "README.md").exists())

    def test_patch_operation_applies(self):
        ws = self.make_workspace().setup()
        patch = (
            "--- a/module.py\n"
            "+++ b/module.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b\n"
        )
        ws.rebuild([FileOperation("modify", "module.py", patch=patch)], make_plan("module.py"))
        self.assertEqual(ws.read_candidate("module.py"), FIXED_MODULE)

    def test_malformed_patch_is_rejected(self):
        ws = self.make_workspace().setup()
        bad = "--- a/module.py\n+++ b/module.py\n@@ -1,2 +1,2 @@\n-nonexistent line\n+other\n"
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild([FileOperation("modify", "module.py", patch=bad)], make_plan("module.py"))

    def test_duplicate_paths_rejected(self):
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [
                    FileOperation("modify", "module.py", content=FIXED_MODULE),
                    FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE),
                ],
                make_plan("module.py"),
            )

    def test_unsupported_action_rejected(self):
        """``rename`` is not part of the FileOperation vocabulary; it must be refused."""
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [FileOperation("rename", "module.py", content="")],
                make_plan("module.py"),
            )

    def test_create_over_existing_file_rejected(self):
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [FileOperation("create", "module.py", content=FIXED_MODULE)],
                make_plan("module.py"),
            )

    def test_modify_missing_file_rejected(self):
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [FileOperation("modify", "ghost.py", content="x = 1\n")],
                make_plan("ghost.py"),
            )

    def test_rebuild_is_deterministic_not_cumulative(self):
        """BASE + CURRENT OPS, never patch-on-patch accumulation."""
        ws = self.make_workspace().setup()
        ws.rebuild(
            [
                FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE),
                FileOperation("create", "scratch.py", content="tmp = 1\n"),
            ],
            make_plan("module.py", create=["scratch.py"]),
        )
        self.assertTrue((ws.root / "scratch.py").exists())

        # Second candidate no longer includes scratch.py -> it must disappear,
        # and module.py must be derived from BASE, not from the prior candidate.
        ws.rebuild(
            [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            make_plan("module.py", create=["scratch.py"]),
        )
        self.assertFalse((ws.root / "scratch.py").exists())
        self.assertEqual(ws.read_candidate("module.py"), FIXED_MODULE)
        self.assertEqual(ws.mutated_paths, ["module.py"])

    def test_rebuild_equals_fresh_materialisation(self):
        """Rebuild-after-revert must equal a candidate built from scratch."""
        ops = [FileOperation("modify", "module.py", content=FIXED_MODULE)]
        plan = make_plan("module.py")

        reused = self.make_workspace().setup()
        reused.rebuild([FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)], plan)
        reused.rebuild(ops, plan)

        fresh = self.make_workspace().setup()
        fresh.rebuild(ops, plan)

        self.assertEqual(snapshot_tree(reused.root), snapshot_tree(fresh.root))

    def test_delete_then_restore_on_rebuild(self):
        ws = self.make_workspace().setup()
        ws.rebuild([FileOperation("delete", "README.md")], make_plan("README.md"))
        self.assertFalse((ws.root / "README.md").exists())
        ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], make_plan("module.py", "README.md"))
        self.assertTrue((ws.root / "README.md").exists())
        self.assertEqual((ws.root / "README.md").read_text(encoding="utf-8"), "tiny\n")

    def test_base_is_frozen_against_concurrent_authoritative_edits(self):
        """A later authoritative edit must not leak into a candidate rebuild."""
        ws = self.make_workspace().setup()
        plan = make_plan("module.py")
        ws.rebuild([FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)], plan)

        # Someone edits the authoritative file while the session is running.
        (self.base_dir / "module.py").write_text("SABOTAGE = True\n", encoding="utf-8")

        ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], plan)
        self.assertEqual(ws.read_candidate("module.py"), FIXED_MODULE)
        self.assertNotIn("SABOTAGE", ws.diff())
        self.assertIn("-    return a - b", ws.diff())

    def test_rebuild_count_tracked(self):
        ws = self.make_workspace().setup()
        plan = make_plan("module.py")
        ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], plan)
        ws.rebuild([FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)], plan)
        self.assertEqual(ws.rebuild_count, 2)


# ---------------------------------------------------------------------------
# 3. Safety: scope, protected paths, traversal
# ---------------------------------------------------------------------------


class TestCandidateSafetyAndScope(TempProjectCase):
    def setUp(self) -> None:
        super().setUp()
        (self.base_dir / "local_agent").mkdir()
        (self.base_dir / "local_agent" / "tool_engine.py").write_text(
            "ENGINE = 1\n", encoding="utf-8"
        )
        (self.base_dir / "local_agent" / "approval.py").write_text(
            "APPROVAL = 1\n", encoding="utf-8"
        )

    def test_out_of_plan_operation_rejected(self):
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [FileOperation("modify", "README.md", content="hacked\n")],
                make_plan("module.py"),
            )
        self.assertEqual((ws.root / "README.md").read_text(encoding="utf-8"), "tiny\n")

    def test_protected_tool_engine_rejected_out_of_scope(self):
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [FileOperation("modify", "local_agent/tool_engine.py", content="HACKED = 1\n")],
                make_plan("module.py"),
            )
        self.assertEqual(
            (ws.root / "local_agent" / "tool_engine.py").read_text(encoding="utf-8"),
            "ENGINE = 1\n",
        )

    def test_protected_approval_rejected_by_protected_paths(self):
        """Even when the plan allows it, protected_paths blocks the candidate."""
        ws = self.make_workspace(protected_paths={"local_agent/approval.py"}).setup()
        ws_ok = self.make_workspace().setup()
        plan = make_plan("local_agent/approval.py")
        op = FileOperation("modify", "local_agent/approval.py", content="HACKED = 1\n")

        # Sanity: without protected_paths and inside plan scope, it would apply.
        ws_ok.rebuild([op], plan)
        self.assertEqual(
            (ws_ok.root / "local_agent" / "approval.py").read_text(encoding="utf-8"),
            "HACKED = 1\n",
        )

        # Note protected_paths only blocks when the path is NOT in plan scope,
        # matching CodingAgent._validate_path exactly (no duplicated logic).
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild([op], make_plan("module.py"))

    def test_path_traversal_rejected(self):
        ws = self.make_workspace().setup()
        for evil in ("../escape.py", "../../escape.py", "sub/../../escape.py"):
            with self.assertRaises((SandboxViolation, UnsafeModificationError)):
                ws.rebuild(
                    [FileOperation("create", evil, content="pwn\n")],
                    make_plan(create=[evil]),
                )
        self.assertFalse((self.base_dir.parent / "escape.py").exists())

    def test_absolute_path_rejected(self):
        ws = self.make_workspace().setup()
        absolute = str(self.base_dir.parent / "abs_escape.py")
        with self.assertRaises((SandboxViolation, UnsafeModificationError)):
            ws.rebuild(
                [FileOperation("create", absolute, content="pwn\n")],
                make_plan(create=[absolute]),
            )
        self.assertFalse(Path(absolute).exists())

    def test_drive_letter_path_rejected(self):
        ws = self.make_workspace().setup()
        with self.assertRaises((SandboxViolation, UnsafeModificationError)):
            ws.rebuild(
                [FileOperation("create", "C:/Windows/pwn.py", content="pwn\n")],
                make_plan(create=["C:/Windows/pwn.py"]),
            )

    def test_backslash_paths_normalise(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [FileOperation("modify", "tests\\test_module.py", content=MODULE_TEST)],
            make_plan("tests/test_module.py"),
        )
        self.assertEqual(changed, ["tests/test_module.py"])

    def test_git_directory_is_protected(self):
        (self.base_dir / ".git").mkdir()
        ws = self.make_workspace().setup()
        with self.assertRaises((UnsafeModificationError, PermissionError)):
            ws.rebuild(
                [FileOperation("create", ".git/config", content="[core]\n")],
                make_plan(create=[".git/config"]),
            )

    def test_failed_rebuild_leaves_candidate_at_base(self):
        ws = self.make_workspace().setup()
        with self.assertRaises(UnsafeModificationError):
            ws.rebuild(
                [FileOperation("modify", "README.md", content="hacked\n")],
                make_plan("module.py"),
            )
        self.assertEqual(snapshot_tree(ws.root), snapshot_tree(self.base_dir))


# ---------------------------------------------------------------------------
# 4. Candidate diff
# ---------------------------------------------------------------------------


class TestCandidateDiff(TempProjectCase):
    def test_diff_reflects_base_to_candidate_only(self):
        ws = self.make_workspace().setup()
        ws.rebuild(
            [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            make_plan("module.py"),
        )
        diff = ws.diff()
        self.assertIn("a/module.py", diff)
        self.assertIn("+    return a + b", diff)
        self.assertIn("-    return a - b", diff)
        self.assertNotIn("README", diff)

    def test_diff_ignores_unrelated_authoritative_changes(self):
        ws = self.make_workspace().setup()
        ws.rebuild(
            [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            make_plan("module.py"),
        )
        # Someone edits an unrelated authoritative file after the mirror exists.
        (self.base_dir / "README.md").write_text("changed elsewhere\n", encoding="utf-8")
        diff = ws.diff()
        self.assertNotIn("README", diff)
        self.assertIn("module.py", diff)

    def test_diff_empty_when_no_operations(self):
        ws = self.make_workspace().setup()
        self.assertEqual(ws.diff(), "")


# ---------------------------------------------------------------------------
# 5. REAL behavioural validation (non-negotiable)
# ---------------------------------------------------------------------------


class TestRealProspectiveValidation(TempProjectCase):
    """Executes real subprocesses against candidate contents."""

    def test_broken_candidate_genuinely_fails_and_fixed_one_passes(self):
        ws = self.make_workspace().setup()
        validator = ProspectiveValidator()
        plan = make_plan("module.py")

        # Candidate 1: still wrong -> the real pytest run must fail.
        changed = ws.rebuild(
            [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)], plan
        )
        first = validator.validate(ws, changed)
        self.assertFalse(first.passed, first.render_feedback())
        self.assertEqual(first.failed_tier, "targeted_tests")
        self.assertGreaterEqual(first.commands_run, 2)
        failing = [r for r in first.failures]
        self.assertTrue(any("pytest" in r.display() for r in failing))
        self.assertIn("test_add", first.render_feedback())

        # Candidate 2: correct -> the same real pytest run must pass.
        changed = ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], plan)
        second = validator.validate(ws, changed)
        self.assertTrue(second.passed, second.render_feedback())
        self.assertEqual(second.failures, [])

        # And the authoritative tree still contains the ORIGINAL bug.
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_validation_reads_candidate_not_authoritative_tree(self):
        """Proof the test process imported candidate bytes, not base bytes."""
        ws = self.make_workspace().setup()
        plan = make_plan("module.py")
        changed = ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], plan)

        # The authoritative module is still broken; if validation ran against it
        # the suite would fail. It passes -> candidate contents were used.
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)
        report = validator_report = ProspectiveValidator().validate(ws, changed)
        self.assertTrue(report.passed, validator_report.render_feedback())

    def test_syntax_tier_catches_invalid_python_before_tests(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [FileOperation("modify", "module.py", content="def add(a, b)\n    return a\n")],
            make_plan("module.py"),
        )
        report = ProspectiveValidator().validate(ws, changed)
        self.assertFalse(report.passed)
        self.assertEqual(report.failed_tier, "syntax")
        # Fail fast: targeted tests were never reached.
        self.assertNotIn("targeted_tests", report.tiers_run)

    def test_commands_execute_with_candidate_cwd(self):
        ws = self.make_workspace().setup()
        result = ws.run(["python", "-c", "import os; print(os.getcwd())"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(Path(result.stdout.strip()).resolve(), ws.root)

    def test_missing_executable_is_skipped_not_failed(self):
        ws = self.make_workspace().setup()
        validator = ProspectiveValidator()
        spec = CommandSpec(name="ghost", command=("definitely_not_a_real_binary_xyz",))
        result = validator._execute(ws, spec, "static_analysis")
        self.assertTrue(result.skipped)
        self.assertTrue(result.succeeded)

    def test_no_targeted_tests_still_runs_syntax_tier(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [FileOperation("create", "orphan.py", content="Z = 1\n")],
            make_plan(create=["orphan.py"]),
        )
        report = ProspectiveValidator().validate(ws, changed)
        self.assertIn("syntax", report.tiers_run)
        self.assertTrue(report.passed)

    def test_non_python_change_produces_empty_but_passing_report(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [FileOperation("modify", "README.md", content="updated\n")],
            make_plan("README.md"),
        )
        report = ProspectiveValidator().validate(ws, changed)
        self.assertTrue(report.passed)
        self.assertEqual(report.commands_run, 0)

    def test_feedback_is_bounded_and_structured(self):
        ws = self.make_workspace().setup()
        changed = ws.rebuild(
            [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)],
            make_plan("module.py"),
        )
        report = ProspectiveValidator(max_output_chars=200).validate(ws, changed)
        self.assertEqual(report.max_output_chars, 200)
        feedback = report.render_feedback()
        self.assertIn("Candidate validation FAILED", feedback)
        self.assertIn("targeted_tests", feedback)
        self.assertLess(len(feedback), 4000)


# ---------------------------------------------------------------------------
# 6. Isolation between concurrent candidates
# ---------------------------------------------------------------------------


class TestCandidateConcurrencyIsolation(TempProjectCase):
    def test_two_workspaces_have_distinct_roots(self):
        a = self.make_workspace().setup()
        b = self.make_workspace().setup()
        self.assertNotEqual(a.root, b.root)

    def test_concurrent_candidates_do_not_cross_contaminate(self):
        a = self.make_workspace().setup()
        b = self.make_workspace().setup()
        plan = make_plan("module.py")
        a.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], plan)
        b.rebuild([FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)], plan)
        self.assertEqual(a.read_candidate("module.py"), FIXED_MODULE)
        self.assertEqual(b.read_candidate("module.py"), STILL_BROKEN_MODULE)
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_concurrent_commands_run_in_their_own_directory(self):
        workspaces = [self.make_workspace().setup() for _ in range(3)]
        errors: list[str] = []
        barrier = threading.Barrier(len(workspaces))

        def probe(ws: CandidateWorkspace) -> tuple[Path, Path]:
            barrier.wait(timeout=30)
            result = ws.run(["python", "-c", "import os; print(os.getcwd())"])
            if result.exit_code != 0:
                errors.append(result.stderr)
            return ws.root, Path(result.stdout.strip()).resolve()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            outcomes = list(pool.map(probe, workspaces))

        self.assertEqual(errors, [])
        for expected_root, actual_cwd in outcomes:
            self.assertEqual(actual_cwd, expected_root)
        self.assertEqual(len({cwd for _, cwd in outcomes}), 3)

    def test_concurrent_validation_sees_own_candidate_only(self):
        good = self.make_workspace().setup()
        bad = self.make_workspace().setup()
        plan = make_plan("module.py")
        good_changed = good.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], plan)
        bad_changed = bad.rebuild(
            [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)], plan
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            good_future = pool.submit(ProspectiveValidator().validate, good, good_changed)
            bad_future = pool.submit(ProspectiveValidator().validate, bad, bad_changed)
            good_report = good_future.result()
            bad_report = bad_future.result()

        self.assertTrue(good_report.passed, good_report.render_feedback())
        self.assertFalse(bad_report.passed)

    def test_parallel_agent_sessions_are_fully_independent(self):
        """Two concurrent implementation sessions on two distinct base trees."""
        roots: list[Path] = []
        for _ in range(2):
            root = Path(tempfile.mkdtemp(prefix="agentpar_")).resolve()
            make_tiny_project(root)
            self.addCleanup(shutil.rmtree, root, ignore_errors=True)
            roots.append(root)

        def session(root: Path) -> tuple[ImplementationResult, Path]:
            workspace = CandidateWorkspace(root)
            filesystem = ProjectFilesystem(root)
            agent = InteractiveCodingAgent(
                filesystem=filesystem,
                registry=ToolRegistry(root, filesystem=filesystem),
                sandbox=workspace,
                cleanup_sandbox=False,
            )
            provider = ScriptedProvider(
                responses=[
                    [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)],
                    [FileOperation("modify", "module.py", content=FIXED_MODULE)],
                ]
            )
            result = agent.execute(
                provider=provider,
                task_objective="fix",
                plan=make_plan("module.py"),
                context=ProjectContext(root=str(root)),
            )
            candidate_root = workspace.root
            workspace.cleanup()
            return result, candidate_root

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(session, roots))

        candidate_roots = {candidate for _, candidate in outcomes}
        self.assertEqual(len(candidate_roots), 2)
        for (result, candidate_root), root in zip(outcomes, roots):
            self.assertTrue(result.success, result.error_message)
            self.assertEqual(result.candidate_iterations, 2)
            self.assertFalse(candidate_root.exists())
            # Each authoritative tree still has the original bug.
            self.assertEqual((root / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_independent_cleanup(self):
        a = self.make_workspace().setup()
        b = self.make_workspace().setup()
        a_root, b_root = a.root, b.root
        a.cleanup()
        self.assertFalse(a_root.exists())
        self.assertTrue(b_root.exists())
        self.assertTrue((b_root / "module.py").exists())

    def test_workspace_registry_is_rooted_at_candidate(self):
        ws = self.make_workspace().setup()
        ws.rebuild([FileOperation("modify", "module.py", content=FIXED_MODULE)], make_plan("module.py"))
        result = ws.registry.execute(
            ToolCall(call_id="c1", tool_name="read_file_range", arguments={"path": "module.py"})
        )
        self.assertFalse(result.is_error)
        self.assertIn("return a + b", result.output)
        # The authoritative registry still sees the bug.
        base_registry = ToolRegistry(self.base_dir)
        base_result = base_registry.execute(
            ToolCall(call_id="c2", tool_name="read_file_range", arguments={"path": "module.py"})
        )
        self.assertIn("return a - b", base_result.output)


# ---------------------------------------------------------------------------
# 7. InteractiveCodingAgent prospective loop
# ---------------------------------------------------------------------------


class TestInteractiveAgentProspectiveLoop(TempProjectCase):
    def make_agent(self, provider_responses, **kwargs) -> tuple[InteractiveCodingAgent, ScriptedProvider, CandidateWorkspace]:
        workspace = self.make_workspace()
        filesystem = ProjectFilesystem(self.base_dir)
        registry = ToolRegistry(self.base_dir, filesystem=filesystem)
        provider = ScriptedProvider(responses=provider_responses)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=registry,
            sandbox=workspace,
            cleanup_sandbox=False,
            **kwargs,
        )
        return agent, provider, workspace

    def run_agent(self, agent, provider, plan=None) -> ImplementationResult:
        return agent.execute(
            provider=provider,
            task_objective="Fix add() so 2 + 3 == 5",
            plan=plan or make_plan("module.py"),
            context=ProjectContext(root=str(self.base_dir)),
        )

    def test_refinement_first_candidate_fails_second_passes(self):
        """The full loop: propose -> apply -> real validation -> refine -> pass."""
        agent, provider, ws = self.make_agent(
            [
                [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)],
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            ]
        )
        result = self.run_agent(agent, provider)

        self.assertTrue(result.success, result.error_message)
        self.assertTrue(result.final_candidate_success)
        self.assertEqual(
            result.termination_reason,
            ImplementationTerminationReason.CANDIDATE_VALIDATION_PASSED,
        )
        self.assertEqual(result.candidate_iterations, 2)
        self.assertEqual(result.candidate_validation_attempts, 2)
        self.assertEqual(result.candidate_validation_failures, 1)
        self.assertEqual(result.candidate_validation_successes, 1)
        self.assertEqual(result.candidate_recovery_attempts, 1)
        self.assertGreater(result.validation_commands_run, 0)
        self.assertEqual(result.candidate_files_changed, ["module.py"])
        self.assertEqual(result.file_operations[0].content, FIXED_MODULE)

        # The model genuinely observed the real failure.
        second_history = provider.histories[1]
        feedback = "\n".join(res.output for _, res in second_history)
        self.assertIn("Candidate validation FAILED", feedback)
        self.assertIn("pytest", feedback)

        # The candidate holds the fix; the authoritative tree still has the bug.
        self.assertEqual(ws.read_candidate("module.py"), FIXED_MODULE)
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_all_attempts_fail_produces_structured_failure_and_no_operations(self):
        agent, provider, _ = self.make_agent(
            [
                [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)],
                [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE + "\n")],
            ],
            max_candidate_iterations=2,
        )
        base_before = snapshot_tree(self.base_dir)
        result = self.run_agent(agent, provider)

        self.assertFalse(result.success)
        self.assertIsNone(result.file_operations)
        self.assertFalse(result.final_candidate_success)
        self.assertEqual(
            result.termination_reason,
            ImplementationTerminationReason.CANDIDATE_BUDGET_EXHAUSTED,
        )
        self.assertEqual(result.failure_category, "budget_exhaustion")
        self.assertTrue(result.is_recoverable_failure)
        self.assertEqual(result.candidate_validation_failures, 2)
        self.assertEqual(snapshot_tree(self.base_dir), base_before)

    def test_single_iteration_budget_terminates_immediately(self):
        agent, provider, _ = self.make_agent(
            [[FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)]],
            max_candidate_iterations=1,
        )
        result = self.run_agent(agent, provider)
        self.assertFalse(result.success)
        self.assertEqual(result.candidate_iterations, 1)
        self.assertEqual(
            result.termination_reason,
            ImplementationTerminationReason.CANDIDATE_BUDGET_EXHAUSTED,
        )

    def test_default_candidate_budget(self):
        agent, _, _ = self.make_agent([])
        self.assertEqual(agent.max_candidate_iterations, DEFAULT_MAX_CANDIDATE_ITERATIONS)
        self.assertTrue(agent.prospective_validation_enabled)

    def test_first_candidate_passing_short_circuits(self):
        agent, provider, _ = self.make_agent(
            [[FileOperation("modify", "module.py", content=FIXED_MODULE)]]
        )
        result = self.run_agent(agent, provider)
        self.assertTrue(result.success)
        self.assertEqual(result.candidate_iterations, 1)
        self.assertEqual(result.candidate_recovery_attempts, 0)

    def test_out_of_scope_candidate_reported_as_invalid_operations(self):
        agent, provider, _ = self.make_agent(
            [[FileOperation("modify", "README.md", content="nope\n")]],
            max_candidate_iterations=1,
        )
        result = self.run_agent(agent, provider)
        self.assertFalse(result.success)
        self.assertEqual(
            result.termination_reason,
            ImplementationTerminationReason.CANDIDATE_INVALID_OPERATIONS,
        )
        self.assertEqual(result.failure_category, "invalid_operations")
        self.assertEqual(
            (self.base_dir / "README.md").read_text(encoding="utf-8"), "tiny\n"
        )

    def test_invalid_operations_are_fed_back_for_refinement(self):
        agent, provider, _ = self.make_agent(
            [
                [FileOperation("modify", "README.md", content="nope\n")],
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            ],
            max_candidate_iterations=2,
        )
        result = self.run_agent(agent, provider)
        self.assertTrue(result.success, result.error_message)
        feedback = "\n".join(res.output for _, res in provider.histories[1])
        self.assertIn("could NOT be applied", feedback)

    def test_tools_operate_against_candidate_root(self):
        """The agent's exploration tools must see candidate state, not base state."""
        agent, provider, ws = self.make_agent(
            [
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
                ToolCall(
                    call_id="t1",
                    tool_name="read_file_range",
                    arguments={"path": "module.py", "start_line": 1, "end_line": 5},
                ),
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            ],
            max_candidate_iterations=3,
        )
        # First response passes immediately, so re-order: make the first fail.
        provider.responses = [
            [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)],
            ToolCall(
                call_id="t1",
                tool_name="read_file_range",
                arguments={"path": "module.py", "start_line": 1, "end_line": 5},
            ),
            [FileOperation("modify", "module.py", content=FIXED_MODULE)],
        ]
        result = self.run_agent(agent, provider)
        self.assertTrue(result.success, result.error_message)

        reads = [
            res.output
            for _, res in provider.histories[-1]
            if _.tool_name == "read_file_range"
        ]
        self.assertTrue(reads)
        # The read observed the FIRST (broken) candidate, proving the tool was
        # rooted at the candidate tree rather than the authoritative one.
        self.assertIn("return a * b", reads[0])

    def test_provider_failure_during_initial_generation_propagates(self):
        agent, provider, ws = self.make_agent([])
        provider.raise_on_step = 1
        with self.assertRaises(ProviderError):
            self.run_agent(agent, provider)
        # Sandbox is torn down even though the loop raised.
        agent.cleanup_sandbox = True
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_provider_failure_during_refinement_propagates(self):
        agent, provider, _ = self.make_agent(
            [[FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)]],
            max_candidate_iterations=3,
        )
        provider.raise_on_step = 2
        with self.assertRaises(ProviderError):
            self.run_agent(agent, provider)
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_sandbox_cleanup_happens_on_provider_failure(self):
        workspace = self.make_workspace()
        filesystem = ProjectFilesystem(self.base_dir)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
            sandbox=workspace,
            cleanup_sandbox=True,
        )
        provider = ScriptedProvider(raise_on_step=1)
        with self.assertRaises(ProviderError):
            self.run_agent(agent, provider)
        self.assertFalse(workspace.is_active)

    def test_validation_crash_does_not_touch_authoritative_tree(self):
        class ExplodingValidator(ProspectiveValidator):
            def validate(self, workspace, changed_files, repository_map=None):
                raise RuntimeError("validator exploded")

        agent, provider, _ = self.make_agent(
            [[FileOperation("modify", "module.py", content=FIXED_MODULE)]]
        )
        agent.validator = ExplodingValidator()
        before = snapshot_tree(self.base_dir)
        with self.assertRaises(RuntimeError):
            self.run_agent(agent, provider)
        self.assertEqual(snapshot_tree(self.base_dir), before)

    def test_candidate_setup_failure_is_structured(self):
        class BrokenWorkspace(CandidateWorkspace):
            def setup(self):
                raise CandidateWorkspaceError("no disk space")

        filesystem = ProjectFilesystem(self.base_dir)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
            sandbox=BrokenWorkspace(self.base_dir),
        )
        result = self.run_agent(agent, ScriptedProvider())
        self.assertFalse(result.success)
        self.assertEqual(
            result.termination_reason,
            ImplementationTerminationReason.CANDIDATE_SETUP_FAILED,
        )
        self.assertTrue(result.prospective_validation_used)

    def test_descriptor_is_deterministic_and_serialisable(self):
        agent, provider, _ = self.make_agent(
            [[FileOperation("modify", "module.py", content=FIXED_MODULE)]]
        )
        result = self.run_agent(agent, provider)
        descriptor = result.candidate_descriptor
        self.assertEqual(descriptor["base_root"], str(self.base_dir))
        self.assertEqual(descriptor["iteration"], 1)
        self.assertTrue(descriptor["validation_passed"])
        self.assertEqual(len(descriptor["operations_digest"]), 16)

        restored = ImplementationResult.from_dict(result.to_dict())
        self.assertEqual(restored.candidate_descriptor, descriptor)
        self.assertEqual(restored.candidate_iterations, result.candidate_iterations)
        self.assertTrue(restored.final_candidate_success)
        self.assertIsNotNone(restored.candidate_validation_report)

    def test_prompt_mentions_prospective_validation_only_when_enabled(self):
        agent, _, _ = self.make_agent([])
        self.assertIn("PROSPECTIVE VALIDATION", agent.build_prompt("obj"))

        filesystem = ProjectFilesystem(self.base_dir)
        plain = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
        )
        self.assertNotIn("PROSPECTIVE VALIDATION", plain.build_prompt("obj"))

    def test_single_shot_fallback_still_used_without_tool_capability(self):
        workspace = self.make_workspace()
        filesystem = ProjectFilesystem(self.base_dir)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
            sandbox=workspace,
        )
        provider = ScriptedProvider(
            supports_tools=False,
            single_shot_ops=[FileOperation("modify", "module.py", content=FIXED_MODULE)],
        )
        result = self.run_agent(agent, provider)
        self.assertTrue(result.success)
        self.assertTrue(result.used_fallback)
        self.assertEqual(
            result.termination_reason,
            ImplementationTerminationReason.SINGLE_SHOT_FALLBACK,
        )
        self.assertFalse(result.prospective_validation_used)


# ---------------------------------------------------------------------------
# 8. Backward compatibility (modes A and B)
# ---------------------------------------------------------------------------


class TestBackwardCompatibility(TempProjectCase):
    def test_mode_b_phase_415_behaviour_unchanged(self):
        """interactive_implementation on, prospective validation off."""
        filesystem = ProjectFilesystem(self.base_dir)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
        )
        self.assertFalse(agent.prospective_validation_enabled)
        provider = ScriptedProvider(
            responses=[[FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)]]
        )
        result = agent.execute(
            provider=provider,
            task_objective="fix",
            plan=make_plan("module.py"),
            context=ProjectContext(root=str(self.base_dir)),
        )
        # Phase 4.15 accepts a syntactically valid but behaviourally wrong patch.
        self.assertTrue(result.success)
        self.assertEqual(result.termination_reason, "completed")
        self.assertFalse(result.prospective_validation_used)
        self.assertEqual(result.candidate_iterations, 0)
        self.assertEqual(result.validation_commands_run, 0)
        self.assertFalse(result.final_candidate_success)

    def test_mode_b_precheck_refinement_still_works(self):
        filesystem = ProjectFilesystem(self.base_dir)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
        )
        provider = ScriptedProvider(
            responses=[
                [FileOperation("modify", "module.py", content="def add(a, b)\n  bad\n")],
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            ]
        )
        result = agent.execute(
            provider=provider,
            task_objective="fix",
            plan=make_plan("module.py"),
            context=ProjectContext(root=str(self.base_dir)),
        )
        self.assertTrue(result.success)
        feedback = "\n".join(res.output for _, res in provider.histories[1])
        self.assertIn("Pre-mutation check rejected", feedback)

    def test_mode_c_precheck_runs_before_candidate_build(self):
        """Syntax errors are still caught cheaply, before any candidate build."""
        workspace = self.make_workspace()
        filesystem = ProjectFilesystem(self.base_dir)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.base_dir, filesystem=filesystem),
            sandbox=workspace,
            cleanup_sandbox=False,
        )
        provider = ScriptedProvider(
            responses=[
                [FileOperation("modify", "module.py", content="def add(a, b)\n  bad\n")],
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            ]
        )
        result = agent.execute(
            provider=provider,
            task_objective="fix",
            plan=make_plan("module.py"),
            context=ProjectContext(root=str(self.base_dir)),
        )
        self.assertTrue(result.success)
        # Only one candidate was ever built (the syntactically valid one).
        self.assertEqual(result.candidate_iterations, 1)


# ---------------------------------------------------------------------------
# 9. Configuration
# ---------------------------------------------------------------------------


class TestProspectiveConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.project = Path(tempfile.mkdtemp(prefix="agentcfg_")).resolve()
        self.addCleanup(shutil.rmtree, self.project, ignore_errors=True)

    def test_defaults_preserve_backward_compatibility(self):
        config = AgentConfig.from_environment(self.project)
        config.validate()
        self.assertFalse(config.prospective_validation_enabled)
        self.assertEqual(config.max_candidate_iterations, 2)
        self.assertEqual(config.candidate_validation_timeout_seconds, 120)

    def test_overrides_apply(self):
        config = AgentConfig.from_environment(
            self.project,
            prospective_validation_enabled=True,
            max_candidate_iterations=5,
            candidate_validation_timeout_seconds=45,
        )
        config.validate()
        self.assertTrue(config.prospective_validation_enabled)
        self.assertEqual(config.max_candidate_iterations, 5)
        self.assertEqual(config.candidate_validation_timeout_seconds, 45)

    def test_environment_variables(self):
        env = {
            "AGENT_PROSPECTIVE_VALIDATION": "true",
            "AGENT_MAX_CANDIDATE_ITERATIONS": "3",
            "AGENT_CANDIDATE_VALIDATION_TIMEOUT": "77",
        }
        original = {key: os.environ.get(key) for key in env}
        os.environ.update(env)
        try:
            config = AgentConfig.from_environment(self.project)
            config.validate()
            self.assertTrue(config.prospective_validation_enabled)
            self.assertEqual(config.max_candidate_iterations, 3)
            self.assertEqual(config.candidate_validation_timeout_seconds, 77)
        finally:
            for key, value in original.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    def test_invalid_values_rejected(self):
        with self.assertRaises(ValueError):
            AgentConfig.from_environment(self.project, max_candidate_iterations=0)
        config = AgentConfig.from_environment(self.project)
        config.max_candidate_iterations = 0
        with self.assertRaises(ValueError):
            config.validate()

    def test_validate_does_not_falsely_reject_defaults(self):
        """Regression guard for the Phase 4.15 validate() bug."""
        config = AgentConfig.from_environment(self.project)
        config.validate()  # must not raise
        self.assertTrue(config.max_implementation_tool_steps >= 1)

    def test_cli_flags(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, include_provider_args=False)
        args = parser.parse_args(
            [
                "--project", str(self.project),
                "--prospective-validation", "true",
                "--max-candidate-iterations", "4",
                "--interactive-implementation", "true",
            ]
        )
        config = config_from_args(args)
        self.assertTrue(config.prospective_validation_enabled)
        self.assertEqual(config.max_candidate_iterations, 4)
        self.assertTrue(config.interactive_implementation)

    def test_cli_defaults_leave_feature_off(self):
        parser = argparse.ArgumentParser()
        add_common_arguments(parser, include_provider_args=False)
        args = parser.parse_args(["--project", str(self.project)])
        config = config_from_args(args)
        self.assertFalse(config.prospective_validation_enabled)
        self.assertFalse(config.interactive_implementation)


# ---------------------------------------------------------------------------
# 10. Telemetry model
# ---------------------------------------------------------------------------


class TestTelemetryModel(unittest.TestCase):
    def test_new_fields_default_to_inert_values(self):
        result = ImplementationResult()
        self.assertFalse(result.prospective_validation_used)
        self.assertEqual(result.candidate_iterations, 0)
        self.assertEqual(result.candidate_files_changed, [])
        self.assertEqual(result.candidate_descriptor, {})
        self.assertIsNone(result.candidate_validation_report)
        self.assertFalse(result.final_candidate_success)

    def test_roundtrip_preserves_candidate_telemetry(self):
        result = ImplementationResult(
            success=True,
            prospective_validation_used=True,
            candidate_iterations=3,
            candidate_validation_attempts=3,
            candidate_validation_successes=1,
            candidate_validation_failures=2,
            candidate_recovery_attempts=2,
            candidate_files_changed=["a.py", "b.py"],
            candidate_elapsed_seconds=1.25,
            candidate_cleanup_failures=1,
            validation_commands_run=6,
            validation_runtime_seconds=0.75,
            final_candidate_success=True,
            candidate_descriptor={"iteration": 3},
            candidate_validation_report={"passed": True},
        )
        restored = ImplementationResult.from_dict(result.to_dict())
        self.assertEqual(restored.candidate_iterations, 3)
        self.assertEqual(restored.candidate_files_changed, ["a.py", "b.py"])
        self.assertEqual(restored.candidate_cleanup_failures, 1)
        self.assertEqual(restored.validation_commands_run, 6)
        self.assertAlmostEqual(restored.validation_runtime_seconds, 0.75)
        self.assertTrue(restored.final_candidate_success)
        self.assertEqual(restored.candidate_descriptor, {"iteration": 3})
        self.assertEqual(restored.candidate_validation_report, {"passed": True})

    def test_termination_reason_vocabulary_extended_not_replaced(self):
        reason = ImplementationTerminationReason
        for legacy in ("completed", "single_shot_fallback", "max_steps_exceeded", "no_operations"):
            self.assertIn(legacy, reason.CATEGORIES)
        self.assertEqual(
            reason.categorize(reason.CANDIDATE_VALIDATION_PASSED), "none"
        )
        self.assertEqual(
            reason.categorize(reason.CANDIDATE_VALIDATION_FAILED),
            "candidate_validation_failure",
        )
        self.assertEqual(
            reason.categorize(reason.CANDIDATE_BUDGET_EXHAUSTED), "budget_exhaustion"
        )
        self.assertIn(reason.CANDIDATE_VALIDATION_FAILED, reason.FAILURE_REASONS)
        self.assertNotIn(reason.CANDIDATE_VALIDATION_PASSED, reason.FAILURE_REASONS)

    def test_validation_report_serialisation(self):
        report = CandidateValidationReport(
            passed=False,
            failed_tier="targeted_tests",
            changed_files=["module.py"],
            results=[
                CandidateCommandResult(
                    name="pytest",
                    command=("pytest", "tests/test_module.py"),
                    tier="targeted_tests",
                    exit_code=1,
                    stdout="assert 6 == 5",
                )
            ],
        )
        data = report.to_dict()
        self.assertFalse(data["passed"])
        self.assertEqual(data["failed_tier"], "targeted_tests")
        self.assertEqual(data["results"][0]["exit_code"], 1)
        self.assertIn("assert 6 == 5", report.render_feedback())


# ---------------------------------------------------------------------------
# 11. Orchestrator end-to-end integration
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration(TempProjectCase):
    def _build_orchestrator(self, **config_overrides):
        from local_agent.orchestrator import Orchestrator
        from local_agent.storage import JsonFileStorage

        storage_dir = Path(tempfile.mkdtemp(prefix="agentstore_")).resolve()
        self.addCleanup(shutil.rmtree, storage_dir, ignore_errors=True)
        settings: dict[str, Any] = {
            "provider": "mock",
            "interactive_implementation": True,
            "prospective_validation_enabled": True,
            "max_iterations": 1,
        }
        settings.update(config_overrides)
        config = AgentConfig.from_environment(self.base_dir, **settings)
        config.validate()
        storage = JsonFileStorage(storage_dir)
        orchestrator = Orchestrator(
            config, storage, None, threading.Lock(), threading.Lock()
        )
        return orchestrator, config

    def _make_task(self) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Task(
            task_id="t-prospective",
            objective="Fix add() so that add(2, 3) == 5",
            status=TaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

    def test_full_orchestrator_pipeline_applies_verified_candidate(self):
        """orchestrator -> InteractiveCodingAgent -> sandbox -> validation -> apply."""
        orchestrator, _ = self._build_orchestrator()
        provider = ScriptedProvider(
            responses=[
                [FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)],
                [FileOperation("modify", "module.py", content=FIXED_MODULE)],
            ]
        )
        # Route every specialist role to the scripted provider.
        orchestrator.router.execute_with_fallback = (  # type: ignore[assignment]
            lambda role, action, stage_name: action(provider)
        )

        report = RunReport(project=ProjectContext(root=str(self.base_dir)))
        task = self._make_task()
        plan = make_plan("module.py")
        context = ProjectContext(root=str(self.base_dir))

        operations, _history = orchestrator._execute_code_generation(
            task, plan, context, failure=None, review=None,
            stage_name="implementation", report=report,
        )

        self.assertIsNotNone(operations)
        self.assertEqual(operations[0].content, FIXED_MODULE)
        # Only the verified result came back; the tree is STILL unmodified,
        # because applying is the job of the unchanged approval/apply pipeline.
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

        impl = report.implementation_result
        self.assertIsNotNone(impl)
        self.assertTrue(impl.prospective_validation_used)
        self.assertTrue(impl.final_candidate_success)
        self.assertEqual(impl.candidate_iterations, 2)

        # Now the existing pipeline applies it for real.
        agent = CodingAgent(orchestrator.filesystem)
        agent.apply(operations, plan)
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), FIXED_MODULE)

    def test_orchestrator_surfaces_failed_candidate_as_provider_error(self):
        orchestrator, _ = self._build_orchestrator(max_candidate_iterations=1)
        provider = ScriptedProvider(
            responses=[[FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)]]
        )
        orchestrator.router.execute_with_fallback = (  # type: ignore[assignment]
            lambda role, action, stage_name: action(provider)
        )

        report = RunReport(project=ProjectContext(root=str(self.base_dir)))
        with self.assertRaises(ProviderError):
            orchestrator._execute_code_generation(
                self._make_task(),
                make_plan("module.py"),
                ProjectContext(root=str(self.base_dir)),
                failure=None,
                review=None,
                stage_name="implementation",
                report=report,
            )
        self.assertEqual((self.base_dir / "module.py").read_text(encoding="utf-8"), BUGGY_MODULE)

    def test_no_candidate_directories_leak_into_project(self):
        orchestrator, _ = self._build_orchestrator()
        provider = ScriptedProvider(
            responses=[[FileOperation("modify", "module.py", content=FIXED_MODULE)]]
        )
        orchestrator.router.execute_with_fallback = (  # type: ignore[assignment]
            lambda role, action, stage_name: action(provider)
        )
        orchestrator._execute_code_generation(
            self._make_task(),
            make_plan("module.py"),
            ProjectContext(root=str(self.base_dir)),
            failure=None,
            review=None,
            stage_name="implementation",
            report=RunReport(project=ProjectContext(root=str(self.base_dir))),
        )
        leaked = [p.name for p in self.base_dir.iterdir() if p.name.startswith("agentcand_")]
        self.assertEqual(leaked, [])

    def test_checkpoint_round_trips_candidate_telemetry(self):
        """The candidate descriptor survives checkpoint -> resume."""
        orchestrator, _ = self._build_orchestrator()
        provider = ScriptedProvider(
            responses=[[FileOperation("modify", "module.py", content=FIXED_MODULE)]]
        )
        orchestrator.router.execute_with_fallback = (  # type: ignore[assignment]
            lambda role, action, stage_name: action(provider)
        )
        report = RunReport(project=ProjectContext(root=str(self.base_dir)))
        task = self._make_task()
        orchestrator.storage.save_task(task)
        orchestrator._execute_code_generation(
            task,
            make_plan("module.py"),
            ProjectContext(root=str(self.base_dir)),
            failure=None,
            review=None,
            stage_name="implementation",
            report=report,
        )
        self.assertIsNotNone(report.implementation_result)

        checkpoint = orchestrator._create_checkpoint(
            task, None, "after candidate loop",
            ProjectContext(root=str(self.base_dir)), report,
        )
        stored = checkpoint.continuation_context.get("implementation_result")
        self.assertIsInstance(stored, dict)
        restored = ImplementationResult.from_dict(stored)
        self.assertTrue(restored.prospective_validation_used)
        self.assertTrue(restored.final_candidate_success)
        self.assertEqual(restored.candidate_iterations, 1)
        self.assertEqual(restored.candidate_descriptor["base_root"], str(self.base_dir))
        # No filesystem snapshot is persisted; only a deterministic description.
        self.assertNotIn("candidate_root", stored)

    def test_prospective_disabled_falls_back_to_phase_415(self):
        orchestrator, _ = self._build_orchestrator(prospective_validation_enabled=False)
        provider = ScriptedProvider(
            responses=[[FileOperation("modify", "module.py", content=STILL_BROKEN_MODULE)]]
        )
        orchestrator.router.execute_with_fallback = (  # type: ignore[assignment]
            lambda role, action, stage_name: action(provider)
        )
        report = RunReport(project=ProjectContext(root=str(self.base_dir)))
        operations, _ = orchestrator._execute_code_generation(
            self._make_task(),
            make_plan("module.py"),
            ProjectContext(root=str(self.base_dir)),
            failure=None,
            review=None,
            stage_name="implementation",
            report=report,
        )
        self.assertEqual(operations[0].content, STILL_BROKEN_MODULE)
        self.assertFalse(report.implementation_result.prospective_validation_used)


# ---------------------------------------------------------------------------
# 12. Realistic acceptance scenario
# ---------------------------------------------------------------------------

ACCEPTANCE_MODELS = (
    "from dataclasses import dataclass\n"
    "\n"
    "\n"
    "@dataclass\n"
    "class User:\n"
    "    user_id: int\n"
    "    name: str\n"
)

ACCEPTANCE_REPOSITORY = (
    "from src.models import User\n"
    "\n"
    "\n"
    "class UserRepository:\n"
    "    def __init__(self):\n"
    "        self._users = {1: User(1, 'ada'), 2: User(2, 'linus')}\n"
    "\n"
    "    def get(self, user_id):\n"
    "        return self._users.get(user_id)\n"
)

# Old API: get_name(user_id) -> str. New API: get_name(user_id, upper=False).
ACCEPTANCE_SERVICE_V1 = (
    "from src.repository import UserRepository\n"
    "\n"
    "\n"
    "class UserService:\n"
    "    def __init__(self, repository=None):\n"
    "        self.repository = repository or UserRepository()\n"
    "\n"
    "    def get_name(self, user_id):\n"
    "        user = self.repository.get(user_id)\n"
    "        return user.name if user else ''\n"
)

# Imperfect candidate: signature changed but the flag is ignored.
ACCEPTANCE_SERVICE_BROKEN = (
    "from src.repository import UserRepository\n"
    "\n"
    "\n"
    "class UserService:\n"
    "    def __init__(self, repository=None):\n"
    "        self.repository = repository or UserRepository()\n"
    "\n"
    "    def get_name(self, user_id, upper=False):\n"
    "        user = self.repository.get(user_id)\n"
    "        return user.name if user else ''\n"
)

ACCEPTANCE_SERVICE_FIXED = (
    "from src.repository import UserRepository\n"
    "\n"
    "\n"
    "class UserService:\n"
    "    def __init__(self, repository=None):\n"
    "        self.repository = repository or UserRepository()\n"
    "\n"
    "    def get_name(self, user_id, upper=False):\n"
    "        user = self.repository.get(user_id)\n"
    "        name = user.name if user else ''\n"
    "        return name.upper() if upper else name\n"
)

ACCEPTANCE_SERVICE_TEST = (
    "from src.service import UserService\n"
    "\n"
    "\n"
    "def test_get_name():\n"
    "    assert UserService().get_name(1) == 'ada'\n"
    "\n"
    "\n"
    "def test_get_name_upper():\n"
    "    assert UserService().get_name(1, upper=True) == 'ADA'\n"
)

ACCEPTANCE_REPO_TEST = (
    "from src.repository import UserRepository\n"
    "\n"
    "\n"
    "def test_get():\n"
    "    assert UserRepository().get(2).name == 'linus'\n"
)


class TestRealisticAcceptanceScenario(unittest.TestCase):
    """Change a service API and update callers, proven by real candidate runs."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp(prefix="agentacc_")).resolve()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "src").mkdir()
        (self.root / "tests").mkdir()
        (self.root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "src" / "models.py").write_text(ACCEPTANCE_MODELS, encoding="utf-8")
        (self.root / "src" / "repository.py").write_text(ACCEPTANCE_REPOSITORY, encoding="utf-8")
        (self.root / "src" / "service.py").write_text(ACCEPTANCE_SERVICE_V1, encoding="utf-8")
        (self.root / "tests" / "test_service.py").write_text(
            ACCEPTANCE_SERVICE_TEST, encoding="utf-8"
        )
        (self.root / "tests" / "test_repository.py").write_text(
            ACCEPTANCE_REPO_TEST, encoding="utf-8"
        )

    def test_imperfect_candidate_fails_then_refined_candidate_passes(self):
        workspace = CandidateWorkspace(self.root)
        self.addCleanup(workspace.cleanup)
        filesystem = ProjectFilesystem(self.root)
        agent = InteractiveCodingAgent(
            filesystem=filesystem,
            registry=ToolRegistry(self.root, filesystem=filesystem),
            sandbox=workspace,
            cleanup_sandbox=False,
            max_candidate_iterations=2,
        )
        provider = ScriptedProvider(
            responses=[
                [FileOperation("modify", "src/service.py", content=ACCEPTANCE_SERVICE_BROKEN)],
                [FileOperation("modify", "src/service.py", content=ACCEPTANCE_SERVICE_FIXED)],
            ]
        )
        plan = Plan(
            objective="Add an upper flag to UserService.get_name and keep callers working",
            files_to_inspect=["src/service.py", "tests/test_service.py"],
            files_likely_to_change=["src/service.py"],
            files_likely_to_create=[],
            steps=["change API"],
            validation_strategy=["pytest"],
            risks=[],
        )
        before = snapshot_tree(self.root)

        result = agent.execute(
            provider=provider,
            task_objective="Change the UserService API to support upper-casing and keep all callers working",
            plan=plan,
            context=ProjectContext(root=str(self.root)),
        )

        self.assertTrue(result.success, result.error_message)
        self.assertTrue(result.final_candidate_success)
        self.assertEqual(result.candidate_iterations, 2)
        self.assertEqual(result.candidate_validation_failures, 1)
        self.assertEqual(result.candidate_validation_successes, 1)

        # The model saw the real assertion failure from the first candidate.
        feedback = "\n".join(res.output for _, res in provider.histories[1])
        self.assertIn("test_get_name_upper", feedback)
        self.assertIn("Candidate validation FAILED", feedback)

        # The authoritative tree is byte-for-byte unchanged so far.
        self.assertEqual(snapshot_tree(self.root), before)

        # Only the existing mutation path changes the real repository.
        CodingAgent(filesystem).apply(result.file_operations, plan)
        self.assertEqual(
            (self.root / "src" / "service.py").read_text(encoding="utf-8"),
            ACCEPTANCE_SERVICE_FIXED,
        )
        real_run = CommandRunner(self.root, 120).run(
            CommandSpec("pytest", ("pytest", "tests", "-q"))
        )
        self.assertEqual(real_run.exit_code, 0, real_run.stdout + real_run.stderr)


if __name__ == "__main__":
    unittest.main()
