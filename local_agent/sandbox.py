"""Phase 4.16: staged overlay execution sandbox and prospective validation.

This module gives the implementation agent a genuine

    PROPOSE -> APPLY TO ISOLATED CANDIDATE TREE -> RUN REAL VALIDATION
            -> OBSERVE -> REFINE -> REBUILD -> REVALIDATE -> FINALIZE

loop. The key abstraction is :class:`CandidateWorkspace`, which materialises

    BASE TREE + CANDIDATE OPERATIONS = PROSPECTIVE TREE

in a disposable directory that the session explicitly owns.

Design notes
------------
* **Disposable filtered mirror.** The base tree is copied once per session into
  a private temporary root with cache/VCS/agent directories filtered out. A
  git temp worktree was rejected because the candidate must reflect the current
  *working tree* (including uncommitted edits from earlier iterations) and
  because ``git worktree add`` mutates shared ``.git`` metadata, which is
  hostile to parallel workers. A pure in-memory overlay was rejected because
  real ``pytest``/``ruff`` subprocesses cannot see it - observing real
  behaviour is the entire point of this phase.
* **Deterministic rebuild.** ``rebuild()`` reverts exactly the paths the
  previous iteration mutated back to their base bytes and then applies the new
  operation set, so the candidate is always ``BASE + CURRENT OPERATIONS`` and
  never patch-on-patch accumulation.
* **No global state.** Each workspace owns its root, its
  :class:`~local_agent.filesystem.ProjectFilesystem`, its
  :class:`~local_agent.commands.CommandRunner` (which passes an explicit
  ``cwd``) and its :class:`~local_agent.tools.ToolRegistry`. The process
  working directory is never mutated.
* **No duplicated safety logic.** Candidate operations are applied through the
  very same :class:`~local_agent.coding_agent.CodingAgent` ``prepare`` /
  ``apply_prepared`` pipeline the authoritative tree uses, so plan scope,
  protected paths, secret files and traversal protection behave identically.

CRITICAL INVARIANT: nothing in this module ever writes to the authoritative
project tree. A validated candidate only yields structured ``FileOperation``
objects, which still flow through the unchanged approval/apply pipeline.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .commands import CommandRunner
from .filesystem import ProjectFilesystem
from .models import CommandSpec, FileOperation, Plan
from .tools import ToolRegistry

LOGGER = logging.getLogger(__name__)

#: Directories never mirrored into a candidate tree. Copying these wholesale
#: would be slow, would drag stale bytecode/caches into every candidate and, in
#: the case of ``.git``/``.agent_worktrees``, would be actively dangerous.
EXCLUDED_DIRECTORY_NAMES: frozenset[str] = frozenset({
    ".git", ".hg", ".svn",
    ".agent_data", ".agent_worktrees",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".cache",
    "node_modules", ".venv", "venv", "env",
    "dist", "build", "coverage", "target", ".next", ".nuxt", ".gradle", ".idea",
})

#: Blobs larger than this are not source and are skipped during mirroring.
DEFAULT_MAX_MIRROR_FILE_BYTES = 8 * 1024 * 1024

#: Prefix for candidate roots, kept short so Windows MAX_PATH has headroom.
CANDIDATE_DIR_PREFIX = "agentcand_"

#: Exit code CommandRunner returns when the executable is not installed.
_EXECUTABLE_MISSING_EXIT_CODE = 127

#: Validation tier names, in the order :meth:`ProspectiveValidator.validate`
#: runs them. Named constants rather than inline strings so that
#: :attr:`CandidateValidationReport.has_test_evidence` cannot drift out of sync
#: with the tier that actually produces test evidence.
TIER_SYNTAX = "syntax"
TIER_TARGETED_TESTS = "targeted_tests"
TIER_STATIC_ANALYSIS = "static_analysis"


class CandidateWorkspaceError(RuntimeError):
    """Raised when a candidate tree cannot be materialised or rebuilt."""


@dataclass
class CandidateCommandResult:
    """One real command executed against the candidate tree."""

    name: str
    command: tuple[str, ...]
    tier: str
    exit_code: int
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""
    skipped: bool = False
    skip_reason: str = ""

    @property
    def succeeded(self) -> bool:
        return self.skipped or self.exit_code == 0

    def display(self) -> str:
        return " ".join(self.command)

    def failure_summary(self, max_chars: int = 1200) -> str:
        """Concise, model-facing summary of why this command failed."""
        blob = ""
        if self.stdout:
            blob += self.stdout
        if self.stderr:
            blob += ("\n" if blob else "") + self.stderr
        blob = blob.strip()
        if not blob:
            return f"(no output; exit code {self.exit_code})"
        if len(blob) <= max_chars:
            return blob
        head = max_chars // 3
        tail = max_chars - head
        return f"{blob[:head]}\n... [output trimmed] ...\n{blob[-tail:]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": list(self.command),
            "tier": self.tier,
            "exit_code": self.exit_code,
            "duration_seconds": round(self.duration_seconds, 4),
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
        }


@dataclass
class CandidateValidationReport:
    """Outcome of running real validation against a candidate tree."""

    passed: bool = True
    results: list[CandidateCommandResult] = field(default_factory=list)
    tiers_run: list[str] = field(default_factory=list)
    failed_tier: str | None = None
    changed_files: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    error_message: str | None = None
    #: Per-failure output budget applied by :meth:`render_feedback` by default.
    max_output_chars: int = 1200
    # --- Phase 4.17: semantic change-impact context (additive, optional) ---
    #: Impact confidence that shaped this run's targeted tier, when semantic
    #: analysis was enabled. Empty string when it was not.
    impact_confidence: str = ""
    #: Recommended validation scope from the same analysis.
    impact_scope: str = ""
    #: Compact, human-readable rendering of the impact report.
    impact_summary: str = ""

    @property
    def executed_results(self) -> list[CandidateCommandResult]:
        return [r for r in self.results if not r.skipped]

    @property
    def commands_run(self) -> int:
        return len(self.executed_results)

    @property
    def failures(self) -> list[CandidateCommandResult]:
        return [r for r in self.results if not r.succeeded]

    @property
    def executed_test_commands(self) -> list[CandidateCommandResult]:
        """Targeted-test commands that genuinely ran (not skipped)."""
        return [
            r for r in self.results
            if r.tier == TIER_TARGETED_TESTS and not r.skipped
        ]

    @property
    def has_test_evidence(self) -> bool:
        """Whether this report rests on at least one real test execution.

        A skipped command counts as *succeeded* - a missing linter must not block
        an implementation - which means ``passed`` alone cannot distinguish
        "every selected test passed" from "the test runner was not installed, so
        nothing ran". Callers that reason about how much post-apply validation is
        still required must consult this instead of trusting ``passed``.
        """
        return bool(self.executed_test_commands)

    def render_feedback(self, max_chars_per_failure: int | None = None) -> str:
        """Structured, compact feedback suitable for feeding back to the model."""
        max_chars_per_failure = (
            self.max_output_chars if max_chars_per_failure is None else max_chars_per_failure
        )
        lines: list[str] = []
        if self.passed:
            lines.append("Candidate validation PASSED against the proposed edits.")
            if not self.has_test_evidence:
                # Never let a pass imply test coverage it does not have.
                lines.append(
                    "NOTE: no test command actually executed (none was selected, or the "
                    "runner is unavailable), so this pass is not evidence that behaviour "
                    "is correct."
                )
        else:
            lines.append(
                "Candidate validation FAILED. Your proposed edits were applied to an "
                "isolated candidate copy of the repository and the commands below were "
                "actually executed against that copy. The real project was NOT modified."
            )
        if self.error_message:
            lines.append(f"Harness error: {self.error_message}")
        if self.changed_files:
            lines.append(f"Files changed by the candidate: {', '.join(self.changed_files)}")
        if self.impact_summary:
            # Telling the model *why* these particular tests ran is what lets it
            # reason about which of its edits caused a failure.
            lines.append(self.impact_summary)
        if self.failed_tier:
            lines.append(f"First failing validation tier: {self.failed_tier}")

        for result in self.results:
            if result.skipped:
                lines.append(f"[{result.tier}] SKIPPED {result.display()} ({result.skip_reason})")
                continue
            status = "PASS" if result.succeeded else "FAIL"
            lines.append(
                f"[{result.tier}] {status} exit={result.exit_code} "
                f"({result.duration_seconds:.2f}s): {result.display()}"
            )
            if not result.succeeded:
                lines.append(result.failure_summary(max_chars_per_failure))

        if not self.passed:
            lines.append(
                "Diagnose the failure above, inspect the specific file(s) involved, and "
                "re-emit ONLY the corrected file operations. Do not regenerate unrelated files."
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "tiers_run": list(self.tiers_run),
            "failed_tier": self.failed_tier,
            "changed_files": list(self.changed_files),
            "elapsed_seconds": round(self.elapsed_seconds, 4),
            "error_message": self.error_message,
            "results": [r.to_dict() for r in self.results],
            "impact_confidence": self.impact_confidence,
            "impact_scope": self.impact_scope,
            "has_test_evidence": self.has_test_evidence,
        }


class CandidateWorkspace:
    """An isolated, disposable prospective tree: BASE + CANDIDATE OPERATIONS.

    The workspace is explicitly owned by whoever constructs it. It never mutates
    the base tree, never changes the process working directory, and shares no
    state with any other workspace instance.
    """

    def __init__(
        self,
        base_root: str | Path,
        *,
        workspace_parent: str | Path | None = None,
        protected_paths: set[str] | None = None,
        command_timeout_seconds: int = 120,
        max_file_bytes: int = DEFAULT_MAX_MIRROR_FILE_BYTES,
        excluded_directories: Iterable[str] | None = None,
        semantic_index: Any | None = None,
    ):
        self.base_root = Path(base_root).expanduser().resolve()
        if not self.base_root.is_dir():
            raise CandidateWorkspaceError(f"base tree does not exist: {self.base_root}")
        self.base_filesystem = ProjectFilesystem(self.base_root)
        self.workspace_parent = (
            Path(workspace_parent).expanduser().resolve() if workspace_parent else None
        )
        self.protected_paths = set(protected_paths or set())
        self.command_timeout_seconds = int(command_timeout_seconds)
        self.max_file_bytes = int(max_file_bytes)
        self.excluded_directories = {
            name.lower() for name in (excluded_directories or EXCLUDED_DIRECTORY_NAMES)
        }
        self.semantic_index = semantic_index

        self._root: Path | None = None
        self._filesystem: ProjectFilesystem | None = None
        self._runner: CommandRunner | None = None
        self._registry: ToolRegistry | None = None
        self._mutated_paths: list[str] = []
        #: Frozen BASE content for every path a candidate has ever touched.
        #: ``None`` means the path did not exist in BASE.
        self._base_bytes: dict[str, bytes | None] = {}
        self._closed = False

        # Telemetry
        self.files_mirrored = 0
        self.bytes_mirrored = 0
        self.rebuild_count = 0
        self.cleanup_failures = 0
        self.setup_seconds = 0.0

    # -- lifecycle ---------------------------------------------------------

    @property
    def root(self) -> Path:
        if self._root is None:
            raise CandidateWorkspaceError("candidate workspace has not been set up")
        return self._root

    @property
    def is_active(self) -> bool:
        return self._root is not None and not self._closed

    @property
    def filesystem(self) -> ProjectFilesystem:
        if self._filesystem is None:
            raise CandidateWorkspaceError("candidate workspace has not been set up")
        return self._filesystem

    @property
    def runner(self) -> CommandRunner:
        if self._runner is None:
            raise CandidateWorkspaceError("candidate workspace has not been set up")
        return self._runner

    @property
    def registry(self) -> ToolRegistry:
        """Tool registry rooted at the candidate tree.

        Composed around the unmodified :class:`ToolRegistry`; ``ToolEngine`` and
        the registry itself stay candidate-agnostic. Because both the filesystem
        and the command runner are rooted here, ``read_file_range``,
        ``grep_code``, ``find_files``, ``search_symbols`` and
        ``run_command_sandbox`` all observe candidate state and every subprocess
        gets ``cwd=<candidate root>``.
        """
        if self._registry is None:
            raise CandidateWorkspaceError("candidate workspace has not been set up")
        return self._registry

    def setup(self) -> "CandidateWorkspace":
        """Materialise the filtered base mirror. Idempotent."""
        if self._root is not None:
            return self
        if self._closed:
            raise CandidateWorkspaceError("candidate workspace was already closed")

        started = time.perf_counter()
        if self.workspace_parent is not None:
            self.workspace_parent.mkdir(parents=True, exist_ok=True)
        root = Path(
            tempfile.mkdtemp(
                prefix=CANDIDATE_DIR_PREFIX,
                dir=str(self.workspace_parent) if self.workspace_parent else None,
            )
        ).resolve()

        try:
            self._mirror_tree(self.base_root, root)
        except OSError as exc:
            shutil.rmtree(root, ignore_errors=True)
            raise CandidateWorkspaceError(f"could not materialise candidate tree: {exc}") from exc

        self._root = root
        self._filesystem = ProjectFilesystem(root)
        self._runner = CommandRunner(root, self.command_timeout_seconds)
        self._registry = ToolRegistry(
            root,
            filesystem=self._filesystem,
            command_runner=self._runner,
            semantic_index=self.semantic_index,
        )
        self.setup_seconds = time.perf_counter() - started
        LOGGER.debug(
            "Materialised candidate workspace %s (%d files, %d bytes, %.3fs)",
            root, self.files_mirrored, self.bytes_mirrored, self.setup_seconds,
        )
        return self

    def _mirror_tree(self, source: Path, destination: Path) -> None:
        """Copy ``source`` into ``destination`` skipping caches, VCS and symlinks.

        Symlinks and Windows junctions are skipped entirely (matching the
        repository scanner) so a link cannot smuggle content in from outside the
        project or create an infinite recursion.
        """
        skip_roots = {destination}
        if self.workspace_parent is not None:
            skip_roots.add(self.workspace_parent)

        stack: list[tuple[Path, Path]] = [(source, destination)]
        while stack:
            src_dir, dst_dir = stack.pop()
            dst_dir.mkdir(parents=True, exist_ok=True)
            try:
                entries = list(src_dir.iterdir())
            except OSError:
                continue
            for entry in entries:
                if entry.is_symlink():
                    continue
                try:
                    resolved = entry.resolve()
                except OSError:
                    continue
                if any(resolved == skip or skip in resolved.parents for skip in skip_roots):
                    continue
                # Excluded names are skipped whatever their type: inside a Git
                # worktree ``.git`` is a *file* pointing back at the real
                # repository's admin directory, and mirroring it would make the
                # candidate look like a second worktree of the authoritative
                # repository.
                if entry.name.lower() in self.excluded_directories:
                    continue
                if entry.is_dir():
                    stack.append((entry, dst_dir / entry.name))
                    continue
                if not entry.is_file():
                    continue
                try:
                    size = entry.stat().st_size
                except OSError:
                    continue
                if size > self.max_file_bytes:
                    continue
                try:
                    shutil.copy2(entry, dst_dir / entry.name)
                except OSError:
                    continue
                self.files_mirrored += 1
                self.bytes_mirrored += size

    def cleanup(self) -> bool:
        """Remove the candidate tree. Idempotent; safe to call after exceptions.

        Uses the retry-with-backoff strategy Phase 4.14's ``WorktreeManager``
        established for Windows, where antivirus scanners and lingering
        ``pytest`` child handles routinely hold files open for a short while.
        """
        root = self._root
        self._closed = True
        self._registry = None
        self._runner = None
        self._filesystem = None
        self._root = None
        self._mutated_paths = []
        self._base_bytes = {}

        if root is None:
            return True
        # Defensive: never delete anything that is the base tree or contains it.
        if root == self.base_root or root in self.base_root.parents:
            LOGGER.error("Refusing to clean candidate root that contains the base tree: %s", root)
            self.cleanup_failures += 1
            return False

        for attempt in range(4):
            if not root.exists():
                return True
            try:
                # ``onexc`` replaced the deprecated ``onerror`` in Python 3.12.
                shutil.rmtree(root, onexc=_force_remove)
            except TypeError:  # pragma: no cover - older interpreters
                shutil.rmtree(root, ignore_errors=True)
            except OSError:
                pass
            if not root.exists():
                return True
            time.sleep(0.05 * (attempt + 1))

        if root.exists():
            shutil.rmtree(root, ignore_errors=True)
        if root.exists():
            self.cleanup_failures += 1
            LOGGER.warning("Could not fully clean candidate workspace: %s", root)
            return False
        return True

    def __enter__(self) -> "CandidateWorkspace":
        return self.setup()

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.cleanup()
        return False

    # -- candidate construction -------------------------------------------

    def rebuild(
        self,
        operations: list[FileOperation],
        plan: Plan | None = None,
    ) -> list[str]:
        """Reset the candidate to BASE and apply ``operations``.

        Operations go through the unmodified
        :class:`~local_agent.coding_agent.CodingAgent` pipeline rooted at the
        candidate, so plan-scope checks, protected-path checks, traversal
        protection and patch validation behave exactly as they will when the
        result is finally applied for real.

        Returns the list of candidate-relative paths the operations changed.
        Raises :class:`~local_agent.coding_agent.UnsafeModificationError`,
        :class:`~local_agent.coding_agent.PatchValidationError` or
        :class:`~local_agent.filesystem.SandboxViolation` exactly as the real
        pipeline would.
        """
        # Imported lazily: coding_agent imports nothing from this module, but
        # keeping the dependency one-directional avoids any import cycle risk.
        from .coding_agent import CodingAgent

        if not self.is_active:
            raise CandidateWorkspaceError("candidate workspace is not active")

        self._reset_mutated_paths()
        agent = CodingAgent(self.filesystem, protected_paths=self.protected_paths)
        prepared = agent.prepare(list(operations), plan)
        # The candidate is at BASE right now, so this snapshot freezes the true
        # base bytes. Reverts read from it rather than from the live
        # authoritative tree, so a concurrent edit to the real repository can
        # never leak into a later candidate iteration.
        self._record_base_bytes(change.path for change in prepared)
        changed = agent.apply_prepared(prepared)
        self._mutated_paths = list(changed)
        self._purge_derived_caches()
        self.rebuild_count += 1
        return list(changed)

    def _purge_derived_caches(self) -> None:
        """Drop bytecode/test caches produced by a previous candidate iteration.

        Without this a rebuilt candidate can silently execute the *previous*
        candidate's code: CPython validates ``__pycache__`` entries on source
        (mtime, size), and two candidate revisions of the same file frequently
        share both - identical length and written inside the same clock second.
        The stale ``.pyc`` then wins and validation would observe the wrong
        program. Purging is cheap because the candidate tree is small.
        """
        root = self._root
        if root is None:
            return
        for name in ("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"):
            for directory in root.rglob(name):
                if directory.is_dir():
                    shutil.rmtree(directory, ignore_errors=True)

    def _record_base_bytes(self, relatives: Iterable[str]) -> None:
        for relative in relatives:
            if relative in self._base_bytes:
                continue
            candidate_path = self.root / relative
            try:
                self._base_bytes[relative] = (
                    candidate_path.read_bytes() if candidate_path.is_file() else None
                )
            except OSError as exc:
                raise CandidateWorkspaceError(
                    f"could not snapshot base content for '{relative}': {exc}"
                ) from exc

    def _reset_mutated_paths(self) -> None:
        """Revert previously applied candidate paths to their frozen base bytes."""
        for relative in self._mutated_paths:
            candidate_path = self.root / relative
            original = self._base_bytes.get(relative)
            try:
                if original is not None:
                    candidate_path.parent.mkdir(parents=True, exist_ok=True)
                    candidate_path.write_bytes(original)
                elif candidate_path.exists():
                    candidate_path.unlink()
            except OSError as exc:
                raise CandidateWorkspaceError(
                    f"could not reset candidate path '{relative}': {exc}"
                ) from exc
        self._mutated_paths = []

    @property
    def mutated_paths(self) -> list[str]:
        return list(self._mutated_paths)

    def base_contents(self, relatives: Iterable[str] | None = None) -> dict[str, str | None]:
        """Frozen BASE text for candidate-touched paths (Phase 4.17).

        A key mapped to ``None`` means the path did not exist in BASE, i.e. the
        candidate created it - that distinction is what lets
        :func:`~local_agent.semantic_impact.diff_python_symbols` tell an *added*
        symbol from a *modified* one exactly, instead of guessing.

        Only paths actually present in the frozen snapshot are returned; a
        caller asking about an unknown path gets no key, and the impact analyzer
        treats that absence as a degradation rather than as "the file is new".
        """
        wanted = (
            list(self._base_bytes)
            if relatives is None
            else [relative for relative in relatives if relative in self._base_bytes]
        )
        return {relative: _decode_or_none(self._base_bytes[relative]) for relative in wanted}

    def diff(self) -> str:
        """Unified diff of BASE -> CANDIDATE, limited to candidate operations.

        Deliberately independent of Git so it reflects only what this candidate
        did, never unrelated modifications present in the authoritative tree.
        """
        import difflib

        pieces: list[str] = []
        for relative in sorted(self._mutated_paths):
            original = _decode_or_empty(self._base_bytes.get(relative))
            current = _read_text_or_empty(self.root / relative)
            pieces.extend(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        return "".join(pieces)

    def read_candidate(self, relative: str) -> str:
        """Read a file as it exists in the candidate tree."""
        return self.filesystem.read_file(relative)

    def run(self, command: tuple[str, ...] | list[str], name: str | None = None) -> Any:
        """Execute a command with ``cwd`` explicitly set to the candidate root."""
        tokens = tuple(str(token) for token in command)
        spec = CommandSpec(name=name or " ".join(tokens), command=tokens)
        return self.runner.run(spec)


def _force_remove(func, path, exc) -> None:  # pragma: no cover - Windows path
    """``shutil.rmtree`` error handler that clears read-only bits and retries."""
    import os
    import stat

    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def _read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _to_universal_newlines(text: str) -> str:
    """Normalise CRLF/CR to LF, exactly as :meth:`Path.read_text` does.

    The BASE snapshot is kept as raw bytes, but every *candidate*-side read goes
    through ``read_text``, which translates newlines. Decoding BASE verbatim
    would therefore make the two sides incomparable on a CRLF checkout: the
    unified diff in :meth:`CandidateWorkspace.diff` would report every line of
    every file as changed, and symbol-level comparison would depend on
    ``splitlines`` happening to hide the difference. Normalising here keeps both
    sides in one text representation.
    """
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _decode_or_empty(data: bytes | None) -> str:
    if not data:
        return ""
    try:
        return _to_universal_newlines(data.decode("utf-8"))
    except UnicodeDecodeError:
        return ""


def _decode_or_none(data: bytes | None) -> str | None:
    """``None`` means "did not exist in BASE"; an undecodable blob also yields
    ``None`` so it is never mistaken for empty source."""
    if data is None:
        return None
    try:
        return _to_universal_newlines(data.decode("utf-8"))
    except UnicodeDecodeError:
        return None


class ProspectiveValidator:
    """Tiered, real validation of a candidate tree.

    Tiers run in increasing cost order and stop at the first genuine failure:

    ``syntax``
        One ``python -m compileall`` invocation over the changed Python files
        that exist in the candidate.
    ``targeted_tests``
        Test commands selected, when Phase 4.17 semantic impact analysis is
        enabled, from the import/reference graph via
        :class:`~local_agent.semantic_impact.SemanticChangeImpactAnalyzer`, and
        otherwise by the existing
        :class:`~local_agent.validation.ValidationIntelligence`
        ``discover_targeted_commands`` heuristics. Either way the commands are
        rooted at the candidate, so the whole repository suite is never re-run
        per iteration.
    ``static_analysis``
        Configured lint/type-check commands, capped and only when the tool is
        actually installed.

    Every command is executed by the workspace's own ``CommandRunner``, whose
    ``cwd`` is the candidate root - never the authoritative tree.
    """

    #: Whole-repository runners are excluded; the point is targeted validation.
    _WHOLE_REPO_COMMAND_NAMES = frozenset({"pytest", "unittest", "compileall"})

    #: How many targeted commands each recommended scope is allowed, as a
    #: multiple of ``max_targeted_commands``. Candidate-time validation is
    #: always *targeted* by construction (the authoritative full-suite run
    #: happens post-apply in the orchestrator and is never skipped), so a
    #: BROAD recommendation here means "run the widest targeted set we are
    #: allowed", not "run nothing" and not "run the whole suite twice".
    _SCOPE_BUDGET_MULTIPLIER: dict[str, int] = {"targeted": 1, "expanded": 2, "broad": 3}

    def __init__(
        self,
        max_targeted_commands: int = 4,
        max_static_commands: int = 1,
        enable_static_analysis: bool = True,
        max_output_chars: int = 1200,
        max_syntax_files: int = 50,
        semantic_impact_enabled: bool = False,
        max_impact_depth: int = 3,
        max_affected_symbols: int = 200,
        max_affected_tests: int = 8,
    ):
        self.max_syntax_files = max(1, int(max_syntax_files))
        self.max_targeted_commands = max(0, int(max_targeted_commands))
        self.max_static_commands = max(0, int(max_static_commands))
        self.enable_static_analysis = bool(enable_static_analysis)
        self.max_output_chars = int(max_output_chars)
        # Phase 4.17 semantic change-impact analysis. Off by default so modes
        # A/B/C from Phases 4.15/4.16 behave exactly as before.
        self.semantic_impact_enabled = bool(semantic_impact_enabled)
        self.max_impact_depth = max(0, int(max_impact_depth))
        self.max_affected_symbols = max(0, int(max_affected_symbols))
        self.max_affected_tests = max(0, int(max_affected_tests))
        #: Report produced by the most recent :meth:`validate` call, or ``None``
        #: when semantic analysis is disabled or failed. Read by the caller for
        #: telemetry and for the evidence ledger; never mutated by it.
        self.last_impact_report: Any | None = None

    def analyze_impact(
        self, workspace: CandidateWorkspace, changed_files: list[str]
    ) -> Any | None:
        """Run semantic impact analysis against the candidate tree.

        Returns ``None`` when the feature is off. On *any* analysis failure it
        returns a report whose confidence is LOW and whose scope is BROAD rather
        than ``None``, so a failure widens validation instead of disabling it.
        """
        if not self.semantic_impact_enabled:
            return None
        from .semantic_impact import SemanticChangeImpactAnalyzer

        try:
            analyzer = SemanticChangeImpactAnalyzer(
                workspace.root,
                semantic_index=None,
                max_impact_depth=self.max_impact_depth,
                max_affected_symbols=self.max_affected_symbols,
                max_affected_tests=self.max_affected_tests,
            )
            return analyzer.analyze(
                changed_files, base_contents=workspace.base_contents(changed_files)
            )
        except (OSError, ValueError, RecursionError) as exc:
            # Never let an analysis defect break the candidate loop. Falling
            # through with ``None`` makes _targeted_commands use the Phase 4.16
            # lexical path, which is the pre-existing, safe behaviour.
            LOGGER.warning("Semantic impact analysis failed for candidate: %s", exc)
            return None

    def validate(
        self,
        workspace: CandidateWorkspace,
        changed_files: list[str],
        repository_map: Any | None = None,
    ) -> CandidateValidationReport:
        started = time.perf_counter()
        report = CandidateValidationReport(
            changed_files=list(changed_files), max_output_chars=self.max_output_chars
        )
        impact = self.analyze_impact(workspace, changed_files)
        self.last_impact_report = impact
        if impact is not None:
            report.impact_confidence = impact.confidence
            report.impact_scope = impact.recommended_scope
            report.impact_summary = impact.summary()

        for tier_name, specs in (
            (TIER_SYNTAX, self._syntax_commands(workspace, changed_files)),
            (
                TIER_TARGETED_TESTS,
                self._targeted_commands(workspace, changed_files, repository_map, impact),
            ),
            (TIER_STATIC_ANALYSIS, self._static_commands(workspace, repository_map)),
        ):
            if not specs:
                continue
            report.tiers_run.append(tier_name)
            tier_failed = False
            for spec in specs:
                result = self._execute(workspace, spec, tier_name)
                report.results.append(result)
                if not result.succeeded:
                    tier_failed = True
            if tier_failed:
                report.passed = False
                report.failed_tier = tier_name
                break

        report.elapsed_seconds = time.perf_counter() - started
        return report

    # -- tier construction -------------------------------------------------

    def _syntax_commands(
        self, workspace: CandidateWorkspace, changed_files: list[str]
    ) -> list[CommandSpec]:
        python_files = sorted(
            path for path in changed_files
            if path.lower().endswith(".py") and (workspace.root / path).is_file()
        )
        if not python_files:
            return []
        # Bound the argument vector; Windows command lines are capped at ~32k.
        python_files = python_files[:self.max_syntax_files]
        return [
            CommandSpec(
                name="candidate_compileall",
                command=("python", "-m", "compileall", "-q", *python_files),
                reason="Candidate syntax check over changed Python files",
                category="type_check",
                risk="low",
            )
        ]

    def _targeted_commands(
        self,
        workspace: CandidateWorkspace,
        changed_files: list[str],
        repository_map: Any | None,
        impact: Any | None = None,
    ) -> list[CommandSpec]:
        """Select the targeted-test commands for this candidate.

        With Phase 4.17 enabled the ranked, explained
        :class:`~local_agent.semantic_impact.ValidationTarget` list drives the
        selection and the recommended scope sets the budget: weaker evidence
        buys a *larger* set of commands, never a smaller one. Without it, the
        Phase 4.16 lexical behaviour is preserved byte for byte.
        """
        if self.max_targeted_commands <= 0:
            return []

        if impact is not None and getattr(impact, "validation_targets", None):
            from .semantic_impact import TIER_BROAD

            multiplier = self._SCOPE_BUDGET_MULTIPLIER.get(
                getattr(impact, "recommended_scope", "broad"), 3
            )
            budget = self.max_targeted_commands * multiplier
            specs: list[CommandSpec] = []
            for target in impact.validation_targets:
                if target.tier == TIER_BROAD:
                    # The broad fallback means "no test could be associated".
                    # Running the entire suite inside every candidate iteration
                    # would be pathologically slow, and the authoritative
                    # full-suite run still happens post-apply, so the candidate
                    # simply contributes no targeted evidence here.
                    continue
                if not (workspace.root / target.path).is_file():
                    continue
                specs.append(
                    CommandSpec(
                        name=f"impact_{Path(target.path).stem}",
                        command=tuple(target.command),
                        reason=target.selected_because,
                        category="unit_test",
                        risk="low",
                        destructive=False,
                    )
                )
                if len(specs) >= budget:
                    break
            if specs:
                return specs

        from .validation import ValidationIntelligence

        intelligence = ValidationIntelligence(workspace.root)
        targeted = intelligence.discover_targeted_commands(changed_files, repository_map)
        return list(targeted)[: self.max_targeted_commands]

    def _static_commands(
        self, workspace: CandidateWorkspace, repository_map: Any | None
    ) -> list[CommandSpec]:
        if not self.enable_static_analysis or self.max_static_commands <= 0:
            return []
        if repository_map is None:
            return []
        from .validation import ValidationIntelligence

        intelligence = ValidationIntelligence(workspace.root)
        try:
            discovered = intelligence.discover_commands(repository_map)
        except (OSError, ValueError):
            return []

        selected: list[CommandSpec] = []
        for command in discovered:
            if getattr(command, "destructive", False):
                continue
            if command.category not in {"lint", "type_check"}:
                continue
            if command.name.lower() in self._WHOLE_REPO_COMMAND_NAMES:
                continue
            selected.append(
                CommandSpec(
                    name=command.name,
                    command=tuple(command.command),
                    reason=getattr(command, "reason", "static analysis"),
                    category=command.category,
                    risk=getattr(command, "risk", "low"),
                    destructive=False,
                )
            )
            if len(selected) >= self.max_static_commands:
                break
        return selected

    # -- execution ---------------------------------------------------------

    def _execute(
        self, workspace: CandidateWorkspace, spec: CommandSpec, tier: str
    ) -> CandidateCommandResult:
        try:
            execution = workspace.runner.run(spec)
        except PermissionError as exc:
            return CandidateCommandResult(
                name=spec.name,
                command=tuple(spec.command),
                tier=tier,
                exit_code=0,
                skipped=True,
                skip_reason=f"blocked by command policy: {exc}",
            )

        stderr = getattr(execution, "stderr", "") or ""
        exit_code = int(getattr(execution, "exit_code", 1))
        if exit_code == _EXECUTABLE_MISSING_EXIT_CODE and "executable not found" in stderr:
            return CandidateCommandResult(
                name=spec.name,
                command=tuple(spec.command),
                tier=tier,
                exit_code=exit_code,
                skipped=True,
                skip_reason="executable not available in this environment",
            )

        return CandidateCommandResult(
            name=spec.name,
            command=tuple(spec.command),
            tier=tier,
            exit_code=exit_code,
            duration_seconds=float(getattr(execution, "duration_seconds", 0.0) or 0.0),
            stdout=getattr(execution, "stdout", "") or "",
            stderr=stderr,
        )
