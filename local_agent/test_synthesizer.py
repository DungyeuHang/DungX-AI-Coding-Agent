from __future__ import annotations

import ast
import datetime
import logging
from pathlib import Path
import re
import time
from typing import Any

from .commands import CommandRunner
from .filesystem import ProjectFilesystem
from .models import (
    CommandSpec,
    ExportedSymbol,
    ProjectContext,
    RepositoryMap,
    RunReport,
    Subtask,
    TestExecutionRecord,
    VerificationGap,
)
from .providers import AIProvider

LOGGER = logging.getLogger(__name__)

_DANGEROUS_PATTERNS = [
    re.compile(r"\b(rm|del)\s+.*(--force|-f)\b", re.I),
    re.compile(r"\b(git\s+(reset|clean|push|rebase|merge))\b", re.I),
    re.compile(r"\b(os\.system|subprocess\.Popen|subprocess\.call|shutil\.rmtree)\b", re.I),
    re.compile(r"\b(db:migrate:reset|db:drop|prisma\s+migrate\s+reset)\b", re.I),
]

# Phase 4.22: names of calls that only prove a symbol *exists* and is of a
# recognizable shape (callable, a class, not None) -- never that it does
# anything correct. A test built entirely out of these is indistinguishable,
# from the completion engine's point of view, from ``assert True``: it can
# never fail against a wrong-but-present implementation (Attack B/C in the
# Phase 4.22 adversarial matrix: a function that exists but returns invalid
# output). See classify_test_triviality below.
_EXISTENCE_ONLY_CALL_NAMES = frozenset({
    "callable", "isclass", "isfunction", "ismethod", "isroutine", "isbuiltin",
})


def _is_existence_only_assert(test_node: ast.AST) -> bool:
    # ``assert True`` / ``assert 1`` -- unconditionally vacuous.
    if isinstance(test_node, ast.Constant) and bool(test_node.value):
        return True
    # ``assert x is not None`` / ``assert x is None``.
    if (
        isinstance(test_node, ast.Compare)
        and len(test_node.ops) == 1
        and isinstance(test_node.ops[0], (ast.Is, ast.IsNot))
        and len(test_node.comparators) == 1
        and isinstance(test_node.comparators[0], ast.Constant)
        and test_node.comparators[0].value is None
    ):
        return True
    # ``assert callable(x)`` / ``assert inspect.isclass(x)`` etc.
    if isinstance(test_node, ast.Call):
        func = test_node.func
        name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
        if name in _EXISTENCE_ONLY_CALL_NAMES:
            return True
    return False


def _function_has_nontrivial_assertion(fn: ast.AST) -> bool:
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            if not _is_existence_only_assert(node.test):
                return True
        elif isinstance(node, ast.With):
            # ``with pytest.raises(...):`` genuinely exercises behavior
            # (an expected exception path), not merely existence.
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    func = call.func
                    name = func.attr if isinstance(func, ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
                    if name == "raises":
                        return True
    return False


def classify_test_triviality(code: str) -> bool:
    """True when this test code cannot possibly distinguish a correct
    implementation from a superficially-matching wrong one: every ``test_``
    function's assertions only check that a symbol exists/is callable/is
    non-None, never an actual return value, output, or side effect.

    Fails closed: unparseable code, or a fixture with no ``test_`` function
    at all, is treated as trivial (never granted behavioral-proof strength).
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    test_funcs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test_")
    ]
    if not test_funcs:
        return True
    for fn in test_funcs:
        if not _function_has_nontrivial_assertion(fn):
            return True
    return False


class VerificationGapAnalyzer:
    """
    Deterministically analyzes whether changed files and exported symbols
    have actual targeted executable test coverage or represent a verification gap.
    """

    def __init__(self, project_root: str | Path, filesystem: ProjectFilesystem | None = None):
        self.project_root = Path(project_root)
        self.filesystem = filesystem or ProjectFilesystem(self.project_root)

    def analyze(
        self,
        changed_files: list[str],
        exported_symbols: list[ExportedSymbol],
        context: ProjectContext,
        targeted_commands: list[CommandSpec] | None = None,
    ) -> VerificationGap | None:
        if not changed_files and not exported_symbols:
            return None

        # Gather existing test contents to check for symbol references
        existing_test_files = [f for f in context.test_files if f.endswith((".py", ".ts", ".tsx", ".js", ".jsx"))]
        test_contents: dict[str, str] = {}
        for test_file in existing_test_files:
            try:
                content = self.filesystem.read_file(test_file)
                if content:
                    test_contents[test_file] = content
            except Exception:
                pass

        missing_symbols: list[ExportedSymbol] = []
        untested_files: set[str] = set()
        reasons: list[str] = []

        # Check each exported symbol
        for symbol in exported_symbols:
            symbol_tested = False
            # Check if any test file references the symbol name
            for test_file, content in test_contents.items():
                # Word boundary match for the symbol name
                if re.search(rf"\b{re.escape(symbol.name)}\b", content):
                    symbol_tested = True
                    break

            if not symbol_tested:
                missing_symbols.append(symbol)
                untested_files.add(symbol.file_path)
                reasons.append(f"Exported {symbol.kind} '{symbol.name}' in '{symbol.file_path}' has no references in test files.")

        # Check changed non-test files that have no matching test file
        for changed in changed_files:
            if changed in context.test_files:
                continue
            p = Path(changed)
            stem = p.stem
            has_matching_test = any(
                stem in t or f"test_{stem}" in t or f"{stem}_test" in t
                for t in existing_test_files
            )
            if not has_matching_test and changed not in untested_files:
                untested_files.add(changed)
                reasons.append(f"Modified source file '{changed}' has no corresponding test file in the repository.")

        if not missing_symbols and not untested_files:
            return None

        severity = "high" if len(missing_symbols) >= 3 else ("medium" if missing_symbols else "low")
        return VerificationGap(
            missing_test_symbols=missing_symbols[:20],
            untested_files=sorted(list(untested_files))[:20],
            reasons=reasons[:20],
            severity=severity,
        )


class TestSynthesizer:
    """
    Synthesizes minimal, bounded, deterministic behavioral test fixtures for untested symbols.
    """
    __test__ = False

    def __init__(
        self,
        project_root: str | Path,
        provider: AIProvider | None = None,
        filesystem: ProjectFilesystem | None = None,
        max_synthetic_test_chars: int = 4000,
    ):
        self.project_root = Path(project_root)
        self.provider = provider
        self.filesystem = filesystem or ProjectFilesystem(self.project_root)
        self.max_synthetic_test_chars = max_synthetic_test_chars

    def synthesize_test(
        self,
        subtask: Subtask,
        gap: VerificationGap,
        context: ProjectContext,
    ) -> str | None:
        """Generates executable test code for the symbols in the verification gap."""
        if not gap.missing_test_symbols:
            return None

        # Filter to python symbols for synthesis
        py_symbols = [s for s in gap.missing_test_symbols if s.file_path.endswith(".py")]
        if not py_symbols:
            return None

        # 1. If provider is available, attempt AI test proposal
        if self.provider and (hasattr(self.provider, "generate_plan") or hasattr(self.provider, "send_message")):
            try:
                proposed_code = self._synthesize_with_provider(subtask, gap, context)
                if proposed_code and self._validate_test_code(proposed_code):
                    return proposed_code[:self.max_synthetic_test_chars]
            except Exception as e:
                LOGGER.debug("Provider test synthesis failed: %s, falling back to deterministic template", e)

        # 2. Deterministic Template Fallback
        deterministic_code = self._generate_deterministic_template(gap)
        if deterministic_code and self._validate_test_code(deterministic_code):
            return deterministic_code[:self.max_synthetic_test_chars]

        return None

    def _synthesize_with_provider(
        self,
        subtask: Subtask,
        gap: VerificationGap,
        context: ProjectContext,
    ) -> str | None:
        prompt_lines = [
            f"Generate a minimal, self-contained, executable pytest unit test verifying the behavioral correctness of these newly implemented symbols:",
        ]
        for s in gap.missing_test_symbols[:5]:
            prompt_lines.append(f"- {s.kind} `{s.name}` in `{s.file_path}`: signature `{s.signature}`")

        prompt_lines.extend([
            "\nRequirements:",
            "1. Output ONLY executable Python test code inside a ```python code block.",
            "2. Import the symbols from their respective modules.",
            "3. Test basic instantiation, valid inputs, expected return types, and core invariants.",
            "4. Do NOT use mocks where real execution is possible.",
            "5. Keep the test minimal and fast (max 50 lines).",
        ])
        prompt = "\n".join(prompt_lines)

        resp = None
        if hasattr(self.provider, "send_message"):
            resp = self.provider.send_message(prompt)
        elif hasattr(self.provider, "generate_plan"):
            resp = self.provider.generate_plan(prompt, context)

        raw_text = ""
        if isinstance(resp, str):
            raw_text = resp
        elif hasattr(resp, "steps") and resp.steps:
            raw_text = "\n".join(str(step) for step in resp.steps)
        elif hasattr(resp, "objective") and resp.objective:
            raw_text = resp.objective
        else:
            raw_text = str(resp) if resp else ""

        code_match = re.search(r"```(?:python)?\s*([\s\S]*?)\s*```", raw_text)
        if code_match:
            return code_match.group(1).strip()
        elif "def test_" in raw_text:
            return raw_text.strip()
        return None

    def _generate_deterministic_template(self, gap: VerificationGap) -> str:
        if not gap.missing_test_symbols:
            return ""

        lines = [
            "# Auto-synthesized behavioral verification fixture",
            "import pytest",
            "import inspect",
            "",
        ]

        # Group symbols by module
        by_file: dict[str, list[ExportedSymbol]] = {}
        for s in gap.missing_test_symbols:
            by_file.setdefault(s.file_path, []).append(s)

        for file_path, symbols in by_file.items():
            if not file_path.endswith(".py"):
                continue
            # Convert file path to module path e.g. local_agent/foo.py -> local_agent.foo
            mod_path = file_path[:-3].replace("/", ".").replace("\\", ".")
            symbol_names = [s.name for s in symbols]
            lines.append(f"from {mod_path} import {', '.join(symbol_names)}")
            lines.append("")

            for s in symbols:
                test_func_name = f"test_behavior_{s.name.lower()}"
                lines.append(f"def {test_func_name}():")
                if s.kind == "class":
                    lines.append(f"    # Verify class instantiation and interface invariants")
                    lines.append(f"    assert inspect.isclass({s.name})")
                    lines.append(f"    try:")
                    lines.append(f"        instance = {s.name}()")
                    lines.append(f"        assert instance is not None")
                    lines.append(f"    except TypeError:")
                    lines.append(f"        # Requires constructor arguments; verified class definition")
                    lines.append(f"        pass")
                elif s.kind in {"function", "async_function"}:
                    lines.append(f"    # Verify callable interface")
                    lines.append(f"    assert callable({s.name})")
                lines.append("")

        return "\n".join(lines)

    def _validate_test_code(self, code: str) -> bool:
        """Validates that the test code is syntactically valid and safe."""
        if not code or len(code) > self.max_synthetic_test_chars:
            return False

        if "def test_" not in code:
            return False

        # Security check: dangerous command patterns
        for pat in _DANGEROUS_PATTERNS:
            if pat.search(code):
                LOGGER.warning("Synthesized test contained dangerous pattern: %s", pat.pattern)
                return False

        # Syntax check via AST
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            LOGGER.warning("Synthesized test failed AST validation: %s", e)
            return False


class BehavioralVerifier:
    """
    Executes synthesized verification fixtures in sandboxed environment and produces TestExecutionRecord.
    """

    def __init__(
        self,
        project_root: str | Path,
        runner: CommandRunner | None = None,
        filesystem: ProjectFilesystem | None = None,
        timeout_seconds: int = 30,
    ):
        self.project_root = Path(project_root)
        self.runner = runner or CommandRunner(self.project_root)
        self.filesystem = filesystem or ProjectFilesystem(self.project_root)
        self.timeout_seconds = timeout_seconds

    def verify(
        self,
        test_code: str,
        subtask_id: str,
        exercised_symbols: list[ExportedSymbol] | None = None,
    ) -> TestExecutionRecord:
        """Writes ephemeral test fixture, runs it, and captures execution telemetry."""
        clean_subtask = re.sub(r"[^A-Za-z0-9_]", "_", subtask_id)
        fixture_filename = f"tests/_synthetic_test_{clean_subtask}.py"
        fixture_path = self.project_root / fixture_filename

        start_time = time.time()
        exercised_names = [s.name for s in (exercised_symbols or [])]
        trivial = classify_test_triviality(test_code)

        try:
            # Write ephemeral fixture
            fixture_path.parent.mkdir(parents=True, exist_ok=True)
            fixture_path.write_text(test_code, encoding="utf-8")

            # Run using CommandRunner (pytest)
            cmd = ("pytest", "-q", fixture_filename)
            cmd_spec = CommandSpec(
                name=f"synthetic_test_{clean_subtask}",
                command=cmd,
                reason="Behavioral verification for synthesized test fixture",
                category="unit_test",
                risk="low",
                destructive=False,
            )

            result = self.runner.run(cmd_spec)
            duration = time.time() - start_time

            status = "passed" if result.exit_code == 0 else "failed"
            failure_cls = "" if result.exit_code == 0 else "SYNTHETIC_TEST_FAILURE"

            return TestExecutionRecord(
                test_id=f"synth-{clean_subtask}",
                command=" ".join(cmd),
                status=status,
                exit_code=result.exit_code,
                duration_seconds=round(duration, 3),
                stdout_summary=result.stdout[:500] if result.stdout else "",
                stderr_summary=result.stderr[:500] if result.stderr else "",
                synthesized=True,
                exercised_symbols=exercised_names,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                failure_classification=failure_cls,
                trivial=trivial,
            )

        except Exception as e:
            duration = time.time() - start_time
            return TestExecutionRecord(
                test_id=f"synth-{clean_subtask}",
                command=f"pytest -q {fixture_filename}",
                status="failed",
                exit_code=1,
                duration_seconds=round(duration, 3),
                stdout_summary="",
                stderr_summary=str(e)[:500],
                synthesized=True,
                exercised_symbols=exercised_names,
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                failure_classification="VERIFICATION_EXECUTION_ERROR",
                trivial=trivial,
            )
        finally:
            # Clean up ephemeral fixture
            try:
                if fixture_path.exists():
                    fixture_path.unlink()
            except Exception:
                pass
