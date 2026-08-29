from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .models import CommandSpec, ExecutionResult

LOGGER = logging.getLogger(__name__)

#: Bare names that mean "the Python interpreter", not a specific installed one.
_PYTHON_EXECUTABLE_NAMES = frozenset({"python", "python3", "python.exe", "python3.exe"})

#: Console scripts that are equivalent to ``-m <module>``. Used only when the
#: wrapper script is genuinely absent from PATH.
_PYTHON_MODULE_SCRIPTS: dict[str, str] = {
    "pytest": "pytest",
    "unittest": "unittest",
    "coverage": "coverage",
    "mypy": "mypy",
    "ruff": "ruff",
    "flake8": "flake8",
    "pylint": "pylint",
    "pyflakes": "pyflakes",
    "black": "black",
}


def _module_is_importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def resolve_executable(command: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """Map a logical command onto argv that runs under the *current* interpreter.

    Returns ``(argv, note)``; ``note`` is empty when nothing was rewritten.

    A bare ``python`` or ``pytest`` name only resolves when the environment
    happens to expose that console script on PATH. It does not on a Windows
    ``py``-launcher-only install, inside an embedded interpreter, or wherever the
    venv's ``Scripts``/``bin`` directory is not active. Reporting "executable not
    found" for the *test runner* is the dangerous case: a skipped command counts
    as succeeded, so validation would report success having executed nothing.
    Falling back to :data:`sys.executable` removes that hole - it is by
    definition present and is the same interpreter, and therefore the same
    installed packages, that the agent itself is running under.

    Only the argv handed to the OS is rewritten. The logical command is
    preserved for display, telemetry and evidence fingerprints, so a candidate's
    evidence stays comparable with the post-apply run's.
    """
    if not command:
        return command, ""
    head = str(command[0])
    if shutil.which(head) is not None:
        return command, ""
    lowered = Path(head).name.lower()
    if lowered in _PYTHON_EXECUTABLE_NAMES:
        return (
            (sys.executable, *command[1:]),
            f"'{head}' is not on PATH; used the running interpreter",
        )
    module = _PYTHON_MODULE_SCRIPTS.get(lowered)
    if module and _module_is_importable(module):
        return (
            (sys.executable, "-m", module, *command[1:]),
            f"'{head}' console script is not on PATH; ran it as '-m {module}'",
        )
    return command, ""


class UnsafeCommandError(PermissionError):
    """Raised when a command violates the conservative command policy."""


class CommandRunner:
    def __init__(self, root: str | Path, timeout_seconds: int = 120):
        self.root = Path(root).resolve()
        self.timeout_seconds = timeout_seconds

    def validate(self, command: tuple[str, ...]) -> None:
        if not command:
            raise UnsafeCommandError("empty command")
        executable = command[0].lower()
        display = " ".join(command).lower()
        if executable in {"rm", "rmdir", "del", "format", "shutdown", "reboot"}:
            raise UnsafeCommandError(f"blocked command: {display}")
        if len(command) >= 2 and executable == "git" and command[1].lower() in {"reset", "clean", "push"}:
            raise UnsafeCommandError(f"blocked command: {display}")
        if any(token in {"&&", "||", ";", "|", ">", "<", "`", "$()"} for token in command):
            raise UnsafeCommandError("shell operators are not allowed")

    def run(self, spec: CommandSpec) -> ExecutionResult:
        self.validate(spec.command)
        started = time.monotonic()
        argv, rewrite_note = resolve_executable(tuple(spec.command))
        if rewrite_note:
            LOGGER.debug("Command %r: %s", spec.command, rewrite_note)
        if shutil.which(argv[0]) is None and not Path(argv[0]).is_file():
            return ExecutionResult(spec.display(), 127, stderr=f"executable not found: {spec.command[0]}")
        try:
            import os
            env = os.environ.copy()
            current_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{self.root}{os.pathsep}{current_pp}" if current_pp else str(self.root)
            result = subprocess.run(
                list(argv), cwd=self.root, capture_output=True, text=True,
                timeout=self.timeout_seconds, shell=False, env=env,
            )
            returncode = result.returncode
            if returncode == 5 and "NO TESTS RAN" in (result.stderr or result.stdout):
                returncode = 0
            return ExecutionResult(spec.display(), returncode, result.stdout, result.stderr, time.monotonic() - started)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            return ExecutionResult(spec.display(), 124, stdout if isinstance(stdout, str) else stdout.decode(errors="replace"), stderr if isinstance(stderr, str) else stderr.decode(errors="replace"), time.monotonic() - started, True)
        except OSError as exc:
            return ExecutionResult(spec.display(), 126, stderr=str(exc), duration_seconds=time.monotonic() - started)
