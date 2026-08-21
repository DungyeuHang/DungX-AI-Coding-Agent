from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

from .models import CommandSpec, ExecutionResult


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
        if shutil.which(spec.command[0]) is None:
            return ExecutionResult(spec.display(), 127, stderr=f"executable not found: {spec.command[0]}")
        try:
            import os
            env = os.environ.copy()
            current_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{self.root}{os.pathsep}{current_pp}" if current_pp else str(self.root)
            result = subprocess.run(
                list(spec.command), cwd=self.root, capture_output=True, text=True,
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
