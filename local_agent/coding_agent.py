from __future__ import annotations

from pathlib import Path

from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation
from .models import FileOperation, Plan


class UnsafeModificationError(PermissionError):
    pass


class CodingAgent:
    VALID_ACTIONS = {"write", "create", "delete"}

    def __init__(self, filesystem: ProjectFilesystem, protected_paths: set[str] | None = None):
        self.filesystem = filesystem
        self.protected_paths = {self._normalize(path) for path in (protected_paths or set())}

    def apply(self, operations: list[FileOperation], plan: Plan | None = None) -> list[str]:
        allowed = set(plan.files_likely_to_change + plan.files_likely_to_create) if plan else set()
        normalized_allowed = {self._normalize(item) for item in allowed}
        changed: list[str] = []
        for operation in operations:
            action = operation.action.lower().strip()
            relative = self._normalize(operation.path)
            if action not in self.VALID_ACTIONS:
                raise UnsafeModificationError(f"unsupported file operation: {operation.action}")
            if not relative or relative == "." or relative.startswith("../"):
                raise SandboxViolation(f"invalid relative path: {operation.path}")
            is_protected = any(relative == protected or relative.startswith(protected.rstrip("/") + "/") for protected in self.protected_paths)
            if is_protected and relative not in {self._normalize(item) for item in allowed}:
                raise UnsafeModificationError(f"refusing to overwrite unrelated existing change: {relative}")
            if normalized_allowed and relative not in normalized_allowed:
                raise UnsafeModificationError(f"file operation is outside the approved plan: {relative}")
            if action in {"write", "create"} and operation.content is None:
                raise UnsafeModificationError(f"{action} requires complete file content: {relative}")
            try:
                if action == "write":
                    self.filesystem.write_file(relative, operation.content or "")
                elif action == "create":
                    self.filesystem.create_file(relative, operation.content or "")
                else:
                    self.filesystem.delete_file(relative)
            except (ProtectedPathError, SandboxViolation) as exc:
                raise UnsafeModificationError(str(exc)) from exc
            changed.append(relative)
        return changed

    @staticmethod
    def _normalize(path: str) -> str:
        value = path.replace("\\", "/")
        if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
            return value
        return Path(value).as_posix()
