from __future__ import annotations

import difflib
from pathlib import Path

from .filesystem import ProjectFilesystem, ProtectedPathError, SandboxViolation, SECRET_NAMES
from .models import FileOperation, Plan, PlanAmendment, PreparedChange, ScopeExpansionProposal
from .patching import PatchApplicationError, UnifiedPatchApplier

PROTECTED_DIRS = {".git", ".hg", ".svn", ".agent_data"}


class UnsafeModificationError(PermissionError):
    pass


class PatchValidationError(UnsafeModificationError):
    """A strict patch failure with enough detail for a focused repair request."""

    def __init__(self, path: str, original: str, patch: str, reason: str):
        self.path = path
        self.original = original
        self.patch = patch
        self.reason = reason
        super().__init__(f"invalid patch for {path}: {reason}")


class ScopeAmendmentGuard:
    """Deterministic validator for scope expansion proposals."""

    def __init__(
        self,
        filesystem: ProjectFilesystem,
        max_total_amendments: int = 5,
        max_scope_growth_factor: float = 2.0,
    ):
        self.filesystem = filesystem
        self.max_total_amendments = max_total_amendments
        self.max_scope_growth_factor = max_scope_growth_factor

    def evaluate(
        self,
        proposal: ScopeExpansionProposal,
        active_plan: Plan,
        initial_scope_count: int = 1,
        max_total_amendments: int | None = None,
        max_scope_growth_factor: float | None = None,
    ) -> tuple[bool, str]:
        """
        Deterministic validation of a scope expansion proposal against 10 safety invariants.
        Returns (is_valid, reason).
        """
        eff_max_amendments = max_total_amendments if max_total_amendments is not None else self.max_total_amendments
        eff_max_growth = max_scope_growth_factor if max_scope_growth_factor is not None else self.max_scope_growth_factor

        # 1. Empty path rejection
        if not proposal.path or not proposal.path.strip():
            return False, "Scope expansion rejected: proposal path cannot be empty"

        raw_path = proposal.path.strip()

        # Check path traversal markers before normalization
        if ".." in Path(raw_path).parts or raw_path.startswith(("/", "\\")) or (len(raw_path) > 1 and raw_path[1] == ":"):
            return False, f"Scope expansion rejected: path traversal or absolute path not allowed: {raw_path}"

        # 2. Guard path traversal and security sandbox
        try:
            resolved = self.filesystem.resolve(raw_path)
            rel_path = resolved.relative_to(self.filesystem.root).as_posix()
        except (SandboxViolation, ValueError, ProtectedPathError) as exc:
            return False, f"Scope expansion rejected: sandbox violation: {exc}"

        # 3. Project-root confinement & Protected directory rejection
        rel_parts = resolved.relative_to(self.filesystem.root).parts
        for part in rel_parts:
            if part.lower() in PROTECTED_DIRS:
                return False, f"Scope expansion rejected: protected directory in path: {part}"

        # 4. Secret / protected file rejection
        if resolved.name.lower() in {name.lower() for name in SECRET_NAMES}:
            return False, f"Scope expansion rejected: secret file cannot be added to plan: {resolved.name}"

        # 5. Duplicate scope rejection
        allowed = active_plan.allowed_paths if hasattr(active_plan, "allowed_paths") else set(active_plan.files_likely_to_change + active_plan.files_likely_to_create)
        if rel_path in allowed:
            return False, f"Scope expansion rejected: path '{rel_path}' is already in plan"

        # 6. Existing / non-existing consistency
        exists = self.filesystem.file_exists(rel_path)
        if not proposal.is_create and not exists:
            return False, f"Scope expansion rejected: cannot modify non-existent file '{rel_path}' (set is_create=True)"
        if proposal.is_create and exists:
            return False, f"Scope expansion rejected: cannot create already existing file '{rel_path}'"

        # 7. Total amendments budget limit
        current_amendments = len(getattr(active_plan, "amendments", []))
        if current_amendments >= eff_max_amendments:
            return False, f"Scope expansion rejected: maximum amendment budget reached ({eff_max_amendments})"

        # 8. Scope growth factor limit
        current_scope_size = len(allowed)
        effective_initial = max(1, initial_scope_count)
        max_allowed_scope = int(effective_initial * eff_max_growth) + 2
        if (current_scope_size + 1) > max_allowed_scope:
            return False, f"Scope expansion rejected: scope growth limit reached (growth factor {eff_max_growth}x exceeded: current={current_scope_size}, max={max_allowed_scope})"

        return True, f"Approved scope expansion for '{rel_path}'"


class CodingAgent:
    VALID_ACTIONS = {"write", "create", "modify", "delete"}

    def __init__(self, filesystem: ProjectFilesystem, protected_paths: set[str] | None = None):
        self.filesystem = filesystem
        self.protected_paths = {self._normalize(path) for path in (protected_paths or set())}
        self._originals: dict[str, str | None] = {}

    def find_unlisted_operations(self, operations: list[FileOperation], plan: Plan | None = None) -> list[FileOperation]:
        """Find operations in the batch that target paths outside the plan's allowed scope."""
        if isinstance(plan, dict):
            allowed = set(plan.get("files_likely_to_change", []) + plan.get("files_likely_to_create", []))
        elif plan:
            allowed = plan.allowed_paths if hasattr(plan, "allowed_paths") else set(getattr(plan, "files_likely_to_change", []) + getattr(plan, "files_likely_to_create", []))
        else:
            allowed = set()
        normalized_allowed = {self._normalize(item) for item in allowed}
        unlisted = []
        for op in operations:
            rel = self._normalize(op.path)
            if normalized_allowed and rel not in normalized_allowed:
                unlisted.append(op)
        return unlisted

    def prepare(self, operations: list[FileOperation], plan: Plan | None = None) -> list[PreparedChange]:
        """Validate all operations and calculate results without writing files."""
        if isinstance(plan, dict):
            allowed = set(plan.get("files_likely_to_change", []) + plan.get("files_likely_to_create", []))
        elif plan:
            allowed = plan.allowed_paths if hasattr(plan, "allowed_paths") else set(getattr(plan, "files_likely_to_change", []) + getattr(plan, "files_likely_to_create", []))
        else:
            allowed = set()
        normalized_allowed = {self._normalize(item) for item in allowed}
        prepared: list[PreparedChange] = []
        seen: set[str] = set()
        for operation in operations:
            action = operation.action.lower().strip()
            relative = self._normalize(operation.path)
            self._validate_path(relative, action, normalized_allowed)
            if relative in seen:
                raise UnsafeModificationError(f"multiple operations target the same file: {relative}")
            seen.add(relative)
            exists = self.filesystem.file_exists(relative)
            original = self.filesystem.read_file(relative) if exists else None
            if action == "create" and exists:
                raise UnsafeModificationError(f"file already exists: {relative}")
            if action == "modify" and not exists:
                raise UnsafeModificationError(f"cannot modify missing file: {relative}")
            if action == "delete" and not exists:
                raise UnsafeModificationError(f"cannot delete missing file: {relative}")
            try:
                resulting = self._resulting_content(action, operation, original, relative)
            except PatchApplicationError as exc:
                raise PatchValidationError(relative, original or "", operation.patch or "", str(exc)) from exc
            diff = "".join(difflib.unified_diff(
                (original or "").splitlines(keepends=True),
                (resulting or "").splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
            prepared.append(PreparedChange(action, relative, original, resulting, diff, operation.reason))
        return prepared

    def apply_prepared(self, changes: list[PreparedChange]) -> list[str]:
        """Write a previously validated batch, recording originals for review."""
        for change in changes:
            if change.path not in self._originals:
                self._originals[change.path] = change.original
        applied: list[PreparedChange] = []
        try:
            for change in changes:
                if change.action == "delete":
                    self.filesystem.delete_file(change.path)
                elif change.action == "create":
                    self.filesystem.create_file(change.path, change.resulting or "")
                else:
                    self.filesystem.write_file(change.path, change.resulting or "")
                applied.append(change)
        except (ProtectedPathError, SandboxViolation, OSError) as exc:
            for change in reversed(applied):
                try:
                    if change.original is None:
                        self.filesystem.delete_file(change.path)
                    else:
                        self.filesystem.write_file(change.path, change.original)
                except (ProtectedPathError, SandboxViolation, OSError):
                    pass
            raise UnsafeModificationError(f"could not apply {change.path}: {exc}; applied changes were rolled back") from exc
        return [change.path for change in changes]

    def apply(self, operations: list[FileOperation], plan: Plan | None = None) -> list[str]:
        return self.apply_prepared(self.prepare(operations, plan))

    def diff(self) -> str:
        pieces: list[str] = []
        for relative, original in self._originals.items():
            path = self.filesystem.resolve(relative)
            current = self.filesystem.read_file(relative) if path.exists() else ""
            pieces.extend(difflib.unified_diff(
                (original or "").splitlines(keepends=True), current.splitlines(keepends=True),
                fromfile=f"a/{relative}", tofile=f"b/{relative}",
            ))
        return "".join(pieces)

    def _validate_path(self, relative: str, action: str, allowed: set[str]) -> None:
        if action not in self.VALID_ACTIONS:
            raise UnsafeModificationError(f"unsupported file operation: {action}")
        if not relative or relative == "." or relative.startswith("../") or relative.startswith("/") or (len(relative) > 1 and relative[1] == ":") or ".." in Path(relative).parts:
            raise SandboxViolation(f"invalid relative path: {relative}")
        is_protected = any(relative == protected or relative.startswith(protected.rstrip("/") + "/") for protected in self.protected_paths)
        if is_protected and relative not in allowed:
            raise UnsafeModificationError(f"refusing to overwrite unrelated existing change: {relative}")
        if allowed and relative not in allowed:
            raise UnsafeModificationError(f"file operation is outside the approved plan: {relative}")

    @staticmethod
    def _resulting_content(action: str, operation: FileOperation, original: str | None, relative: str) -> str | None:
        if action == "delete":
            return None
        if action in {"write", "create", "modify"} and operation.content is not None:
            return operation.content
        if operation.patch is not None:
            return UnifiedPatchApplier().apply(original or "", operation.patch, expected_path=relative)
        raise PatchApplicationError(f"{action} requires complete content or a unified patch: {relative}")

    @staticmethod
    def _normalize(path: str) -> str:
        value = path.replace("\\", "/")
        if value.startswith("/") or (len(value) > 1 and value[1] == ":"):
            return value
        return Path(value).as_posix()
