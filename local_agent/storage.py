from __future__ import annotations

import datetime
import json
import os
import shutil
import threading
import uuid
from abc import ABC, abstractmethod # noqa: F401
from pathlib import Path
from typing import Any, TYPE_CHECKING, TypeVar

from .models import Checkpoint, ProjectMemory, ProviderConfig, RepositoryKnowledgeGraph, SchedulerState, SemanticIndex, Subtask, Task

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a runtime import cycle
    from .maintenance import MaintenanceStore
    from .validation_lifecycle import ValidationLifecycleStore
    from .validation_telemetry import ValidationTelemetryStore

T = TypeVar("T", Task, Checkpoint)

def _to_jsonable(obj: Any) -> Any:
    """JSON default handler: serialize datetime/date as ISO 8601 strings.

    This is a safety net so that any datetime that leaks through a model's
    ``to_dict`` (e.g. nested inside ``asdict`` output) round-trips correctly
    instead of raising ``TypeError: Object of type datetime is not JSON
    serializable``.
    """
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


class TaskStorage(ABC):
    @abstractmethod
    def save_task(self, task: Task) -> None:
        pass

    @abstractmethod
    def load_task(self, task_id: str) -> Task:
        pass

    @abstractmethod
    def list_tasks(self) -> list[Task]:
        pass

    @abstractmethod
    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        pass

    def list_checkpoints_for_task(self, task_id: str) -> list[Checkpoint]:
        return []

    def load_latest_checkpoint(self, task_id: str, subtask_id: str | None = None) -> Checkpoint | None:
        return None

    @abstractmethod
    def save_scheduler_state(self, state: SchedulerState) -> None:
        pass

    @abstractmethod
    def load_scheduler_state(self) -> SchedulerState:
        pass

    @abstractmethod
    def save_provider_configs(self, configs: list[ProviderConfig]) -> None:
        pass

    @abstractmethod
    def load_provider_configs(self) -> list[ProviderConfig]:
        pass
    
    @abstractmethod
    def save_semantic_index(self, semantic_index: SemanticIndex) -> None:
        pass

    @abstractmethod
    def load_semantic_index(self) -> SemanticIndex:
        pass

    @abstractmethod
    def save_project_memory(self, memory: ProjectMemory) -> None:
        pass

    @abstractmethod
    def load_project_memory(self) -> ProjectMemory:
        pass

    def save_knowledge_graph(self, graph: RepositoryKnowledgeGraph) -> None:
        pass

    def load_knowledge_graph(self) -> RepositoryKnowledgeGraph:
        return RepositoryKnowledgeGraph()

    def save_validation_telemetry(self, store: "ValidationTelemetryStore") -> None:
        pass

    def load_validation_telemetry(self) -> "ValidationTelemetryStore":
        from .validation_telemetry import ValidationTelemetryStore
        return ValidationTelemetryStore()

    def save_validation_lifecycle(self, store: "ValidationLifecycleStore") -> None:
        """Phase 4.20. Default no-op, matching every other optional store here:
        a ``TaskStorage`` implementation that predates this method keeps working
        and simply retains no lifecycle history."""
        pass

    def load_validation_lifecycle(self) -> "ValidationLifecycleStore":
        from .validation_lifecycle import ValidationLifecycleStore
        return ValidationLifecycleStore()

    def save_maintenance(self, store: "MaintenanceStore") -> None:
        """Phase 4.21. Default no-op, matching every other optional store here."""
        pass

    def load_maintenance(self) -> "MaintenanceStore":
        from .maintenance import MaintenanceStore
        return MaintenanceStore()

class JsonFileStorage(TaskStorage):
    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.tasks_dir = self.base_dir / "tasks"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self.base_dir.mkdir(parents=True, exist_ok=True) # Ensure base_dir exists for semantic_index.json
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.checkpoints_dir / f"{checkpoint_id}.json"

    def _scheduler_state_path(self) -> Path:
        return self.base_dir / "scheduler_state.json"
    
    def _semantic_index_path(self) -> Path:
        return self.base_dir / "semantic_index.json"

    def _provider_configs_path(self) -> Path:
        return self.base_dir / "providers.json"

    def _project_memory_path(self) -> Path:
        return self.base_dir / "project_memory.json"

    def _knowledge_graph_path(self) -> Path:
        return self.base_dir / "knowledge_graph.json"

    def _validation_telemetry_path(self) -> Path:
        return self.base_dir / "validation_telemetry.json"

    def _validation_lifecycle_path(self) -> Path:
        return self.base_dir / "validation_lifecycle.json"

    def _maintenance_path(self) -> Path:
        return self.base_dir / "maintenance.json"

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        temp_path = path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=_to_jsonable)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (AttributeError, OSError):
                    pass
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    def save_task(self, task: Task) -> None:
        self._atomic_write(self._task_path(task.task_id), task.to_dict())

    def load_task(self, task_id: str) -> Task:
        path = self._task_path(task_id)
        if not path.exists():
            raise FileNotFoundError(f"Task with ID {task_id} not found.")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return Task.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load task {task_id} due to malformed data: {e}")

    def list_tasks(self) -> list[Task]:
        tasks = []
        for task_file in self.tasks_dir.glob("*.json"):
            try:
                tasks.append(self.load_task(task_file.stem))
            except (FileNotFoundError, ValueError):
                # Skip corrupted or deleted task files
                pass
        return tasks

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        self._atomic_write(self._checkpoint_path(checkpoint.checkpoint_id), checkpoint.to_dict())

    def load_checkpoint(self, checkpoint_id: str) -> Checkpoint:
        path = self._checkpoint_path(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint with ID {checkpoint_id} not found.")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return Checkpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # TypeError covers Checkpoint.from_dict's `cls(**d)` call raising on
            # a schema-mismatched or corrupted record (a required field like
            # current_state_description missing, or an unexpected key from a
            # different schema version) -- classified as the same "malformed
            # data" condition as unparseable JSON, not a crash.
            raise ValueError(f"Failed to load checkpoint {checkpoint_id} due to malformed data: {e}")

    def list_checkpoints_for_task(self, task_id: str) -> list[Checkpoint]:
        checkpoints: list[Checkpoint] = []
        for cp_file in self.checkpoints_dir.glob("*.json"):
            try:
                cp = self.load_checkpoint(cp_file.stem)
                if cp.task_id == task_id:
                    checkpoints.append(cp)
            except Exception:
                pass
        checkpoints.sort(key=lambda c: (c.timestamp, c.checkpoint_id))
        return checkpoints

    def load_latest_checkpoint(self, task_id: str, subtask_id: str | None = None) -> Checkpoint | None:
        cps = self.list_checkpoints_for_task(task_id)
        if subtask_id is not None:
            cps = [cp for cp in cps if cp.subtask_id == subtask_id]
        return cps[-1] if cps else None

    def save_scheduler_state(self, state: SchedulerState) -> None:
        self._atomic_write(self._scheduler_state_path(), state.to_dict())

    def load_scheduler_state(self) -> SchedulerState:
        path = self._scheduler_state_path()
        if not path.exists():
            return SchedulerState()
        try:
            with path.open("r", encoding="utf-8") as f:
                return SchedulerState.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return SchedulerState() # Return default state if file is corrupted

    def save_provider_configs(self, configs: list[ProviderConfig]) -> None:
        data = [c.to_dict() for c in configs]
        self._atomic_write(self._provider_configs_path(), {"providers": data})

    def load_provider_configs(self) -> list[ProviderConfig]:
        path = self._provider_configs_path()
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return [ProviderConfig.from_dict(c) for c in data.get("providers", [])]
        except (json.JSONDecodeError, KeyError):
            return []

    def save_semantic_index(self, semantic_index: SemanticIndex) -> None:
        self._atomic_write(self._semantic_index_path(), semantic_index.to_dict())

    def load_semantic_index(self) -> SemanticIndex:
        path = self._semantic_index_path()
        if not path.exists():
            return SemanticIndex()
        try:
            with path.open("r", encoding="utf-8") as f:
                return SemanticIndex.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError) as e:
            # Log error but return empty index for robustness
            print(f"Warning: Failed to load semantic index due to malformed data: {e}. Returning empty index.")
            return SemanticIndex()

    def save_project_memory(self, memory: ProjectMemory) -> None:
        # The lock must be handled by the caller (Scheduler/Orchestrator)
        # to ensure the read-modify-write cycle is atomic.
        self._atomic_write(self._project_memory_path(), memory.to_dict())

    def load_project_memory(self) -> ProjectMemory:
        path = self._project_memory_path()
        if not path.exists():
            return ProjectMemory()
        try:
            # Reading is generally safe without a lock if writes are atomic,
            # but the caller should still lock if performing a read-modify-write.
            with path.open("r", encoding="utf-8") as f:
                return ProjectMemory.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load project memory due to malformed data: {e}. Returning empty memory.")
            return ProjectMemory()

    def save_knowledge_graph(self, graph: RepositoryKnowledgeGraph) -> None:
        self._atomic_write(self._knowledge_graph_path(), graph.to_dict())

    def load_knowledge_graph(self) -> RepositoryKnowledgeGraph:
        path = self._knowledge_graph_path()
        if not path.exists():
            return RepositoryKnowledgeGraph()
        try:
            with path.open("r", encoding="utf-8") as f:
                return RepositoryKnowledgeGraph.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Quarantine corrupted file
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            corrupt_path = self.base_dir / f"knowledge_graph.json.corrupt.{timestamp}"
            try:
                shutil.copy2(path, corrupt_path)
            except Exception:
                pass
            print(f"Warning: Failed to load knowledge graph due to malformed data ({e}). Quarantined to {corrupt_path.name} and returning clean graph.")
            return RepositoryKnowledgeGraph()

    def save_validation_telemetry(self, store: "ValidationTelemetryStore") -> None:
        self._atomic_write(self._validation_telemetry_path(), store.to_dict())

    def load_validation_telemetry(self) -> "ValidationTelemetryStore":
        from .validation_telemetry import ValidationTelemetryStore
        path = self._validation_telemetry_path()
        if not path.exists():
            return ValidationTelemetryStore()
        try:
            with path.open("r", encoding="utf-8") as f:
                return ValidationTelemetryStore.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Quarantine corrupted file, same policy as the knowledge graph.
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            corrupt_path = self.base_dir / f"validation_telemetry.json.corrupt.{timestamp}"
            try:
                shutil.copy2(path, corrupt_path)
            except Exception:
                pass
            print(f"Warning: Failed to load validation telemetry due to malformed data ({e}). Quarantined to {corrupt_path.name} and returning an empty store.")
            return ValidationTelemetryStore()

    def save_validation_lifecycle(self, store: "ValidationLifecycleStore") -> None:
        self._atomic_write(self._validation_lifecycle_path(), store.to_dict())

    def load_validation_lifecycle(self) -> "ValidationLifecycleStore":
        from .validation_lifecycle import ValidationLifecycleStore
        path = self._validation_lifecycle_path()
        if not path.exists():
            return ValidationLifecycleStore()
        try:
            with path.open("r", encoding="utf-8") as f:
                return ValidationLifecycleStore.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError) as e:
            # Same quarantine policy as the knowledge graph and the telemetry
            # store: keep the bad file for forensics, return an empty store, and
            # mark it corrupt so downstream analysis stays conservative rather
            # than treating "no data" as "clean history".
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            corrupt_path = self.base_dir / f"validation_lifecycle.json.corrupt.{timestamp}"
            try:
                shutil.copy2(path, corrupt_path)
            except Exception:
                pass
            print(f"Warning: Failed to load validation lifecycle history due to malformed data ({e}). Quarantined to {corrupt_path.name} and returning an empty store.")
            store = ValidationLifecycleStore()
            store.corrupted_records_skipped = 1
            return store
    def save_maintenance(self, store: "MaintenanceStore") -> None:
        self._atomic_write(self._maintenance_path(), store.to_dict())

    def load_maintenance(self) -> "MaintenanceStore":
        from .maintenance import MaintenanceStore
        path = self._maintenance_path()
        if not path.exists():
            return MaintenanceStore()
        try:
            with path.open("r", encoding="utf-8") as f:
                return MaintenanceStore.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError, ValueError, UnicodeDecodeError, OSError) as e:
            # Same quarantine policy as every other optional store: keep the bad
            # file for forensics, return an empty one, and mark it corrupt so
            # the learning layer treats "no data" as untrustworthy rather than
            # as a clean history.
            timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            corrupt_path = self.base_dir / f"maintenance.json.corrupt.{timestamp}"
            try:
                shutil.copy2(path, corrupt_path)
            except Exception:
                pass
            print(f"Warning: Failed to load maintenance history due to malformed data ({e}). Quarantined to {corrupt_path.name} and returning an empty store.")
            store = MaintenanceStore()
            store.corrupted_records_skipped = 1
            return store
