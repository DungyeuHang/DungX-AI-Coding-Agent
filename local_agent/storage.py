from __future__ import annotations

import datetime
import json
import os
import shutil
import threading
import uuid
from abc import ABC, abstractmethod # noqa: F401
from pathlib import Path
from typing import Any, TypeVar

from .models import Checkpoint, ProjectMemory, ProviderConfig, RepositoryKnowledgeGraph, SchedulerState, SemanticIndex, Subtask, Task

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

class JsonFileStorage(TaskStorage):
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

    def _atomic_write(self, path: Path, data: dict[str, Any]) -> None:
        temp_path = path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=False, default=_to_jsonable)
            os.replace(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
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
        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Failed to load checkpoint {checkpoint_id} due to malformed data: {e}")

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