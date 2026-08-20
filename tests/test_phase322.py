from __future__ import annotations

import datetime
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.models import (
    Memory,
    MemoryCategory,
    ProjectContext,
    ProjectMemory,
    RunReport,
    Task,
    TaskStatus,
)
from local_agent.orchestrator import Orchestrator
from local_agent.storage import JsonFileStorage


class Phase322_ProjectMemoryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")
        self.base_config = AgentConfig.from_environment(self.root)
        self.memory_lock = threading.Lock()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.root)

    def _create_task(self, task_id: str, objective: str) -> Task:
        now = datetime.datetime.now(datetime.timezone.utc)
        return Task(task_id=task_id, objective=objective, status=TaskStatus.PENDING, created_at=now, updated_at=now)

    def test_memory_persistence_and_reload(self):
        # Arrange
        memory = ProjectMemory(memories=[
            Memory(
                memory_id="mem1",
                category=MemoryCategory.FILE_ROLE,
                content="This is the main app file.",
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                source_task_id="task1",
                related_path="src/app.py"
            )
        ])

        # Act
        self.storage.save_project_memory(memory)
        reloaded_memory = self.storage.load_project_memory()

        # Assert
        self.assertEqual(len(reloaded_memory.memories), 1)
        self.assertEqual(reloaded_memory.memories[0].memory_id, "mem1")
        self.assertEqual(reloaded_memory.memories[0].related_path, "src/app.py")

    def test_orchestrator_creates_memory_on_successful_run(self):
        # Arrange
        task = self._create_task("task1", "Implement feature X")
        report = RunReport(project=mock.MagicMock(), completed=True, changed_files=["src/feature.py"])
        
        # Mock scheduler and locks for Orchestrator
        mock_scheduler = mock.MagicMock()
        orchestrator = Orchestrator(self.base_config, self.storage, mock_scheduler, threading.Lock(), self.memory_lock)

        # Act
        orchestrator._create_memories_from_run(task, report)

        # Assert
        memory = self.storage.load_project_memory()
        self.assertEqual(len(memory.memories), 1)
        self.assertEqual(memory.memories[0].category, MemoryCategory.FILE_ROLE)
        self.assertEqual(memory.memories[0].related_path, "src/feature.py")
        self.assertIn("Implement feature X", memory.memories[0].content)

    def test_context_selector_uses_memory_to_boost_score(self):
        # Arrange
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("main app")
        (self.root / "src" / "utils.py").write_text("utility functions")

        memory = ProjectMemory(memories=[
            Memory(
                memory_id="mem1", category=MemoryCategory.FILE_ROLE,
                content="The app.py file is critical for user authentication.",
                timestamp=datetime.datetime.now(datetime.timezone.utc),
                source_task_id="task1", related_path="src/app.py"
            )
        ])
        project_context = ProjectContext(root=str(self.root), source_files=["src/app.py", "src/utils.py"])
        
        # Act: Task is about "login", which doesn't appear in file paths
        task_objective = "Fix the login button"
        selector = ContextSelector(self.root, project_memory=memory)
        result_context = selector.select(task_objective, project_context)

        # Assert
        selected_items = result_context.metadata["context_selection"]["selected_items"]
        self.assertTrue(len(selected_items) > 0)
        
        app_item = next((item for item in selected_items if item["path"] == "src/app.py"), None)
        utils_item = next((item for item in selected_items if item["path"] == "src/utils.py"), None)

        self.assertIsNotNone(app_item)
        self.assertIn("project memory: file_role", app_item["reason"])
        # The memory boost should make app.py score higher than utils.py, which has no memory
        if utils_item:
            self.assertGreater(app_item["score"], utils_item["score"])

    def test_concurrent_memory_writes_are_safe(self):
        # Arrange
        self.storage.save_project_memory(ProjectMemory())

        def worker(task_id: str):
            with self.memory_lock:
                memory = self.storage.load_project_memory()
                time.sleep(0.01) # Encourage race condition if lock is broken
                new_mem = Memory(memory_id=str(uuid.uuid4()), category=MemoryCategory.PROJECT_CONVENTION, content=f"Memory from {task_id}", timestamp=datetime.datetime.now(datetime.timezone.utc), source_task_id=task_id)
                memory.memories.append(new_mem)
                self.storage.save_project_memory(memory)

        thread1 = threading.Thread(target=worker, args=("task1",))
        thread2 = threading.Thread(target=worker, args=("task2",))

        # Act
        thread1.start()
        thread2.start()
        thread1.join()
        thread2.join()

        # Assert
        final_memory = self.storage.load_project_memory()
        self.assertEqual(len(final_memory.memories), 2)
        contents = {m.content for m in final_memory.memories}
        self.assertIn("Memory from task1", contents)
        self.assertIn("Memory from task2", contents)