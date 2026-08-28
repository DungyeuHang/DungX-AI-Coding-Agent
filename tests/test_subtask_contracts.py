from __future__ import annotations

import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock

from local_agent.coding_agent import CodingAgent, UnsafeModificationError
from local_agent.context import ContextSelector
from local_agent.contract_extractor import ContractExtractor
from local_agent.filesystem import ProjectFilesystem
from local_agent.models import (
    Checkpoint,
    CommandSpec,
    ExecutionResult,
    ExportedSymbol,
    FileOperation,
    Plan,
    ProjectContext,
    RunReport,
    Subtask,
    SubtaskContract,
    SubtaskStatus,
    Task,
    TaskPlan,
    TaskStatus,
    ValidationPlan,
)
from local_agent.planner import Planner
from local_agent.providers import AIProvider


class TestSubtaskContractModels(unittest.TestCase):
    def test_exported_symbol_creation_and_bounds(self):
        long_sig = "def very_long_function(" + "arg: int, " * 100 + ") -> None:"
        long_desc = "A" * 600
        sym = ExportedSymbol(
            symbol_id="test.py::func",
            name="func",
            kind="function",
            file_path="test.py",
            signature=long_sig,
            description=long_desc,
        )
        self.assertEqual(sym.name, "func")
        self.assertEqual(sym.kind, "function")
        self.assertLessEqual(len(sym.signature), 500)
        self.assertTrue(sym.signature.endswith("..."))
        self.assertLessEqual(len(sym.description), 500)
        self.assertTrue(sym.description.endswith("..."))

    def test_exported_symbol_serialization(self):
        sym = ExportedSymbol(
            symbol_id="auth.py::AuthService",
            name="AuthService",
            kind="class",
            file_path="auth.py",
            signature="class AuthService(BaseService):",
            description="Handles user authentication.",
        )
        data = sym.to_dict()
        restored = ExportedSymbol.from_dict(data)
        self.assertEqual(restored.symbol_id, sym.symbol_id)
        self.assertEqual(restored.name, sym.name)
        self.assertEqual(restored.kind, sym.kind)
        self.assertEqual(restored.file_path, sym.file_path)
        self.assertEqual(restored.signature, sym.signature)
        self.assertEqual(restored.description, sym.description)

    def test_subtask_contract_creation_and_bounds(self):
        symbols = [
            ExportedSymbol(symbol_id=f"f.py::s{i}", name=f"s{i}", kind="function", file_path="f.py")
            for i in range(25)
        ]
        modified = [f"mod_{i}.py" for i in range(30)]
        created = [f"create_{i}.py" for i in range(30)]
        val_cmds = [f"pytest test_{i}.py" for i in range(15)]
        notes = [f"Note {i}" for i in range(15)]

        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Subtask 1",
            exported_symbols=symbols,
            modified_files=modified,
            created_files=created,
            validation_commands=val_cmds,
            architectural_notes=notes,
        )
        self.assertEqual(len(contract.exported_symbols), 10)
        self.assertEqual(len(contract.modified_files), 20)
        self.assertEqual(len(contract.created_files), 20)
        self.assertEqual(len(contract.validation_commands), 10)
        self.assertEqual(len(contract.architectural_notes), 10)

    def test_subtask_contract_serialization(self):
        contract = SubtaskContract(
            subtask_id="sub-auth",
            title="Implement Auth",
            exported_symbols=[
                ExportedSymbol(
                    symbol_id="auth.py::login",
                    name="login",
                    kind="function",
                    file_path="auth.py",
                    signature="def login(user: str, pwd: str) -> bool:",
                    description="Authenticates user.",
                )
            ],
            modified_files=["app.py"],
            created_files=["auth.py"],
            validation_commands=["pytest tests/test_auth.py"],
            architectural_notes=["Token based authentication"],
        )
        data = contract.to_dict()
        self.assertEqual(data["subtask_id"], "sub-auth")
        self.assertEqual(len(data["exported_symbols"]), 1)
        self.assertEqual(data["created_files"], ["auth.py"])

        restored = SubtaskContract.from_dict(data)
        self.assertEqual(restored.subtask_id, contract.subtask_id)
        self.assertEqual(restored.title, contract.title)
        self.assertEqual(len(restored.exported_symbols), 1)
        self.assertEqual(restored.exported_symbols[0].name, "login")
        self.assertEqual(restored.created_files, ["auth.py"])
        self.assertEqual(restored.validation_commands, ["pytest tests/test_auth.py"])

    def test_subtask_contract_format_for_prompt(self):
        contract = SubtaskContract(
            subtask_id="sub-rate",
            title="Add Rate Limiter",
            exported_symbols=[
                ExportedSymbol(
                    symbol_id="limiter.py::RateLimiter",
                    name="RateLimiter",
                    kind="class",
                    file_path="limiter.py",
                    signature="class RateLimiter:",
                    description="Token bucket limiter",
                )
            ],
            modified_files=["config.py"],
            created_files=["limiter.py"],
            validation_commands=["pytest tests/test_limiter.py"],
            architectural_notes=["Default rate is 60 req/min"],
        )
        formatted = contract.format_for_prompt(max_chars=500)
        self.assertIn("Subtask Contract: 'Add Rate Limiter'", formatted)
        self.assertIn("Created Files: limiter.py", formatted)
        self.assertIn("[class] `class RateLimiter:` in `limiter.py`", formatted)
        self.assertIn("Verified Validation Commands: pytest tests/test_limiter.py", formatted)
        self.assertIn("Default rate is 60 req/min", formatted)

    def test_subtask_model_contract_field_and_backward_compatibility(self):
        # Legacy dict without contract field
        legacy_data = {
            "subtask_id": "legacy-1",
            "status": "PENDING",
            "title": "Legacy Subtask",
            "goal": "Legacy goal",
        }
        subtask = Subtask.from_dict(legacy_data)
        self.assertIsNone(subtask.contract)

        # Serializing subtask with contract
        subtask.contract = SubtaskContract(
            subtask_id="legacy-1",
            title="Legacy Subtask",
            created_files=["legacy.py"],
        )
        serialized = subtask.to_dict()
        self.assertIn("contract", serialized)
        self.assertEqual(serialized["contract"]["created_files"], ["legacy.py"])

        # Restoring
        restored = Subtask.from_dict(serialized)
        self.assertIsNotNone(restored.contract)
        self.assertEqual(restored.contract.created_files, ["legacy.py"])


class TestTaskPlanUpstreamContractResolution(unittest.TestCase):
    def test_single_dependency_contract_resolution(self):
        sub_a = Subtask(
            subtask_id="sub-a",
            title="Subtask A",
            status=SubtaskStatus.COMPLETED,
            contract=SubtaskContract(subtask_id="sub-a", title="Subtask A", created_files=["a.py"]),
        )
        sub_b = Subtask(
            subtask_id="sub-b",
            title="Subtask B",
            status=SubtaskStatus.PENDING,
            dependencies=["sub-a"],
        )
        plan = TaskPlan(objective="Test Plan", subtasks=[sub_a, sub_b])
        contracts = plan.get_upstream_contracts("sub-b")
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].subtask_id, "sub-a")
        self.assertEqual(contracts[0].created_files, ["a.py"])

    def test_multi_dependency_deterministic_contract_resolution(self):
        sub_a = Subtask(
            subtask_id="sub-a",
            title="Subtask A",
            status=SubtaskStatus.COMPLETED,
            contract=SubtaskContract(subtask_id="sub-a", title="Subtask A", created_files=["a.py"]),
        )
        sub_b = Subtask(
            subtask_id="sub-b",
            title="Subtask B",
            status=SubtaskStatus.COMPLETED,
            contract=SubtaskContract(subtask_id="sub-b", title="Subtask B", created_files=["b.py"]),
        )
        sub_c = Subtask(
            subtask_id="sub-c",
            title="Subtask C",
            status=SubtaskStatus.PENDING,
            dependencies=["sub-a", "sub-b"],
        )
        plan = TaskPlan(objective="Test Plan", subtasks=[sub_a, sub_b, sub_c])
        contracts = plan.get_upstream_contracts("sub-c")
        self.assertEqual(len(contracts), 2)
        self.assertEqual([c.subtask_id for c in contracts], ["sub-a", "sub-b"])

    def test_excludes_superseded_pruned_and_non_completed_dependencies(self):
        sub_superseded = Subtask(
            subtask_id="sub-old",
            title="Old Subtask",
            status=SubtaskStatus.SUPERSEDED,
            contract=SubtaskContract(subtask_id="sub-old", title="Old Subtask", created_files=["old.py"]),
        )
        sub_pruned = Subtask(
            subtask_id="sub-pruned",
            title="Pruned Subtask",
            status=SubtaskStatus.PRUNED,
            contract=SubtaskContract(subtask_id="sub-pruned", title="Pruned Subtask", created_files=["pruned.py"]),
        )
        sub_failed = Subtask(
            subtask_id="sub-failed",
            title="Failed Subtask",
            status=SubtaskStatus.FAILED,
            contract=SubtaskContract(subtask_id="sub-failed", title="Failed Subtask"),
        )
        sub_pending = Subtask(
            subtask_id="sub-pending",
            title="Pending Subtask",
            status=SubtaskStatus.PENDING,
            contract=SubtaskContract(subtask_id="sub-pending", title="Pending Subtask"),
        )
        sub_running = Subtask(
            subtask_id="sub-running",
            title="Running Subtask",
            status=SubtaskStatus.RUNNING,
            contract=SubtaskContract(subtask_id="sub-running", title="Running Subtask"),
        )
        sub_completed = Subtask(
            subtask_id="sub-valid",
            title="Valid Subtask",
            status=SubtaskStatus.COMPLETED,
            contract=SubtaskContract(subtask_id="sub-valid", title="Valid Subtask", created_files=["valid.py"]),
        )
        sub_consumer = Subtask(
            subtask_id="sub-consumer",
            title="Consumer Subtask",
            status=SubtaskStatus.PENDING,
            dependencies=["sub-old", "sub-pruned", "sub-failed", "sub-pending", "sub-running", "sub-valid"],
        )
        plan = TaskPlan(
            objective="Test DAG Plan",
            subtasks=[sub_superseded, sub_pruned, sub_failed, sub_pending, sub_running, sub_completed, sub_consumer],
        )
        contracts = plan.get_upstream_contracts("sub-consumer")
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].subtask_id, "sub-valid")


class TestContractExtractor(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_python_ast_class_and_function_extraction(self):
        code = '''"""Module docstring."""

class UserService:
    """Manages user operations."""
    def get_user(self, user_id: str) -> dict:
        return {"id": user_id}

async def fetch_user_data(user_id: str, timeout: int = 5) -> dict:
    """Fetches user data asynchronously."""
    return {}

def _internal_helper(x: int) -> int:
    return x * 2

UserMapping = dict[str, UserService]
'''
        self.filesystem.create_file("services/user.py", code)
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-user", title="Add User Service")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["services/user.py"])

        contract = extractor.extract_contract(subtask, report, preexisting_files=set())
        self.assertEqual(contract.subtask_id, "sub-user")
        self.assertIn("services/user.py", contract.created_files)

        sym_names = [s.name for s in contract.exported_symbols]
        self.assertIn("UserService", sym_names)
        self.assertIn("fetch_user_data", sym_names)
        self.assertNotIn("_internal_helper", sym_names)

        user_svc = next(s for s in contract.exported_symbols if s.name == "UserService")
        self.assertEqual(user_svc.kind, "class")
        self.assertEqual(user_svc.signature, "class UserService:")
        self.assertEqual(user_svc.description, "Manages user operations.")

        fetch_func = next(s for s in contract.exported_symbols if s.name == "fetch_user_data")
        self.assertEqual(fetch_func.kind, "function")
        self.assertTrue(fetch_func.signature.startswith("async def fetch_user_data("))
        self.assertIn("-> dict", fetch_func.signature)
        self.assertEqual(fetch_func.description, "Fetches user data asynchronously.")

    def test_javascript_typescript_regex_extraction(self):
        ts_code = '''
export interface UserConfig {
    id: string;
    roles: string[];
}

export class AuthManager {
    constructor() {}
}

export async function authenticate(token: string): Promise<boolean> {
    return true;
}

export const DEFAULT_TIMEOUT = 5000;
'''
        self.filesystem.create_file("src/auth.ts", ts_code)
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-ts", title="Add TypeScript Auth")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["src/auth.ts"])

        contract = extractor.extract_contract(subtask, report, preexisting_files=set())
        sym_names = [s.name for s in contract.exported_symbols]
        self.assertIn("UserConfig", sym_names)
        self.assertIn("AuthManager", sym_names)
        self.assertIn("authenticate", sym_names)
        self.assertIn("DEFAULT_TIMEOUT", sym_names)

    def test_validation_command_and_architectural_notes_extraction(self):
        self.filesystem.create_file("api.py", "def get_data(): pass\n")
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-api", title="Create API")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["api.py"])
        report.validation_plan = ValidationPlan(
            commands=[],
            primary_commands=[CommandSpec("pytest", ("pytest", "tests/test_api.py"), "test")],
            secondary_commands=[CommandSpec("pytest-k", ("pytest", "-k", "api"), "test")],
            skipped_commands=[],
            reasons=["test"],
            risk_level="low",
        )

        contract = extractor.extract_contract(subtask, report, preexisting_files=set())
        self.assertIn("pytest tests/test_api.py", contract.validation_commands)
        self.assertIn("pytest -k api", contract.validation_commands)
        self.assertTrue(any("api.py" in note for note in contract.architectural_notes))

    def test_syntax_error_graceful_handling(self):
        self.filesystem.create_file("broken.py", "def broken_syntax(:\n")
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-broken", title="Broken Syntax")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["broken.py"])

        contract = extractor.extract_contract(subtask, report)
        self.assertEqual(len(contract.exported_symbols), 0)
        self.assertEqual(contract.modified_files, ["broken.py"])


class TestContextAndPlannerIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_context_selector_prioritizes_upstream_contract_files(self):
        self.filesystem.create_file("upstream_module.py", "class UpstreamWorker: pass\n")
        self.filesystem.create_file("unrelated_file.py", "# Unrelated content\n")
        context = ProjectContext(root=str(self.root), source_files=["upstream_module.py", "unrelated_file.py"])

        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Create Upstream Worker",
            created_files=["upstream_module.py"],
            exported_symbols=[
                ExportedSymbol(symbol_id="upstream_module.py::UpstreamWorker", name="UpstreamWorker", kind="class", file_path="upstream_module.py")
            ],
        )

        selector = ContextSelector(self.root)
        selected_context = selector.select(
            task="Integrate worker",
            context=context,
            subtask_goal="Connect downstream component to UpstreamWorker",
            upstream_contracts=[contract],
        )
        self.assertIn("upstream_module.py", selected_context.metadata["selected_files"])

    def test_planner_create_subtask_plan_formats_upstream_contracts(self):
        provider = MagicMock(spec=AIProvider)
        provider.generate_plan.return_value = Plan(objective="Subtask plan", steps=["Step 1"])

        planner = Planner(provider)
        subtask = Subtask(subtask_id="sub-2", title="Use Limiter", goal="Apply rate limiting")
        context = ProjectContext(root=str(self.root))
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Rate Limiter",
            created_files=["limiter.py"],
            exported_symbols=[
                ExportedSymbol(symbol_id="limiter.py::RateLimiter", name="RateLimiter", kind="class", file_path="limiter.py", signature="class RateLimiter:")
            ],
        )

        planner.create_subtask_plan(subtask, context, upstream_contracts=[contract])
        # Verify provider received the UPSTREAM SUBTASK CONTRACTS in the task prompt
        provider_call_arg = provider.generate_plan.call_args[0][0]
        self.assertIn("UPSTREAM SUBTASK CONTRACTS", provider_call_arg)
        self.assertIn("RateLimiter", provider_call_arg)
        self.assertIn("limiter.py", provider_call_arg)


class TestSecurityAndCheckpointInvariants(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_contract_cannot_bypass_scope_amendment_or_unsafe_modification(self):
        self.filesystem.create_file("allowed.py", "x = 1\n")
        self.filesystem.create_file("unauthorized.py", "secret = True\n")
        coding_agent = CodingAgent(self.filesystem)
        plan = Plan(objective="Allowed edits", steps=["Edit allowed"], files_likely_to_change=["allowed.py"])

        # Operation targeting unauthorized.py must raise UnsafeModificationError even if contract mentions it
        ops = [
            FileOperation(path="unauthorized.py", action="write", content="secret = False\n")
        ]
        with self.assertRaises(UnsafeModificationError):
            coding_agent.prepare(ops, plan)

    def test_checkpoint_subtask_contract_persistence_and_restoration(self):
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Step 1",
            created_files=["module_a.py"],
            validation_commands=["pytest tests/test_a.py"],
        )
        subtask_1 = Subtask(subtask_id="sub-1", title="Step 1", status=SubtaskStatus.COMPLETED, contract=contract)
        subtask_2 = Subtask(subtask_id="sub-2", title="Step 2", status=SubtaskStatus.PENDING, dependencies=["sub-1"])
        plan = TaskPlan(objective="Overall Task", subtasks=[subtask_1, subtask_2])

        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-100", objective="Overall Task", status=TaskStatus.PENDING, created_at=now, updated_at=now, plan=plan)

        # Create checkpoint continuation context with task_plan
        checkpoint = Checkpoint(
            checkpoint_id="chk-1",
            task_id="task-100",
            subtask_id="sub-1",
            timestamp=datetime.datetime.now(datetime.timezone.utc),
            current_state_description="Completed subtask 1",
            continuation_context={"task_plan": task.plan.to_dict()},
        )

        chk_dict = checkpoint.to_dict()
        restored_chk = Checkpoint.from_dict(chk_dict)
        restored_tp = TaskPlan.from_dict(restored_chk.continuation_context["task_plan"])

        upstream_contracts = restored_tp.get_upstream_contracts("sub-2")
        self.assertEqual(len(upstream_contracts), 1)
        self.assertEqual(upstream_contracts[0].subtask_id, "sub-1")
        self.assertEqual(upstream_contracts[0].created_files, ["module_a.py"])
        self.assertEqual(upstream_contracts[0].validation_commands, ["pytest tests/test_a.py"])


class TestSubtaskContractEdgeCasesAndExtractionLimits(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exported_symbol_empty_defaults(self):
        sym = ExportedSymbol(symbol_id="a.py::X", name="X", kind="class", file_path="a.py")
        self.assertEqual(sym.signature, "")
        self.assertEqual(sym.description, "")

    def test_subtask_contract_empty_defaults(self):
        contract = SubtaskContract(subtask_id="s1", title="Title")
        self.assertEqual(contract.exported_symbols, [])
        self.assertEqual(contract.modified_files, [])
        self.assertEqual(contract.created_files, [])
        self.assertEqual(contract.validation_commands, [])
        self.assertEqual(contract.architectural_notes, [])
        self.assertIsInstance(contract.created_at, datetime.datetime)

    def test_subtask_contract_format_for_prompt_truncation(self):
        contract = SubtaskContract(
            subtask_id="s1",
            title="A very long contract title",
            architectural_notes=["Note A" * 50, "Note B" * 50],
        )
        formatted = contract.format_for_prompt(max_chars=50)
        self.assertLessEqual(len(formatted), 50)
        self.assertTrue(formatted.endswith("..."))

    def test_task_plan_get_upstream_contracts_target_not_found(self):
        plan = TaskPlan(objective="Test", subtasks=[])
        contracts = plan.get_upstream_contracts("non_existent_id")
        self.assertEqual(contracts, [])

    def test_task_plan_get_upstream_contracts_no_dependencies(self):
        sub = Subtask(subtask_id="s1", title="Independent Subtask", dependencies=[])
        plan = TaskPlan(objective="Test", subtasks=[sub])
        contracts = plan.get_upstream_contracts("s1")
        self.assertEqual(contracts, [])

    def test_task_plan_get_upstream_contracts_dependency_missing_from_plan(self):
        sub = Subtask(subtask_id="s2", title="Depends on missing", dependencies=["s_missing"])
        plan = TaskPlan(objective="Test", subtasks=[sub])
        contracts = plan.get_upstream_contracts("s2")
        self.assertEqual(contracts, [])

    def test_task_plan_get_upstream_contracts_dependency_has_no_contract(self):
        sub_1 = Subtask(subtask_id="s1", title="Completed without contract", status=SubtaskStatus.COMPLETED, contract=None)
        sub_2 = Subtask(subtask_id="s2", title="Consumer", status=SubtaskStatus.PENDING, dependencies=["s1"])
        plan = TaskPlan(objective="Test", subtasks=[sub_1, sub_2])
        contracts = plan.get_upstream_contracts("s2")
        self.assertEqual(contracts, [])

    def test_contract_extractor_limits_symbols_to_ten(self):
        # Create a Python file with 20 functions
        code_lines = [f"def func_{i}(x: int) -> int:\n    return {i}\n" for i in range(20)]
        self.filesystem.create_file("many_funcs.py", "\n".join(code_lines))

        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="s-many", title="Many Functions")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["many_funcs.py"])

        contract = extractor.extract_contract(subtask, report)
        self.assertLessEqual(len(contract.exported_symbols), 10)
        self.assertEqual(len(contract.exported_symbols), 10)

    def test_contract_extractor_fallback_when_file_not_found(self):
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="s-missing", title="Missing File")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["non_existent.py"])

        contract = extractor.extract_contract(subtask, report)
        self.assertEqual(len(contract.exported_symbols), 0)
        self.assertEqual(contract.modified_files, ["non_existent.py"])

    def test_contract_extractor_type_alias_and_constants(self):
        code = '''
MAX_RETRIES: int = 5
EndpointMap: dict[str, str] = {}
_PRIVATE_CONST = 100
'''
        self.filesystem.create_file("constants.py", code)
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="s-const", title="Constants")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["constants.py"])

        contract = extractor.extract_contract(subtask, report)
        sym_names = [s.name for s in contract.exported_symbols]
        self.assertIn("MAX_RETRIES", sym_names)
        self.assertIn("EndpointMap", sym_names)
        self.assertNotIn("_PRIVATE_CONST", sym_names)

    def test_subtask_contract_deserialization_with_dict_symbols(self):
        data = {
            "subtask_id": "s-dict",
            "title": "Dict Symbols",
            "exported_symbols": [
                {
                    "symbol_id": "a.py::func",
                    "name": "func",
                    "kind": "function",
                    "file_path": "a.py",
                    "signature": "def func():",
                    "description": "test",
                }
            ],
            "modified_files": ["a.py"],
            "created_files": [],
            "validation_commands": [],
            "architectural_notes": [],
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        contract = SubtaskContract.from_dict(data)
        self.assertEqual(len(contract.exported_symbols), 1)
        self.assertIsInstance(contract.exported_symbols[0], ExportedSymbol)
        self.assertEqual(contract.exported_symbols[0].name, "func")


class TestOrchestratorContractIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.filesystem = ProjectFilesystem(self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_contract_extractor_synthesizes_all_fields_cohesively(self):
        code = '''class DBConnection:
    """Manages database connection pooling."""
    def query(self, sql: str) -> list:
        return []

def init_db(url: str) -> DBConnection:
    """Initializes the database."""
    return DBConnection()
'''
        self.filesystem.create_file("db.py", code)
        extractor = ContractExtractor(filesystem=self.filesystem, project_root=self.root)
        subtask = Subtask(subtask_id="sub-db", title="Initialize DB Engine", goal="Set up DB")
        report = RunReport(project=ProjectContext(root=str(self.root)), changed_files=["db.py"])
        report.validation_plan = ValidationPlan(
            commands=[],
            primary_commands=[CommandSpec("test-db", ("pytest", "tests/test_db.py"), "test db")],
            secondary_commands=[],
            skipped_commands=[],
            reasons=["db tests"],
            risk_level="low",
        )

        contract = extractor.extract_contract(subtask, report, preexisting_files=set())
        self.assertEqual(contract.subtask_id, "sub-db")
        self.assertEqual(contract.title, "Initialize DB Engine")
        self.assertEqual(len(contract.exported_symbols), 2)
        self.assertEqual(contract.created_files, ["db.py"])
        self.assertEqual(contract.validation_commands, ["pytest tests/test_db.py"])
        self.assertTrue(len(contract.architectural_notes) >= 2)

    def test_downstream_task_plan_contract_chaining(self):
        # A -> B -> C
        sub_a = Subtask(
            subtask_id="A",
            title="A",
            status=SubtaskStatus.COMPLETED,
            contract=SubtaskContract(subtask_id="A", title="A", created_files=["a.py"]),
        )
        sub_b = Subtask(
            subtask_id="B",
            title="B",
            status=SubtaskStatus.COMPLETED,
            dependencies=["A"],
            contract=SubtaskContract(subtask_id="B", title="B", created_files=["b.py"]),
        )
        sub_c = Subtask(
            subtask_id="C",
            title="C",
            status=SubtaskStatus.PENDING,
            dependencies=["B"],
        )
        plan = TaskPlan(objective="Chain", subtasks=[sub_a, sub_b, sub_c])

        # C only depends on B, so C only receives B's contract
        c_contracts = plan.get_upstream_contracts("C")
        self.assertEqual(len(c_contracts), 1)
        self.assertEqual(c_contracts[0].subtask_id, "B")

        # If C is rewired to depend on both A and B:
        sub_c.dependencies = ["A", "B"]
        c_contracts_all = plan.get_upstream_contracts("C")
        self.assertEqual(len(c_contracts_all), 2)
        self.assertEqual([c.subtask_id for c in c_contracts_all], ["A", "B"])


if __name__ == "__main__":
    unittest.main()
