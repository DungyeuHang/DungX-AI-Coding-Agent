import datetime
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from local_agent.config import AgentConfig
from local_agent.context import ContextSelector
from local_agent.knowledge import KnowledgeGraphManager, compute_file_hash, sanitize_text
from local_agent.models import (
    ArchitecturalInvariant,
    BehavioralAssertion,
    CommandSpec,
    ExportedSymbol,
    FailurePatternRecord,
    KnowledgeFileNode,
    KnowledgeSymbolNode,
    Plan,
    ProjectContext,
    ProjectMemory,
    RepositoryKnowledgeGraph,
    RunReport,
    Subtask,
    SubtaskContract,
    Task,
    TaskPlan,
    TaskStatus,
    TestExecutionRecord,
)
from local_agent.orchestrator import Orchestrator
from local_agent.planner import Planner
from local_agent.storage import JsonFileStorage


class TestKnowledgeGraphModels(unittest.TestCase):
    """Unit tests for Phase 4.13 data structures and serialization."""

    def test_behavioral_assertion_serialization(self):
        assertion = BehavioralAssertion(
            assertion_id="assert-1",
            description="Returns status 200 on login",
            test_command="pytest tests/test_auth.py",
            status="passed",
            commit_sha="abc1234",
            verified_at=datetime.datetime(2026, 8, 28, 12, 0, 0, tzinfo=datetime.timezone.utc),
        )
        d = assertion.to_dict()
        self.assertEqual(d["assertion_id"], "assert-1")
        self.assertEqual(d["status"], "passed")
        self.assertEqual(d["commit_sha"], "abc1234")

        restored = BehavioralAssertion.from_dict(d)
        self.assertEqual(restored.assertion_id, assertion.assertion_id)
        self.assertEqual(restored.description, assertion.description)
        self.assertEqual(restored.test_command, assertion.test_command)
        self.assertEqual(restored.status, "passed")
        self.assertEqual(restored.commit_sha, "abc1234")
        self.assertEqual(restored.verified_at, assertion.verified_at)

    def test_knowledge_symbol_node_serialization(self):
        assertion = BehavioralAssertion(
            assertion_id="assert-2",
            description="Computes hash correctly",
            test_command="pytest tests/test_hash.py",
            status="passed",
        )
        symbol = KnowledgeSymbolNode(
            symbol_id="src/utils.py::hash_data",
            name="hash_data",
            kind="function",
            file_path="src/utils.py",
            signature="def hash_data(data: bytes) -> str",
            docstring="Computes SHA256",
            content_hash="deadbeef",
            verified_behaviors=[assertion],
            confidence=0.95,
            provenance="behavioral_test",
            last_verified_at=datetime.datetime.now(datetime.timezone.utc),
        )
        d = symbol.to_dict()
        self.assertEqual(d["symbol_id"], "src/utils.py::hash_data")
        self.assertEqual(len(d["verified_behaviors"]), 1)

        restored = KnowledgeSymbolNode.from_dict(d)
        self.assertEqual(restored.symbol_id, symbol.symbol_id)
        self.assertEqual(len(restored.verified_behaviors), 1)
        self.assertEqual(restored.verified_behaviors[0].assertion_id, "assert-2")
        self.assertEqual(restored.provenance, "behavioral_test")

    def test_knowledge_file_node_serialization(self):
        file_node = KnowledgeFileNode(
            path="src/auth.py",
            content_hash="abcde",
            language=".py",
            module_role="authentication",
            exported_symbol_ids=["src/auth.py::login"],
            dependencies=["src/utils.py"],
            dependents=["src/app.py"],
            validation_commands=["pytest tests/test_auth.py"],
            risk_level="high_risk",
            last_modified_task_id="task-100",
            last_modified_at=datetime.datetime.now(datetime.timezone.utc),
        )
        d = file_node.to_dict()
        self.assertEqual(d["path"], "src/auth.py")
        self.assertEqual(d["risk_level"], "high_risk")

        restored = KnowledgeFileNode.from_dict(d)
        self.assertEqual(restored.path, "src/auth.py")
        self.assertEqual(restored.exported_symbol_ids, ["src/auth.py::login"])
        self.assertEqual(restored.dependencies, ["src/utils.py"])
        self.assertEqual(restored.risk_level, "high_risk")

    def test_architectural_invariant_serialization(self):
        inv = ArchitecturalInvariant(
            invariant_id="inv-1",
            scope="repository",
            target_path="*",
            rule_text="Never import local_agent.tool_engine in approval.py",
            enforcement_type="dependency",
            source_task_id="task-1",
            confidence=1.0,
        )
        d = inv.to_dict()
        restored = ArchitecturalInvariant.from_dict(d)
        self.assertEqual(restored.invariant_id, "inv-1")
        self.assertEqual(restored.enforcement_type, "dependency")
        self.assertEqual(restored.confidence, 1.0)

    def test_failure_pattern_record_serialization(self):
        pat = FailurePatternRecord(
            pattern_id="pat-1",
            error_signature="TypeError: unsupported operand type",
            failing_command="pytest tests/test_calc.py",
            root_cause_summary="Missing string conversion",
            successful_repair_summary="Cast operand with int()",
            affected_files=["src/calc.py"],
            occurrence_count=3,
            confidence=0.9,
        )
        d = pat.to_dict()
        restored = FailurePatternRecord.from_dict(d)
        self.assertEqual(restored.pattern_id, "pat-1")
        self.assertEqual(restored.occurrence_count, 3)
        self.assertEqual(restored.affected_files, ["src/calc.py"])

    def test_repository_knowledge_graph_full_roundtrip(self):
        graph = RepositoryKnowledgeGraph(repo_id="test-repo")
        graph.files["src/main.py"] = KnowledgeFileNode(path="src/main.py", content_hash="hash1")
        graph.symbols["src/main.py::main"] = KnowledgeSymbolNode(
            symbol_id="src/main.py::main",
            name="main",
            file_path="src/main.py",
        )
        graph.invariants.append(
            ArchitecturalInvariant(
                invariant_id="inv-1",
                rule_text="Keep modular structure",
            )
        )
        graph.failure_patterns.append(
            FailurePatternRecord(
                pattern_id="pat-1",
                error_signature="AssertionError",
            )
        )

        d = graph.to_dict()
        restored = RepositoryKnowledgeGraph.from_dict(d)
        self.assertEqual(restored.repo_id, "test-repo")
        self.assertIn("src/main.py", restored.files)
        self.assertIn("src/main.py::main", restored.symbols)
        self.assertEqual(len(restored.invariants), 1)
        self.assertEqual(len(restored.failure_patterns), 1)


class TestStorageAndQuarantine(unittest.TestCase):
    """Unit tests for storage persistence and corruption quarantine."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage = JsonFileStorage(self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_clean_knowledge_graph(self):
        graph = RepositoryKnowledgeGraph(repo_id="my-repo")
        graph.files["app.py"] = KnowledgeFileNode(path="app.py", content_hash="h123")
        self.storage.save_knowledge_graph(graph)

        loaded = self.storage.load_knowledge_graph()
        self.assertEqual(loaded.repo_id, "my-repo")
        self.assertIn("app.py", loaded.files)

    def test_missing_file_returns_empty_graph(self):
        loaded = self.storage.load_knowledge_graph()
        self.assertIsInstance(loaded, RepositoryKnowledgeGraph)
        self.assertEqual(len(loaded.files), 0)

    def test_corrupted_json_quarantines_and_returns_empty_graph(self):
        kg_path = Path(self.temp_dir) / "knowledge_graph.json"
        kg_path.write_text("{corrupt json garbage!!!", encoding="utf-8")

        loaded = self.storage.load_knowledge_graph()
        self.assertIsInstance(loaded, RepositoryKnowledgeGraph)
        self.assertEqual(len(loaded.files), 0)

        # Confirm quarantine file created
        corrupt_files = list(Path(self.temp_dir).glob("knowledge_graph.json.corrupt.*"))
        self.assertGreaterEqual(len(corrupt_files), 1)
        self.assertIn("garbage", corrupt_files[0].read_text(encoding="utf-8"))


class TestKnowledgeGraphManager(unittest.TestCase):
    """Unit tests for KnowledgeGraphManager logic."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "proj"
        self.project_root.mkdir()
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")
        self.manager = KnowledgeGraphManager(self.storage, self.project_root)

        # Create dummy project files
        self.file1 = self.project_root / "math_utils.py"
        self.file1.write_text("def add(a: int, b: int) -> int:\n    return a + b\n", encoding="utf-8")
        self.file2 = self.project_root / "service.py"
        self.file2.write_text("def run():\n    pass\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sync_with_scan_discovers_and_hashes_files(self):
        context = ProjectContext(
            root=str(self.project_root),
            source_files=["math_utils.py", "service.py"],
            test_files=[],
            config_files=[],
        )
        self.manager.sync_with_scan(context)
        graph = self.manager.get_graph()

        self.assertIn("math_utils.py", graph.files)
        self.assertIn("service.py", graph.files)
        self.assertEqual(graph.files["math_utils.py"].content_hash, compute_file_hash(self.file1.read_bytes()))

    def test_sync_with_scan_purges_deleted_files_and_symbols(self):
        context = ProjectContext(
            root=str(self.project_root),
            source_files=["math_utils.py", "service.py"],
            test_files=[],
            config_files=[],
        )
        self.manager.sync_with_scan(context)

        # Add a symbol to service.py
        graph = self.manager.get_graph()
        graph.symbols["service.py::run"] = KnowledgeSymbolNode(
            symbol_id="service.py::run",
            name="run",
            file_path="service.py",
        )
        graph.files["service.py"].exported_symbol_ids.append("service.py::run")

        # Delete service.py on disk
        self.file2.unlink()

        new_context = ProjectContext(
            root=str(self.project_root),
            source_files=["math_utils.py"],
            test_files=[],
            config_files=[],
        )
        self.manager.sync_with_scan(new_context)

        self.assertNotIn("service.py", graph.files)
        self.assertNotIn("service.py::run", graph.symbols)

    def test_sync_with_scan_invalidates_stale_behavioral_assertions_on_hash_mismatch(self):
        # 1. Setup initial contract and symbol
        contract = SubtaskContract(
            subtask_id="sub-1",
            title="Add math utils",
            modified_files=["math_utils.py"],
            exported_symbols=[
                ExportedSymbol(
                    symbol_id="math_utils.py::add",
                    name="add",
                    kind="function",
                    file_path="math_utils.py",
                    signature="def add(a: int, b: int) -> int",
                    verified=True,
                )
            ],
            behavioral_evidence=[
                TestExecutionRecord(
                    test_id="test_add",
                    command="pytest tests/test_math.py",
                    status="passed",
                    exit_code=0,
                    exercised_symbols=["add"],
                )
            ],
        )
        self.manager.promote_subtask_contract(contract)
        graph = self.manager.get_graph()
        self.assertEqual(len(graph.symbols["math_utils.py::add"].verified_behaviors), 1)

        # 2. Modify math_utils.py on disk
        self.file1.write_text("def add(a: int, b: int) -> int:\n    # modified\n    return a + b + 0\n", encoding="utf-8")

        # 3. Sync scan
        context = ProjectContext(root=str(self.project_root), source_files=["math_utils.py"], test_files=[], config_files=[])
        self.manager.sync_with_scan(context)

        # Assertions demoted/cleared because content changed
        self.assertEqual(len(graph.symbols["math_utils.py::add"].verified_behaviors), 0)

    def test_promote_subtask_contract_with_evidence_and_invariants(self):
        contract = SubtaskContract(
            subtask_id="sub-10",
            title="Implement User Registration",
            created_files=["auth.py"],
            modified_files=[],
            exported_symbols=[
                ExportedSymbol(
                    symbol_id="auth.py::register_user",
                    name="register_user",
                    kind="function",
                    file_path="auth.py",
                    signature="def register_user(email: str) -> bool",
                    description="Registers a new user",
                    verified=True,
                )
            ],
            validation_commands=["pytest tests/test_auth.py"],
            architectural_notes=["User emails must be normalized to lowercase."],
            behavioral_evidence=[
                TestExecutionRecord(
                    test_id="test_reg",
                    command="pytest tests/test_auth.py",
                    status="passed",
                    exit_code=0,
                    exercised_symbols=["register_user"],
                )
            ],
        )
        self.manager.promote_subtask_contract(contract)
        graph = self.manager.get_graph()

        self.assertIn("auth.py", graph.files)
        self.assertIn("pytest tests/test_auth.py", graph.files["auth.py"].validation_commands)
        self.assertIn("auth.py::register_user", graph.symbols)
        sym = graph.symbols["auth.py::register_user"]
        self.assertEqual(sym.provenance, "behavioral_test")
        self.assertEqual(len(sym.verified_behaviors), 1)

        self.assertEqual(len(graph.invariants), 1)
        self.assertIn("normalized to lowercase", graph.invariants[0].rule_text)

    def test_secret_sanitization_in_promotions_and_failure_patterns(self):
        secret_note = "Do not share sk-abcdef12345678901234567890 or AIzaSyA1234567890123456789012345678901 in logs."
        contract = SubtaskContract(
            subtask_id="sub-sec",
            title="Secured Subtask",
            modified_files=["secret_mod.py"],
            architectural_notes=[secret_note],
        )
        self.manager.promote_subtask_contract(contract)
        graph = self.manager.get_graph()

        self.assertEqual(len(graph.invariants), 1)
        rule_text = graph.invariants[0].rule_text
        self.assertNotIn("sk-abcdef", rule_text)
        self.assertNotIn("AIzaSyA", rule_text)
        self.assertIn("[REDACTED_SECRET]", rule_text)

        # Test failure pattern sanitization
        self.manager.record_failure_pattern(
            failing_command="pytest --token=Bearer secrettoken123456789",
            error_text="Auth error with token: ghp_123456789012345678901234567890123456",
            repair_summary="Updated config password='supersecretpassword'",
            affected_files=["auth.py"],
        )
        pat = graph.failure_patterns[0]
        self.assertNotIn("secrettoken123456789", pat.failing_command)
        self.assertNotIn("ghp_123456", pat.error_signature)
        self.assertNotIn("supersecretpassword", pat.successful_repair_summary)

    def test_promote_run_report_with_repair_history(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-r1", objective="Fix calculation error", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        context = ProjectContext(root=str(self.project_root), source_files=["calc.py"], test_files=[], config_files=[])
        report = RunReport(
            project=context,
            completed=True,
            changed_files=["calc.py"],
        )
        plan = Plan(objective="Fix calculation error", steps=["Fix bug"])
        plan.repair_history = [
            {
                "status": "success",
                "command": "pytest tests/test_calc.py",
                "error": "ZeroDivisionError: division by zero",
                "summary": "Added check for zero divisor before division",
            }
        ]
        report.plan = plan

        self.manager.promote_run_report(task, report)
        graph = self.manager.get_graph()

        self.assertEqual(len(graph.failure_patterns), 1)
        pat = graph.failure_patterns[0]
        self.assertEqual(pat.failing_command, "pytest tests/test_calc.py")
        self.assertIn("ZeroDivisionError", pat.error_signature)
        self.assertIn("Added check for zero divisor", pat.successful_repair_summary)

        # Repeating the same failure increments occurrence_count
        self.manager.promote_run_report(task, report)
        self.assertEqual(len(graph.failure_patterns), 1)
        self.assertEqual(graph.failure_patterns[0].occurrence_count, 2)

    def test_query_context_knowledge_respects_max_chars_cap(self):
        # Create a large knowledge base
        for i in range(50):
            sym = KnowledgeSymbolNode(
                symbol_id=f"math_utils.py::func_{i}",
                name=f"func_{i}",
                file_path="math_utils.py",
                signature=f"def func_{i}(arg: int) -> int",
                content_hash=compute_file_hash(self.file1.read_bytes()),
                verified_behaviors=[
                    BehavioralAssertion(
                        assertion_id=f"a-{i}",
                        description=f"Behavioral test passing for func_{i} with comprehensive parameter permutations",
                        test_command=f"pytest tests/test_func_{i}.py",
                        status="passed",
                    )
                ],
            )
            self.manager.get_graph().symbols[sym.symbol_id] = sym

        # Query with tight max_chars cap
        formatted_text, syms, invs, pats = self.manager.query_context_knowledge(
            task_objective="Use func_1 and func_2 and math_utils",
            changed_files=["math_utils.py"],
            max_chars=500,
        )
        self.assertLessEqual(len(formatted_text), 500)
        self.assertTrue(len(syms) > 0)

    def test_compaction_enforces_capacity_limits(self):
        graph = self.manager.get_graph()
        # Add 1500 symbols
        for i in range(1500):
            graph.symbols[f"sym_{i}"] = KnowledgeSymbolNode(
                symbol_id=f"sym_{i}",
                name=f"sym_{i}",
                file_path="math_utils.py",
                confidence=0.5 if i < 1000 else 0.95,
                provenance="behavioral_test" if i >= 1000 else "ast_scan",
            )

        # Add 150 invariants
        for i in range(150):
            graph.invariants.append(
                ArchitecturalInvariant(
                    invariant_id=f"inv_{i}",
                    rule_text=f"Rule {i}",
                    confidence=0.5 if i < 100 else 1.0,
                )
            )

        self.manager.compact(max_symbols=1000, max_invariants=100)
        self.assertLessEqual(len(graph.symbols), 1000)
        self.assertLessEqual(len(graph.invariants), 100)

        # High confidence & behavioral symbols preserved
        self.assertIn("sym_1499", graph.symbols)


class TestContextSelectorAndPlannerIntegration(unittest.TestCase):
    """Unit tests for ContextSelector and Planner integration with persistent knowledge."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "proj"
        self.project_root.mkdir()
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")

        self.auth_file = self.project_root / "auth.py"
        self.auth_file.write_text("def authenticate_token(token: str) -> bool:\n    return bool(token)\n", encoding="utf-8")

        self.manager = KnowledgeGraphManager(self.storage, self.project_root)
        self.manager.sync_with_scan(ProjectContext(root=str(self.project_root), source_files=["auth.py"], test_files=[], config_files=[]))

        # Add verified symbol to knowledge graph
        sym = KnowledgeSymbolNode(
            symbol_id="auth.py::authenticate_token",
            name="authenticate_token",
            file_path="auth.py",
            signature="def authenticate_token(token: str) -> bool",
            content_hash=compute_file_hash(self.auth_file.read_bytes()),
            verified_behaviors=[
                BehavioralAssertion(
                    assertion_id="assert-auth",
                    description="Validates JWT token correctly",
                    test_command="pytest tests/test_auth.py",
                    status="passed",
                )
            ],
            confidence=1.0,
            provenance="behavioral_test",
        )
        self.manager.get_graph().symbols[sym.symbol_id] = sym

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_context_selector_boosts_and_enriches_metadata(self):
        selector = ContextSelector(
            self.project_root,
            knowledge_manager=self.manager,
            knowledge_graph=self.manager.get_graph(),
        )
        context = ProjectContext(root=str(self.project_root), source_files=["auth.py"], test_files=[], config_files=[])
        result = selector.select(
            task="Call authenticate_token to verify user login session",
            context=context,
        )

        self.assertIn("persistent_knowledge", result.metadata)
        self.assertIn("authenticate_token", result.metadata["persistent_knowledge"])
        self.assertIn("Validates JWT token correctly", result.metadata["persistent_knowledge"])

    def test_planner_injects_persistent_knowledge_into_prompt(self):
        provider = MagicMock()
        provider.generate_plan.return_value = Plan(objective="Test plan", steps=["Step 1"])
        planner = Planner(provider)

        context = ProjectContext(
            root=str(self.project_root),
            source_files=["auth.py"],
            test_files=[],
            config_files=[],
            metadata={"persistent_knowledge": "VERIFIED REPOSITORY INTERFACES:\n  - `authenticate_token`"},
        )
        plan = planner.create_plan_for_task("Implement login route", context)
        self.assertIsInstance(plan, Plan)

        # Verify the prompt sent to provider included persistent knowledge
        call_args = provider.generate_plan.call_args[0]
        self.assertIn("PERSISTENT REPOSITORY KNOWLEDGE:", call_args[0])
        self.assertIn("authenticate_token", call_args[0])


class TestOrchestratorKnowledgeLifecycle(unittest.TestCase):
    """End-to-end unit tests for Orchestrator integration."""

    def setUp(self):
        import threading
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "proj"
        self.project_root.mkdir()
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")

        # Create a sample file
        (self.project_root / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

        self.config = AgentConfig.from_environment(
            self.project_root,
            provider="mock",
            knowledge_graph_enabled=True,
        )
        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()
        self.orchestrator = Orchestrator(
            self.config,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_orchestrator_initializes_knowledge_manager(self):
        self.assertIsNotNone(self.orchestrator.knowledge_manager)
        self.assertIsInstance(self.orchestrator.knowledge_manager, KnowledgeGraphManager)

    def test_orchestrator_syncs_on_scan(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        task = Task(task_id="task-1", objective="Test task", status=TaskStatus.PENDING, created_at=now, updated_at=now)
        self.storage.save_task(task)
        report = self.orchestrator.run(task)

        # Knowledge graph should have synced calc.py
        graph = self.orchestrator.knowledge_manager.get_graph()
        self.assertIn("calc.py", graph.files)

    def test_knowledge_graph_disabled_via_config(self):
        cfg = AgentConfig.from_environment(
            self.project_root,
            provider="mock",
            knowledge_graph_enabled=False,
        )
        orch = Orchestrator(
            cfg,
            storage=self.storage,
            scheduler=None,
            repo_lock=self.repo_lock,
            memory_lock=self.memory_lock,
        )
        self.assertIsNone(orch.knowledge_manager)


class TestCrossTaskKnowledgeRetention(unittest.TestCase):
    """Verifies that knowledge is durably preserved and transferred across separate tasks."""

    def setUp(self):
        import threading
        self.temp_dir = tempfile.mkdtemp()
        self.project_root = Path(self.temp_dir) / "proj"
        self.project_root.mkdir()
        self.storage = JsonFileStorage(Path(self.temp_dir) / "storage")

        # Create module file
        self.auth_file = self.project_root / "auth.py"
        self.auth_file.write_text("def generate_jwt(user_id: str) -> str:\n    return f'token_{user_id}'\n", encoding="utf-8")

        self.repo_lock = threading.Lock()
        self.memory_lock = threading.Lock()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_cross_task_retention_across_separate_orchestrator_runs(self):
        # 1. First Orchestrator run promotes contract
        config1 = AgentConfig.from_environment(self.project_root, provider="mock")
        orch1 = Orchestrator(config1, self.storage, None, self.repo_lock, self.memory_lock)

        contract = SubtaskContract(
            subtask_id="sub-auth-1",
            title="Create JWT generator",
            modified_files=["auth.py"],
            exported_symbols=[
                ExportedSymbol(
                    symbol_id="auth.py::generate_jwt",
                    name="generate_jwt",
                    kind="function",
                    file_path="auth.py",
                    signature="def generate_jwt(user_id: str) -> str",
                    description="Generates JWT auth token",
                    verified=True,
                )
            ],
            architectural_notes=["Tokens must expire in 24 hours."],
            behavioral_evidence=[
                TestExecutionRecord(
                    test_id="test_jwt",
                    command="pytest tests/test_auth.py",
                    status="passed",
                    exit_code=0,
                    exercised_symbols=["generate_jwt"],
                )
            ],
        )
        orch1.knowledge_manager.promote_subtask_contract(contract)
        orch1.knowledge_manager.save()

        # 2. Second Orchestrator in separate lifecycle loads the knowledge
        config2 = AgentConfig.from_environment(self.project_root, provider="mock")
        orch2 = Orchestrator(config2, self.storage, None, self.repo_lock, self.memory_lock)

        graph2 = orch2.knowledge_manager.get_graph()
        self.assertIn("auth.py::generate_jwt", graph2.symbols)
        sym = graph2.symbols["auth.py::generate_jwt"]
        self.assertEqual(sym.provenance, "behavioral_test")
        self.assertEqual(len(sym.verified_behaviors), 1)
        self.assertEqual(len(graph2.invariants), 1)

        # Context query in task 2 returns verified knowledge
        text, syms, invs, pats = orch2.knowledge_manager.query_context_knowledge("Implement login using generate_jwt")
        self.assertIn("VERIFIED REPOSITORY INTERFACES", text)
        self.assertIn("generate_jwt", text)
        self.assertIn("ARCHITECTURAL INVARIANTS", text)

    def test_stale_hash_invalidation_demotes_assertions(self):
        config = AgentConfig.from_environment(self.project_root, provider="mock")
        orch = Orchestrator(config, self.storage, None, self.repo_lock, self.memory_lock)

        contract = SubtaskContract(
            subtask_id="sub-auth-1",
            title="Create JWT generator",
            modified_files=["auth.py"],
            exported_symbols=[
                ExportedSymbol(
                    symbol_id="auth.py::generate_jwt",
                    name="generate_jwt",
                    kind="function",
                    file_path="auth.py",
                    signature="def generate_jwt(user_id: str) -> str",
                    verified=True,
                )
            ],
            behavioral_evidence=[
                TestExecutionRecord(
                    test_id="test_jwt",
                    command="pytest tests/test_auth.py",
                    status="passed",
                    exit_code=0,
                    exercised_symbols=["generate_jwt"],
                )
            ],
        )
        orch.knowledge_manager.promote_subtask_contract(contract)
        orch.knowledge_manager.save()

        # External disk edit alters file hash
        self.auth_file.write_text("def generate_jwt(user_id: str) -> str:\n    # altered implementation\n    return 'changed'\n", encoding="utf-8")

        # Invalidate via sync_with_scan
        context = orch.analyzer.scan()
        orch.knowledge_manager.sync_with_scan(context)

        sym = orch.knowledge_manager.get_graph().symbols["auth.py::generate_jwt"]
        self.assertEqual(len(sym.verified_behaviors), 0)
        self.assertLess(sym.confidence, 1.0)


if __name__ == "__main__":
    unittest.main()

