from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from local_agent.context import ContextSelector
from local_agent.models import FileIndex, ProjectContext, SemanticIndex, SymbolDefinition, SymbolLocation
from local_agent.repository import RepositoryIntelligence


class Phase3155Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.root = self.temp_dir / "repo"
        self.root.mkdir()
        self.files = {
            "src/app.js": "export function fetchData() {}\nclass Widget { render() {} }\n",
            "src/service.ts": "export function fetchData() {}\nexport class UserService { save() {} }\n",
            "src/component.jsx": "export const Button = () => <button />;\n",
            "src/view.tsx": "export const Dashboard = () => <main />;\n",
            "src/python_module.py": "def python_function():\n    pass\n",
            "README.md": "semantic indexing fixture",
        }
        for relative, content in self.files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        def symbol(name: str, kind: str, line: int, parent: str | None = None) -> SymbolDefinition:
            return SymbolDefinition(
                name=name,
                kind=kind,
                location=SymbolLocation(start_line=line, end_line=line + 1),
                parent=parent,
            )

        self.semantic_index = SemanticIndex(
            files={
                "src/app.js": FileIndex(
                    path="src/app.js",
                    language="JavaScript",
                    content_hash="js-app",
                    symbols=[
                        symbol("fetchData", "function", 1),
                        symbol("Widget", "class", 2),
                        symbol("render", "method", 2, "Widget"),
                    ],
                ),
                "src/service.ts": FileIndex(
                    path="src/service.ts",
                    language="TypeScript",
                    content_hash="ts-service",
                    symbols=[
                        symbol("fetchData", "function", 1),
                        symbol("UserService", "class", 2),
                        symbol("save", "method", 2, "UserService"),
                    ],
                ),
                "src/component.jsx": FileIndex(
                    path="src/component.jsx",
                    language="JavaScript",
                    content_hash="jsx-component",
                    symbols=[symbol("Button", "function", 1)],
                ),
                "src/view.tsx": FileIndex(
                    path="src/view.tsx",
                    language="TypeScript",
                    content_hash="tsx-view",
                    symbols=[symbol("Dashboard", "function", 1)],
                ),
                "src/python_module.py": FileIndex(
                    path="src/python_module.py",
                    language="Python",
                    content_hash="python-module",
                    symbols=[symbol("python_function", "function", 1)],
                ),
            }
        )
        self.context = ProjectContext(
            root=str(self.root),
            source_files=list(self.files),
            metadata={"semantic_index": self.semantic_index},
        )
        self.selector = ContextSelector(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_javascript_and_typescript_symbol_models(self) -> None:
        javascript = self.semantic_index.files["src/app.js"]
        typescript = self.semantic_index.files["src/service.ts"]
        self.assertEqual(javascript.language, "JavaScript")
        self.assertEqual(typescript.language, "TypeScript")
        self.assertEqual(javascript.symbols[0].kind, "function")
        self.assertEqual(typescript.symbols[0].kind, "function")

    def test_classes_methods_and_exported_declarations(self) -> None:
        app_symbols = self.semantic_index.files["src/app.js"].symbols
        service_symbols = self.semantic_index.files["src/service.ts"].symbols
        self.assertEqual([symbol.name for symbol in app_symbols], ["fetchData", "Widget", "render"])
        self.assertEqual(app_symbols[1].kind, "class")
        self.assertEqual(app_symbols[2].kind, "method")
        self.assertEqual(app_symbols[2].parent, "Widget")
        self.assertEqual(service_symbols[1].name, "UserService")
        self.assertEqual(service_symbols[2].parent, "UserService")

    def test_jsx_and_tsx_file_support(self) -> None:
        self.assertEqual(self.semantic_index.files["src/component.jsx"].language, "JavaScript")
        self.assertEqual(self.semantic_index.files["src/view.tsx"].language, "TypeScript")
        self.assertEqual(self.semantic_index.find_file_for_symbol("Button"), ["src/component.jsx"])
        self.assertEqual(self.semantic_index.find_file_for_symbol("Dashboard"), ["src/view.tsx"])

    def test_multiple_definitions_and_deterministic_ordering(self) -> None:
        matches = self.semantic_index.find_symbols({"fetchData"})
        self.assertEqual([path for path, _ in matches], ["src/app.js", "src/service.ts"])
        self.assertEqual([symbol.name for _, symbol in self.semantic_index.search_symbols("service")], ["UserService"])

    def test_context_selector_prioritizes_javascript_and_typescript_symbols(self) -> None:
        selected = self.selector.select("Fix fetchData", self.context)
        items = selected.metadata["context_selection"]["selected_items"]
        scores = {item["path"]: item["score"] for item in items}
        self.assertGreater(scores["src/app.js"], scores["README.md"] if "README.md" in scores else 0.0)
        self.assertGreater(scores["src/service.ts"], scores["README.md"] if "README.md" in scores else 0.0)
        reasons = [reason for item in items for reason in item["reason"]]
        self.assertTrue(any(reason.startswith("semantic symbol definition match:") for reason in reasons))

    def test_unavailable_semantic_index_falls_back_without_semantic_reason(self) -> None:
        self.context.metadata["semantic_index"] = None
        selected = self.selector.select("Fix fetchData", self.context)
        items = selected.metadata["context_selection"]["selected_items"]
        self.assertTrue(items)
        self.assertFalse(any("semantic symbol definition match:" in reason for item in items for reason in item["reason"]))
        self.assertLess(max(item["score"] for item in items), 0.5)

    def test_python_symbol_and_phase3154_behavior_regression(self) -> None:
        found = self.semantic_index.find_symbol("python_function")
        self.assertTrue(all(isinstance(symbol, SymbolDefinition) for symbol in found))
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].name, "python_function")

    def test_repository_recognizes_js_ts_without_treesitter(self) -> None:
        intelligence = RepositoryIntelligence(self.root)
        context = intelligence.scan()
        self.assertIn("src/app.js", context.source_files)
        self.assertIn("src/service.ts", context.source_files)
        self.assertIsNotNone(context.metadata.get("semantic_index"))


if __name__ == "__main__":
    unittest.main()
