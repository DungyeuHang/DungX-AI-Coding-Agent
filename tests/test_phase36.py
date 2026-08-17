from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_agent.analyzer import RepositoryAnalyzer
from local_agent.context import ContextSelector


class Phase36Tests(unittest.TestCase):
    def _repository(self) -> Path:
        root = Path(tempfile.mkdtemp())
        (root / "package.json").write_text(json.dumps({
            "name": "retrieval-fixture",
            "dependencies": {"react": "18.0.0", "react-router-dom": "6.0.0", "firebase": "11.0.0"},
            "devDependencies": {"vitest": "2.0.0"},
        }), encoding="utf-8")
        (root / "src" / "app").mkdir(parents=True)
        (root / "src" / "app" / "layouts").mkdir(parents=True)
        (root / "src" / "components").mkdir(parents=True)
        (root / "src" / "pages").mkdir(parents=True)
        (root / "src" / "pages" / "__tests__").mkdir(parents=True)
        (root / "functions" / "src").mkdir(parents=True)
        (root / "functions" / "tests").mkdir(parents=True)
        (root / "src" / "app" / "router.tsx").write_text(
            "import { createBrowserRouter } from 'react-router-dom';\n"
            "import { RootLayout } from './layouts/RootLayout';\n"
            "import { AppShellLayout } from './layouts/AppShellLayout';\n"
            "import { HomePage } from '../pages/HomePage';\n"
            "import { AboutPage } from '../pages/AboutPage';\n"
            "export const router = createBrowserRouter([{ path: '/', element: <RootLayout />, children: [\n"
            "{ path: 'home', element: <HomePage /> }, { path: 'about', element: <AboutPage /> },\n"
            "{ path: 'app', element: <AppShellLayout /> } ] }]);\n", encoding="utf-8"
        )
        (root / "src" / "app" / "layouts" / "RootLayout.tsx").write_text(
            "import { Outlet } from 'react-router-dom';\nimport { Navigation } from '../../components/Navigation';\nexport function RootLayout() { return <><Navigation /><Outlet /></>; }\n", encoding="utf-8"
        )
        (root / "src" / "app" / "layouts" / "AppShellLayout.tsx").write_text(
            "import { Outlet } from 'react-router-dom';\nimport { Navigation } from '../../components/Navigation';\nexport function AppShellLayout() { return <><Navigation /><Outlet /></>; }\n", encoding="utf-8"
        )
        (root / "src" / "components" / "Navigation.tsx").write_text(
            "import { NavLink } from 'react-router-dom';\nexport function Navigation() { return <nav><NavLink to='/home'>Home</NavLink><NavLink to='/about'>About</NavLink></nav>; }\n", encoding="utf-8"
        )
        (root / "src" / "pages" / "HomePage.tsx").write_text(
            "import { Card } from '../components/Card';\nexport function HomePage() { return <Card>Student home</Card>; }\n", encoding="utf-8"
        )
        (root / "src" / "pages" / "AboutPage.tsx").write_text(
            "import { Card } from '../components/Card';\nexport function AboutPage() { return <Card>About the application</Card>; }\n", encoding="utf-8"
        )
        (root / "src" / "components" / "Card.tsx").write_text(
            "export function Card({ children }: { children: string }) { return <section>{children}</section>; }\n", encoding="utf-8"
        )
        (root / "src" / "pages" / "__tests__" / "AboutPage.test.tsx").write_text(
            "import { AboutPage } from '../AboutPage';\ntest('renders about page', () => expect(AboutPage).toBeDefined());\n", encoding="utf-8"
        )
        (root / "src" / "pages" / "__tests__" / "HomePage.test.tsx").write_text(
            "import { HomePage } from '../HomePage';\ntest('renders home page', () => expect(HomePage).toBeDefined());\n", encoding="utf-8"
        )
        (root / "functions" / "src" / "membershipHandlers.ts").write_text(
            "import { getFirestore } from 'firebase-admin/firestore';\nexport function handleMembership() { return getFirestore(); }\n", encoding="utf-8"
        )
        (root / "functions" / "tests" / "membershipHandlers.test.ts").write_text(
            "import { handleMembership } from '../src/membershipHandlers';\ntest('handles membership', () => expect(handleMembership).toBeDefined());\n", encoding="utf-8"
        )
        (root / "functions" / "src" / "unrelated.ts").write_text("export const value = 1;\n", encoding="utf-8")
        return root

    def _select(self, root: Path, task: str, max_files: int = 12):
        context = RepositoryAnalyzer(root).analyze()
        ContextSelector(root, max_files=max_files, max_chars=50000, max_file_chars=3000, max_tokens=12000, dependency_depth=1).select(task, context)
        return context

    def test_about_navigation_retrieval_prefers_ui_architecture(self):
        context = self._select(self._repository(), "Add an About page and register it in navigation.")
        selected = context.metadata["selected_files"]
        expected = {
            "src/app/router.tsx",
            "src/app/layouts/RootLayout.tsx",
            "src/app/layouts/AppShellLayout.tsx",
            "src/components/Navigation.tsx",
            "src/pages/AboutPage.tsx",
            "package.json",
        }
        self.assertTrue(expected.issubset(set(selected)), selected)
        self.assertNotIn("functions/src/membershipHandlers.ts", selected[:6])
        reasons = {reason for item in context.metadata["context_selection"]["selected_items"] for reason in item["reason"]}
        self.assertIn("router/navigation architecture", reasons)
        self.assertIn("direct router relationship", reasons)

    def test_student_home_retrieval_includes_page_dependencies_and_tests(self):
        context = self._select(self._repository(), "Fix the student home page.")
        selected = set(context.metadata["selected_files"])
        self.assertIn("src/pages/HomePage.tsx", selected)
        self.assertIn("src/components/Card.tsx", selected)
        self.assertIn("src/pages/__tests__/HomePage.test.tsx", selected)

    def test_firebase_membership_task_prefers_backend_files(self):
        context = self._select(self._repository(), "Fix Firebase membership handling.")
        selected = context.metadata["selected_files"]
        self.assertIn("functions/src/membershipHandlers.ts", selected)
        self.assertIn("functions/tests/membershipHandlers.test.ts", selected)
        self.assertLess(selected.index("functions/src/membershipHandlers.ts"), selected.index("src/app/router.tsx") if "src/app/router.tsx" in selected else len(selected))

    def test_retrieval_preserves_context_budgets(self):
        context = self._select(self._repository(), "Add an About page and register it in navigation.", max_files=4)
        previews = context.metadata["selected_file_previews"]
        selection = context.metadata["context_selection"]
        self.assertLessEqual(len(previews), 4)
        self.assertLessEqual(sum(len(value.encode("utf-8")) for value in previews.values()), 50000)
        self.assertLessEqual(selection["estimated_tokens"], 12000)


if __name__ == "__main__":
    unittest.main()
