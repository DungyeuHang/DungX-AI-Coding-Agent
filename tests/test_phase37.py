from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from local_agent.context import ContextSelector
from local_agent.impact import ChangeImpactAnalyzer
from local_agent.repository import RepositoryIntelligence
from local_agent.models import ChangeImpact


class Phase37Tests(unittest.TestCase):
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
        (root / "src" / "auth").mkdir(parents=True)
        (root / "functions" / "src").mkdir(parents=True)
        (root / "functions" / "tests").mkdir(parents=True)
        (root / "src" / "app" / "router.tsx").write_text(
            "import { createBrowserRouter } from 'react-router-dom';\n"
            "import { RootLayout } from './layouts/RootLayout';\n"
            "import { HomePage } from '../pages/HomePage';\n"
            "export const router = createBrowserRouter([{ path: '/', element: <RootLayout />, children: [\n"
            "{ path: 'home', element: <HomePage /> } ] }]);\n", encoding="utf-8"
        )
        (root / "src" / "components" / "Navigation.tsx").write_text(
            "import { NavLink } from 'react-router-dom';\nexport function Navigation() { return <nav><NavLink to='/home'>Home</NavLink></nav>; }\n", encoding="utf-8"
        )
        (root / "src" / "pages" / "HomePage.tsx").write_text(
            "import { Card } from '../components/Card';\nexport function HomePage() { return <Card>Student home</Card>; }\n", encoding="utf-8"
        )
        (root / "src" / "components" / "Card.tsx").write_text(
            "export function Card({ children }: { children: string }) { return <section>{children}</section>; }\n", encoding="utf-8"
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
        (root / "src" / "auth" / "authService.ts").write_text("export function signIn() {}\n", encoding="utf-8")
        return root

    def _analyze_impact(self, root: Path, task: str) -> ChangeImpact:
        # This is a simplified analysis flow for testing purposes.
        # RepositoryIntelligence scans the project to build the repository map.
        context = RepositoryIntelligence(root).scan()
        # ContextSelector narrows down the files for the task.
        ContextSelector(root, max_files=12).select(task, context)
        return ChangeImpactAnalyzer(root).analyze(task, context)

    def test_add_about_page_impact_analysis(self):
        impact = self._analyze_impact(self._repository(), "Add an About page and register it in navigation.")
        targets = {t.path: t for t in impact.targets}
        self.assertIn("src/pages/AboutPage.tsx", targets)
        self.assertEqual(targets["src/pages/AboutPage.tsx"].role, "create")
        self.assertEqual(targets["src/pages/AboutPage.tsx"].risk, "low")
        self.assertIn("src/app/router.tsx", targets)
        self.assertEqual(targets["src/app/router.tsx"].role, "modify")
        self.assertEqual(targets["src/app/router.tsx"].risk, "medium")
        self.assertIn("src/components/Navigation.tsx", targets)
        self.assertEqual(targets["src/components/Navigation.tsx"].role, "modify")
        self.assertNotIn("functions/src/membershipHandlers.ts", targets)

    def test_fix_home_page_impact_analysis(self):
        impact = self._analyze_impact(self._repository(), "Fix the student home page.")
        targets = {t.path: t for t in impact.targets}
        self.assertIn("src/pages/HomePage.tsx", targets)
        self.assertEqual(targets["src/pages/HomePage.tsx"].role, "modify")
        self.assertIn("src/components/Card.tsx", targets)
        self.assertEqual(targets["src/components/Card.tsx"].role, "architecture")
        self.assertIn("src/pages/__tests__/HomePage.test.tsx", targets)
        self.assertEqual(targets["src/pages/__tests__/HomePage.test.tsx"].role, "test")

    def test_fix_firebase_backend_impact_analysis(self):
        impact = self._analyze_impact(self._repository(), "Fix Firebase membership handling.")
        targets = {t.path: t for t in impact.targets}
        self.assertIn("functions/src/membershipHandlers.ts", targets)
        self.assertEqual(targets["functions/src/membershipHandlers.ts"].role, "modify")
        self.assertIn("functions/tests/membershipHandlers.test.ts", targets)
        self.assertEqual(targets["functions/tests/membershipHandlers.test.ts"].role, "test")
        self.assertNotIn("src/pages/HomePage.tsx", targets)

    def test_auth_change_risk_is_high(self):
        impact = self._analyze_impact(self._repository(), "Change authentication logic.")
        target = next(t for t in impact.targets if t.path == "src/auth/authService.ts")
        self.assertEqual(target.risk, "high")

    def test_dependency_change_risk_is_high(self):
        impact = self._analyze_impact(self._repository(), "Update a package dependency.")
        target = next(t for t in impact.targets if t.path == "package.json")
        self.assertEqual(target.risk, "high")


if __name__ == "__main__":
    unittest.main()