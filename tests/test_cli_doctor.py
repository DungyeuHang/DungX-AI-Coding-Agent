import io
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from local_agent.cli import main as cli_main
from local_agent.models import ProviderConfig
from local_agent.storage import JsonFileStorage


class TestDoctorCommand(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.storage = JsonFileStorage(self.root / ".agent_data")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.root)

    def test_doctor_command_healthy_environment(self):
        # Arrange
        provider_configs = [ProviderConfig(provider_id="mock", priority=10, enabled=True)]
        self.storage.save_provider_configs(provider_configs)
        with mock.patch("local_agent.git.GitIntegration") as MockGit:
            mock_git_instance = MockGit.return_value
            mock_git_instance.is_repository.return_value = True
            mock_git_instance.branch.return_value = "main"
            mock_git_instance.status.return_value = "## main...origin/main\nnothing to commit, working tree clean"

            # Act
            with mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
                cli_main(["doctor", "--project", str(self.root)])
                output = fake_out.getvalue()

        # Assert
        self.assertIn("DungX AI Coding Agent Health Check", output)
        self.assertIn("Runtime Environment", output)
        self.assertIn("Project and Storage", output)
        self.assertIn("(Writable)", output)
        self.assertIn("Git Repository", output)
        self.assertIn("Working Tree: Clean", output)
        self.assertIn("Provider Status", output)
        self.assertIn("Configured Providers: 1", output)

    def test_doctor_command_no_providers_configured(self):
        # Arrange: No providers are saved to storage

        # Act
        with mock.patch("sys.stdout", new=io.StringIO()) as fake_out:
            with mock.patch("local_agent.git.GitIntegration") as MockGit:
                MockGit.return_value.is_repository.return_value = False
                cli_main(["doctor", "--project", str(self.root)])
                output = fake_out.getvalue()

        # Assert
        self.assertIn("Not a Git repository", output)
        self.assertIn("Configured Providers: 0", output)