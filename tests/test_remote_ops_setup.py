import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class RemoteOpsSetupTests(unittest.TestCase):
    def test_gitignore_ignores_remote_session_file(self):
        content = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".claude/remote_session.json", content)

    def test_remote_ops_script_exists_and_supports_core_commands(self):
        path = ROOT / "scripts" / "remote_ops.py"
        self.assertTrue(path.exists(), f"Missing script: {path}")
        content = path.read_text(encoding="utf-8")
        self.assertIn("save-session", content)
        self.assertIn("show-session", content)
        self.assertIn("download", content)
        self.assertIn("upload", content)
        self.assertIn("exec", content)
        self.assertIn("paramiko", content)

    def test_project_init_mentions_remote_ops(self):
        content = (ROOT / ".claude" / "PROJECT_INIT.md").read_text(encoding="utf-8")
        self.assertIn("remote_ops.py", content)
        self.assertIn("remote_session.json", content)


if __name__ == "__main__":
    unittest.main()
