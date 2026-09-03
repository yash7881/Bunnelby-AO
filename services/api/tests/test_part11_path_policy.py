from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services.api.app.local_files.path_policy import (
    ApprovedRoot,
    PathPolicy,
    is_secret_file,
)


class Part11PathPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "Documents"
        self.root.mkdir()
        self.policy = PathPolicy((ApprovedRoot("documents", self.root),))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_regular_file_inside_approved_root_is_allowed(self) -> None:
        target = self.root / "Work" / "Resume.pdf"
        target.parent.mkdir()
        target.write_bytes(b"safe fixture")
        decision = self.policy.check_file(target)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.root_alias, "documents")
        self.assertEqual(decision.relative_path, "Work/Resume.pdf")

    def test_traversal_and_sibling_prefix_are_rejected(self) -> None:
        outside = self.root.parent / "Documents-private" / "secret.txt"
        outside.parent.mkdir()
        outside.write_text("no", encoding="utf-8")
        self.assertFalse(self.policy.check_file(self.root / ".." / outside.parent.name / outside.name).allowed)

    def test_excluded_directory_is_rejected(self) -> None:
        target = self.root / "project" / ".git" / "config"
        target.parent.mkdir(parents=True)
        target.write_text("not indexed", encoding="utf-8")
        decision = self.policy.check_file(target)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "excluded_directory")

    def test_secret_names_are_rejected_before_open(self) -> None:
        for name in (".env", ".env.production", "id_rsa", "client_secret_test.json", "vault.kdbx", "token.json"):
            with self.subTest(name=name):
                target = self.root / name
                target.write_text("must never be read", encoding="utf-8")
                decision = self.policy.check_file(target)
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.reason, "secret_file")

    def test_symlink_escape_is_rejected_when_supported(self) -> None:
        outside = Path(self.temp.name) / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("no", encoding="utf-8")
        link = self.root / "link"
        try:
            os.symlink(outside, link, target_is_directory=True)
        except (OSError, NotImplementedError):
            # Standard Windows accounts may not have SeCreateSymbolicLinkPrivilege.
            # Exercise the same branch deterministically instead of hiding it
            # behind a skipped test.
            link.mkdir()
            local = link / "secret.txt"
            local.write_text("no", encoding="utf-8")
            from services.api.app.local_files import path_policy

            real_check = path_policy._is_reparse_point
            with patch.object(
                path_policy,
                "_is_reparse_point",
                side_effect=lambda path: path == link or real_check(path),
            ):
                decision = self.policy.check_file(local)
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.reason, "reparse_point")
            return
        decision = self.policy.check_file(link / "secret.txt")
        self.assertFalse(decision.allowed)
        self.assertIn(decision.reason, {"reparse_point", "outside_approved_roots"})

    def test_unknown_alias_and_arbitrary_path_are_not_trusted(self) -> None:
        self.assertIsNone(self.policy.root("c_drive"))
        with self.assertRaises(ValueError):
            ApprovedRoot("documents/../../windows", self.root)

    def test_secret_matcher_is_case_insensitive(self) -> None:
        self.assertTrue(is_secret_file(Path("Credentials.PROD.JSON")))
        self.assertTrue(is_secret_file(Path("PRIVATE.KEY")))
        self.assertFalse(is_secret_file(Path("credentials-guide.md")))


if __name__ == "__main__":
    unittest.main()
