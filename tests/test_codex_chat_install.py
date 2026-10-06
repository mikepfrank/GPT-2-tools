"""Publication checks use temporary files and fake service/network operations."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SOURCE = Path(__file__).parents[1] / "deploy/codex-chat/install.py"
spec = importlib.util.spec_from_file_location("codex_chat_installer", SOURCE)
installer = importlib.util.module_from_spec(spec)
# The operator helper runs on Linux. Unit tests also run on the Windows laptop;
# mocked lock operations never exercise the platform flock implementation.
if os.name == "nt":
    with mock.patch.dict(sys.modules, {"fcntl": types.SimpleNamespace()}):
        spec.loader.exec_module(installer)
else:
    spec.loader.exec_module(installer)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "Caddyfile"
        self.config.write_bytes(b"# Existing sites\ngpt2-chat.com { respond old }\n")
        self.original = self.config.read_bytes()
        self.mounted = self.root / "mounted-Caddyfile"
        os.link(self.config, self.mounted)
        self.path = "/" + "opaque_beta_component_123456"
        self.operations = []
        self.in_lock = False
        self.pids = dict(zip(installer.OTHERS, ["101", "102", "103", "104"]))

        @contextmanager
        def lock():
            self.assertFalse(self.in_lock)
            self.in_lock = True
            try:
                yield
            finally:
                self.in_lock = False

        def run(args):
            self.assertTrue(self.in_lock)
            self.operations.append(args)
            return ""

        self.addCleanup(mock.patch.stopall)
        mock.patch.object(installer, "CONFIG", self.config).start()
        mock.patch.object(installer, "OPS", self.root / "ops").start()
        mock.patch.object(installer, "base_path", return_value=self.path).start()
        mock.patch.object(installer, "mounted_config", return_value=self.mounted).start()
        mock.patch.object(installer, "proxy_lock", lock).start()
        mock.patch.object(installer, "other_pids", return_value=self.pids).start()
        mock.patch.object(installer, "check_health", return_value=True).start()
        mock.patch.object(installer, "check_existing_health", return_value=True).start()
        mock.patch.object(installer, "run", side_effect=run).start()
        mock.patch.object(installer.os, "chown", create=True).start()
        self.readiness = mock.patch.object(installer, "wait_for_health").start()

    def publish(self):
        with redirect_stdout(io.StringIO()) as output:
            installer.publish()
        return output.getvalue()

    def signals(self):
        return [operation for operation in self.operations if operation[:2] == ["systemctl", "kill"]]

    def test_publish_keeps_bind_mounted_inode_and_validates_before_write(self):
        inode = self.config.stat().st_ino
        original_write = installer.write_proxy_bytes
        fsync = mock.patch.object(installer.os, "fsync", wraps=os.fsync).start()

        def checked_write(data):
            self.assertTrue(self.in_lock)
            validates = [operation for operation in self.operations if "validate" in operation]
            self.assertEqual(len(validates), 2)
            self.assertEqual(self.config.read_bytes(), self.original)
            return original_write(data)

        mock.patch.object(installer, "write_proxy_bytes", side_effect=checked_write).start()
        # The hard link independently verifies the old mounted inode receives
        # the new bytes. No filesystem-wide mocking is needed for that check.
        output = self.publish()
        expected = self.original + installer.beta_block(self.path)
        self.assertEqual(self.config.stat().st_ino, inode)
        self.assertEqual(self.mounted.read_bytes(), expected)
        self.assertEqual((self.root / "ops/Caddyfile.original").read_bytes(), self.original)
        self.assertEqual(len(self.signals()), 1)
        self.assertGreaterEqual(fsync.call_count, 2)
        self.assertEqual(self.readiness.call_args.kwargs, {"include_beta": True})
        self.assertIn("Beta HTTPS ready", output)
        self.assertFalse(self.config.with_name("Caddyfile.gpt2-codex-candidate").exists())

    def test_failed_public_readiness_restores_and_verifies_original_sites(self):
        def readiness(path, *, include_beta):
            self.assertTrue(self.in_lock)
            if include_beta:
                self.assertNotEqual(self.mounted.read_bytes(), self.original)
                raise RuntimeError("New beta did not become ready")
            self.assertEqual(self.mounted.read_bytes(), self.original)

        self.readiness.side_effect = readiness
        inode = self.config.stat().st_ino
        with self.assertRaisesRegex(RuntimeError, "New beta did not become ready"):
            self.publish()
        self.assertEqual(self.config.stat().st_ino, inode)
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(self.mounted.read_bytes(), self.original)
        self.assertEqual(len(self.signals()), 2)
        self.assertEqual([call.kwargs["include_beta"] for call in self.readiness.call_args_list], [True, False])
        self.assertEqual(len([operation for operation in self.operations if "validate" in operation]), 3)

    def test_failed_rollback_readiness_reports_unverified_rollback(self):
        self.readiness.side_effect = RuntimeError("HTTPS unavailable")
        with self.assertRaisesRegex(RuntimeError, "rollback could not be verified"):
            self.publish()
        self.assertEqual(self.mounted.read_bytes(), self.original)
        self.assertEqual(len(self.signals()), 2)

    def test_repeat_publish_still_checks_beta_and_original_sites(self):
        self.config.write_bytes(self.original + installer.beta_block(self.path))
        expected = self.config.read_bytes()
        output = self.publish()
        self.readiness.assert_called_once_with(self.path, include_beta=True)
        self.assertEqual(self.config.read_bytes(), expected)
        self.assertEqual(len(self.signals()), 0)
        self.assertIn("Beta HTTPS ready", output)
        self.readiness.side_effect = RuntimeError("Beta is unavailable")
        with self.assertRaisesRegex(RuntimeError, "Beta is unavailable"):
            self.publish()

    def test_foreign_or_changed_existing_beta_block_is_never_overwritten(self):
        self.config.write_bytes(self.original + installer.beta_block("/different_opaque_beta_path"))
        before = self.config.read_bytes()
        with self.assertRaisesRegex(RuntimeError, "existing beta proxy block differs"):
            self.publish()
        self.assertEqual(self.config.read_bytes(), before)
        self.assertEqual(self.operations, [])

    def test_proxy_mount_drift_and_unhealthy_existing_sites_stop_before_mutation(self):
        self.mounted.unlink()
        self.mounted.write_bytes(b"stale mounted configuration")
        with self.assertRaisesRegex(RuntimeError, "mounted proxy configuration bytes disagree"):
            self.publish()
        self.assertEqual(self.config.read_bytes(), self.original)
        self.assertEqual(self.operations, [])
        self.mounted.write_bytes(self.original)
        installer.check_existing_health.return_value = False
        with self.assertRaisesRegex(RuntimeError, "existing HTTPS service is not ready"):
            self.publish()
        self.assertEqual(self.operations, [])


class ReadinessTests(unittest.TestCase):
    def test_poller_retries_until_both_beta_and_existing_sites_are_ready(self):
        with mock.patch.object(installer, "check_existing_health", side_effect=[True, False, True]) as existing, \
                mock.patch.object(installer, "check_health", side_effect=[False, True, True]) as beta, \
                mock.patch.object(installer.time, "sleep") as sleep:
            installer.wait_for_health("/private_component", include_beta=True)
        self.assertEqual(existing.call_count, 3)
        self.assertEqual(beta.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_rollback_poller_checks_only_original_sites_and_never_beta(self):
        with mock.patch.object(installer, "check_existing_health", side_effect=[False, True]) as existing, \
                mock.patch.object(installer, "check_health") as beta, \
                mock.patch.object(installer.time, "sleep"):
            installer.wait_for_health("/private_component", include_beta=False)
        self.assertEqual(existing.call_count, 2)
        beta.assert_not_called()


if __name__ == "__main__":
    unittest.main()
