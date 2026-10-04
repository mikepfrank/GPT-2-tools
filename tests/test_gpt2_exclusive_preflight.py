"""Model-free startup guard checks; no real systemd, SSH, or cgroup calls."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "gpt2_exclusive_preflight", Path(__file__).resolve().parents[1] / "deploy" / "gpt2-exclusive-preflight.py"
)
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


class ExclusivePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.values = {"LoadState": "loaded", "ActiveState": "inactive", "MainPID": "0", "ControlGroup": ""}

    def query(self, unit):
        self.assertEqual(unit, "gpt2-preview.service")
        return dict(self.values)

    def verify(self, query=None):
        guard.verify_other_unit("gpt2-preview.service", cgroup_root=self.root, query=query or self.query)

    def group(self, child="", content=""):
        directory = self.root / "system.slice" / "gpt2-preview.service" / child
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "cgroup.procs").write_text(content)
        return directory

    def test_deleted_stopped_cgroup_is_allowed_and_state_is_rechecked(self):
        calls = []
        def query(unit):
            calls.append(unit)
            return self.query(unit)
        self.verify(query)
        self.assertEqual(len(calls), 2)

    def test_empty_root_and_recursive_groups_are_allowed(self):
        self.group()
        self.group("child/grandchild")
        self.group("child")
        self.values["ControlGroup"] = "/system.slice/gpt2-preview.service"
        self.verify()

    def test_process_in_root_or_descendant_is_rejected(self):
        root = self.group()
        child = self.group("child")
        for group in (root, child):
            with self.subTest(group=group.name):
                (group / "cgroup.procs").write_text("123\n")
                with self.assertRaisesRegex(RuntimeError, "still has processes"):
                    self.verify()
                (group / "cgroup.procs").write_text("")

    def test_transitional_failed_missing_and_nonzero_pid_are_rejected(self):
        for name, value in (("ActiveState", "active"), ("ActiveState", "activating"),
                            ("ActiveState", "deactivating"), ("ActiveState", "failed"),
                            ("LoadState", "not-found"), ("MainPID", "123"), ("MainPID", "")):
            with self.subTest(name=name, value=value):
                original = self.values[name]
                self.values[name] = value
                with self.assertRaisesRegex(RuntimeError, "fully inactive"):
                    self.verify()
                self.values[name] = original

    def test_invalid_paths_are_rejected(self):
        for value in ("relative", "/", "/system.slice/../other", "/system.slice/./other", "/system.slice//other"):
            with self.subTest(value=value):
                self.values["ControlGroup"] = value
                with self.assertRaisesRegex(RuntimeError, "path is invalid"):
                    self.verify()

    def test_missing_or_unreadable_process_file_is_rejected(self):
        directory = self.group()
        (directory / "cgroup.procs").unlink()
        with self.assertRaises(FileNotFoundError):
            self.verify()
        with patch.object(Path, "read_text", side_effect=PermissionError("fixture unreadable")):
            with self.assertRaises(PermissionError):
                self.verify()

    def test_walk_errors_are_not_silently_skipped(self):
        self.group()
        def bad_walk(directory, *, onerror, followlinks):
            self.assertFalse(followlinks)
            onerror(PermissionError("fixture unreadable descendant"))
        with patch.object(guard.os, "walk", bad_walk):
            with self.assertRaises(PermissionError):
                self.verify()

    def test_descendant_symlink_is_rejected_instead_of_skipped(self):
        self.group()
        self.group("child")
        with patch.object(Path, "is_symlink", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "cgroup symlink"):
                self.verify()

    def test_state_change_after_recursive_check_is_rejected(self):
        states = [dict(self.values), {**self.values, "ActiveState": "activating"}]
        with self.assertRaisesRegex(RuntimeError, "fully inactive"):
            self.verify(lambda unit: states.pop(0))
        states = [dict(self.values), {**self.values, "ControlGroup": "/system.slice/gpt2-preview.service"}]
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            self.verify(lambda unit: states.pop(0))

    def test_property_query_is_bounded_and_requires_complete_output(self):
        output = "LoadState=loaded\nActiveState=inactive\nMainPID=0\nControlGroup=\n"
        with patch.object(guard.subprocess, "check_output", return_value=output) as query:
            self.assertEqual(guard.unit_properties("gpt2-preview.service"), self.values)
            self.assertEqual(query.call_args.kwargs["timeout"], 10)
        with patch.object(guard.subprocess, "check_output", return_value="ActiveState=inactive\n"):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                guard.unit_properties("gpt2-preview.service")

    def test_main_denies_query_errors_without_starting_any_service(self):
        with patch.object(guard, "verify_other_unit", side_effect=subprocess.TimeoutExpired("systemctl", 10)):
            with patch.object(guard.sys, "stderr") as output:
                self.assertEqual(guard.main(["gpt2-preview.service"]), 1)
                self.assertTrue(output.write.called)


if __name__ == "__main__":
    unittest.main()
