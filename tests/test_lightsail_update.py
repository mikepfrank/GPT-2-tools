"""Model-free production-update checks; no SSH, systemd, or model execution."""
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("lightsail_update_helper", ROOT / "scripts/lightsail-update.py")
update = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(update)
promote, preview = update.promote, update.preview
FIXTURE_SPEC = importlib.util.spec_from_file_location("update_promotion_fixtures", Path(__file__).with_name("test_lightsail_promote.py"))
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(fixtures)
# Reuse fixture behavior against the update helper's imported modules, so all
# patched deployment paths refer to the same synthetic installation.
fixtures.promote, fixtures.preview = promote, preview
preview.CHECKOUT = PurePosixPath("/home/gpt-2/GPT-2-streaming-preview")
promote.ORIGINAL_CHECKOUT = PurePosixPath("/home/gpt-2/GPT-2-tools")
SOURCE = update.NEW_SOURCE
FixtureHost = fixtures.FakeHost


class BaselineHost(FixtureHost):
    override_install = promote.Host.override_install
    override_remove = promote.Host.override_remove
    start = preview.Host.start
    stop = preview.Host.stop
    streaming_ready = promote.Host.streaming_ready

    def __init__(self, *args):
        super().__init__(*args)
        self.source = update.OLD_SOURCE


fixtures.FakeHost = BaselineHost


class FakeHost(FixtureHost):
    override_remove = promote.Host.override_remove

    def __init__(self, original):
        self.__dict__.update(original.__dict__)
        self.events = []
        self.clock, self.sleep = lambda: self.seconds, self.advance
        self.loaded_dropin = self.dropin
        self.release_source = SOURCE
        self.release_files = {"src/gpt2_local/chat.py": promote.digest(b"reviewed synthetic release\n"),
                              "src/gpt2_local/public_chat.py": promote.digest(b"synthetic public backend\n")}
        self.next_pid = 12

    def inventory(self):
        current = super().inventory()
        if self.configured:
            start, pre = promote.dropin_exec(self.loaded_dropin)
            working = next(line.removeprefix("WorkingDirectory=") for line in self.loaded_dropin.decode().splitlines()
                           if line.startswith("WorkingDirectory="))
            current["production_exec"], current["production_pre"] = start, pre
            current["properties"][promote.PROD] = self.base_properties[promote.PROD].replace(
                str(promote.ORIGINAL_CHECKOUT), working,
            )
        return current

    def release_inventory(self, path, source):
        promote.require(self.release_source == source, "Release source differs")
        return {"source": self.release_source, "files": dict(self.release_files)}

    def show(self, unit, name):
        if name == "DropInPaths":
            return str(self.override_path) if self.configured and unit == promote.PROD else ""
        return super().show(unit, name)

    def run(self, args):
        output = super().run(args)
        if args == ["systemctl", "daemon-reload"]:
            self.loaded_dropin = self.override_path.read_bytes() if self.override_path.exists() else b""
        elif args[:2] == ["systemctl", "start"]:
            self.pids[args[2]] = self.next_pid
            self.next_pid += 1
        return output

    def override_replace(self, data, expected_hash):
        return update.Host.override_replace(self, data, expected_hash)

    def start(self, service):
        return preview.Host.start(self, service)

    def stop(self, service):
        return preview.Host.stop(self, service)

    def streaming_ready(self, service, expected):
        return promote.Host.streaming_ready(self, service, expected)


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.PromotionTests(methodName="runTest")
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.controller.promote()
        self.root, self.config, self.override = self.fixture.root, self.fixture.config, self.fixture.override
        self.promotion_stage = self.fixture.stage
        self.promotion_manifest = (self.promotion_stage / "manifest.json").read_bytes()
        self.old_dropin = self.override.read_bytes()
        self.host = FakeHost(self.fixture.host)
        self.release = PurePosixPath("/home/gpt-2/GPT-2-releases/seed-78ebbdc")
        self.stage = self.root / "update-stage"
        for item in (patch.object(update, "CONFIG", self.config), patch.object(update, "OVERRIDE", self.override),
                     patch.object(update, "RELEASE", self.release),
                     patch.object(update, "OLD_OVERRIDE_SHA", promote.digest(self.old_dropin))):
            item.start()
            self.addCleanup(item.stop)
        if os.name == "nt":
            def deployment_path(value):
                # Release paths are systemd POSIX arguments, rather than paths
                # on this Windows test machine. Fixture stage paths stay native.
                path = Path(value)
                return PurePosixPath(path.as_posix()) if path.as_posix().startswith("/home/") else path
            item = patch.object(update, "Path", deployment_path)
            item.start()
            self.addCleanup(item.stop)
            native_open = os.open
            def root_write_access(path, flags, *args, **kwargs):
                # POSIX root can write a root-owned 0440 file without changing
                # its mode. Emulate that permission on NTFS for existing files.
                own = Path(path) == self.override and flags & os.O_WRONLY and self.override.exists()
                if own:
                    self.override.chmod(0o600)
                try:
                    return native_open(path, flags, *args, **kwargs)
                finally:
                    if own:
                        self.override.chmod(0o440)
            item = patch.object(update.os, "open", root_write_access)
            item.start()
            self.addCleanup(item.stop)
        self.controller = update.Controller(self.host, self.stage, self.promotion_stage, self.release, SOURCE)
        self.controller.emit = lambda phase, **details: self.host.events.append((phase, details, self.host.seconds))
        self.prepare()

    def prepare(self):
        with patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_gid=0))}):
            self.controller.prepare()

    def commands(self, verb, unit=None):
        prefix = ["systemctl", verb] + ([unit] if unit else [])
        return [event for event in self.host.events if event[0] == "command" and event[1][:len(prefix)] == prefix]

    def assert_no_preview_control(self):
        self.assertFalse(self.commands("start", promote.PREVIEW))
        self.assertFalse(self.commands("stop", promote.PREVIEW))
        self.assertEqual(self.host.pids[promote.PREVIEW], 0)

    def test_prepare_stages_only_and_preserves_old_polling_rollback_evidence(self):
        self.assertFalse(self.commands("start") or self.commands("stop") or self.commands("daemon-reload"))
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual(self.config.read_bytes(), fixtures.ORIGINAL + promote.redirect_site())
        self.assertEqual((self.promotion_stage / "manifest.json").read_bytes(), self.promotion_manifest)
        self.assertEqual(self.controller.manifest["schema_version"], 3)
        self.assertEqual(self.controller.manifest["phase"], "prepared")

    def test_update_and_rollback_each_retain_results_and_only_restart_production(self):
        began = self.host.seconds
        self.host.connections = 2
        self.controller.update()
        self.assertGreaterEqual(self.commands("stop", promote.PROD)[0][2] - began, 604)
        self.assertNotEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual(self.config.read_bytes(), fixtures.ORIGINAL + promote.redirect_site())
        began = self.host.seconds
        self.controller.rollback()
        self.assertGreaterEqual(self.commands("stop", promote.PROD)[-1][2] - began, 600)
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual((self.promotion_stage / "manifest.json").read_bytes(), self.promotion_manifest)
        self.assert_no_preview_control()

    def test_old_polling_rollback_still_works_after_update_rollback(self):
        self.controller.update()
        self.controller.rollback()
        self.assertEqual((self.promotion_stage / "manifest.json").read_bytes(), self.promotion_manifest)
        old = promote.Controller(self.host, self.promotion_stage)
        old.emit = lambda phase, **details: self.host.events.append((phase, details, self.host.seconds))
        old.rollback()
        self.assertFalse(self.override.exists())
        self.assertFalse(self.host.configured)
        self.assertEqual(self.host.health(promote.PROD)["capabilities"]["streaming"], False)
        self.assertEqual(self.config.read_bytes(), fixtures.ORIGINAL + promote.redirect_site())

    def test_failure_after_exposure_retains_new_results_before_restoring_old_build(self):
        began = self.host.seconds
        self.host.fail_redirect = True
        with self.assertRaises(promote.PreviewError):
            self.controller.update()
        stopped = self.commands("stop", promote.PROD)
        self.assertGreaterEqual(stopped[0][2] - began, 600)
        self.assertGreaterEqual(stopped[-1][2] - self.commands("start", promote.PROD)[0][2], 600)
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual(self.host.model_pids(), {self.host.pid(promote.PROD)})
        self.assertEqual(self.config.read_bytes(), fixtures.ORIGINAL + promote.redirect_site())
        self.assert_no_preview_control()

    def test_undrainable_jobs_are_never_stopped_or_replaced(self):
        self.host.block_prod = True
        with self.assertRaises(promote.PreviewError):
            self.controller.update()
        self.assertFalse(self.commands("stop", promote.PROD))
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.host.assert_gate(preview.PROD_ORIGIN)
        self.assert_no_preview_control()

    def test_pid_zero_pending_restart_is_cancelled_before_old_build_recovers(self):
        self.host.fail_retry_ready = True
        with self.assertRaises(promote.PreviewError):
            self.controller.update()
        self.assertGreaterEqual(len(self.commands("stop", promote.PROD)), 2)
        self.assertEqual(self.host.active[promote.PROD], "active")
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual(self.host.model_pids(), {self.host.pid(promote.PROD)})
        self.assert_no_preview_control()

    def test_extra_model_is_rejected_before_any_live_write_or_service_control(self):
        self.host.extra_model = True
        began = len(self.host.events)
        with self.assertRaises(promote.PreviewError):
            self.controller.update()
        self.assertFalse(any(event[0] in ("config", "command") for event in self.host.events[began:]))
        self.assertEqual(self.override.read_bytes(), self.old_dropin)

    def test_memory_guard_prevents_any_new_model_start(self):
        self.host.memory = 7.9
        with self.assertRaises(promote.PreviewError):
            self.controller.update()
        self.assertFalse(self.commands("start"))
        self.assert_no_preview_control()

    def test_source_candidate_override_and_loaded_setting_drift_prevent_live_writes(self):
        for change in ("source", "files", "candidate", "override", "properties"):
            with self.subTest(change=change):
                old_source, old_properties = self.host.release_source, dict(self.host.base_properties)
                old_files = dict(self.host.release_files)
                candidate = next(self.stage.glob("*.caddy"))
                old_candidate = candidate.read_bytes()
                if change == "source":
                    self.host.release_source = "c" * 40
                elif change == "files":
                    self.host.release_files["src/gpt2_local/chat.py"] = "d" * 64
                elif change == "candidate":
                    candidate.chmod(0o600)
                    candidate.write_bytes(b"tampered")
                    candidate.chmod(0o440)
                elif change == "override":
                    self.override.chmod(0o600)
                    self.override.write_bytes(b"foreign override")
                    self.override.chmod(0o440)
                else:
                    self.host.base_properties[promote.PROD] += "\nMemoryMax=1"
                began = len(self.host.events)
                with self.assertRaises(promote.PreviewError):
                    self.controller.update()
                self.assertFalse(any(event[0] in ("config", "command") for event in self.host.events[began:]))
                self.host.release_source, self.host.base_properties = old_source, old_properties
                self.host.release_files = old_files
                for path, data in ((candidate, old_candidate), (self.override, self.old_dropin)):
                    path.chmod(0o600)
                    path.write_bytes(data)
                    path.chmod(0o440)

    def test_actual_override_short_write_restores_exact_prior_bytes_and_loaded_selection(self):
        self.host.pids[promote.PROD] = 0
        self.host.active[promote.PROD] = "inactive"
        inode = self.override.stat().st_ino
        native_fdopen = os.fdopen
        class ShortWriter:
            def __init__(self, stream):
                self.stream = stream
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.stream.__exit__(*args)
            def __getattr__(self, name):
                return getattr(self.stream, name)
            def write(self, data):
                return self.stream.write(data[:len(data) // 2])
        opened = 0
        def short_once(*args, **kwargs):
            nonlocal opened
            opened += 1
            stream = native_fdopen(*args, **kwargs)
            return ShortWriter(stream) if opened == 1 else stream
        with patch.object(update.os, "fdopen", short_once), self.assertRaisesRegex(promote.PreviewError, "Short override write"):
            self.host.override_replace((self.stage / "new.conf").read_bytes(), promote.digest(self.old_dropin))
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual(self.override.stat().st_ino, inode)
        self.assertEqual(self.host.loaded_dropin, self.old_dropin)
        self.assertEqual(len(self.commands("daemon-reload")), 1)

    def test_actual_failed_daemon_reload_restores_exact_prior_override(self):
        self.host.pids[promote.PROD] = 0
        self.host.active[promote.PROD] = "inactive"
        self.host.fail_daemon_reload = True
        inode = self.override.stat().st_ino
        with self.assertRaisesRegex(promote.PreviewError, "daemon reload failure"):
            self.host.override_replace((self.stage / "new.conf").read_bytes(), promote.digest(self.old_dropin))
        self.assertEqual(self.override.read_bytes(), self.old_dropin)
        self.assertEqual(self.override.stat().st_ino, inode)
        self.assertEqual(self.host.loaded_dropin, self.old_dropin)
        self.assertEqual(len(self.commands("daemon-reload")), 2)

    def test_manifest_failure_preserves_previous_complete_recovery_record(self):
        before = (self.stage / "manifest.json").read_bytes()
        def partial_dump(value, stream, **kwargs):
            stream.write('{"phase":')
            raise OSError("fixture interrupted manifest write")
        for failure in (patch.object(update.json, "dump", partial_dump),
                        patch.object(update.os, "replace", side_effect=OSError("fixture manifest rename failure"))):
            with failure, self.assertRaises(OSError):
                self.controller.record("new_selected")
            self.assertEqual((self.stage / "manifest.json").read_bytes(), before)
            self.assertEqual(json.loads(before)["phase"], "prepared")
            self.assertEqual(self.override.read_bytes(), self.old_dropin)
            self.assertFalse(self.commands("daemon-reload"))
            self.controller.manifest = json.loads(before)

    def drift_during_retention(self, change):
        began, pid = self.host.seconds, self.host.pid(promote.PROD)
        advance = self.host.advance
        changed = False
        foreign = self.old_dropin + b"\n# externally changed during result retention\n"
        def drift(seconds):
            nonlocal changed
            advance(seconds)
            if not changed and self.host.seconds - began >= 600:
                changed = True
                if change == "release":
                    self.host.release_files["src/gpt2_local/chat.py"] = "d" * 64
                else:
                    self.override.chmod(0o600)
                    self.override.write_bytes(foreign)
                    self.override.chmod(0o440)
        self.host.sleep = drift
        with self.assertRaises(promote.PreviewError):
            self.controller.update()
        self.assertTrue(changed)
        self.assertFalse(self.commands("stop") or self.commands("start") or self.commands("daemon-reload"))
        self.assertEqual(self.host.model_pids(), {pid})
        self.assertEqual(self.host.pid(promote.PROD), pid)
        self.assertEqual(self.override.read_bytes(), self.old_dropin if change == "release" else foreign)
        self.host.assert_gate(preview.PROD_ORIGIN)
        self.assertEqual((self.promotion_stage / "manifest.json").read_bytes(), self.promotion_manifest)
        self.assert_no_preview_control()

    def test_release_drift_during_retention_leaves_existing_model_gated_and_running(self):
        self.drift_during_retention("release")

    def test_override_drift_during_retention_is_not_overwritten_or_stopped(self):
        self.drift_during_retention("override")


class CandidateTests(unittest.TestCase):
    def test_canonical_gate_preserves_existing_test_redirect_and_chess_bytes(self):
        active = fixtures.ORIGINAL + promote.redirect_site()
        staged = update.candidates(active)
        self.assertEqual(staged["active"], active)
        self.assertEqual(staged["gated"], active.replace(
            b"\ngpt2-chat.com {\n", b"\ngpt2-chat.com {\n" + preview.generation_gate().encode(), 1,
        ))
        self.assertTrue(staged["gated"].endswith(promote.redirect_site()))
        self.assertEqual(staged["gated"].count(b"@gpt2_preview_generation {"), 1)


if __name__ == "__main__":
    unittest.main()
