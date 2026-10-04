import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tempfile
from contextlib import contextmanager
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("lightsail_promote_helper", Path(__file__).resolve().parents[1] / "scripts/lightsail-promote.py")
promote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote)
preview = promote.preview
# Target deployment paths have POSIX spelling even in Windows model-free tests.
preview.CHECKOUT = PurePosixPath("/home/gpt-2/GPT-2-streaming-preview")
promote.ORIGINAL_CHECKOUT = PurePosixPath("/home/gpt-2/GPT-2-tools")
ORIGINAL = (b"{\n admin off\n}\n\nastraplayschess.com {\n encode gzip\n respond ok\n}\n"
            b"\narcturuschess.com {\n respond ok\n}\n\ngpt2-chat.com {\n    rewrite * ?\n"
            b"    encode gzip\n    reverse_proxy 127.0.0.1:8899 {\n        header_up Host {http.request.hostport}\n    }\n}\n")
MODEL = {"id": "openai-community/gpt2-xl", "revision": "15ea56dee5df4983c59b2538573817e1667135e2",
         "context_tokens": 1024, "precision": "FP32", "device": "cpu"}


class FakeHost(promote.Host):
    def __init__(self, config, installed, override, dropin):
        super().__init__(drain_timeout=8)
        self.config, self.installed, self.override_path = config, installed, override
        self.seconds, self.events = 0, []
        self.clock, self.sleep = lambda: self.seconds, self.advance
        self.pids = {promote.PROD: 0, promote.PREVIEW: 20}
        self.active = {promote.PROD: "inactive", promote.PREVIEW: "active"}
        self.configured = False
        self.dropin = dropin
        self.base_exec = "/original/python -m gpt2_local.public_chat --port 8899"
        self.source = "5389a30" + "a" * 33
        self.original_source = promote.ORIGINAL_SOURCE
        self.extra_model = self.block_preview = self.block_prod = False
        self.fail_prod_ready = self.fail_redirect = self.fail_daemon_reload = False
        self.fail_retry_ready = self.bad_tls_capability = False
        self.queue_sequences = {promote.PROD: [], promote.PREVIEW: []}
        self.connections = 0
        self.memory = 12
        self.leftover = False
        self.base_properties = {promote.PROD: "User=gpt-2\nWorkingDirectory=" + str(promote.ORIGINAL_CHECKOUT) + "\nEnvironment=OMP_NUM_THREADS=2\nMemoryMax=9663676416",
                                promote.PREVIEW: "User=gpt-2\nWorkingDirectory=preview\n",
                                **{unit: "untouched" for unit in (preview.CADDY, *preview.CHESS)}}

    def advance(self, seconds):
        self.seconds += seconds

    @contextmanager
    def lock(self):
        self.events.append(("lock", self.seconds))
        yield
        self.events.append(("unlock", self.seconds))

    def inventory(self):
        files = {str(path): promote.digest(path.read_bytes()) for path in self.installed}
        if self.override_path.exists():
            files[str(self.override_path)] = promote.digest(self.override_path.read_bytes())
        properties = dict(self.base_properties)
        if self.configured:
            properties[promote.PROD] = properties[promote.PROD].replace(str(promote.ORIGINAL_CHECKOUT), str(preview.CHECKOUT))
        start, pre = promote.dropin_exec(self.dropin)
        return {"files": files, "properties": properties, "source": self.source, "original_source": self.original_source,
                "caddy_pid": 30, "inode": self.config.stat().st_ino, "config_attributes": [0, 0, 0o100600],
                "chess_pids": {preview.CHESS[0]: 40, preview.CHESS[1]: 50}, "production_pid": self.pids[promote.PROD],
                "preview_pid": self.pids[promote.PREVIEW], "config_sha256": promote.digest(self.config.read_bytes()),
                "production_exec": start if self.configured else self.base_exec, "production_pre": pre if self.configured else ""}

    def pid(self, unit):
        return self.pids[unit]

    def show(self, unit, name):
        if name == "ActiveState":
            return self.active[unit]
        if name == "WorkingDirectory":
            return str(preview.CHECKOUT if self.configured else promote.ORIGINAL_CHECKOUT)
        if name == "FragmentPath":
            return str(self.installed[0])
        if name == "DropInPaths":
            return ""
        raise AssertionError(name)

    def cgroup_empty(self, unit):
        return not self.pids[unit] and not self.leftover

    def model_pids(self):
        return {pid for pid in self.pids.values() if pid} | ({999} if self.extra_model else set())

    def available_gib(self):
        return self.memory

    def run(self, args):
        self.events.append(("command", args, self.seconds))
        if args[:2] == ["systemctl", "stop"]:
            self.pids[args[2]] = 0
            self.active[args[2]] = "inactive"
        elif args[:2] == ["systemctl", "start"]:
            if any(self.pids.values()):
                raise AssertionError("two models would overlap")
            self.pids[args[2]] = 11 if args[2] == promote.PROD else 21
            self.active[args[2]] = "active"
        elif args == ["systemctl", "daemon-reload"]:
            self.configured = self.override_path.exists()
            if self.fail_daemon_reload:
                self.fail_daemon_reload = False
                raise promote.PreviewError("fixture daemon reload failure")
        elif args[0] == "cp":
            Path(args[-1]).write_bytes(Path(args[-2]).read_bytes())
        else:
            raise AssertionError(args)
        return ""

    def validate(self, path):
        self.events.append(("validate", path.name))

    def unit_check(self, path):
        self.events.append(("unit_validate", path.name))

    def write_config(self, data, snapshot):
        self.config.write_bytes(data)
        self.events.append(("config", promote.digest(data), self.seconds))

    def reload(self):
        self.events.append(("reload", self.seconds))

    def chess_health(self):
        pass

    def health(self, service):
        sequence = self.queue_sequences[service]
        blocked = self.block_prod if service == promote.PROD else self.block_preview
        queue = sequence.pop(0) if sequence else {"waiting": 1 if blocked else 0, "running": blocked}
        return {"status": "ready", "model": MODEL, "limits": {"result_ttl_seconds": 600, "max_reply_tokens": 256},
                "queue": queue, "capabilities": {"streaming": service == promote.PREVIEW or self.configured}}

    def request(self, origin, path, *, port=None, body=None):
        service = promote.PROD if origin == preview.PROD_ORIGIN else promote.PREVIEW
        state = self.health(service)
        if self.bad_tls_capability:
            state["capabilities"]["streaming"] = False
        return 200, json.dumps(state)

    def ready(self, service):
        if service == promote.PROD and self.fail_retry_ready:
            self.fail_retry_ready = False
            self.pids[service] = 0
            self.active[service] = "activating"
            raise promote.PreviewError("fixture scheduled restart")
        if service == promote.PROD and self.fail_prod_ready:
            self.fail_prod_ready = False
            raise promote.PreviewError("fixture readiness failure")
        return self.health(service)

    def settled_connections(self, service):
        if self.connections:
            self.connections -= 1
            return False
        return True

    def gate_check(self, origin):
        self.events.append(("gate", origin, self.seconds))
        self.assert_gate(origin)

    def assert_gate(self, origin):
        text = self.config.read_text()
        header = origin.removeprefix("https://") + " {\n"
        body = text.split(header, 1)[1].split("\n}\n", 1)[0]
        if "method POST" not in body or "accepted\\\":false" not in body:
            raise AssertionError("generation gate missing")

    def redirect_check(self):
        if self.fail_redirect:
            self.fail_redirect = False
            raise promote.PreviewError("fixture exposed-production check failure")
        self.events.append(("redirect_verified", self.seconds))


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "Caddyfile"
        self.config.write_bytes(preview.candidates(ORIGINAL)["preview-active"])
        self.override = self.root / "systemd/90-gpt2-streaming.conf"
        self.guard = self.root / "generic-guard.py"
        self.guard.write_bytes(promote.GUARD_TEMPLATE.read_bytes())
        self.template = self.root / "dropin.example"
        data = promote.DROPIN_TEMPLATE.read_bytes().replace(promote.GUARD.as_posix().encode(), str(self.guard).encode())
        self.template.write_bytes(data)
        self.base_unit = self.root / "gpt2-public.service"
        self.base_unit.write_bytes(b"[Service]\nUser=gpt-2\nExecStart=/original/python\n")
        self.env = self.root / "original.env"
        self.env.write_bytes(b"untouched\n")
        self.host = FakeHost(self.config, [self.base_unit, self.env, self.guard], self.override, data)
        self.old_stage = self.root / "old-stage"
        self.old_stage.mkdir()
        old_snapshot = self.host.inventory()
        old_snapshot["files"].pop(str(self.guard))
        old_snapshot["config_sha256"] = promote.digest(ORIGINAL)
        previous = {"schema_version": 1, "phase": "active", "snapshot": old_snapshot,
                    "candidates": {name: promote.digest(value) for name, value in preview.candidates(ORIGINAL).items()}}
        (self.old_stage / "manifest.json").write_text(json.dumps(previous))
        (self.old_stage / "manifest.json").chmod(0o600)
        (self.old_stage / "original.caddy").write_bytes(ORIGINAL)
        (self.old_stage / "original.caddy").chmod(0o440)
        self.stage = self.root / "new-stage"
        self.patches = [patch.object(promote, "CONFIG", self.config), patch.object(preview, "CONFIG", self.config),
                        patch.object(promote, "OVERRIDE", self.override), patch.object(promote, "GUARD", self.guard),
                        patch.object(promote, "DROPIN_TEMPLATE", self.template), patch.object(promote.os, "chown", create=True)]
        # NTFS read-only deletion differs from POSIX unlink permissions. Model
        # that one operation explicitly while retaining all helper checks.
        unlink = Path.unlink
        if os.name == "nt":
            def posix_unlink(path, *args, **kwargs):
                if path.exists() and not path.is_symlink():
                    path.chmod(0o600)
                return unlink(path, *args, **kwargs)
            self.patches.append(patch.object(Path, "unlink", posix_unlink))
            # Windows chmod/stat cannot represent a POSIX root-private 0600
            # manifest. Linux runs exercise the real ownership/mode predicate.
            self.patches.append(patch.object(promote, "private_root_file",
                                            lambda path: promote.require(path.is_file() and not path.is_symlink(), "Invalid fixture manifest")))
            native_stat = Path.stat
            def protected_directory_metadata(path, *args, **kwargs):
                result = native_stat(path, *args, **kwargs)
                if path == self.override.parent:
                    return SimpleNamespace(st_uid=0, st_gid=result.st_gid, st_ino=result.st_ino,
                                           st_mode=result.st_mode & ~0o022)
                return result
            self.patches.append(patch.object(Path, "stat", protected_directory_metadata))
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.controller = promote.Controller(self.host, self.stage, self.old_stage)
        self.controller.emit = lambda phase, **details: self.host.events.append((phase, details, self.host.seconds))
        self.prepare()

    def prepare(self):
        with patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_gid=0))}):
            self.controller.prepare()

    def commands(self, verb, unit=None):
        prefix = ["systemctl", verb] + ([unit] if unit else [])
        return [event for event in self.host.events if event[0] == "command" and event[1][:len(prefix)] == prefix]

    def test_candidates_preserve_exact_original_public_bytes_and_redirect(self):
        data = promote.candidates(ORIGINAL)
        self.assertEqual(data["production-active"], ORIGINAL + promote.redirect_site())
        self.assertEqual(data["rollback-active"], data["production-active"])
        prefix = ORIGINAL.split(b"gpt2-chat.com {", 1)[0]
        self.assertTrue(all(value.startswith(prefix) for value in data.values()))
        self.assertIn(b"redir https://gpt2-chat.com{uri} 307", data["production-active"])
        self.assertIn(b"header Cache-Control no-store", data["production-active"])
        self.assertIn(b"    encode gzip\n", data["production-active"])
        self.assertIn(b"    rewrite * ?\n", data["production-active"])
        self.assertIn(b"method POST\n        path /api/message /api/regenerate", data["production-gated"])

    def test_prepare_only_stages_immutable_candidates_and_preservation_snapshots(self):
        self.assertFalse(self.commands("start") or self.commands("stop") or self.commands("daemon-reload"))
        self.assertEqual(self.config.read_bytes(), preview.candidates(ORIGINAL)["preview-active"])
        self.assertFalse(self.override.exists())
        self.assertEqual(len(list(self.stage.glob("*.caddy"))), 7)
        self.assertEqual(self.controller.manifest["phase"], "prepared")
        self.assertEqual(set(self.controller.manifest["snapshots"]), {str(file) for file in self.host.installed})

    def test_prepare_rejects_new_inherited_dropin_before_staging_or_live_commands(self):
        extra = self.root / "foreign-dropin.conf"
        extra.write_bytes(b"[Service]\nNice=0\n")
        self.host.installed.append(extra)
        controller = promote.Controller(self.host, self.root / "rejected-stage", self.old_stage)
        previous = len(self.host.events)
        with patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_gid=0))}), \
                self.assertRaisesRegex(promote.PreviewError, "Files/drop-ins changed"):
            controller.prepare()
        self.assertFalse(controller.stage.exists())
        self.assertFalse(any(event[0] == "command" for event in self.host.events[previous:]))

    def test_prepare_rejects_loaded_setting_drift_before_staging_or_live_commands(self):
        self.host.base_properties[promote.PROD] += "\nMemoryHigh=1"
        controller = promote.Controller(self.host, self.root / "rejected-stage", self.old_stage)
        previous = len(self.host.events)
        with patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_gid=0))}), \
                self.assertRaisesRegex(promote.PreviewError, "Loaded settings changed"):
            controller.prepare()
        self.assertFalse(controller.stage.exists())
        self.assertFalse(any(event[0] == "command" for event in self.host.events[previous:]))

    def test_promotion_retains_preview_results_and_uses_only_one_model(self):
        self.host.connections = 2
        self.controller.promote()
        self.assertGreaterEqual(self.commands("stop", promote.PREVIEW)[0][2], 604)
        self.assertEqual(self.host.pids, {promote.PROD: 11, promote.PREVIEW: 0})
        self.assertEqual(self.controller.manifest["phase"], "promoted")
        self.assertEqual(self.override.read_bytes(), self.template.read_bytes())
        self.assertEqual(self.config.read_bytes(), ORIGINAL + promote.redirect_site())
        self.assertEqual(self.host.events[-1][0], "unlock")

    def test_rollback_retains_production_results_removes_only_own_override(self):
        self.controller.promote()
        began = self.host.seconds
        self.controller.rollback()
        self.assertGreaterEqual(self.commands("stop", promote.PROD)[0][2] - began, 600)
        self.assertFalse(self.override.exists())
        self.assertFalse(self.host.configured)
        self.assertEqual(self.controller.manifest["phase"], "rolled_back")
        self.assertEqual(self.config.read_bytes(), ORIGINAL + promote.redirect_site())
        self.assertEqual(self.base_unit.read_bytes(), b"[Service]\nUser=gpt-2\nExecStart=/original/python\n")
        self.assertTrue(self.guard.exists())

    def test_failure_after_production_exposure_preserves_jobs_before_preview_recovery(self):
        self.host.fail_redirect = True
        with self.assertRaisesRegex(promote.PreviewError, "exposed-production"):
            self.controller.promote()
        self.assertGreaterEqual(self.commands("stop", promote.PROD)[0][2], 1200)
        self.assertEqual(self.host.pids, {promote.PROD: 0, promote.PREVIEW: 21})
        self.assertFalse(self.override.exists())
        self.assertEqual(self.config.read_bytes(), preview.candidates(ORIGINAL)["preview-active"])
        self.assertEqual(self.controller.manifest["phase"], "preview_recovered")

    def test_pid_zero_scheduled_restart_is_stopped_before_preview_recovers(self):
        self.host.fail_retry_ready = True
        with self.assertRaisesRegex(promote.PreviewError, "scheduled restart"):
            self.controller.promote()
        self.assertTrue(self.commands("stop", promote.PROD))
        self.assertEqual(self.host.active[promote.PROD], "inactive")
        self.assertEqual(self.host.model_pids(), {21})

    def test_undrainable_preview_is_not_stopped(self):
        self.host.block_preview = True
        with self.assertRaisesRegex(promote.PreviewError, "did not drain"):
            self.controller.promote()
        self.assertFalse(self.commands("stop", promote.PREVIEW))
        self.assertEqual(self.host.pids[promote.PREVIEW], 20)
        self.assertEqual(self.config.read_bytes(), preview.candidates(ORIGINAL)["preview-active"])

    def test_undrainable_production_is_left_gated_in_recovery(self):
        self.host.fail_redirect = self.host.block_prod = True
        with self.assertRaisesRegex(promote.PreviewError, "Cannot safely recover"):
            self.controller.promote()
        self.assertFalse(self.commands("stop", promote.PROD))
        self.assertEqual(self.host.pids, {promote.PROD: 11, promote.PREVIEW: 0})
        self.assertEqual(self.config.read_bytes(), promote.candidates(ORIGINAL)["production-gated"])

    def test_tls_streaming_capability_is_required_before_opening_production(self):
        self.host.bad_tls_capability = True
        with self.assertRaises(promote.PreviewError):
            self.controller.promote()
        self.assertFalse(any(event[0] == "config" and event[1] == promote.digest(ORIGINAL + promote.redirect_site()) for event in self.host.events))

    def test_source_guard_or_candidate_drift_prevents_any_live_transition(self):
        for change in ("source", "original_source", "guard", "candidate"):
            with self.subTest(change=change):
                self.host.source = self.controller.manifest["snapshot"]["source"]
                self.host.original_source = promote.ORIGINAL_SOURCE
                if change == "source":
                    self.host.source += "changed"
                elif change == "original_source":
                    self.host.original_source += "changed"
                elif change == "guard":
                    self.guard.write_bytes(b"changed")
                else:
                    path = self.stage / "production-active.caddy"
                    path.chmod(0o600)
                    path.write_bytes(b"changed")
                previous = len(self.host.events)
                with self.assertRaises(promote.PreviewError):
                    self.controller.promote()
                self.assertFalse(any(event[0] == "command" for event in self.host.events[previous:]))
                self.guard.write_bytes(promote.GUARD_TEMPLATE.read_bytes())

    def test_foreign_override_is_never_removed(self):
        self.override.parent.mkdir()
        self.override.write_bytes(b"foreign")
        self.override.chmod(0o440)
        with self.assertRaisesRegex(promote.PreviewError, "foreign or changed"):
            self.host.override_remove(self.controller.manifest["dropin_sha256"])
        self.assertEqual(self.override.read_bytes(), b"foreign")
        self.assertFalse(self.commands("daemon-reload"))

    def test_actual_override_install_cleans_partial_write_and_failed_reload(self):
        self.host.pids[promote.PREVIEW] = 0
        self.host.active[promote.PREVIEW] = "inactive"
        fdopen = os.fdopen
        class ShortWriter:
            def __init__(self, *args):
                self.file = fdopen(*args)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return self.file.__exit__(*args)
            def write(self, data):
                return self.file.write(data[:len(data) // 2])
            def flush(self):
                self.file.flush()
            def fileno(self):
                return self.file.fileno()
        with patch.object(promote.os, "fdopen", ShortWriter):
            with self.assertRaisesRegex(promote.PreviewError, "bytes differ"):
                self.host.override_install(self.template.read_bytes())
        self.assertFalse(self.override.exists())
        self.assertFalse(self.host.configured)
        self.host.fail_daemon_reload = True
        with self.assertRaisesRegex(promote.PreviewError, "daemon reload"):
            self.host.override_install(self.template.read_bytes())
        self.assertFalse(self.override.exists())
        self.assertFalse(self.host.configured)


if __name__ == "__main__":
    unittest.main()
