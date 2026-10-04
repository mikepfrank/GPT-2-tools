"""Model-free lifecycle checks: no systemd, SSH, network or model calls."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("lightsail_preview_helper", Path(__file__).resolve().parents[1] / "scripts" / "lightsail-preview.py")
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)
ORIGINAL = b"{\n admin off\n}\n\nastraplayschess.com {\n respond ok\n}\n\ngpt2-chat.com {\n    rewrite * ?\n    encode gzip\n    reverse_proxy 127.0.0.1:8899 {\n        header_up Host {http.request.hostport}\n    }\n}\n"


class FakeHost(preview.Host):
    def __init__(self, config):
        self.config, self.events, self.seconds = config, [], 0
        self.clock = lambda: self.seconds
        self.sleep = lambda seconds: setattr(self, "seconds", self.seconds + seconds)
        self.drain_timeout, self.readiness_timeout = 8, 8
        self.pids = {preview.PROD: 10, preview.PREVIEW: 0, preview.CADDY: 30, preview.CHESS[0]: 40, preview.CHESS[1]: 50}
        self.active = {preview.PROD: "active", preview.PREVIEW: "inactive"}
        self.extra_model = False
        self.memory = 12
        self.queue_sequence = []
        self.fail_readiness = False
        self.fail_preview_drain = False
        self.pending_connection_checks = 0
        self.fail_reload_once = False
        self.fail_write_once = False
        self.leftover_cgroup = False
        self.loaded_source = "5389a30aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

    @contextmanager
    def lock(self):
        self.events.append(("lock", self.seconds))
        yield
        self.events.append(("unlock", self.seconds))

    def inventory(self):
        return {"files": {"unit": "untouched", "env": "untouched", "preflight": "untouched"}, "properties": {},
                "source": self.loaded_source, "caddy_pid": 30, "inode": self.config.stat().st_ino,
                "config_attributes": [0, 0, 0o100600], "chess_pids": {preview.CHESS[0]: 40, preview.CHESS[1]: 50},
                "production_pid": self.pids[preview.PROD], "config_sha256": preview.digest(self.config.read_bytes())}

    def pid(self, unit):
        return self.pids[unit]

    def model_pids(self):
        result = {pid for unit, pid in self.pids.items() if unit in (preview.PROD, preview.PREVIEW) and pid}
        return result | ({999} if self.extra_model else set())

    def show(self, unit, name):
        return self.active[unit] if name == "ActiveState" else ""

    def cgroup_empty(self, unit):
        return self.pids[unit] == 0 and not self.leftover_cgroup

    def available_gib(self):
        return self.memory

    def run(self, args):
        self.events.append(("command", args, self.seconds))
        if args[:2] == ["systemctl", "stop"]:
            self.pids[args[2]] = 0
            self.active[args[2]] = "inactive"
        elif args[:2] == ["systemctl", "start"]:
            self.pids[args[2]] = 20 if args[2] == preview.PREVIEW else 11
            self.active[args[2]] = "active"
        elif args[0] == "cp":
            Path(args[-1]).write_bytes(Path(args[-2]).read_bytes())
        else:
            raise AssertionError(args)
        return ""

    def validate(self, path):
        self.events.append(("validate", path.name))

    def write_config(self, data, snapshot):
        if self.fail_write_once:
            self.fail_write_once = False
            self.config.write_bytes(data[:len(data) // 2])
            raise preview.PreviewError("fixture short write")
        self.config.write_bytes(data)

    def reload(self):
        self.events.append(("reload",))
        if self.fail_reload_once:
            self.fail_reload_once = False
            raise preview.PreviewError("fixture reload failure")

    def chess_health(self):
        pass

    def gate_check(self, origin):
        self.events.append(("gate", origin, self.seconds))

    def health(self, service):
        if service == preview.PREVIEW and self.fail_preview_drain:
            queue = {"waiting": 1, "running": True}
        else:
            queue = self.queue_sequence.pop(0) if self.queue_sequence else {"waiting": 0, "running": False}
        return {"queue": queue, "limits": {"result_ttl_seconds": 600}, "status": "ready"}

    def settled_connections(self, service):
        if self.pending_connection_checks:
            self.pending_connection_checks -= 1
            return False
        return True

    def ready(self, service):
        if service == preview.PREVIEW and self.fail_readiness:
            self.fail_readiness = False
            raise preview.PreviewError("fixture readiness failure")
        return self.health(service)


class FakeController(preview.Controller):
    def load(self):
        pass

    def record(self, phase):
        self.manifest["phase"] = phase

    def emit(self, phase, **details):
        self.host.events.append((phase, details, self.host.seconds))

    def apply(self, name):
        self.integrity()
        self.host.config.write_bytes(self.data[name])
        self.host.events.append(("apply", name, self.host.seconds))
        self.record(name)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = Path(self.directory.name) / "Caddyfile"
        self.config.write_bytes(ORIGINAL)
        self.host = FakeHost(self.config)
        self.controller = FakeController(self.host, self.directory.name)
        self.controller.data = preview.candidates(ORIGINAL)
        self.controller.manifest = {"snapshot": self.host.inventory(),
                                    "candidates": {name: preview.digest(data) for name, data in self.controller.data.items()}}

    def tearDown(self):
        self.directory.cleanup()

    def test_candidates_preserve_chess_and_gate_exact_generation_paths(self):
        values = preview.candidates(ORIGINAL)
        prefix = ORIGINAL.split(b"gpt2-chat.com {", 1)[0]
        for data in values.values():
            self.assertTrue(data.startswith(prefix))
        production = values["production-gated"].decode()
        self.assertIn("method POST\n        path /api/message /api/regenerate", production)
        self.assertIn('\\"accepted\\":false', production)
        self.assertIn("reverse_proxy 127.0.0.1:8899", production)
        test = values["preview-active"].decode().split("test.gpt2-chat.com {", 1)[1]
        self.assertNotIn("encode", test)
        self.assertIn("rewrite * ?", test)
        self.assertIn("header_up -Forwarded", test)
        self.assertIn("flush_interval -1", test)
        self.assertIn("127.0.0.1:8898", test)

    def test_prepare_does_not_call_any_start_stop_or_apply(self):
        stage = Path(self.directory.name) / "stage"
        controller = preview.Controller(self.host, stage)
        with patch.object(preview, "CONFIG", self.config), patch.dict(sys.modules, {"pwd": SimpleNamespace(getpwnam=lambda name: SimpleNamespace(pw_uid=0, pw_gid=0))}), \
                patch.object(preview.os, "chown", create=True), patch.object(controller, "emit"):
            controller.prepare()
        self.assertEqual(self.config.read_bytes(), ORIGINAL)
        self.assertFalse(any(event[0] == "command" for event in self.host.events))
        self.assertEqual(len(list(stage.glob("*.caddy"))), 5)
        self.assertEqual(self.host.pids[preview.PROD], 10)
        self.assertEqual(self.host.pids[preview.PREVIEW], 0)

    def test_activation_keeps_results_full_ttl_before_stop_and_single_model(self):
        self.host.pending_connection_checks = 2
        self.controller.activate()
        stop = next(event for event in self.host.events if event[0] == "command" and event[1] == ["systemctl", "stop", preview.PROD])
        self.assertGreaterEqual(stop[2], 604)
        self.assertEqual(self.host.pids[preview.PROD], 0)
        self.assertEqual(self.host.pids[preview.PREVIEW], 20)
        self.assertEqual(self.controller.manifest["phase"], "active")
        self.assertEqual(self.host.events[-1][0], "unlock")

    def test_new_work_resets_retention_timer(self):
        idle = {"waiting": 0, "running": False}
        self.host.queue_sequence = [idle, idle, idle, {"waiting": 0, "running": True}, idle, idle]
        self.controller.retain(preview.PROD)
        self.assertGreaterEqual(self.host.seconds, 606)

    def test_restore_keeps_preview_results_then_returns_original_bytes(self):
        self.controller.activate()
        began = self.host.seconds
        self.controller.restore()
        stop = next(event for event in self.host.events if event[0] == "command" and event[1] == ["systemctl", "stop", preview.PREVIEW])
        self.assertGreaterEqual(stop[2] - began, 600)
        self.assertEqual(self.config.read_bytes(), ORIGINAL)
        self.assertEqual(self.host.pids[preview.PREVIEW], 0)
        self.assertEqual(self.host.pids[preview.PROD], 11)

    def test_activation_readiness_failure_restores_original_without_overlap(self):
        self.host.fail_readiness = True
        with self.assertRaisesRegex(preview.PreviewError, "fixture readiness"):
            self.controller.activate()
        self.assertEqual(self.config.read_bytes(), ORIGINAL)
        self.assertEqual(self.host.model_pids(), {11})
        self.assertEqual(self.controller.manifest["phase"], "restored")
        preview_stop = next(event for event in self.host.events if event[0] == "command" and event[1] == ["systemctl", "stop", preview.PREVIEW])
        self.assertGreaterEqual(preview_stop[2], 1200)

    def test_undrainable_production_is_never_stopped(self):
        self.host.queue_sequence = [{"waiting": 1, "running": True}] * 20
        with self.assertRaisesRegex(preview.PreviewError, "did not drain"):
            self.controller.activate()
        self.assertEqual(self.host.pids[preview.PROD], 10)
        self.assertFalse(any(event[0] == "command" for event in self.host.events))
        self.assertEqual(self.config.read_bytes(), ORIGINAL)

    def test_undrainable_preview_is_left_running_without_starting_production(self):
        self.controller.activate()
        self.host.fail_preview_drain = True
        previous = len(self.host.events)
        with self.assertRaisesRegex(preview.PreviewError, "did not drain"):
            self.controller.restore()
        self.assertEqual(self.host.pids[preview.PREVIEW], 20)
        self.assertEqual(self.host.pids[preview.PROD], 0)
        self.assertFalse(any(event[0] == "command" for event in self.host.events[previous:]))

    def test_extra_model_transitional_state_and_low_memory_block_start(self):
        self.host.extra_model = True
        with self.assertRaisesRegex(preview.PreviewError, "Unexpected model"):
            self.controller.activate()
        self.assertEqual(self.config.read_bytes(), ORIGINAL)
        self.host.extra_model = False
        self.host.pids[preview.PROD] = 0
        self.host.active[preview.PROD] = "deactivating"
        with self.assertRaisesRegex(preview.PreviewError, "fully inactive"):
            self.host.start(preview.PREVIEW)
        self.host.active[preview.PROD] = "inactive"
        self.host.leftover_cgroup = True
        with self.assertRaisesRegex(preview.PreviewError, "empty cgroup"):
            self.host.start(preview.PREVIEW)
        self.host.leftover_cgroup = False
        self.host.memory = 7.9
        with self.assertRaisesRegex(preview.PreviewError, "8 GiB"):
            self.host.start(preview.PREVIEW)
        self.assertFalse(any(event[0] == "command" for event in self.host.events))

    def actual_controller(self):
        controller = preview.Controller(self.host, self.directory.name)
        controller.manifest = self.controller.manifest
        for name, data in self.controller.data.items():
            path = Path(self.directory.name) / (name + ".caddy")
            if path.exists():
                path.chmod(0o600)
            path.write_bytes(data)
            path.chmod(0o440)
        return controller

    def test_actual_apply_rejects_changed_candidate_before_live_write(self):
        controller = self.actual_controller()
        path = Path(self.directory.name) / "production-gated.caddy"
        path.chmod(0o600)
        path.write_bytes(b"changed")
        with patch.object(preview, "CONFIG", self.config):
            with self.assertRaisesRegex(preview.PreviewError, "Candidate changed"):
                controller.apply("production-gated")
        self.assertEqual(self.config.read_bytes(), ORIGINAL)
        self.assertFalse(any(event[0] == "command" for event in self.host.events))

    def test_actual_apply_restores_bytes_on_short_write_or_failed_reload(self):
        for failure in ("fail_write_once", "fail_reload_once"):
            with self.subTest(failure=failure):
                controller = self.actual_controller()
                original_inode = self.config.stat().st_ino
                setattr(self.host, failure, True)
                with patch.object(preview, "CONFIG", self.config), patch.object(controller, "record"):
                    with self.assertRaisesRegex(preview.PreviewError, "fixture"):
                        controller.apply("production-gated")
                self.assertEqual(self.config.read_bytes(), ORIGINAL)
                self.assertEqual(self.config.stat().st_ino, original_inode)
                self.assertEqual(self.host.events[-1][0], "reload")


if __name__ == "__main__":
    unittest.main()
