#!/usr/bin/env python3
"""Prepare, activate, or restore the authorized single-model Lightsail preview.

--prepare writes and validates staged candidates only. Activation gates/drains
production and preserves its result TTL before stopping it. Restore does the
same for preview. Run with the installed Python 3.12 as root; neither unit is
installed, enabled, disabled, or modified by this helper. See the stage manifest
and stdout phase records. Never rerun historical one-off provisioning scripts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.error
import urllib.request


CONFIG = Path("/home/astra/.config/astra-chess/Caddyfile")
LOCK = Path("/run/lock/astra-caddy-config.lock")
BINARY = Path("/home/astra/.local/bin/caddy")
CHECKOUT = Path("/home/gpt-2/GPT-2-streaming-preview")
SOURCE = "5389a30"
PROD, PREVIEW, CADDY = "gpt2-public.service", "gpt2-preview.service", "astra-caddy.service"
CHESS = ("astra-chess.service", "or-chess.service")
UNITS = (PROD, PREVIEW, CADDY, *CHESS)
PROD_ORIGIN, TEST_ORIGIN = "https://gpt2-chat.com", "https://test.gpt2-chat.com"
ENV_FILES = (Path("/etc/gpt2-public.env"), Path("/etc/gpt2-preview.env"))
PREFLIGHT = Path("/usr/local/libexec/gpt2-preview-preflight.py")
CHECKPOINT_SHA = "0f8b28eb05a8075f48b61b6f35332978c74fc7763fa9fb4051a1c30511736a6a"
USER_ENV = ["sudo", "-u", "astra", "env", "HOME=/home/astra/.local/share/astra-caddy",
            "XDG_DATA_HOME=/home/astra/.local/share/astra-caddy",
            "XDG_CONFIG_HOME=/home/astra/.local/share/astra-caddy/config", str(BINARY)]
GATE_BODY = json.dumps({"error": "GPT-2 is paused for a controlled preview; your saved chat is unchanged.", "accepted": False}, separators=(",", ":"))


class PreviewError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise PreviewError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generation_gate():
    return ("    @gpt2_preview_generation {\n        method POST\n"
            "        path /api/message /api/regenerate\n    }\n"
            "    header @gpt2_preview_generation Content-Type application/json\n"
            "    header @gpt2_preview_generation Cache-Control no-store\n"
            "    header @gpt2_preview_generation Retry-After 60\n"
            f"    respond @gpt2_preview_generation {json.dumps(GATE_BODY)} 503\n")


def maintenance():
    return ("    header Content-Type application/json\n    header Cache-Control no-store\n"
            "    header Retry-After 60\n" + f"    respond {json.dumps(GATE_BODY)} 503\n")


def candidates(original):
    """Only insert into the canonical site and append the new test site."""
    text = original.decode("utf-8")
    header = "gpt2-chat.com {\n"
    require(text.count(header) == 1, "Canonical site header is not unique; review configuration manually")
    require("test.gpt2-chat.com" not in text, "Test site already exists; review/merge it manually")
    test_header = "\n# Controlled streaming preview; separate browser-storage origin.\ntest.gpt2-chat.com {\n    rewrite * ?\n"
    proxy = ("    reverse_proxy 127.0.0.1:8898 {\n        flush_interval -1\n"
             "        header_up Host {http.request.hostport}\n"
             "        header_up X-Real-IP {http.request.remote.host}\n"
             "        header_up -X-Forwarded-For\n        header_up -Forwarded\n    }\n}\n")
    def make(prod, test):
        return (text.replace(header, header + prod, 1) + test_header + test).encode("utf-8")
    return {
        "original": original,
        "production-gated": make(generation_gate(), maintenance() + "}\n"),
        "preview-gated": make(maintenance(), generation_gate() + proxy),
        "preview-active": make(maintenance(), proxy),
        "both-maintenance": make(maintenance(), maintenance() + "}\n"),
    }


class Host:
    def __init__(self, *, drain_timeout=360, readiness_timeout=120):
        self.drain_timeout, self.readiness_timeout = drain_timeout, readiness_timeout
        self.clock, self.sleep = time.monotonic, time.sleep

    def run(self, args):
        return subprocess.run(args, check=True, text=True, capture_output=True, timeout=30).stdout.strip()

    def show(self, unit, name):
        return self.run(["systemctl", "show", "-p", name, "--value", unit])

    def pid(self, unit):
        return int(self.show(unit, "MainPID"))

    @contextmanager
    def lock(self):
        import fcntl
        with LOCK.open("a") as stream:
            end = self.clock() + 10
            while True:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    require(self.clock() < end, "Shared Caddy maintenance lock is busy; no transition started")
                    self.sleep(.25)
            yield

    def inventory(self):
        files = {}
        properties = {}
        for unit in UNITS:
            fragment = self.show(unit, "FragmentPath")
            require(fragment, f"Missing installed unit: {unit}")
            for path in (fragment, *self.show(unit, "DropInPaths").split()):
                files[path] = digest(Path(path).read_bytes())
            properties[unit] = self.run(["systemctl", "show", unit,
                "-p", "User", "-p", "Group", "-p", "WorkingDirectory",
                "-p", "EnvironmentFiles", "-p", "Environment", "-p", "MemoryHigh", "-p", "MemoryMax",
                "-p", "MemorySwapMax", "-p", "Restart", "-p", "UnitFileState"])
        for path in ENV_FILES:
            files[str(path)] = digest(path.read_bytes())
        files[str(PREFLIGHT)] = digest(PREFLIGHT.read_bytes())
        require(PREFLIGHT.stat().st_uid == 0 and not PREFLIGHT.stat().st_mode & 0o022,
                "Installed model-exclusivity guard must remain root-owned and not writable by other users")
        proxy = self.pid(CADDY)
        mounted = Path(f"/proc/{proxy}/root{CONFIG}")
        data = CONFIG.read_bytes()
        require(proxy > 0 and data == mounted.read_bytes(), "Caddy host/mounted bytes do not agree")
        inode = CONFIG.stat().st_ino
        require(inode == mounted.stat().st_ino, "Caddy bind-mounted inode differs")
        require(digest(BINARY.read_bytes()) == digest(Path(f"/proc/{proxy}/exe").read_bytes()), "Running Caddy binary differs")
        version = self.run([str(BINARY), "version"])
        match = re.match(r"v(\d+)\.(\d+)\.(\d+)", version)
        require(match and tuple(map(int, match.groups())) >= (2, 11, 0), "Caddy >=2.11.0 is required for file reload")
        argv = Path(f"/proc/{proxy}/cmdline").read_bytes().split(b"\0")
        require(b"run" in argv and b"--resume" not in argv and b"--watch" not in argv and str(CONFIG).encode() in argv,
                "Caddy startup is not the expected file-based run")
        require("admin off" in data.decode(), "Caddy admin-off precondition changed")
        head = self.run(["git", "-c", f"safe.directory={CHECKOUT}", "-C", str(CHECKOUT), "rev-parse", "HEAD"])
        require(head.startswith(SOURCE), "Preview checkout differs from reviewed source 5389a30")
        require(not self.run(["git", "-c", f"safe.directory={CHECKOUT}", "-C", str(CHECKOUT), "status", "--porcelain", "--untracked-files=no"]),
                "Preview tracked source has uncommitted changes")
        require(self.show(PREVIEW, "User") == "gpt-2" and self.show(PREVIEW, "WorkingDirectory") == str(CHECKOUT),
                "Preview user/checkout does not match the isolated deployment")
        executable = self.show(PREVIEW, "ExecStart")
        require(str(PREFLIGHT) in self.show(PREVIEW, "ExecStartPre"), "Preview does not use the installed exclusivity guard")
        for part in ("/home/gpt-2/.venvs/gpt2-public/bin/python", "-m gpt2_local.public_chat", "--host 127.0.0.1",
                     "--port 8898", "--offline", "--enable-streaming", "--trusted-proxy 127.0.0.1"):
            require(part in executable, f"Missing expected preview ExecStart option: {part}")
        settings = {}
        for line in ENV_FILES[1].read_text().splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                settings[key.strip()] = value.strip().strip("\"'")
        expected = {"PYTHONPATH": str(CHECKOUT / "src"), "HF_HOME": "/home/gpt-2/.cache/gpt2-public/huggingface",
                    "GPT2_PUBLIC_ORIGIN": TEST_ORIGIN, "GPT2_THREADS": "2", "GPT2_MAX_REPLY_TOKENS": "256"}
        require(all(settings.get(key) == value for key, value in expected.items()), "Preview environment contract differs")
        require(float(settings.get("GPT2_MIN_AVAILABLE_MEMORY_GIB", "0")) == 8, "Preview memory floor is not 8 GiB")
        require(self.show(PREVIEW, "UnitFileState") in ("static", "disabled"), "Preview must not be boot-enabled")
        attributes = CONFIG.stat()
        return {"files": files, "properties": properties, "source": head, "caddy_pid": proxy, "inode": inode,
                "config_attributes": [attributes.st_uid, attributes.st_gid, attributes.st_mode],
                "chess_pids": {unit: self.pid(unit) for unit in CHESS}, "production_pid": self.pid(PROD),
                "config_sha256": digest(data)}

    def validate(self, path):
        require("Valid configuration" in self.run(USER_ENV + ["validate", "--config", str(path), "--adapter", "caddyfile"]),
                "Full Caddy candidate did not validate")

    def reload(self):
        journal = self.run(["journalctl", "-u", CADDY, "-n", "1", "--show-cursor", "--no-pager", "-o", "cat"])
        cursors = [line.removeprefix("-- cursor: ") for line in journal.splitlines() if line.startswith("-- cursor: ")]
        require(len(cursors) == 1, "No unique Caddy journal cursor")
        self.run(["systemctl", "kill", "--kill-whom=main", "--signal=SIGUSR1", CADDY])
        for _ in range(30):
            logs = self.run(["journalctl", "-u", CADDY, "--after-cursor", cursors[0], "--no-pager", "-o", "cat"])
            require("failed to reload config from file" not in logs and "ignored SIGUSR1" not in logs, "Caddy reload failed")
            if "successfully reloaded config from file" in logs:
                return
            self.sleep(.5)
        raise PreviewError("Caddy reload was not confirmed")

    def write_config(self, data, snapshot):
        with CONFIG.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        mounted = Path(f"/proc/{snapshot['caddy_pid']}/root{CONFIG}")
        require(CONFIG.stat().st_ino == mounted.stat().st_ino == snapshot["inode"] and mounted.read_bytes() == data,
                "In-place Caddy write/mount verification failed")

    def request(self, origin, path, *, port=None, body=None):
        host = origin.removeprefix("https://")
        args = ["curl", "--silent", "--show-error", "--max-time", "8", "-w", "\n%{http_code}"]
        if port:
            args += ["-H", "Host: " + host, f"http://127.0.0.1:{port}" + path]
        else:
            args += ["--resolve", host + ":443:127.0.0.1", origin + path]
        if body is not None:
            args += ["-H", "Origin: " + origin, "-H", "Content-Type: application/json", "--data-binary", json.dumps(body)]
        payload, code = self.run(args).rsplit("\n", 1)
        return int(code), payload

    def health(self, service):
        origin, port = (PROD_ORIGIN, 8899) if service == PROD else (TEST_ORIGIN, 8898)
        code, payload = self.request(origin, "/api/health", port=port)
        require(code == 200, f"Direct backend health failed for {service}")
        health = json.loads(payload)
        require(health.get("status") == "ready", f"Backend is not ready: {service}")
        return health

    def settled_connections(self, service):
        port = 8899 if service == PROD else 8898
        rows = self.run(["ss", "-H", "-t", "-a", "-n", f"sport = :{port}"]).splitlines()
        return not any(row.split()[0] not in ("LISTEN", "TIME-WAIT") for row in rows)

    def gate_check(self, origin):
        for path in ("/api/message", "/api/regenerate"):
            code, payload = self.request(origin, path, body={})
            require(code == 503 and json.loads(payload).get("accepted") is False, "Generation maintenance gate is not effective")

    def ready(self, service):
        end = self.clock() + self.readiness_timeout
        origin = PROD_ORIGIN if service == PROD else TEST_ORIGIN
        while True:
            try:
                health = self.health(service)
                code, payload = self.request(origin, "/api/health")
                require(code == 200 and json.loads(payload).get("status") == "ready", "TLS proxy backend is not ready")
                if service == PREVIEW:
                    require(health.get("capabilities", {}).get("streaming") is True, "Preview streaming is disabled")
                return health
            except (PreviewError, subprocess.SubprocessError, ValueError):
                require(self.clock() < end, f"Readiness timed out: {service}")
                self.sleep(2)

    def chess_health(self):
        for origin in ("https://astraplayschess.com", "https://arcturuschess.com"):
            require(self.request(origin, "/")[0] == 200, "Existing chess site health failed")

    def model_pids(self):
        found = set()
        patterns = ("gpt2_local.public_chat", "gpt2_local.chat", "gpt2_local.cli", "gpt2_local.experiment",
                    "benchmark-host.py", "preview-with-parity.py", "/gpt2-xl")
        for directory in Path("/proc").iterdir():
            if not directory.name.isdecimal() or int(directory.name) == os.getpid():
                continue
            try:
                cmd = (directory / "cmdline").read_bytes().decode(errors="replace")
                maps = (directory / "maps").read_text(errors="replace")
            except (FileNotFoundError, ProcessLookupError):
                continue
            if any(pattern in cmd for pattern in patterns) or CHECKPOINT_SHA in maps or "model.safetensors" in maps:
                found.add(int(directory.name))
        return found

    def available_gib(self):
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        available = int(fields["MemAvailable"].split()[0]) * 1024
        root = Path("/sys/fs/cgroup")
        groups = {root}
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) == 3 and parts[:2] == ["0", ""]:
                require(parts[2].startswith("/") and ".." not in parts[2].split("/"), "Invalid cgroup path")
                group = root.joinpath(*parts[2].strip("/").split("/"))
                while True:
                    groups.add(group)
                    if group == root:
                        break
                    group = group.parent
        for group in groups:
            maximum = group / "memory.max"
            if maximum.exists() and maximum.read_text().strip() != "max":
                available = min(available, max(0, int(maximum.read_text()) - int((group / "memory.current").read_text())))
        return available / 1024**3

    def stop(self, service):
        require(service in (PROD, PREVIEW), "Refusing to stop another service")
        self.run(["systemctl", "stop", service])
        require(self.pid(service) == 0 and self.show(service, "ActiveState") == "inactive", "Model service did not stop cleanly")
        require(self.cgroup_empty(service), "Stopped model service still has cgroup processes")

    def cgroup_empty(self, service):
        membership = self.show(service, "ControlGroup")
        if not membership:
            return True
        require(membership.startswith("/") and ".." not in membership.split("/"), "Unexpected service cgroup path")
        directory = Path("/sys/fs/cgroup").joinpath(*membership.strip("/").split("/"))
        return not directory.exists() or not any(file.read_text().strip() for file in directory.rglob("cgroup.procs"))

    def start(self, service):
        require(service in (PROD, PREVIEW), "Refusing to start another service")
        other = PREVIEW if service == PROD else PROD
        require(self.pid(other) == 0 and self.show(other, "ActiveState") == "inactive" and self.cgroup_empty(other),
                "Other model unit is not fully inactive with an empty cgroup")
        require(not self.model_pids(), "A GPT-2 process still exists; refusing a second model")
        require(self.available_gib() >= 8, "Host/cgroup has less than 8 GiB available; memory guard retained")
        self.run(["systemctl", "start", service])


class Controller:
    def __init__(self, host, stage):
        self.host, self.stage = host, Path(stage)
        self.manifest = None

    def emit(self, phase, **details):
        print(json.dumps({"phase": phase, "time_utc": timestamp(), **details}), flush=True)

    def prepare(self):
        import pwd
        with self.host.lock():
            snapshot = self.host.inventory()
            require(self.host.pid(PROD) > 0 and self.host.pid(PREVIEW) == 0, "Prepare requires production active and preview inactive")
            require(self.host.model_pids() == {snapshot["production_pid"]}, "Unexpected GPT-2 process before preparation")
            original = CONFIG.read_bytes()
            self.stage.mkdir(mode=0o711, parents=True, exist_ok=False)
            os.chmod(self.stage, 0o711)
            account = pwd.getpwnam("astra")
            hashes = {}
            for name, data in candidates(original).items():
                path = self.stage / (name + ".caddy")
                path.write_bytes(data)
                os.chown(path, 0, account.pw_gid)
                os.chmod(path, 0o440)
                self.host.validate(path)
                hashes[name] = digest(data)
            self.manifest = {"schema_version": 1, "prepared_at": timestamp(), "snapshot": snapshot, "candidates": hashes}
            self.record("prepared")
            require(self.host.inventory() == snapshot and CONFIG.read_bytes() == original, "Live state changed during preparation")
            self.host.chess_health()
            self.emit("prepared_no_live_changes", stage=str(self.stage), candidates=hashes)

    def record(self, phase):
        self.manifest["phase"] = phase
        self.manifest["updated_at"] = timestamp()
        path = self.stage / "manifest.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(self.manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def load(self):
        path = self.stage / "manifest.json"
        require(path.stat().st_uid == 0 and path.stat().st_mode & 0o077 == 0, "Manifest must remain root-owned/private")
        self.manifest = json.loads(path.read_text())
        require(self.manifest.get("schema_version") == 1, "Unsupported manifest")
        require(set(self.manifest["candidates"]) == {"original", "production-gated", "preview-gated", "preview-active", "both-maintenance"},
                "Candidate set differs")
        for name, expected in self.manifest["candidates"].items():
            require(digest((self.stage / (name + ".caddy")).read_bytes()) == expected, "Staged candidate changed")

    def integrity(self, *, original=False):
        current, saved = self.host.inventory(), self.manifest["snapshot"]
        for field in ("files", "properties", "source", "caddy_pid", "inode", "config_attributes", "chess_pids"):
            require(current[field] == saved[field], f"Preserved state changed: {field}; stop for manual review")
        allowed = {saved["config_sha256"]} if original else set(self.manifest["candidates"].values())
        require(current["config_sha256"] in allowed, "Live Caddy bytes changed outside this transition")
        self.host.chess_health()

    def apply(self, name):
        self.integrity()
        path = self.stage / (name + ".caddy")
        data = path.read_bytes()
        require(digest(data) == self.manifest["candidates"][name], "Candidate changed before validation")
        require(path.stat().st_uid == 0 and not path.stat().st_mode & 0o222, "Candidate must remain root-owned/read-only")
        self.host.validate(path)
        # Backup before every transition, without renaming the bind-mounted file.
        backup = self.stage / f"before-{time.time_ns()}.caddy"
        self.host.run(["cp", "-a", "--no-clobber", str(CONFIG), str(backup)])
        old = CONFIG.read_bytes()
        require(backup.read_bytes() == old, "Transition backup mismatch")
        require(path.read_bytes() == data and digest(data) == self.manifest["candidates"][name], "Candidate changed before live write")
        try:
            self.host.write_config(data, self.manifest["snapshot"])
            self.host.reload()
        except BaseException:
            # A short write or failed signal must not strand partial file bytes.
            self.host.validate(backup)
            self.host.write_config(old, self.manifest["snapshot"])
            self.host.reload()
            raise
        self.record(name)
        self.integrity()
        self.emit("proxy_transition", candidate=name, sha256=digest(data), backup=str(backup))

    def retain(self, service):
        host = self.host
        deadline, idle_since, ttl = host.clock() + host.drain_timeout, None, 600
        while True:
            health = host.health(service)
            ttl = max(600, int(health["limits"]["result_ttl_seconds"]))
            idle = health.get("queue") == {"waiting": 0, "running": False}
            if not idle:
                idle_since = None
                require(host.clock() < deadline, "Accepted jobs did not drain; no model may be stopped")
            elif idle_since is None and host.settled_connections(service):
                # Recheck after all pre-gate upstream connections settled.
                if host.health(service).get("queue") == {"waiting": 0, "running": False}:
                    idle_since = host.clock()
                    self.emit("result_retention_wait", service=service, seconds=ttl)
            elif idle_since is None:
                require(host.clock() < deadline, "Pre-gate connections did not settle; no model may be stopped")
            if idle_since is not None and host.clock() - idle_since >= ttl:
                require(host.health(service)["queue"] == {"waiting": 0, "running": False}, "Queue changed at retention boundary")
                self.emit("drained_and_results_expired", service=service, retained_seconds=ttl)
                return
            host.sleep(2)

    def activate(self):
        self.load()
        with self.host.lock():
            self.integrity(original=True)
            require(self.host.pid(PROD) == self.manifest["snapshot"]["production_pid"] > 0 and self.host.pid(PREVIEW) == 0,
                    "Production/preview process state changed since prepare")
            require(self.host.model_pids() == {self.host.pid(PROD)}, "Unexpected model process before activation")
            try:
                self.apply("production-gated")
                self.host.gate_check(PROD_ORIGIN)
                self.retain(PROD)
                self.host.stop(PROD)
                self.apply("preview-gated")
                self.host.start(PREVIEW)
                self.host.ready(PREVIEW)
                self.host.gate_check(TEST_ORIGIN)
                require(self.host.model_pids() == {self.host.pid(PREVIEW)}, "Model exclusivity failed after startup")
                self.apply("preview-active")
                self.host.ready(PREVIEW)
                self.record("active")
                self.emit("preview_active", origin=TEST_ORIGIN, preview_pid=self.host.pid(PREVIEW),
                          production_paused=True, stage=str(self.stage))
            except BaseException:
                self.emit("activation_failed_attempting_safe_restore")
                try:
                    self._restore()
                except BaseException as recovery:
                    self.emit("recovery_requires_operator", error=str(recovery), stage=str(self.stage))
                    raise PreviewError("Automatic restoration could not safely complete; inspect stage and running model before any stop/start") from recovery
                raise

    def _restore(self):
        self.integrity()
        if self.host.pid(PREVIEW):
            require(not self.host.pid(PROD), "Both model units active; operator review required")
            self.apply("preview-gated")
            self.host.gate_check(TEST_ORIGIN)
            self.retain(PREVIEW)
            self.host.stop(PREVIEW)
        self.apply("both-maintenance")
        if not self.host.pid(PROD):
            self.host.start(PROD)
        # Production is still externally gated while its original backend loads.
        end = self.host.clock() + self.host.readiness_timeout
        while True:
            try:
                self.host.health(PROD)
                break
            except (PreviewError, subprocess.SubprocessError, ValueError):
                require(self.host.clock() < end, "Original production backend did not become ready; maintenance stays in place")
                self.host.sleep(2)
        require(self.host.model_pids() == {self.host.pid(PROD)}, "Original model exclusivity was not restored")
        self.apply("original")
        self.host.ready(PROD)
        self.integrity(original=True)
        self.record("restored")
        self.emit("production_restored", origin=PROD_ORIGIN, production_pid=self.host.pid(PROD),
                  original_config_sha256=self.manifest["snapshot"]["config_sha256"])

    def restore(self):
        self.load()
        with self.host.lock():
            self._restore()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--activate", action="store_true")
    group.add_argument("--restore", action="store_true")
    parser.add_argument("--stage", type=Path, required=True, help="fresh stage directory for prepare; same directory for activate/restore")
    args = parser.parse_args()
    require(os.name == "posix" and os.geteuid() == 0, "Run on Lightsail with the installed Python 3.12 as root")
    controller = Controller(Host(), args.stage.resolve())
    if args.prepare:
        controller.prepare()
    elif args.activate:
        controller.activate()
    else:
        controller.restore()


if __name__ == "__main__":
    main()
