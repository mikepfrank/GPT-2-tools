#!/usr/bin/env python3
"""Promote the accepted single-model preview, or roll production back safely.

Run root-owned tooling with Python 3.12. --prepare changes staged files only;
--promote/--rollback belong in durable systemd-run units. Both preserve accepted
work and completed results before stopping any live model. The old preview
helper's --restore is obsolete after promotion; use this helper's --rollback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("gpt2_reviewed_preview", HERE.with_name("lightsail-preview.py"))
preview = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preview)
require, digest, PreviewError = preview.require, preview.digest, preview.PreviewError
PROD, PREVIEW = preview.PROD, preview.PREVIEW
CONFIG = preview.CONFIG
ORIGINAL_CHECKOUT = Path("/home/gpt-2/GPT-2-tools")
ORIGINAL_SOURCE = "e5971db3b79110cb62e92575a2054a87ba5b98c4"
OVERRIDE = Path("/etc/systemd/system/gpt2-public.service.d/90-gpt2-streaming.conf")
GUARD = Path("/usr/local/libexec/gpt2-exclusive-preflight.py")
DROPIN_TEMPLATE = HERE.parents[1] / "deploy/gpt2-streaming-production.conf.example"
GUARD_TEMPLATE = HERE.parents[1] / "deploy/gpt2-exclusive-preflight.py"
NAMES = {"preview-original", "preview-gated", "both-maintenance", "production-gated",
         "production-active", "rollback-gated", "rollback-active"}


def private_root_file(path):
    metadata = path.lstat()
    require(not path.is_symlink() and metadata.st_uid == 0 and not metadata.st_mode & 0o077,
            f"Expected root-owned private manifest: {path}")


def frozen_root_file(path):
    metadata = path.lstat()
    require(not path.is_symlink() and metadata.st_uid == 0 and not metadata.st_mode & 0o222,
            f"Expected root-owned read-only staged file: {path}")


def production_section(original, *, gate=False):
    text = original.decode("utf-8")
    header = "gpt2-chat.com {\n"
    require(text.count(header) == 1, "Original canonical header is not unique")
    start = text.index(header) + len(header)
    end = text.find("\n}\n", start)
    require(end >= 0, "Original canonical closing line is missing")
    body = text[start:end + 1]
    require("test.gpt2-chat.com" not in text, "Original public configuration already contains test site")
    return (text[:start] + (preview.generation_gate() if gate else "") + body + text[end + 1:]).encode()


def redirect_site():
    return b"\n# Hosted preview accepted; keep this origin reusable for future previews.\ntest.gpt2-chat.com {\n    header Cache-Control no-store\n    redir https://gpt2-chat.com{uri} 307\n}\n"


def maintenance_site():
    return ("\ntest.gpt2-chat.com {\n" + preview.maintenance() + "}\n").encode()


def candidates(original):
    old = preview.candidates(original)
    return {
        "preview-original": old["preview-active"],
        "preview-gated": old["preview-gated"],
        "both-maintenance": old["both-maintenance"],
        "production-gated": production_section(original, gate=True) + maintenance_site(),
        "production-active": original + redirect_site(),
        "rollback-gated": production_section(original, gate=True) + maintenance_site(),
        "rollback-active": original + redirect_site(),
    }


def exec_argv(value):
    """Discard systemd's changing start/exit/PID fields; preserve exact argv."""
    if not value:
        return ""
    match = re.search(r"argv\[\]=(.*?); ignore_errors=", value)
    require(match is not None, "Unexpected systemd ExecStart representation")
    return " ".join(match.group(1).split())


def dropin_exec(data):
    lines = data.decode().splitlines()
    start = [line.removeprefix("ExecStart=") for line in lines if line.startswith("ExecStart=") and line != "ExecStart="]
    pre = [line.removeprefix("ExecStartPre=") for line in lines if line.startswith("ExecStartPre=")]
    require(len(start) == len(pre) == 1, "Reviewed override must contain one command and one startup guard")
    return start[0], pre[0]


class Host(preview.Host):
    def inventory(self):
        current = super().inventory()
        require(GUARD.lstat().st_uid == 0 and not GUARD.is_symlink() and not GUARD.stat().st_mode & 0o022,
                "Production exclusivity guard must remain root-owned and protected")
        current["files"][str(GUARD)] = digest(GUARD.read_bytes())
        head = self.run(["git", "-c", f"safe.directory={ORIGINAL_CHECKOUT}", "-C", str(ORIGINAL_CHECKOUT), "rev-parse", "HEAD"])
        require(head == ORIGINAL_SOURCE, "Original production checkout differs from e5971db")
        require(not self.run(["git", "-c", f"safe.directory={ORIGINAL_CHECKOUT}", "-C", str(ORIGINAL_CHECKOUT),
                              "status", "--porcelain", "--untracked-files=no"]), "Original production tracked source is dirty")
        current.update(original_source=head, preview_pid=self.pid(PREVIEW),
                       production_exec=exec_argv(self.show(PROD, "ExecStart")),
                       production_pre=exec_argv(self.show(PROD, "ExecStartPre")))
        return current

    def unit_check(self, path):
        self.run(["systemd-analyze", "verify", "--man=no", str(path)])

    def override_install(self, data):
        require(self.pid(PROD) == 0 and self.show(PROD, "ActiveState") == "inactive" and self.cgroup_empty(PROD),
                "Production must be fully inactive before installing its override")
        require(not OVERRIDE.exists() and not OVERRIDE.is_symlink(), "Owned override path already exists")
        directory = OVERRIDE.parent
        directory.mkdir(mode=0o755, exist_ok=True)
        require(directory.lstat().st_uid == 0 and not directory.is_symlink() and not directory.stat().st_mode & 0o022,
                "Production drop-in directory is not protected")
        inode = None
        try:
            descriptor = os.open(OVERRIDE, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
            inode = os.fstat(descriptor).st_ino
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(OVERRIDE, 0o440)
            require(OVERRIDE.read_bytes() == data, "Installed override bytes differ")
            self.run(["systemctl", "daemon-reload"])
        except BaseException:
            # Remove only the inode exclusively created here, even after a short
            # write. Never mistake someone else's file for our own partial file.
            if inode is not None and OVERRIDE.exists():
                require(not OVERRIDE.is_symlink() and OVERRIDE.lstat().st_ino == inode and OVERRIDE.stat().st_uid == 0,
                        "Override changed during failed installation; manual review required")
                OVERRIDE.unlink()
                self.run(["systemctl", "daemon-reload"])
            raise

    def override_remove(self, expected):
        require(self.pid(PROD) == 0 and self.show(PROD, "ActiveState") == "inactive" and self.cgroup_empty(PROD),
                "Production must be fully inactive before removing its override")
        if OVERRIDE.exists() or OVERRIDE.is_symlink():
            frozen_root_file(OVERRIDE)
            require(digest(OVERRIDE.read_bytes()) == expected, "Refusing to remove a foreign or changed override")
            OVERRIDE.unlink()
        # Also cancel stale loaded configuration after an earlier reload error.
        self.run(["systemctl", "daemon-reload"])

    def streaming_ready(self, service, expected):
        health = self.ready(service)
        origin = preview.PROD_ORIGIN if service == PROD else preview.TEST_ORIGIN
        code, payload = self.request(origin, "/api/health")
        external = json.loads(payload)
        require(code == 200 and external.get("status") == "ready" and
                external.get("capabilities", {}).get("streaming", False) is expected,
                "TLS backend streaming capability differs from selected build")
        require(health.get("capabilities", {}).get("streaming", False) is expected,
                "Loaded backend streaming capability differs from selected build")
        require(health.get("model") == {"id": "openai-community/gpt2-xl", "revision": "15ea56dee5df4983c59b2538573817e1667135e2",
                                      "context_tokens": 1024, "precision": "FP32", "device": "cpu"},
                "Loaded backend model identity differs")
        require(health["limits"]["max_reply_tokens"] == 256 and health["limits"]["result_ttl_seconds"] == 600,
                "Loaded backend reply/result limits differ")
        require(external.get("model") == health["model"] and external.get("limits") == health["limits"],
                "TLS/backend model identity or limits differ")
        return health

    def redirect_check(self):
        headers = self.run(["curl", "--silent", "--show-error", "--max-time", "8", "--head",
                            "--resolve", "test.gpt2-chat.com:443:127.0.0.1",
                            preview.TEST_ORIGIN + "/preview-redirect-check?x=1"])
        require(re.search(r"^HTTP/\S+ 307(?: |$)", headers) and
                re.search(r"(?im)^location: https://gpt2-chat.com/preview-redirect-check\?x=1\s*$", headers) and
                re.search(r"(?im)^cache-control: no-store\s*$", headers),
                "Test-domain HTTPS redirect did not preserve path/query")


class Controller(preview.Controller):
    def __init__(self, host, stage, previous_stage=None):
        super().__init__(host, stage)
        self.previous_stage = Path(previous_stage) if previous_stage else None

    def read_previous(self):
        require(self.previous_stage is not None, "--preview-stage is required for --prepare")
        manifest = self.previous_stage / "manifest.json"
        private_root_file(manifest)
        previous = json.loads(manifest.read_text())
        require(previous.get("schema_version") == 1 and previous.get("phase") == "active", "Preview stage is not an active reviewed transition")
        original = self.previous_stage / "original.caddy"
        frozen_root_file(original)
        data = original.read_bytes()
        require(digest(data) == previous["candidates"]["original"] == previous["snapshot"]["config_sha256"],
                "Original public proxy snapshot differs")
        require(digest(CONFIG.read_bytes()) == previous["candidates"]["preview-active"] and
                CONFIG.read_bytes() == preview.candidates(data)["preview-active"], "Live proxy is not the accepted preview configuration")
        return previous, data, digest(manifest.read_bytes())

    def prepare(self):
        import pwd
        with self.host.lock():
            require(not OVERRIDE.exists() and not OVERRIDE.is_symlink(), "Production streaming override must be absent at preparation")
            previous, original, previous_hash = self.read_previous()
            snapshot = self.host.inventory()
            for field in ("source", "caddy_pid", "inode", "config_attributes", "chess_pids"):
                require(snapshot[field] == previous["snapshot"][field], f"Preview preservation differs: {field}")
            expected_files = dict(previous["snapshot"]["files"])
            require(str(GUARD) not in expected_files, "Generic production guard unexpectedly exists in old preview snapshot")
            expected_files[str(GUARD)] = digest(GUARD.read_bytes())
            require(snapshot["files"] == expected_files, "Files/drop-ins changed beyond the newly reviewed generic guard")
            require(snapshot["properties"] == previous["snapshot"]["properties"], "Loaded settings changed since accepted preview preparation")
            require(self.host.pid(PREVIEW) > 0 and self.host.model_pids() == {self.host.pid(PREVIEW)}, "Preview must be the sole resident model")
            require(self.host.pid(PROD) == 0 and self.host.show(PROD, "ActiveState") == "inactive" and self.host.cgroup_empty(PROD),
                    "Original production is not fully inactive")
            require(self.host.show(PROD, "WorkingDirectory") == str(ORIGINAL_CHECKOUT) and not snapshot["production_pre"],
                    "Original production effective configuration differs")
            dropin = DROPIN_TEMPLATE.read_bytes()
            start, pre = dropin_exec(dropin)
            require(start.startswith(f"/usr/bin/env PYTHONPATH={preview.CHECKOUT}/src /home/gpt-2/.venvs/gpt2-public/bin/python ") and
                    "--port 8899" in start and "--enable-streaming" in start and
                    pre == f"/usr/bin/python3 {GUARD} {PREVIEW}", "Production override template contract differs")
            require(GUARD.read_bytes() == GUARD_TEMPLATE.read_bytes(), "Installed generic guard differs from reviewed tooling")
            self.stage.mkdir(mode=0o711, parents=True, exist_ok=False)
            os.chmod(self.stage, 0o711)
            account = pwd.getpwnam("astra")
            hashes = {}
            for name, data in candidates(original).items():
                path = self.stage / (name + ".caddy")
                self.freeze(path, data, group=account.pw_gid)
                self.host.validate(path)
                hashes[name] = digest(data)
            self.freeze(self.stage / "streaming.conf", dropin)
            fragment = Path(self.host.show(PROD, "FragmentPath"))
            assembled = fragment.read_bytes()
            for path in self.host.show(PROD, "DropInPaths").split():
                assembled += b"\n" + Path(path).read_bytes()
            assembled += b"\n" + dropin
            unit = self.stage / PROD
            self.freeze(unit, assembled)
            self.host.unit_check(unit)
            snapshots = {}
            for index, (path, sha) in enumerate(snapshot["files"].items()):
                saved = self.stage / f"snapshot-{index:03}.bytes"
                data = Path(path).read_bytes()
                require(digest(data) == sha, "Installed file changed during snapshot")
                self.freeze(saved, data)
                snapshots[path] = saved.name
            self.manifest = {"schema_version": 2, "prepared_at": preview.timestamp(), "snapshot": snapshot, "candidates": hashes,
                             "dropin_sha256": digest(dropin), "streaming_exec": start, "streaming_pre": pre,
                             "assembled_unit_sha256": digest(assembled),
                             "previous_stage": str(self.previous_stage), "previous_manifest_sha256": previous_hash,
                             "snapshots": snapshots, "helper_sha256": digest(HERE.read_bytes()),
                             "preview_helper_sha256": digest(HERE.with_name("lightsail-preview.py").read_bytes())}
            self.record("prepared")
            require(self.host.inventory() == snapshot, "Live state changed during preparation")
            self.host.chess_health()
            self.emit("promotion_prepared_no_live_changes", stage=str(self.stage), source=snapshot["source"],
                      original_source=snapshot["original_source"], old_preview_restore_obsolete_after_promotion=True)

    @staticmethod
    def freeze(path, data, group=0):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o440)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chown(path, 0, group)
        os.chmod(path, 0o440)

    def load(self):
        path = self.stage / "manifest.json"
        private_root_file(path)
        self.manifest = json.loads(path.read_text())
        require(self.manifest.get("schema_version") == 2 and set(self.manifest["candidates"]) == NAMES, "Unsupported promotion manifest")
        require(digest(HERE.read_bytes()) == self.manifest["helper_sha256"] and
                digest(HERE.with_name("lightsail-preview.py").read_bytes()) == self.manifest["preview_helper_sha256"],
                "Promotion/preview tooling changed after preparation")
        for name, sha in self.manifest["candidates"].items():
            path = self.stage / (name + ".caddy")
            frozen_root_file(path)
            require(digest(path.read_bytes()) == sha, "Staged promotion proxy candidate changed")
        path = self.stage / "streaming.conf"
        frozen_root_file(path)
        require(digest(path.read_bytes()) == self.manifest["dropin_sha256"], "Staged override changed")
        path = self.stage / PROD
        frozen_root_file(path)
        require(digest(path.read_bytes()) == self.manifest["assembled_unit_sha256"], "Staged unit-validation candidate changed")
        previous = Path(self.manifest["previous_stage"]) / "manifest.json"
        private_root_file(previous)
        require(digest(previous.read_bytes()) == self.manifest["previous_manifest_sha256"], "Original preview stage changed")
        for installed, staged in self.manifest["snapshots"].items():
            path = self.stage / staged
            frozen_root_file(path)
            require(digest(path.read_bytes()) == self.manifest["snapshot"]["files"][installed], "Preserved snapshot bytes changed")

    def integrity(self, *, original=False):
        current, saved = self.host.inventory(), self.manifest["snapshot"]
        files = dict(current["files"])
        own = files.pop(str(OVERRIDE), None)
        require(files == saved["files"], "Preserved units/drop-ins/environments/guards changed")
        if own is not None:
            frozen_root_file(OVERRIDE)
            require(own == self.manifest["dropin_sha256"], "Owned override changed")
        # During an interrupted daemon-reload the disk and loaded state may
        # differ; only either complete reviewed command/property set is allowed.
        base = saved["properties"][PROD]
        streaming = base.replace("WorkingDirectory=" + str(ORIGINAL_CHECKOUT), "WorkingDirectory=" + str(preview.CHECKOUT))
        mode = current["production_exec"] == self.manifest["streaming_exec"]
        require(current["production_exec"] == (self.manifest["streaming_exec"] if mode else saved["production_exec"]) and
                current["production_pre"] == (self.manifest["streaming_pre"] if mode else saved["production_pre"]),
                "Production loaded command/startup guard changed")
        expected_properties = dict(saved["properties"])
        expected_properties[PROD] = streaming if mode else base
        require(current["properties"] == expected_properties, "Loaded account/environment/resource settings changed")
        for field in ("source", "original_source", "caddy_pid", "inode", "config_attributes", "chess_pids"):
            require(current[field] == saved[field], f"Preserved state changed: {field}")
        require(current["config_sha256"] in self.manifest["candidates"].values(), "Proxy changed outside this promotion")
        self.host.chess_health()
        return own is not None, mode

    def override(self, install):
        self.integrity()
        if install:
            path = self.stage / "streaming.conf"
            frozen_root_file(path)
            data = path.read_bytes()
            require(digest(data) == self.manifest["dropin_sha256"], "Override candidate changed before unit transition")
            unit = self.stage / PROD
            frozen_root_file(unit)
            require(digest(unit.read_bytes()) == self.manifest["assembled_unit_sha256"], "Unit-validation candidate changed")
            self.host.unit_check(unit)
            require(path.read_bytes() == data and digest(data) == self.manifest["dropin_sha256"], "Override changed before installation")
            self.host.override_install(data)
        else:
            self.host.override_remove(self.manifest["dropin_sha256"])
        owned, effective = self.integrity()
        require(owned is install and effective is install, "Production unit did not load selected source/command")
        self.record("override_installed" if install else "override_removed")

    def inactive(self, service):
        require(self.host.pid(service) == 0 and self.host.show(service, "ActiveState") == "inactive" and self.host.cgroup_empty(service),
                "Model unit is not fully inactive with an empty recursive cgroup")

    def stop_active(self, service, gate):
        if self.host.pid(service):
            self.apply(gate)
            self.host.gate_check(preview.PROD_ORIGIN if service == PROD else preview.TEST_ORIGIN)
            self.retain(service)
        else:
            require(self.host.cgroup_empty(service), "PID-zero model unit retains processes; no force stop allowed")
        # Even MainPID=0 may have an on-failure retry pending. Always stop to
        # cancel it, then prove complete inactivity before starting another model.
        self.host.stop(service)
        self.inactive(service)

    def promote(self):
        self.load()
        with self.host.lock():
            owned, effective = self.integrity()
            require(not owned and not effective and self.manifest["phase"] == "prepared", "Promotion already attempted; inspect phase before proceeding")
            self.inactive(PROD)
            require(self.host.pid(PREVIEW) == self.manifest["snapshot"]["preview_pid"] > 0 and
                    self.host.model_pids() == {self.host.pid(PREVIEW)}, "Accepted preview model changed since prepare")
            try:
                self.stop_active(PREVIEW, "preview-gated")
                self.apply("both-maintenance")
                self.override(True)
                self.apply("production-gated")
                self.host.start(PROD)
                self.host.streaming_ready(PROD, True)
                self.host.gate_check(preview.PROD_ORIGIN)
                require(self.host.model_pids() == {self.host.pid(PROD)}, "Promoted model is not exclusive")
                self.apply("production-active")
                self.host.streaming_ready(PROD, True)
                self.host.redirect_check()
                self.integrity()
                self.record("promoted")
                self.emit("production_streaming_active", production_pid=self.host.pid(PROD), origin=preview.PROD_ORIGIN,
                          source=self.manifest["snapshot"]["source"], override_sha256=self.manifest["dropin_sha256"],
                          old_preview_restore_obsolete=True)
            except BaseException:
                self.emit("promotion_failed_attempting_preview_recovery")
                try:
                    self.recover_preview()
                except BaseException as recovery:
                    self.emit("recovery_requires_operator", error=str(recovery), stage=str(self.stage))
                    raise PreviewError("Cannot safely recover preview; inspect gates/model/phase before any stop") from recovery
                raise

    def recover_preview(self):
        self.integrity()
        self.stop_active(PROD, "production-gated")
        if not self.host.pid(PREVIEW):
            self.host.stop(PREVIEW)
            self.inactive(PREVIEW)
        else:
            require(self.host.model_pids() == {self.host.pid(PREVIEW)}, "Preview recovery found another model")
        self.override(False)
        self.apply("preview-gated")
        if not self.host.pid(PREVIEW):
            self.host.start(PREVIEW)
        self.host.streaming_ready(PREVIEW, True)
        self.host.gate_check(preview.TEST_ORIGIN)
        self.apply("preview-original")
        self.host.streaming_ready(PREVIEW, True)
        require(self.host.model_pids() == {self.host.pid(PREVIEW)}, "Recovered preview is not exclusive")
        self.record("preview_recovered")
        self.emit("accepted_preview_restored", preview_pid=self.host.pid(PREVIEW), production_paused=True)

    def rollback(self):
        self.load()
        with self.host.lock():
            self.integrity()
            require(self.host.model_pids() <= {self.host.pid(PROD), self.host.pid(PREVIEW)} - {0} and
                    not (self.host.pid(PROD) and self.host.pid(PREVIEW)), "Unexpected/overlapping models prevent rollback")
            if self.host.pid(PREVIEW):
                self.stop_active(PREVIEW, "preview-gated")
            else:
                self.host.stop(PREVIEW)
                self.inactive(PREVIEW)
            self.stop_active(PROD, "production-gated")
            self.apply("both-maintenance")
            self.override(False)
            self.apply("rollback-gated")
            self.host.start(PROD)
            self.host.streaming_ready(PROD, False)
            self.host.gate_check(preview.PROD_ORIGIN)
            self.apply("rollback-active")
            self.host.streaming_ready(PROD, False)
            self.host.redirect_check()
            require(self.host.model_pids() == {self.host.pid(PROD)}, "Rollback production is not exclusive")
            self.integrity()
            self.record("rolled_back")
            self.emit("original_polling_production_restored", production_pid=self.host.pid(PROD), origin=preview.PROD_ORIGIN,
                      source=self.manifest["snapshot"]["original_source"], test_domain_redirect=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--promote", action="store_true")
    modes.add_argument("--rollback", action="store_true")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--preview-stage", type=Path, help="original active-preview stage; --prepare only")
    args = parser.parse_args()
    require(os.name == "posix" and os.geteuid() == 0, "Run installed Python 3.12 tooling as root on Lightsail")
    require(args.prepare or args.preview_stage is None, "--preview-stage applies only to --prepare")
    controller = Controller(Host(), args.stage.resolve(), args.preview_stage.resolve() if args.preview_stage else None)
    if args.prepare:
        controller.prepare()
    elif args.promote:
        controller.promote()
    else:
        controller.rollback()


if __name__ == "__main__":
    main()
