#!/usr/bin/env python3
"""Update production to the accepted seed-policy release, or restore streaming.

Run root-owned Python 3.12 tooling. --prepare writes staged evidence only.
Run --update/--rollback in durable systemd-run units: both gate new generation,
drain accepted work, and retain completed results for at least 600 seconds.
Only the owned production drop-in changes; the previous source and promotion
stage stay intact. The earlier polling rollback is usable after --rollback.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time


HERE = Path(__file__).resolve()
SPEC = importlib.util.spec_from_file_location("gpt2_reviewed_promotion", HERE.with_name("lightsail-promote.py"))
promote = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(promote)
preview = promote.preview
require, digest, PreviewError = promote.require, promote.digest, promote.PreviewError
PROD, PREVIEW, CONFIG, OVERRIDE = promote.PROD, promote.PREVIEW, promote.CONFIG, promote.OVERRIDE
OLD_SOURCE = "5389a30f5833c8bcaaf0314f1c0e1c63960f309e"
NEW_SOURCE = "78ebbdc254249562187c06db8ec9c3cb7dd0e87b"
RELEASE = Path("/home/gpt-2/GPT-2-releases/seed-78ebbdc")
PROMOTION_STAGE = Path("/home/astra/.config/astra-chess/gpt2-production-stage-20261004")
OLD_OVERRIDE_SHA = "524068351cc10be7557e08344c5e2ee5cc6e92a07dc51f720eac9fc76fed106a"
NAMES = {"active", "gated"}


def candidates(active):
    text = active.decode("utf-8")
    headers = list(re.finditer(r"(?m)^gpt2-chat\.com \{\n", text))
    require(len(headers) == 1 and "@gpt2_preview_generation" not in text,
            "Canonical production site is not unique or already gated")
    position = headers[0].end()
    return {"active": active, "gated": (text[:position] + preview.generation_gate() + text[position:]).encode()}


def updated_dropin(old, release):
    old_path, new_path = preview.CHECKOUT.as_posix(), Path(release).as_posix()
    require(re.fullmatch(r"/home/gpt-2/GPT-2-releases/[A-Za-z0-9_-]+", new_path), "Unsafe release path")
    require(old.count(old_path.encode()) == 2, "Old drop-in checkout references differ")
    new = old.replace(old_path.encode(), new_path.encode()).replace(b"commit 5389a30", b"commit " + NEW_SOURCE.encode())
    old_start, old_pre = promote.dropin_exec(old)
    new_start, new_pre = promote.dropin_exec(new)
    require(new_start == old_start.replace(old_path, new_path) and new_pre == old_pre and
            f"WorkingDirectory={new_path}" in new.decode(), "Update changed more than the selected source")
    return new


class Host(promote.Host):
    def cgroup_empty(self, service):
        membership = self.show(service, "ControlGroup") or f"/system.slice/{service}"
        require(membership.startswith("/") and ".." not in membership.split("/"), "Unexpected service cgroup path")
        directory = Path("/sys/fs/cgroup").joinpath(*membership.strip("/").split("/"))
        try:
            metadata = directory.lstat()
        except FileNotFoundError:
            return True
        require(stat.S_ISDIR(metadata.st_mode) and not directory.is_symlink(), "Unexpected cgroup directory")

        def failed(error):
            raise PreviewError("Cannot inspect recursive service cgroup") from error

        for parent, directories, files in os.walk(directory, onerror=failed, followlinks=False):
            require(not any((Path(parent) / name).is_symlink() for name in directories), "Symlink in service cgroup")
            path = Path(parent) / "cgroup.procs"
            require("cgroup.procs" in files and not path.is_symlink(), "Missing or unsafe cgroup process inventory")
            if path.read_text().strip():
                return False
        return True

    def release_inventory(self, release, source):
        release = Path(release)
        require(release.is_dir() and not release.is_symlink() and release.lstat().st_uid == 0 and
                not release.stat().st_mode & 0o222, "Release directory must be root-owned and read-only")
        args = ["git", "-c", f"safe.directory={release}", "-C", str(release)]
        require(self.run(args + ["rev-parse", "HEAD"]) == source, "Release HEAD differs from accepted exact source")
        require(not self.run(args + ["status", "--porcelain", "--untracked-files=all"]), "Release source is dirty")
        names = self.run(args + ["ls-files", "-z"]).split("\0")
        files = {}
        for name in filter(None, names):
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe tracked release path")
            path = release.joinpath(*relative.parts)
            require(path.is_file() and not path.is_symlink() and path.lstat().st_uid == 0 and
                    not path.stat().st_mode & 0o222, "Tracked release file must be root-owned and read-only")
            parent = path.parent
            while parent != release:
                require(not parent.is_symlink() and parent.lstat().st_uid == 0 and not parent.stat().st_mode & 0o222,
                        "Release ancestor must be root-owned and read-only")
                parent = parent.parent
            files[name] = digest(path.read_bytes())
        require("src/gpt2_local/public_chat.py" in files and "src/gpt2_local/chat.py" in files,
                "Release is missing backend source")
        return {"source": source, "files": files}

    def override_replace(self, data, expected):
        require(self.pid(PROD) == 0 and self.show(PROD, "ActiveState") == "inactive" and self.cgroup_empty(PROD),
                "Production must be fully inactive before changing its override")
        promote.frozen_root_file(OVERRIDE)
        before = OVERRIDE.read_bytes()
        require(digest(before) == expected, "Refusing to replace a foreign or changed override")
        metadata = OVERRIDE.lstat()

        def write_checked(content):
            descriptor = os.open(OVERRIDE, os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(descriptor, "wb") as stream:
                opened = os.fstat(stream.fileno())
                require(opened.st_ino == metadata.st_ino and opened.st_uid == 0 and not opened.st_mode & 0o222,
                        "Override inode/ownership changed during replacement")
                os.ftruncate(stream.fileno(), 0)
                require(stream.write(content) == len(content), "Short override write")
                stream.flush()
                os.fsync(stream.fileno())
            promote.frozen_root_file(OVERRIDE)
            require(OVERRIDE.lstat().st_ino == metadata.st_ino and OVERRIDE.read_bytes() == content,
                    "Installed override bytes/inode differ")

        try:
            write_checked(data)
            self.run(["systemctl", "daemon-reload"])
        except BaseException:
            # Restore the exact prior streaming selection before propagating an
            # error. A changed inode is foreign state and must never be touched.
            promote.frozen_root_file(OVERRIDE)
            require(OVERRIDE.lstat().st_ino == metadata.st_ino, "Override changed; manual recovery required")
            write_checked(before)
            self.run(["systemctl", "daemon-reload"])
            raise


class Controller(preview.Controller):
    def __init__(self, host, stage, promotion_stage=None, release=None, source=None):
        super().__init__(host, stage)
        self.promotion_stage = Path(promotion_stage or PROMOTION_STAGE)
        self.release = Path(release or RELEASE)
        self.source = source or NEW_SOURCE

    freeze = staticmethod(promote.Controller.freeze)

    def record(self, phase):
        self.manifest["phase"] = phase
        self.manifest["updated_at"] = preview.timestamp()
        path = self.stage / "manifest.json"
        temporary = self.stage / f".manifest-{os.getpid()}-{time.time_ns()}.json"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(self.manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == "posix":
            descriptor = os.open(self.stage, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def previous(self):
        controller = promote.Controller(self.host, self.promotion_stage)
        controller.load()
        require(controller.manifest["phase"] == "promoted", "Original promotion stage is not promoted")
        return controller

    def prepare(self):
        import pwd
        with self.host.lock():
            require(self.release == RELEASE and self.source == NEW_SOURCE, "Only the accepted pinned release is authorized")
            previous = self.previous()
            require(previous.integrity() == (True, True), "Original streaming selection differs")
            snapshot = self.host.inventory()
            require(snapshot["source"] == OLD_SOURCE and snapshot["files"].get(str(OVERRIDE)) == OLD_OVERRIDE_SHA,
                    "Current streaming source/owned override differs")
            require(snapshot["config_sha256"] == previous.manifest["candidates"]["production-active"],
                    "Live proxy is not the original active production configuration")
            self.inactive(PREVIEW)
            require(self.host.show(PROD, "ActiveState") == "active" and snapshot["production_pid"] > 0 and
                    self.host.model_pids() == {snapshot["production_pid"]}, "Production must be the sole active model")
            self.host.streaming_ready(PROD, True)
            self.host.redirect_check()
            release_inventory = self.host.release_inventory(self.release, self.source)
            active = CONFIG.read_bytes()
            old = OVERRIDE.read_bytes()
            require(digest(old) == OLD_OVERRIDE_SHA, "Old override changed before staging")
            new = updated_dropin(old, self.release)
            self.stage.mkdir(mode=0o711, parents=True, exist_ok=False)
            os.chmod(self.stage, 0o711)
            account = pwd.getpwnam("astra")
            hashes = {}
            for name, data in candidates(active).items():
                path = self.stage / (name + ".caddy")
                self.freeze(path, data, group=account.pw_gid)
                self.host.validate(path)
                hashes[name] = digest(data)
            self.freeze(self.stage / "old.conf", old)
            self.freeze(self.stage / "new.conf", new)
            assembled = Path(self.host.show(PROD, "FragmentPath")).read_bytes()
            for path in self.host.show(PROD, "DropInPaths").split():
                assembled += b"\n" + (new if Path(path) == OVERRIDE else Path(path).read_bytes())
            require(str(OVERRIDE) in self.host.show(PROD, "DropInPaths").split(), "Owned override is not loaded")
            self.freeze(self.stage / PROD, assembled)
            self.host.unit_check(self.stage / PROD)
            snapshots = {}
            for index, (installed, sha) in enumerate(snapshot["files"].items()):
                data = Path(installed).read_bytes()
                require(digest(data) == sha, "Installed file changed during preparation")
                saved = f"snapshot-{index:03}.bytes"
                self.freeze(self.stage / saved, data)
                snapshots[installed] = saved
            self.manifest = {
                "schema_version": 3, "prepared_at": preview.timestamp(), "snapshot": snapshot,
                "candidates": hashes, "old_sha256": digest(old), "new_sha256": digest(new),
                "old_exec": promote.dropin_exec(old)[0], "new_exec": promote.dropin_exec(new)[0],
                "production_pre": promote.dropin_exec(old)[1], "assembled_unit_sha256": digest(assembled),
                "promotion_stage": str(self.promotion_stage),
                "promotion_manifest_sha256": digest((self.promotion_stage / "manifest.json").read_bytes()),
                "release": self.release.as_posix(), "release_inventory": release_inventory, "snapshots": snapshots,
                "helper_sha256": digest(HERE.read_bytes()),
                "promotion_helper_sha256": digest(HERE.with_name("lightsail-promote.py").read_bytes()),
                "preview_helper_sha256": digest(HERE.with_name("lightsail-preview.py").read_bytes()),
            }
            self.record("prepared")
            require(self.host.inventory() == snapshot and CONFIG.read_bytes() == active and
                    self.host.release_inventory(self.release, self.source) == release_inventory,
                    "Live/release state changed during preparation")
            self.host.chess_health()
            self.emit("version_update_prepared_no_live_changes", stage=str(self.stage), old_source=OLD_SOURCE,
                      new_source=self.source, release=str(self.release))

    def load(self):
        path = self.stage / "manifest.json"
        promote.private_root_file(path)
        self.manifest = json.loads(path.read_text())
        require(self.manifest.get("schema_version") == 3 and set(self.manifest["candidates"]) == NAMES,
                "Unsupported version-update manifest")
        require(self.manifest["release"] == RELEASE.as_posix() and self.manifest["release_inventory"]["source"] == NEW_SOURCE and
                self.manifest["old_sha256"] == OLD_OVERRIDE_SHA and self.manifest["snapshot"]["source"] == OLD_SOURCE,
                "Pinned version-update selection differs")
        self.release, self.source = Path(self.manifest["release"]), NEW_SOURCE
        self.promotion_stage = Path(self.manifest["promotion_stage"])
        require(digest((self.promotion_stage / "manifest.json").read_bytes()) == self.manifest["promotion_manifest_sha256"],
                "Original promotion manifest changed")
        self.previous()
        for file, field in ((HERE, "helper_sha256"), (HERE.with_name("lightsail-promote.py"), "promotion_helper_sha256"),
                            (HERE.with_name("lightsail-preview.py"), "preview_helper_sha256")):
            require(digest(file.read_bytes()) == self.manifest[field], "Reviewed update tooling changed")
        staged = {**{name + ".caddy": sha for name, sha in self.manifest["candidates"].items()},
                  "old.conf": self.manifest["old_sha256"], "new.conf": self.manifest["new_sha256"],
                  PROD: self.manifest["assembled_unit_sha256"]}
        staged.update({saved: self.manifest["snapshot"]["files"][installed]
                       for installed, saved in self.manifest["snapshots"].items()})
        for name, sha in staged.items():
            path = self.stage / name
            promote.frozen_root_file(path)
            require(digest(path.read_bytes()) == sha, "Immutable staged update bytes changed")

    def integrity(self, *, original=False):
        current, saved = self.host.inventory(), self.manifest["snapshot"]
        files, expected = dict(current["files"]), dict(saved["files"])
        owned = files.pop(str(OVERRIDE), None)
        expected.pop(str(OVERRIDE))
        require(files == expected, "Preserved units/drop-ins/environments/guards changed")
        promote.frozen_root_file(OVERRIDE)
        require(owned in (self.manifest["old_sha256"], self.manifest["new_sha256"]), "Owned override changed")
        mode = current["production_exec"] == self.manifest["new_exec"]
        require(current["production_exec"] == self.manifest["new_exec" if mode else "old_exec"] and
                current["production_pre"] == self.manifest["production_pre"], "Loaded production command/startup guard changed")
        properties = dict(saved["properties"])
        if mode:
            properties[PROD] = properties[PROD].replace("WorkingDirectory=" + str(preview.CHECKOUT),
                                                       "WorkingDirectory=" + self.release.as_posix())
        require(current["properties"] == properties, "Loaded production account/environment/resource settings changed")
        for field in ("source", "original_source", "caddy_pid", "inode", "config_attributes", "chess_pids"):
            require(current[field] == saved[field], f"Preserved state changed: {field}")
        require(current["config_sha256"] in self.manifest["candidates"].values(), "Proxy changed outside this version update")
        require(self.host.release_inventory(self.release, self.source) == self.manifest["release_inventory"],
                "Pinned release content changed")
        require(digest((self.promotion_stage / "manifest.json").read_bytes()) == self.manifest["promotion_manifest_sha256"],
                "Original promotion stage changed")
        self.host.chess_health()
        return owned == self.manifest["new_sha256"], mode

    def inactive(self, service):
        require(self.host.pid(service) == 0 and self.host.show(service, "ActiveState") == "inactive" and self.host.cgroup_empty(service),
                "Model unit is not fully inactive with an empty recursive cgroup")

    def exclusive(self):
        self.inactive(PREVIEW)
        require(self.host.model_pids() == {self.host.pid(PROD)} - {0}, "Unexpected GPT-2 process prevents version transition")

    def stop_active(self):
        self.exclusive()
        self.apply("gated")
        self.host.gate_check(preview.PROD_ORIGIN)
        selected_pid = self.host.pid(PROD)
        if selected_pid:
            self.retain(PROD)
            require(self.host.pid(PROD) == selected_pid and self.host.show(PROD, "ActiveState") == "active",
                    "Production process changed during result retention; no stop allowed")
        else:
            require(self.host.cgroup_empty(PROD), "PID-zero production retains processes; no force stop allowed")
        # The retention window is deliberately long. Recheck source, files,
        # loaded properties, proxy gate, and model exclusivity before any stop.
        self.integrity()
        self.exclusive()
        self.host.gate_check(preview.PROD_ORIGIN)
        # Cancel scheduled retries even if the failed service has MainPID zero.
        self.host.stop(PROD)
        self.inactive(PROD)

    def override(self, new):
        disk, effective = self.integrity()
        target = self.manifest["new_sha256" if new else "old_sha256"]
        path = self.stage / ("new.conf" if new else "old.conf")
        promote.frozen_root_file(path)
        data = path.read_bytes()
        require(digest(data) == target, "Override candidate changed before selection")
        self.host.unit_check(self.stage / PROD)
        if disk is not new or effective is not new:
            self.host.override_replace(data, self.manifest["new_sha256" if disk else "old_sha256"])
        require(self.integrity() == (new, new), "Production did not load the selected release")
        self.record("new_selected" if new else "old_selected")

    def restore_old(self):
        self.integrity()
        self.stop_active()
        self.override(False)
        self.host.start(PROD)
        self.host.streaming_ready(PROD, True)
        self.host.gate_check(preview.PROD_ORIGIN)
        self.exclusive()
        self.apply("active")
        self.host.streaming_ready(PROD, True)
        self.host.redirect_check()
        self.integrity()
        # Only the active bytes match the original stage's proxy candidates.
        # Its manifest is never rewritten by this update helper.
        require(self.previous().integrity() == (True, True), "Original polling rollback contract was not restored")
        self.record("old_restored")
        self.emit("previous_streaming_release_restored", source=OLD_SOURCE, production_pid=self.host.pid(PROD),
                  earlier_polling_rollback_available=True)

    def update(self):
        self.load()
        with self.host.lock():
            require(self.manifest["phase"] == "prepared" and self.integrity() == (False, False),
                    "Version update already attempted or current selection changed")
            self.exclusive()
            require(self.host.pid(PROD) == self.manifest["snapshot"]["production_pid"] > 0 and
                    self.host.show(PROD, "ActiveState") == "active", "Prepared production process changed")
            try:
                self.stop_active()
                self.override(True)
                self.host.start(PROD)
                self.host.streaming_ready(PROD, True)
                self.host.gate_check(preview.PROD_ORIGIN)
                self.exclusive()
                self.apply("active")
                self.host.streaming_ready(PROD, True)
                self.host.redirect_check()
                self.integrity()
                self.record("updated")
                self.emit("production_version_updated", source=self.source, release=str(self.release),
                          production_pid=self.host.pid(PROD), override_sha256=self.manifest["new_sha256"])
            except BaseException:
                self.emit("version_update_failed_attempting_previous_streaming_recovery")
                try:
                    self.restore_old()
                except BaseException as recovery:
                    try:
                        self.apply("gated")
                        self.host.gate_check(preview.PROD_ORIGIN)
                    except BaseException as gating:
                        self.emit("gate_requires_operator", error=str(gating))
                    self.emit("recovery_requires_operator", error=str(recovery), stage=str(self.stage))
                    raise PreviewError("Cannot safely restore previous streaming release; leave gated and inspect accepted work") from recovery
                raise

    def rollback(self):
        self.load()
        with self.host.lock():
            try:
                self.restore_old()
            except BaseException:
                # A readiness/integrity error after restoration must not leave
                # an uncertain build accepting fresh work. Never force-stop it.
                try:
                    self.apply("gated")
                    self.host.gate_check(preview.PROD_ORIGIN)
                except BaseException as gating:
                    self.emit("gate_requires_operator", error=str(gating))
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--update", action="store_true")
    modes.add_argument("--rollback", action="store_true")
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--promotion-stage", type=Path, help="unchanged original promotion stage; --prepare only")
    args = parser.parse_args()
    require(os.name == "posix" and os.geteuid() == 0, "Run installed Python 3.12 tooling as root on Lightsail")
    require(args.prepare or args.promotion_stage is None, "--promotion-stage applies only to --prepare")
    controller = Controller(Host(), args.stage.resolve(), args.promotion_stage.resolve() if args.promotion_stage else None)
    if args.prepare:
        controller.prepare()
    elif args.update:
        controller.update()
    else:
        controller.rollback()


if __name__ == "__main__":
    main()
