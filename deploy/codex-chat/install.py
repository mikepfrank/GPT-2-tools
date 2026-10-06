#!/usr/bin/env python3
"""Install/start only the separate beta. Publish Caddy only after private checks.

Run as root from /home/gpt-2/GPT-2-codex. The URL capability and proxy backup
remain in root-owned storage, never source control. --publish locks and appends
one site, validates, reloads in place, and rolls back on failed readiness.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import time
import urllib.request

SOURCE = Path("/home/gpt-2/GPT-2-codex")
TEMPLATES = SOURCE / "deploy/codex-chat"
OPS = Path("/var/lib/gpt2-codex-ops")
ENV = Path("/etc/gpt2-codex.env")
CONFIG = Path("/home/astra/.config/astra-chess/Caddyfile")
CADDY = "/home/astra/.local/bin/caddy"
OTHERS = ("gpt2-public.service", "astra-chess.service", "or-chess.service", "astra-caddy.service")
BETA_ORIGIN = "https://codex-test.gpt2-chat.com"
READINESS_TIMEOUT = 100
# Existing canonical GPT-2 and chess health routes recorded in the handoffs.
EXISTING_HEALTH = (
    ("https://gpt2-chat.com/api/health", {"status": "ready"}),
    ("https://astraplayschess.com/health", {"ok": True}),
    ("https://arcturuschess.com/health", {"ok": True}),
    ("https://arcturus.astraplayschess.com/health", {"ok": True}),
)
CADDY_ENV = ["sudo", "-u", "astra", "env", "HOME=/home/astra/.local/share/astra-caddy",
             "XDG_DATA_HOME=/home/astra/.local/share/astra-caddy",
             "XDG_CONFIG_HOME=/home/astra/.local/share/astra-caddy/config", CADDY]


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def other_pids():
    return {unit: run(["systemctl", "show", unit, "-p", "MainPID", "--value"]) for unit in OTHERS}


def install_file(source, target, mode=0o755):
    target = Path(target)
    target.write_bytes(Path(source).read_bytes().replace(b"\r\n", b"\n"))
    target.chmod(mode)
    os.chown(target, 0, 0)


def install():
    before = other_pids()
    assert all(int(pid) > 0 for pid in before.values()), "An existing service is not running"
    assert run(["id", "-u", "gpt-2"]) == "1015" and run(["id", "-g", "gpt-2"]) == "1018"
    memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    assert int(memory["MemAvailable"].split()[0]) >= 8 * 1024 * 1024, "Insufficient startup memory"
    run(["docker", "image", "inspect", "gpt2-codex-runtime:0.154.0"])
    OPS.mkdir(parents=True, exist_ok=True, mode=0o700)
    for source, target, mode in (("sandbox-launcher.sh", "/usr/local/libexec/gpt2-codex-sandbox", 0o755),
                                 ("network.sh", "/usr/local/libexec/gpt2-codex-network", 0o755),
                                 ("cleanup.sh", "/usr/local/libexec/gpt2-codex-cleanup", 0o755),
                                 ("seccomp.json", "/usr/local/libexec/gpt2-codex-seccomp.json", 0o644)):
        install_file(TEMPLATES / source, target, mode)
    rule = Path("/etc/sudoers.d/gpt2-codex")
    rule.write_text('Defaults!/usr/local/libexec/gpt2-codex-sandbox env_keep += "GPT2_PROXY_TOKEN"\n'
                    'gpt-2 ALL=(root) NOPASSWD: /usr/local/libexec/gpt2-codex-sandbox *\n')
    rule.chmod(0o440)
    run(["visudo", "-cf", str(rule)])
    for name in ("gpt2-codex.service", "gpt2-codex-network.service"):
        install_file(TEMPLATES / name, Path("/etc/systemd/system") / name, 0o644)
    if not ENV.exists():
        capability = "/" + secrets.token_urlsafe(24)
        launcher = ["sudo", "-n", "/usr/local/libexec/gpt2-codex-sandbox", "{session_id}", "{codex_home}", "{workspace}", "{base_url}"]
        ENV.write_text("PYTHONPATH=/home/gpt-2/GPT-2-codex/src\n"
                       "HF_HOME=/home/gpt-2/.cache/gpt2-public/huggingface\n"
                       "TZ=America/Chicago\nGPT2_CODEX_DAILY_LIMIT_USD=10\nGPT2_CODEX_ORIGIN=https://codex-test.gpt2-chat.com\n"
                       f"GPT2_CODEX_BASE_PATH={capability}\n"
                       f"GPT2_CODEX_LAUNCHER='{json.dumps(launcher)}'\n")
        ENV.chmod(0o600)
    run(["systemctl", "daemon-reload"])
    run(["systemctl", "enable", "--now", "gpt2-codex-network.service", "gpt2-codex.service"])
    assert other_pids() == before, "An existing service PID changed"
    (OPS / "initial-services.json").write_text(json.dumps(before))
    print("Separate beta services installed; existing service PIDs unchanged.")


def base_path():
    path = next(line.split("=", 1)[1] for line in ENV.read_text().splitlines() if line.startswith("GPT2_CODEX_BASE_PATH="))
    if not re.fullmatch(r"/[A-Za-z0-9_-]{16,80}", path):
        raise RuntimeError("The beta URL component is invalid")
    return path


def check_health(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.status == 200 and json.load(response).get("application") == "gpt2-codex"


def check_existing_health():
    for url, expected in EXISTING_HEALTH:
        with urllib.request.urlopen(url, timeout=10) as response:
            if response.status != 200:
                return False
            payload = json.load(response)
            if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
                return False
    return True


def wait_for_health(path, *, include_beta):
    deadline = time.monotonic() + READINESS_TIMEOUT
    while True:
        try:
            # Check old sites even while the new domain awaits its certificate.
            existing_ready = check_existing_health()
            beta_ready = not include_beta or check_health(BETA_ORIGIN + path + "/health")
            if existing_ready and beta_ready:
                return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("HTTPS readiness timed out for the beta or existing services" if include_beta
                               else "Existing services failed HTTPS readiness after rollback")
        time.sleep(2)


def beta_block(path):
    return (f"\n# Isolated GPT-2/Codex closed beta.\ncodex-test.gpt2-chat.com {{\n"
            f"    @beta path {path} {path}/*\n    handle @beta {{\n"
            "        reverse_proxy 127.0.0.1:8900 {\n            flush_interval -1\n"
            "            header_up Host {http.request.hostport}\n"
            "            header_up -Forwarded\n            header_up -X-Forwarded-For\n"
            "        }\n    }\n    handle {\n        respond \"Not found\" 404\n    }\n}\n").encode()


def mounted_config(pid):
    return Path(f"/proc/{pid}/root{CONFIG}")


def verify_proxy_bytes(expected, pid):
    if CONFIG.read_bytes() != expected or mounted_config(pid).read_bytes() != expected:
        raise RuntimeError("Host and mounted proxy configuration bytes disagree")


def write_proxy_bytes(data):
    # Caddy bind-mounts this file individually. Replacing its directory entry
    # would leave the running process reading the previous inode.
    with CONFIG.open("r+b") as stream:
        stream.seek(0)
        stream.write(data)
        stream.truncate()
        stream.flush()
        os.fsync(stream.fileno())


def validate_candidate(path):
    run(CADDY_ENV + ["validate", "--config", str(path), "--adapter", "caddyfile"])


def stage_candidate(candidate, data):
    candidate.write_bytes(data)
    stat = CONFIG.stat()
    os.chown(candidate, stat.st_uid, stat.st_gid)
    candidate.chmod(stat.st_mode & 0o777)
    validate_candidate(candidate)


def save_backup(original):
    OPS.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = OPS / "Caddyfile.original"
    with backup.open("wb") as stream:
        stream.write(original)
        stream.flush()
        os.fsync(stream.fileno())
    backup.chmod(0o600)
    (OPS / "Caddyfile.original.sha256").write_text(hashlib.sha256(original).hexdigest())


def reload_proxy():
    run(["systemctl", "kill", "--signal=SIGUSR1", "--kill-who=main", "astra-caddy.service"])


@contextmanager
def proxy_lock():
    with Path("/run/lock/astra-caddy-config.lock").open("a") as lock:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Shared proxy maintenance is busy")
                time.sleep(.25)
        yield


def publish():
    path = base_path()
    if not check_health("http://127.0.0.1:8900" + path + "/health"):
        raise RuntimeError("Private beta is not ready")
    with proxy_lock():
        before = other_pids()
        if not all(int(pid) > 0 for pid in before.values()):
            raise RuntimeError("An existing service is not running")
        original = CONFIG.read_bytes()
        pid = before["astra-caddy.service"]
        verify_proxy_bytes(original, pid)
        if not check_existing_health():
            raise RuntimeError("An existing HTTPS service is not ready")
        expected_block = beta_block(path)
        already_present = b"codex-test.gpt2-chat.com" in original
        if already_present and (expected_block not in original or original.count(b"codex-test.gpt2-chat.com") != 1):
            raise RuntimeError("The existing beta proxy block differs; operator review is required")
        candidate = CONFIG.with_name("Caddyfile.gpt2-codex-candidate")
        try:
            # Validate both snapshots before any live bytes are changed. The
            # same readable candidate is reused to validate rollback bytes.
            validate_candidate(CONFIG)
            if already_present:
                wait_for_health(path, include_beta=True)
                verify_proxy_bytes(original, pid)
                if other_pids() != before:
                    raise RuntimeError("An existing service PID changed")
            else:
                proposed = original + expected_block
                stage_candidate(candidate, proposed)
                save_backup(original)
                try:
                    write_proxy_bytes(proposed)
                    verify_proxy_bytes(proposed, pid)
                    reload_proxy()
                    wait_for_health(path, include_beta=True)
                    verify_proxy_bytes(proposed, pid)
                    if other_pids() != before:
                        raise RuntimeError("An existing service PID changed")
                except BaseException:
                    try:
                        stage_candidate(candidate, original)
                        write_proxy_bytes(original)
                        verify_proxy_bytes(original, pid)
                        reload_proxy()
                        wait_for_health(path, include_beta=False)
                        if other_pids() != before:
                            raise RuntimeError("An existing service PID changed during rollback")
                    except BaseException as rollback_error:
                        raise RuntimeError("Beta publication failed and rollback could not be verified; operator review is required") from rollback_error
                    raise
        finally:
            candidate.unlink(missing_ok=True)
    print("Beta HTTPS ready: " + BETA_ORIGIN + path + "/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args()
    assert os.geteuid() == 0, "Run this operator helper as root"
    os.umask(0o077)
    if args.install:
        install()
    if args.publish:
        publish()
