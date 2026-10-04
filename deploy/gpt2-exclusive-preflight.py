"""Fail closed unless the opposing GPT-2 service and its whole cgroup are stopped.

Run with system Python 3.9 or later. This guard only reads unit/cgroup state;
the maintenance helper owns submission gates, drain, retention and service control.
"""
import argparse
import os
from pathlib import Path
import stat
import subprocess
import sys


UNITS = ("gpt2-public.service", "gpt2-preview.service")
PROPERTIES = ("LoadState", "ActiveState", "MainPID", "ControlGroup")


def unit_properties(unit):
    output = subprocess.check_output(
        ["/usr/bin/systemctl", "show", unit, *["--property=" + name for name in PROPERTIES]],
        text=True, timeout=10,
    )
    values = dict(line.split("=", 1) for line in output.splitlines())
    if set(values) != set(PROPERTIES):
        raise RuntimeError("Opposing unit properties were incomplete")
    return values


def require_stopped(values):
    if (values.get("LoadState") != "loaded" or values.get("ActiveState") != "inactive"
            or values.get("MainPID") != "0"):
        raise RuntimeError("Opposing GPT-2 unit must be loaded, fully inactive, and have MainPID=0")


def verify_other_unit(unit, *, cgroup_root=Path("/sys/fs/cgroup"), query=unit_properties):
    if unit not in UNITS:
        raise RuntimeError("Unrecognized opposing GPT-2 unit")
    before = query(unit)
    require_stopped(before)
    membership = before.get("ControlGroup") or "/system.slice/" + unit
    if (not membership.startswith("/") or membership == "/"
            or any(part in ("", ".", "..") for part in membership.split("/")[1:])):
        raise RuntimeError("Opposing unit cgroup path is invalid")
    directory = cgroup_root.joinpath(*membership.split("/")[1:])
    try:
        attributes = directory.lstat()
    except FileNotFoundError:
        attributes = None  # systemd normally removes the fully stopped cgroup.
    if attributes is not None:
        if not stat.S_ISDIR(attributes.st_mode):
            raise RuntimeError("Opposing unit cgroup is not a directory")

        def walk_error(error):
            raise error

        for name, children, _ in os.walk(directory, onerror=walk_error, followlinks=False):
            current = Path(name)
            if any((current / child).is_symlink() for child in children):
                raise RuntimeError("Unexpected cgroup symlink")
            if (current / "cgroup.procs").read_text().strip():
                raise RuntimeError("Opposing GPT-2 cgroup still has processes")
    after = query(unit)
    require_stopped(after)
    if before != after:
        raise RuntimeError("Opposing unit state changed during the startup check")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("other_unit", choices=UNITS)
    args = parser.parse_args(argv)
    try:
        verify_other_unit(args.other_unit)
    except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
        print("GPT-2 startup refused: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
