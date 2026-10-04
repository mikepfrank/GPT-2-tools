"""Fail closed unless the production GPT-2 unit is fully stopped."""
from pathlib import Path
import subprocess
import sys


def property_value(name):
    return subprocess.check_output(
        ['/usr/bin/systemctl', 'show', '--property=' + name, '--value', 'gpt2-public.service'],
        text=True,
    ).strip()


def main():
    try:
        if property_value('ActiveState') != 'inactive' or property_value('MainPID') != '0':
            raise RuntimeError('Production GPT-2 must be fully inactive with MainPID=0')
        cgroup = property_value('ControlGroup') or '/system.slice/gpt2-public.service'
        processes = Path('/sys/fs/cgroup') / cgroup.lstrip('/') / 'cgroup.procs'
        if processes.exists() and processes.read_text().strip():
            raise RuntimeError('Production GPT-2 still has cgroup processes')
    except (OSError, subprocess.SubprocessError, RuntimeError) as error:
        print('Preview model startup refused: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
