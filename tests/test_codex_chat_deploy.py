from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

from gpt2_local.codex_chat_bridge import DEFAULT_ACTION_TIMEOUT_SECONDS


class SandboxLauncherTests(unittest.TestCase):
    def test_independent_sandbox_deadline_allows_bridge_cleanup(self):
        launcher = (Path(__file__).resolve().parents[1] / "deploy" / "codex-chat" / "sandbox-launcher.sh").read_text()
        timeout = re.search(r"/usr/bin/timeout --signal=TERM --kill-after=5s (\d+)s /usr/bin/docker run", launcher)
        self.assertIsNotNone(timeout, "Cannot locate the sandbox's independent timeout")
        self.assertEqual(int(timeout.group(1)), DEFAULT_ACTION_TIMEOUT_SECONDS + 10)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "Requires a Linux Bash shell")
    def test_background_app_server_retains_json_rpc_stdin(self):
        launcher = (Path(__file__).resolve().parents[1] / "deploy" / "codex-chat" / "sandbox-launcher.sh").read_text()
        # Reuse the real async invocation's redirection with a harmless stdin
        # reader. Dropping <&0 makes Bash send /dev/null to that child, which
        # was enough to leave an otherwise healthy Codex app-server silent.
        suffix = re.search(r'"\$@"([^\n]*)\nchild=\$!', launcher)
        self.assertIsNotNone(suffix, "Cannot locate sandbox launcher's background invocation")
        script = "set -euo pipefail\n/bin/cat" + suffix.group(1) + '\nchild=$!\nwait "$child"\n'
        payload = '{"jsonrpc":"2.0","id":1,"method":"initialize"}\n'
        result = subprocess.run([shutil.which("bash"), "-c", script], input=payload,
                                text=True, capture_output=True, timeout=5, check=True)
        self.assertEqual(result.stdout, payload)


if __name__ == "__main__":
    unittest.main()
