#!/usr/bin/env python3
"""Operator smoke test against the private beta; keeps its transcript private."""
import argparse
import http.client
import json
from pathlib import Path
import time
import uuid

OPS = Path("/var/lib/gpt2-codex-ops")
STATE = OPS / "smoke-session.json"
ORIGIN = "https://codex-test.gpt2-chat.com"


def api(method, route, body=None, cookie=None):
    connection = http.client.HTTPConnection("127.0.0.1", 8900, timeout=15)
    headers = {"Origin": ORIGIN, "Content-Type": "application/json"}
    if cookie:
        headers["Cookie"] = cookie
    try:
        connection.request(method, route, body=json.dumps(body) if body is not None else None, headers=headers)
        response = connection.getresponse()
        value = json.loads(response.read())
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}: {value}")
        return value, response.getheader("Set-Cookie")
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", default="Codex, please say hello in one short sentence. Do not query GPT-2 or call wait_for_human for this smoke test.")
    parser.add_argument("--target", default="codex", choices=("codex", "gpt2", "auto"))
    parser.add_argument("--new", action="store_true")
    args = parser.parse_args()
    path = next(line.split("=", 1)[1] for line in Path("/etc/gpt2-codex.env").read_text().splitlines() if line.startswith("GPT2_CODEX_BASE_PATH="))
    if STATE.exists() and not args.new:
        saved = json.loads(STATE.read_text())
    else:
        state, cookie = api("POST", path + "/api/sessions", {"example_rounds": 0})
        saved = {"session_id": state["session_id"], "cookie": cookie.split(";", 1)[0]}
        STATE.write_text(json.dumps(saved))
        STATE.chmod(0o600)
    endpoint = path + "/api/sessions/" + saved["session_id"]
    api("POST", endpoint + "/presence", {"connected": True}, saved["cookie"])
    api("POST", endpoint + "/control", {"paused": False}, saved["cookie"])
    state, _ = api("POST", endpoint + "/message", {"message": args.message, "target": args.target, "request_id": uuid.uuid4().hex}, saved["cookie"])
    start = last_pulse = time.monotonic()
    last_status = None
    while time.monotonic() - start < 210:
        state, _ = api("GET", endpoint, cookie=saved["cookie"])
        if state["status"] != last_status:
            print("status:", state["status"], flush=True)
            last_status = state["status"]
        if state["status"] == "idle":
            break
        if time.monotonic() - last_pulse >= 10:
            api("POST", endpoint + "/presence", {"connected": True}, saved["cookie"])
            last_pulse = time.monotonic()
        time.sleep(1)
    api("POST", endpoint + "/control", {"paused": True}, saved["cookie"])
    exported, _ = api("GET", endpoint + "/export", cookie=saved["cookie"])
    (OPS / "smoke-export.json").write_text(json.dumps(exported, ensure_ascii=False))
    (OPS / "smoke-export.json").chmod(0o600)
    print("messages:", [(item["role"], item["content"]) for item in state["messages"][-5:]])
    print("context tokens:", state["context_tokens"], "budget:", state["budget"])
    if state.get("error") or state["status"] != "idle":
        raise RuntimeError(state.get("error") or "Smoke turn did not finish")


if __name__ == "__main__":
    main()
