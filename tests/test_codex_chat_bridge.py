import json
import os
from pathlib import Path
import sys
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch

from gpt2_local.codex_chat_bridge import (DEFAULT_ACTION_TIMEOUT_SECONDS, CodexChatBridge,
                                         CodexError, _safe_environment, _stop_process, validate_message)
from gpt2_local.codex_chat_budget import DailyBudgetExceeded


FAKE_SERVER = r'''
import json,os,sys,tomllib,time
scenario=sys.argv[1]
if scenario=="rpc-error":
    print("PRIVATE STARTUP DIAGNOSTIC",file=sys.stderr,flush=True)
def send(value):
    print(json.dumps(value),flush=True)
for line in sys.stdin:
    value=json.loads(line)
    method=value.get("method")
    params=value.get("params",{})
    if method=="initialize":
        if scenario=="rpc-error":
            send({"id":value["id"],"error":{"code":-32600,"message":"PRIVATE RPC DETAIL"}})
        else:
            send({"id":value["id"],"result":{"codexHome":os.environ["CODEX_HOME"]}})
    elif method=="config/read":
        with open(os.path.join(os.environ["CODEX_HOME"],"config.toml"),"rb") as source:
            config=tomllib.load(source)
        if scenario=="config-mismatch":
            config["model"]="gpt-6-astra"
        if scenario=="ambient-mcp":
            config["mcp_servers"]={"ambient":{"enabled":True}}
        send({"id":value["id"],"result":{"config":config}})
    elif method in ("thread/start","thread/resume"):
        send({"id":value["id"],"result":{"thread":{"id":"test-thread"},
            "model":"gpt-6.1-sol","modelProvider":"gpt2_gateway","reasoningEffort":"medium",
            "approvalPolicy":"never","approvalsReviewer":"user",
            "runtimeWorkspaceRoots":[params["cwd"]],
            "sandbox":{"type":"workspaceWrite","networkAccess":False},"instructionSources":[]}})
    elif method=="turn/start":
        send({"id":value["id"],"result":{"turn":{"id":"test-turn"}}})
        send({"method":"turn/started","params":{"threadId":"test-thread","turn":{"id":"test-turn"}}})
        if scenario=="terminal-error":
            send({"method":"error","params":{"threadId":"test-thread","turnId":"test-turn",
                "message":"PRIVATE TERMINAL ERROR","willRetry":False}})
            continue
        if scenario=="hang":
            time.sleep(100)
        if scenario in ("child","child-out-of-order"):
            if scenario=="child":
                send({"method":"thread/started","params":{"thread":{"id":"test-child"}}})
            send({"method":"turn/started","params":{"threadId":"test-child","turn":{"id":"child-turn"}}})
            send({"method":"thread/tokenUsage/updated","params":{"threadId":"test-child",
                  "tokenUsage":{"total":{"totalTokens":999999}}}})
            send({"method":"item/completed","params":{"threadId":"test-child","turnId":"child-turn",
                  "item":{"type":"agentMessage","phase":"final_answer","text":"PRIVATE CHILD FINAL"}}})
            send({"method":"error","params":{"turnId":"child-turn","message":"PRIVATE CHILD ERROR","willRetry":False}})
            send({"method":"turn/completed","params":{"threadId":"test-child","turn":{"id":"child-turn","status":"failed"}}})
            send({"id":1000,"method":"item/tool/call","params":{"threadId":"test-child","turnId":"child-turn",
                  "tool":"emit_message","arguments":{"text":"CHILD MUST STAY PRIVATE"}}})
            continue
        send({"id":999,"method":"item/tool/call","params":{"threadId":"test-thread","turnId":"test-turn",
              "tool":"forbidden" if scenario=="forbidden" else "emit_message","arguments":{"text":"A short public paragraph."}}})
    elif value.get("id") in (1000,1001,1002):
        if value.get("result",{}).get("success") is not False:
            send({"method":"error","params":{"threadId":"test-thread","message":"Child tool unexpectedly accepted","willRetry":False}})
            continue
        if value["id"]<1002:
            send({"id":value["id"]+1,"method":"item/tool/call","params":{"threadId":"test-child","turnId":"child-turn",
                  "tool":"query_gpt2" if value["id"]==1000 else "wait_for_human","arguments":{}}})
        else:
            send({"id":999,"method":"item/tool/call","params":{"threadId":"test-thread","turnId":"test-turn",
                  "tool":"emit_message","arguments":{"text":"A short public paragraph."}}})
    elif value.get("id")==999:
        send({"method":"thread/tokenUsage/updated","params":{"threadId":"test-thread",
              "tokenUsage":{"total":{"totalTokens":123}}}})
        send({"method":"item/completed","params":{"threadId":"test-thread","turnId":"test-turn",
              "item":{"id":"private-final","type":"agentMessage","phase":"final_answer","text":"PRIVATE FINAL DO NOT PUBLISH"}}})
        send({"method":"turn/completed","params":{"threadId":"test-thread","turn":{"id":"test-turn","status":"completed"}}})
'''


class FakeGateway:
    base_url = "http://127.0.0.1:9999/v1"
    def __init__(self):
        self.activations=[]
        self.lifetimes=[]
        self.revocations=[]
    def activate(self, session_id, timeout_seconds):
        self.activations.append(session_id)
        self.lifetimes.append(timeout_seconds)
        return "scoped-test-token"
    def revoke(self, token):
        self.revocations.append(token)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.script=self.root/"fake_server.py"
        self.script.write_text(FAKE_SERVER,encoding="utf-8")
        self.gateway=FakeGateway()

    def bridge(self, scenario="normal", timeout=10):
        # This is a fake stdlib app-server subprocess. It cannot call a model.
        return CodexChatBridge(self.root/"data",[sys.executable,str(self.script),scenario],
                               self.gateway,timeout_seconds=timeout)

    def test_emit_only_public_and_persisted_thread_resume(self):
        calls=[]
        def callback(name,args):
            calls.append((name,args))
            return {"ok":True,"gpt2_context":"Human> Hi\n\nCodex> A short public paragraph."}
        bridge=self.bridge()
        result=bridge.run("chat", "Human addresses Codex",callback)
        self.assertEqual(result["thread_id"],"test-thread")
        self.assertEqual(result["usage"]["total_tokens"],123)
        self.assertEqual(result["final"],"PRIVATE FINAL DO NOT PUBLISH")
        self.assertEqual([args["text"] for name,args in calls if name=="emit_message"],
                         ["A short public paragraph."])
        state=json.loads((self.root/"data/players/chat/bridge-state.json").read_text())
        self.assertEqual(state["thread_id"],"test-thread")
        self.assertEqual(self.gateway.revocations,["scoped-test-token"])
        resumed=bridge.run("chat","New human message",callback)
        self.assertEqual(resumed["thread_id"],"test-thread")
        self.assertEqual(resumed["usage"]["total_tokens"],0)

    def test_default_action_deadline_and_scoped_gateway_lifetime_are_ten_minutes(self):
        # Completing a fake action exercises the production default and gateway
        # activation without waiting ten minutes or calling either model.
        bridge=CodexChatBridge(self.root/"data",[sys.executable,str(self.script),"normal"],self.gateway)
        self.assertEqual(DEFAULT_ACTION_TIMEOUT_SECONDS,600)
        self.assertEqual(bridge.timeout_seconds,600)
        bridge.run("chat","Hello",lambda *_:{})
        self.assertEqual(self.gateway.lifetimes,[600])
        config=tomllib.loads((self.root/"data/players/chat/codex-home/config.toml").read_text())
        self.assertEqual(config["model_providers"]["gpt2_gateway"]["stream_idle_timeout_ms"],180000)

    def test_timeout_upper_bound_remains_ten_minutes(self):
        for timeout in (0,-1,601):
            with self.subTest(timeout=timeout),self.assertRaises(ValueError):
                CodexChatBridge(self.root/"data",[sys.executable],self.gateway,timeout_seconds=timeout)
        self.assertEqual(self.bridge(timeout=600).timeout_seconds,600)

    def test_policy_mismatch_and_ambient_mcp_fail_closed(self):
        for scenario in ("config-mismatch","ambient-mcp","forbidden"):
            calls=[]
            with self.subTest(scenario=scenario),self.assertRaises(CodexError):
                self.bridge(scenario).run(scenario,"Hello",lambda name,args: calls.append(name) or {})
            self.assertNotIn("emit_message",calls)
        self.assertEqual(len(self.gateway.revocations),3)

    def test_cancellation_and_timeout_revoke_token(self):
        with self.assertRaisesRegex(CodexError,"cancelled"):
            self.bridge().run("cancelled","Hello",lambda *_:{},cancelled=lambda:True)
        with self.assertRaisesRegex(CodexError,"timed out") as captured:
            self.bridge("hang",timeout=.3).run("timed-out","Hello",lambda *_:{})
        self.assertEqual(captured.exception.code,"codex_timeout")
        self.assertEqual(len(self.gateway.revocations),2)

    def test_recovery_thread_conflict_is_rejected_before_launch(self):
        bridge=self.bridge()
        bridge.run("chat","Hello",lambda *_:{})
        with self.assertRaisesRegex(CodexError,"disagree"):
            bridge.run("chat","Hello",lambda *_:{},thread_id="different-thread")
        self.assertEqual(len(self.gateway.activations),1)

    def test_operator_key_and_ambient_credentials_never_enter_child(self):
        with patch.dict(os.environ,{"OPENAI_API_KEY":"operator-secret","AWS_SECRET_ACCESS_KEY":"cloud-secret",
                                    "CODEX_HOME":"ambient-home","SMTP_PASSWORD":"mail-secret"}):
            env=_safe_environment(self.root/"private",self.root/"workspace","scoped-test-token")
        self.assertNotIn("OPENAI_API_KEY",env)
        self.assertNotIn("AWS_SECRET_ACCESS_KEY",env)
        self.assertNotIn("SMTP_PASSWORD",env)
        self.assertNotIn("operator-secret",env.values())
        self.assertEqual(env["GPT2_PROXY_TOKEN"],"scoped-test-token")
        self.assertEqual(env["CODEX_HOME"],str(self.root/"private"))

    def test_message_and_launcher_limits(self):
        for text in ("","a\nb","a\rb","x"*701,"word "*101):
            with self.subTest(text=text[:20]),self.assertRaises(ValueError):
                validate_message(text)
        self.assertEqual(validate_message(" A paragraph. "),"A paragraph.")
        with self.assertRaises(ValueError):
            CodexChatBridge(self.root,["codex"],self.gateway)

    def test_privileged_cleanup_closes_stdin_before_denied_signals(self):
        events=[]
        process=Mock()
        process.pid=123
        process.poll.return_value=None
        process.stdin.close.side_effect=lambda:events.append("stdin-closed")
        process.stdout.close.side_effect=lambda:events.append("stdout-closed")
        process.wait.side_effect=[subprocess.TimeoutExpired("launcher",3),subprocess.TimeoutExpired("launcher",3)]
        def denied(*_args):
            events.append("signal-denied")
            raise PermissionError("sudo-owned group")
        with patch("gpt2_local.codex_chat_bridge.os.name","posix"),patch(
                "gpt2_local.codex_chat_bridge.os.killpg",side_effect=denied,create=True),patch(
                "gpt2_local.codex_chat_bridge.signal.SIGKILL",9,create=True):
            _stop_process(process)
        self.assertEqual(events[0],"stdin-closed")
        self.assertEqual(events.count("signal-denied"),2)
        self.assertIn("stdout-closed",events)

    def test_cleanup_releases_session_even_if_revocation_raises(self):
        bridge=self.bridge()
        with patch.object(self.gateway,"revoke",side_effect=RuntimeError("test revocation failure")):
            with self.assertRaisesRegex(RuntimeError,"revocation failure"):
                bridge.run("chat","Hello",lambda *_:{})
        self.assertEqual(bridge._active_sessions,set())
        self.assertEqual(bridge.run("chat","Hello",lambda *_:{})["thread_id"],"test-thread")

    def test_runtime_diagnostics_and_rpc_details_remain_private(self):
        bridge=self.bridge("rpc-error")
        with self.assertRaises(CodexError) as captured:
            bridge.run("chat","Hello",lambda *_:{})
        self.assertNotIn("PRIVATE",str(captured.exception))
        path=self.root/"data/players/chat/runtime-stderr.log"
        logged=path.read_text()
        self.assertIn("PRIVATE STARTUP DIAGNOSTIC",logged)
        self.assertIn("PRIVATE RPC DETAIL",logged)
        self.assertEqual(path.parent,self.root/"data/players/chat")
        if os.name!="nt":
            self.assertEqual(path.stat().st_mode & 0o777,0o600)

    def test_terminal_error_details_remain_private(self):
        with self.assertRaises(CodexError) as captured:
            self.bridge("terminal-error").run("chat","Hello",lambda *_:{})
        self.assertNotIn("PRIVATE",str(captured.exception))
        path=self.root/"data/players/chat/runtime-stderr.log"
        self.assertIn("PRIVATE TERMINAL ERROR",path.read_text())

    def test_child_events_are_private_and_child_host_tools_are_denied(self):
        for scenario in ("child","child-out-of-order"):
            calls=[]
            def callback(name,args):
                calls.append((name,args))
                return {"ok":True}
            with self.subTest(scenario=scenario):
                result=self.bridge(scenario).run(scenario,"Do a background task",callback)
                self.assertEqual(result["thread_id"],"test-thread")
                self.assertEqual(result["usage"]["total_tokens"],123)
                self.assertEqual(result["final"],"PRIVATE FINAL DO NOT PUBLISH")
                self.assertEqual([name for name,_ in calls],["_thread","emit_message"])
                self.assertEqual(calls[1][1]["text"],"A short public paragraph.")
                state=json.loads((self.root/"data/players"/scenario/"bridge-state.json").read_text())
                self.assertEqual(state,{"thread_id":"test-thread","usage_total":123})

    def test_trusted_daily_admission_failure_is_typed_before_revocation(self):
        self.gateway.action_failure=lambda token:{"code":"daily_allowance_exhausted"}
        with self.assertRaises(DailyBudgetExceeded) as captured:
            self.bridge("terminal-error").run("chat","Hello",lambda *_:{})
        self.assertIn("conversation is saved",str(captured.exception))
        self.assertNotIn("PRIVATE",str(captured.exception))
        self.assertEqual(self.gateway.revocations,["scoped-test-token"])
        self.assertTrue((self.root/"data/players/chat/bridge-state.json").exists())

    def test_other_gateway_failure_code_remains_generic(self):
        self.gateway.action_failure=lambda token:{"code":"provider_rate_limit"}
        with self.assertRaises(CodexError) as captured:
            self.bridge("terminal-error").run("chat","Hello",lambda *_:{})
        self.assertNotIn("PRIVATE",str(captured.exception))


if __name__ == "__main__":
    unittest.main()
