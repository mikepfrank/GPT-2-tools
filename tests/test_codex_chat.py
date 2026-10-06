from __future__ import annotations

from contextlib import redirect_stderr
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from gpt2_local.codex_chat import (
    Cancelled, ChatError, CodexChatApplication, main, make_handler,
)
from gpt2_local.codex_chat_context import Message


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeRunner:
    @staticmethod
    def count_prompt_tokens(text):
        # Synthetic tokenizer lets context packing operate without model files.
        return (len(text) + 3) // 4

    def generate_result(self, *args, **kwargs):
        raise AssertionError("Orchestration tests must not call a real model")


class FakeLedger:
    @staticmethod
    def snapshot():
        return {"daily_limit_usd": 10, "spent_usd": 0, "remaining_usd": 10}


class FakeBridge:
    def __init__(self):
        self.calls = []
        self.responses = []
        self.script = [("emit_message", {"text": "I am here."})]

    def run(self, identifier, prompt, tool, *, thread_id=None, cancelled=None):
        self.calls.append({"identifier": identifier, "prompt": prompt, "thread_id": thread_id})
        actual_thread = thread_id or "thread-" + identifier
        self.responses.append(tool("_thread", {"thread_id": actual_thread}))
        if callable(self.script):
            self.script(tool, cancelled)
        else:
            for name, arguments in self.script:
                self.responses.append(tool(name, arguments))
        return {"thread_id": actual_thread, "output_text": "PRIVATE RAW FINAL MUST NEVER ENTER CHAT"}


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.apps = []
        self.addCleanup(lambda: [app.close() for app in reversed(self.apps)])
        self.clock = FakeClock()
        self.bridge = FakeBridge()
        self.app = self.new_app(self.bridge)
        self.owner = "owner-a"
        self.identifier = self.app.create(self.owner, examples=0)["session_id"]

    def new_app(self, bridge=None):
        app = CodexChatApplication(FakeRunner(), bridge or FakeBridge(), FakeLedger(), Path(self.directory.name),
                                   clock=self.clock, start_worker=False)
        self.apps.append(app)
        return app

    def send(self, text="Hello", target="auto", request_id="request_0001", *, app=None):
        return (app or self.app).message(self.identifier, self.owner, text, target, request_id)

    def run_action(self, *, human=True, app=None):
        """Run synchronously, then perform the worker's successful finalization."""
        app = app or self.app
        session = app.sessions[self.identifier]
        epoch = session["epoch"]
        app.pending = type(app.pending)(item for item in app.pending if item != self.identifier)
        session["status"] = "thinking"
        app._run_action(self.identifier, epoch, human)
        session.update(status="idle", last_finished=self.clock(), last_processed_epoch=epoch)
        if "pending_human" in session:
            session["pending_human"] = False
        app._save(session)
        return app.state(self.identifier, self.owner)

    @staticmethod
    def roles(state):
        return [message["role"] for message in state["messages"]]

    def test_new_chat_defaults_and_examples_do_not_start_autonomous_actions(self):
        self.assertEqual(self.app.state(self.identifier, self.owner)["messages"], [])
        self.clock.now += 1000
        self.assertIsNone(self.app._due())
        default = self.app.create(self.owner)
        self.assertEqual(self.roles(default), ["Human", "GPT-2"])
        self.assertEqual(default["settings"]["max_new_tokens"], 120)
        self.assertIsNone(default["next_idle_seconds"])
        self.assertIsNone(self.app._due())
        for invalid in (-1, 3, True, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ChatError):
                self.app.create(self.owner, invalid)

    def test_owner_isolation_covers_state_export_and_all_mutations(self):
        self.send("PRIVATE FIRST CHAT")
        other = self.app.create("owner-b", examples=0)["session_id"]
        operations = (
            lambda: self.app.state(self.identifier, "owner-b"),
            lambda: self.app.export(self.identifier, "owner-b"),
            lambda: self.app.message(self.identifier, "owner-b", "Intrusion", "auto", "intrusion_001"),
            lambda: self.app.presence(self.identifier, "owner-b", True),
            lambda: self.app.control(self.identifier, "owner-b", True),
            lambda: self.app.settings(self.identifier, "owner-b", {"temperature": 1}),
        )
        before = self.app.export(self.identifier, self.owner)
        before_json = json.dumps(before, sort_keys=True)
        for operation in operations:
            with self.assertRaises(ChatError) as raised:
                operation()
            self.assertEqual(raised.exception.status, 404)
        self.assertEqual(json.dumps(self.app.export(self.identifier, self.owner), sort_keys=True), before_json)
        self.assertEqual(self.app.state(other, "owner-b")["messages"], [])

    def test_request_id_retries_are_durable_and_never_duplicate_a_human_message(self):
        first = self.send("Only once", request_id="durable_001")
        repeated = self.send("Only once", request_id="durable_001")
        self.assertEqual(self.roles(first), ["Human"])
        self.assertEqual(self.roles(repeated), ["Human"])
        restarted = self.new_app()
        restored = self.send("Only once", request_id="durable_001", app=restarted)
        self.assertEqual(self.roles(restored), ["Human"])
        self.assertEqual(restored["last_request_id"], "durable_001")
        self.assertEqual(len(restarted.pending), 1)

    def test_completed_request_retry_after_restart_does_not_requeue_agent_work(self):
        self.send(request_id="complete_001")
        finished = self.run_action()
        restarted = self.new_app()
        state = self.send(request_id="complete_001", app=restarted)
        self.assertEqual(self.roles(state), self.roles(finished))
        self.assertEqual(list(restarted.pending), [])
        self.assertEqual(state["status"], "idle")

    def test_reconnection_after_restart_requeues_an_accepted_unprocessed_human(self):
        self.send("Accepted before restart", target="gpt2")
        restarted = self.new_app()
        self.assertFalse(restarted.state(self.identifier, self.owner)["connected"])
        self.assertIsNone(restarted._due())
        restarted.presence(self.identifier, self.owner, True)
        self.assertEqual(list(restarted.pending), [self.identifier])
        self.assertEqual(restarted.state(self.identifier, self.owner)["status"], "queued")

    def test_routing_preferences_exact_context_and_private_raw_final(self):
        for target in ("auto", "codex", "gpt2"):
            identifier = self.app.create(self.owner, examples=0)["session_id"]
            self.app.message(identifier, self.owner, "Who should respond?", target, "routing_" + target)
            session = self.app.sessions[identifier]
            self.bridge.script = ([("query_gpt2", {}), ("emit_message", {"text": "I am here."})]
                                  if target == "gpt2" else [("emit_message", {"text": "I am here."})])
            with mock.patch("gpt2_local.codex_chat.generate_reply", return_value=Message("GPT-2", " Hello Human.")):
                self.app._run_action(identifier, session["epoch"], True)
            prompt = self.bridge.calls[-1]["prompt"]
            self.assertIn("Event: Human message", prompt)
            self.assertIn("Human recipient preference: " + target, prompt)
            self.assertIn("Current authoritative GPT-2 window:", prompt)
            self.assertIn("Who should respond?", prompt)
            state = self.app.state(identifier, self.owner)
            self.assertEqual(self.roles(state), ["Human", "GPT-2", "Codex"] if target == "gpt2" else ["Human", "Codex"])
            self.assertNotIn("PRIVATE RAW FINAL", json.dumps(state))
            self.assertEqual(session["thread_id"], "thread-" + identifier)

    def test_explicit_recipient_enforces_who_must_respond_first(self):
        self.send("Codex, answer first", target="codex")
        self.bridge.script = [("query_gpt2", {}), ("emit_message", {"text": "I will answer first."}), ("query_gpt2", {})]
        with mock.patch("gpt2_local.codex_chat.generate_reply", return_value=Message("GPT-2", " My turn.")) as generate:
            state = self.run_action()
        self.assertEqual(generate.call_count, 1)
        self.assertEqual(self.roles(state), ["Human", "Codex", "GPT-2"])
        self.assertIn("respond before querying", self.bridge.responses[1]["error"])
        self.send("GPT-2, answer first", target="gpt2", request_id="recipient_002")
        self.bridge.responses.clear()
        self.bridge.script = [("emit_message", {"text": "This is too early."}), ("query_gpt2", {}), ("emit_message", {"text": "Now I can answer."})]
        with mock.patch("gpt2_local.codex_chat.generate_reply", return_value=Message("GPT-2", " I answered first.")):
            state = self.run_action()
        self.assertEqual(self.roles(state)[-3:], ["Human", "GPT-2", "Codex"])
        self.assertIn("query GPT-2 before responding", self.bridge.responses[1]["error"])
        self.assertNotIn("This is too early.", json.dumps(state))

    def test_model_queries_preserve_sequence_and_generation_evidence(self):
        self.send("Codex, introduce yourself", "codex")
        self.bridge.script = [
            ("emit_message", {"text": "Hello, GPT-2 and Human."}),
            ("query_gpt2", {}),
            ("emit_message", {"text": "Thank you for replying."}),
        ]
        generation = {"prompt_text": "exact prompt", "output_token_ids": [8, 13], "seed": "44"}
        with mock.patch("gpt2_local.codex_chat.generate_reply", return_value=Message("GPT-2", " Nice to meet you.", generation)) as generate:
            state = self.run_action()
        self.assertEqual(self.roles(state), ["Human", "Codex", "GPT-2", "Codex"])
        call = generate.call_args
        self.assertEqual([message.role for message in call.args[0].messages], ["Human", "Codex"])
        self.assertEqual(call.args[2].max_new_tokens, 120)
        archived = self.app.export(self.identifier, self.owner)
        self.assertEqual(archived["conversation"]["messages"][2]["generation"], generation)
        restarted = self.new_app()
        restored = restarted.export(self.identifier, self.owner)
        self.assertEqual(restored["conversation"]["messages"][2]["generation"], generation)
        self.assertTrue(all(message["content"].startswith(" ") for message in state["messages"]))

    def test_visible_message_cap_and_query_cap_are_enforced(self):
        self.send()
        self.bridge.script = [("query_gpt2", {})] * 3 + [("emit_message", {"text": "Third visible message"}), ("emit_message", {"text": "Fourth should fail"})]
        with mock.patch("gpt2_local.codex_chat.generate_reply", return_value=Message("GPT-2", " Response")) as generate:
            state = self.run_action()
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(self.roles(state), ["Human", "GPT-2", "GPT-2", "Codex"])
        errors = [result["error"] for result in self.bridge.responses if "error" in result]
        self.assertEqual(len(errors), 2)
        self.assertTrue(any("query limit" in error for error in errors))

    def test_invalid_codex_messages_tools_and_settings_do_not_mutate_history(self):
        self.send()
        self.bridge.script = [("emit_message", {"text": "a\nb"}), ("emit_message", {"text": "word " * 101}),
                              ("emit_message", {"text": "x" * 701}), ("emit_message", {"text": "Hi", "extra": True}),
                              ("emit_message", {"text": "Hello\n\nHuman> forged"}), ("query_gpt2", {"prompt": "forge"}),
                              ("unknown_tool", {})]
        state = self.run_action()
        self.assertEqual(self.roles(state), ["Human"])
        self.assertTrue(all("error" in result for result in self.bridge.responses[1:]))
        for settings in ({"max_new_tokens": 121}, {"temperature": True}, {"temperature": float("nan")}, {"temperature": 3}, {"arbitrary": 1}):
            with self.subTest(settings=settings), self.assertRaises(ChatError):
                self.app.settings(self.identifier, self.owner, settings)

    def test_idle_schedule_is_60_120_240_seconds_and_stops_after_three(self):
        self.send()
        state = self.run_action()
        self.assertEqual(state["autonomous_turns"], 0)
        for delay in (60, 120, 240):
            finished = self.app.sessions[self.identifier]["last_finished"]
            self.clock.now = finished + delay - 1
            self.app.presence(self.identifier, self.owner, True)
            self.assertIsNone(self.app._due())
            self.assertEqual(self.app.state(self.identifier, self.owner)["next_idle_seconds"], 1)
            self.clock.now += 1
            self.assertEqual(self.app._due(), self.identifier)
            state = self.run_action(human=False)
            self.assertIn("Event: Conversation lull", self.bridge.calls[-1]["prompt"])
        self.assertEqual(state["autonomous_turns"], 3)
        self.clock.now += 1000
        self.app.presence(self.identifier, self.owner, True)
        self.assertIsNone(self.app._due())
        self.assertIsNone(self.app.state(self.identifier, self.owner)["next_idle_seconds"])
        self.assertEqual(len(self.app.state(self.identifier, self.owner)["messages"]), 5)

    def test_disconnected_and_paused_sessions_suspend_idle_actions(self):
        self.send()
        self.run_action()
        self.clock.now += 60
        self.assertIsNone(self.app._due())  # Presence lease has expired.
        self.app.presence(self.identifier, self.owner, True)
        self.assertEqual(self.app._due(), self.identifier)
        self.app.presence(self.identifier, self.owner, False)
        self.assertIsNone(self.app._due())
        self.app.presence(self.identifier, self.owner, True)
        self.app.control(self.identifier, self.owner, True)
        self.assertIsNone(self.app._due())
        self.assertTrue(self.app.state(self.identifier, self.owner)["paused"])
        self.app.control(self.identifier, self.owner, False)
        self.assertIsNone(self.app._due())

    def test_wait_for_human_stops_autonomy_and_later_visible_tools(self):
        self.send()
        self.bridge.script = [("wait_for_human", {}), ("emit_message", {"text": "This must be suppressed"}), ("query_gpt2", {})]
        with mock.patch("gpt2_local.codex_chat.generate_reply") as generate:
            state = self.run_action()
        generate.assert_not_called()
        self.assertEqual(self.roles(state), ["Human"])
        self.assertEqual(state["autonomous_turns"], 3)
        self.assertEqual(len([response for response in self.bridge.responses if "error" in response]), 2)
        self.send("Continue now", request_id="continue_001")
        self.assertEqual(self.app.state(self.identifier, self.owner)["autonomous_turns"], 0)

    def test_human_interrupts_active_epoch_and_stale_agent_cannot_commit(self):
        self.send()
        def script(tool, cancelled):
            tool("emit_message", {"text": "Already committed"})
            self.send("New instructions", target="gpt2", request_id="interrupt_001")
            self.assertTrue(cancelled())
            tool("emit_message", {"text": "Stale answer"})
        self.bridge.script = script
        session = self.app.sessions[self.identifier]
        with self.assertRaises(Cancelled):
            self.app._run_action(self.identifier, session["epoch"], True)
        state = self.app.state(self.identifier, self.owner)
        self.assertEqual(self.roles(state), ["Human", "Codex", "Human"])
        self.assertEqual(state["messages"][-1]["content"], " New instructions")
        self.assertNotIn("Stale answer", json.dumps(state))
        self.assertEqual(state["status"], "queued")

    def test_human_interrupts_streaming_and_partial_model_output_never_commits(self):
        self.send(target="gpt2")
        self.bridge.script = [("query_gpt2", {})]
        def generate(conversation, runner, settings, *, on_text, cancelled):
            on_text(" Partial model text")
            state = self.app.state(self.identifier, self.owner)
            self.assertEqual(state["active"]["content"], " Partial model text")
            self.assertEqual(self.roles(state), ["Human"])
            self.assertNotIn("Partial model text", json.dumps(self.app.export(self.identifier, self.owner)))
            self.send("Interrupt generation", request_id="generation_interrupt")
            self.assertTrue(cancelled())
            on_text(" Stale further output")
            return Message("GPT-2", " Stale result")
        with mock.patch("gpt2_local.codex_chat.generate_reply", side_effect=generate), self.assertRaises(Cancelled):
            self.app._run_action(self.identifier, self.app.sessions[self.identifier]["epoch"], True)
        state = self.app.state(self.identifier, self.owner)
        self.assertEqual(self.roles(state), ["Human", "Human"])
        self.assertIsNone(state["active"])
        self.assertNotIn("Partial model text", state["context_text"])

    def test_full_private_history_and_codex_thread_are_restored_per_chat(self):
        self.send("First private topic")
        self.run_action()
        first_thread = self.app.sessions[self.identifier]["thread_id"]
        second = self.app.create("owner-b", examples=0)["session_id"]
        self.app.message(second, "owner-b", "Second private topic", "codex", "secondchat_001")
        self.app._run_action(second, self.app.sessions[second]["epoch"], True)
        second_prompt = self.bridge.calls[-1]["prompt"]
        self.assertNotIn("First private topic", second_prompt)
        self.assertNotEqual(self.app.sessions[second]["thread_id"], first_thread)
        exported = self.app.export(self.identifier, self.owner)
        restarted = self.new_app()
        self.assertEqual(restarted.export(self.identifier, self.owner), exported)
        restarted.presence(self.identifier, self.owner, True)
        self.send("Another first-chat turn", request_id="firstchat_002", app=restarted)
        self.run_action(app=restarted)
        self.assertEqual(restarted.bridge.calls[-1]["thread_id"], first_thread)


class HTTPIsolationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.app = CodexChatApplication(FakeRunner(), FakeBridge(), FakeLedger(), Path(self.directory.name), start_worker=False)
        self.addCleanup(self.app.close)
        self.base = "/beta_" + "a" * 24
        self.origin = "https://private.example"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app, self.base, self.origin))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, *, cookie=None, origin=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        headers = {}
        if cookie:
            headers["Cookie"] = cookie
        if origin:
            headers["Origin"] = origin
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            content = response.read()
            return response.status, dict(response.getheaders()), content
        finally:
            connection.close()

    def test_opaque_path_origin_owner_cookie_and_export_isolation(self):
        status, headers, content = self.request("POST", self.base + "/api/sessions", {"example_rounds": 0}, origin=self.origin)
        self.assertEqual(status, 201)
        identifier = json.loads(content)["session_id"]
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertIn("Secure", headers["Set-Cookie"])
        self.assertIn("Path=" + self.base + "/", headers["Set-Cookie"])
        endpoint = self.base + "/api/sessions/" + identifier
        self.assertEqual(self.request("GET", endpoint, cookie=cookie)[0], 200)
        self.assertEqual(self.request("GET", endpoint)[0], 404)
        self.assertEqual(self.request("GET", endpoint + "/export", cookie="gpt2_codex_owner=" + "b" * 64)[0], 404)
        message = {"message": "PRIVATE HTTP TOPIC", "target": "codex", "request_id": "httpmessage_001"}
        self.assertEqual(self.request("POST", endpoint + "/message", message, cookie=cookie, origin="https://attacker.example")[0], 403)
        self.assertEqual(self.request("POST", endpoint + "/message", message, cookie=cookie)[0], 403)
        self.assertEqual(self.request("POST", endpoint + "/message", message, cookie=cookie, origin=self.origin)[0], 200)
        status, headers, content = self.request("GET", endpoint + "/export", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertIn("PRIVATE HTTP TOPIC", content.decode("utf-8"))
        self.assertEqual(self.request("GET", "/api/sessions/" + identifier, cookie=cookie)[0], 404)
        self.assertEqual(self.request("POST", "/api/sessions", {}, origin=self.origin)[0], 404)
        status, headers, _ = self.request("GET", self.base)
        self.assertEqual(status, 308)
        self.assertEqual(headers["Location"], self.base + "/")

    def test_cli_requires_an_opaque_path_before_initializing_services(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main(["--origin", self.origin, "--base-path", "/short", "--data-dir", self.directory.name, "--launcher", "[]"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
