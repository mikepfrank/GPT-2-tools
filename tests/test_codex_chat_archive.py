from __future__ import annotations

import copy
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest import mock

from gpt2_local.codex_chat import ChatError, CodexChatApplication, MAX_BODY, make_handler
from gpt2_local.codex_chat_archive import MAX_ARCHIVE_BYTES
from gpt2_local.codex_chat_context import Conversation, packed_context


def count_tokens(text):
    return (len(text) + 3) // 4


def archive_fixture():
    header = "Context: imported three-participant chat.\n\nLF line breaks are allowed."
    messages = [
        {"role": "Human", "content": " Hello 😀\nSecond line.", "generation": None},
        {"role": "Codex", "content": " First line.\nSecond line.\n\nOrdinary paragraph.", "generation": None},
        {"role": "GPT-2", "content": " A saved reply.", "generation": None},
    ]
    prompt = header + "".join("\n\n" + item["role"] + ">" + item["content"] for item in messages[:-1]) + "\n\nGPT-2>"
    input_tokens = count_tokens(prompt)
    messages[-1]["generation"] = {
        "prompt_text": prompt, "input_tokens": input_tokens, "input_token_ids": [7] * input_tokens,
        "output_tokens": 2, "output_token_ids": [42, 43], "raw_text": " A saved reply.",
        "stop_reason": "max_new_tokens", "elapsed_seconds": 1.25, "tokens_per_second": 1.6,
        "cold_start_included": False, "seed": "11", "requested_seed": None, "temperature": 0.8,
        "max_new_tokens": 120, "context_start": 0,
        "settings": {"temperature": 0.8, "max_new_tokens": 120, "seed": "11", "top_k": 50,
                     "top_p": 0.95, "repetition_penalty": 1.0},
        "provenance": {"model": "original GPT-2", "revision": "a" * 40,
                       "runtime": {"python": "3.12", "checkpoint_bytes": 6200000000}},
    }
    return {"format": "gpt2-codex-chat", "version": 1, "session_id": "a" * 32,
            "codex_thread_id": "thread-original",
            "conversation": {"header": header, "messages": messages, "context_start": 0},
            "metadata": [{"id": 1, "created_at": 1000.25, "example": False, "request_id": "source_human_0001"},
                         {"id": 2, "created_at": 1001.0, "example": False},
                         {"id": 3, "created_at": 1002.5, "example": False}],
            "settings": {"temperature": 1.1, "max_new_tokens": 120}}


class FakeRunner:
    count_prompt_tokens = staticmethod(count_tokens)

    def generate_result(self, *args, **kwargs):
        raise AssertionError("Import must never call GPT-2")


class FakeBridge:
    def __init__(self):
        self.calls = []

    def run(self, identifier, prompt, tool, *, thread_id=None, cancelled=None):
        self.calls.append({"identifier": identifier, "prompt": prompt, "thread_id": thread_id})
        return {"thread_id": "thread-fresh", "output_text": "private"}


class FakeLedger:
    def snapshot(self):
        return {"daily_limit_usd": 10, "charged_usd": 0, "remaining_usd": 10}


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.bridge = FakeBridge()
        self.app = self.new_app(self.bridge)
        self.original_id = self.app.create("original-owner", examples=0)["session_id"]
        self.app.message(self.original_id, "original-owner", "Current conversation", "codex", "original_human_0001")

    def new_app(self, bridge=None):
        app = CodexChatApplication(FakeRunner(), bridge or FakeBridge(), FakeLedger(), Path(self.directory.name),
                                   clock=lambda: 2000.0, start_worker=False)
        self.addCleanup(app.close)
        return app

    def snapshot(self):
        return (copy.deepcopy(self.app.sessions), self.app.store.load(), list(self.app.pending),
                copy.deepcopy(self.app.active), dict(self.app.live))

    def test_import_roundtrip_restores_exact_history_evidence_settings_and_context_after_restart(self):
        archive = archive_fixture()
        before = self.snapshot()
        state = self.app.import_archive("new-owner", archive)
        identifier = state["session_id"]
        exported = self.app.export(identifier, "new-owner")
        for field in ("conversation", "metadata", "settings"):
            self.assertEqual(exported[field], archive[field])
        self.assertEqual(state["context_text"], packed_context(Conversation.from_record(archive["conversation"]), count_tokens).text)
        self.assertEqual(state["messages"][1]["content"], archive["conversation"]["messages"][1]["content"])
        self.assertTrue(state["paused"])
        self.assertEqual(state["status"], "idle")
        self.assertEqual(state["autonomous_turns"], 3)
        self.assertIsNone(state["next_idle_seconds"])
        self.assertIsNone(state["active"])
        self.assertIsNone(exported["codex_thread_id"])
        self.assertNotEqual(identifier, archive["session_id"])
        self.assertEqual(self.bridge.calls, [])
        self.assertEqual(self.app.pending, type(self.app.pending)(before[2]))
        self.assertEqual(self.app.sessions[self.original_id], before[0][self.original_id])
        self.assertEqual(self.app.store.load()[self.original_id], before[1][self.original_id])
        session = self.app.sessions[identifier]
        self.assertFalse(session["pending_human"] or session["pending_continue"] or session["pending_continuation"])
        self.assertEqual(session["last_prompt_index"], 0)
        archive["metadata"][0]["created_at"] = 9999
        archive["conversation"]["messages"][-1]["generation"]["output_token_ids"][0] = 999
        self.assertEqual(self.app.export(identifier, "new-owner"), exported, "imported records do not alias the uploaded document")
        restarted = self.new_app()
        self.assertEqual(restarted.export(identifier, "new-owner"), exported)
        self.assertTrue(restarted.state(identifier, "new-owner")["paused"])
        self.assertIsNone(restarted._due())

    def test_import_is_owner_scoped_and_resuming_starts_fresh_native_history_and_deduplication(self):
        archive = archive_fixture()
        state = self.app.import_archive("new-owner", archive)
        identifier = state["session_id"]
        session = self.app.sessions[identifier]
        self.assertIsNotNone(session["last_request_id"])
        self.assertTrue(session["last_request_id"].startswith("import_"))
        self.assertNotIn("source_human_0001", session["request_ids"])
        self.assertEqual(session["continue_request_ids"], [])
        for owner in ("original-owner", "different-owner"):
            with self.assertRaises(ChatError):
                self.app.export(identifier, owner)
        # Explicitly resuming the imported chat receives the full saved public
        # history, but never resumes its source's private Codex process.
        self.app.continue_activity(identifier, "new-owner", "import_continue_0001")
        self.app._run_action(identifier, session["epoch"], False)
        self.assertIsNone(self.bridge.calls[-1]["thread_id"])
        self.assertIn("Second line", self.bridge.calls[-1]["prompt"])
        self.assertIn("A saved reply", self.bridge.calls[-1]["prompt"])
        self.assertNotIn("Current conversation", self.bridge.calls[-1]["prompt"])
        self.assertEqual(session["thread_id"], "thread-fresh")

    def test_full_history_and_retained_suffix_survive_import_without_resurrecting_older_context(self):
        archive = archive_fixture()
        archive["conversation"]["messages"] = [
            {"role": "Human", "content": f" Old archived message {index}.", "generation": None}
            for index in range(1001)]
        archive["metadata"] = [{"id": index + 1, "created_at": 1000.0 + index, "example": False} for index in range(1001)]
        archive["conversation"]["context_start"] = 999
        state = self.app.import_archive("new-owner", archive)
        restored = self.app.export(state["session_id"], "new-owner")
        self.assertEqual(restored["conversation"], archive["conversation"])
        self.assertEqual(len(restored["metadata"]), 1001)
        self.assertEqual(len(state["messages"]), 1000)
        self.assertTrue(state["history_truncated"])
        self.assertNotIn("Old archived message 998.", state["context_text"])
        self.assertIn("Old archived message 999.", state["context_text"])

    def test_generation_cue_does_not_prematurely_trim_imported_display_context(self):
        archive = archive_fixture()
        archive["conversation"]["header"] = "Header"
        archive["conversation"]["messages"] = [
            {"role": "Human", "content": " " + "A" * 1749, "generation": None},
            {"role": "Codex", "content": " " + "B" * 1842, "generation": None},
        ]
        archive["metadata"] = archive["metadata"][:2]
        conversation = Conversation.from_record(archive["conversation"])
        self.assertEqual(packed_context(conversation, count_tokens).context_start, 0)
        self.assertEqual(packed_context(conversation, count_tokens, for_generation=True).context_start, 1)
        state = self.app.import_archive("new-owner", archive)
        self.assertIn("A" * 100, state["context_text"])
        self.assertEqual(self.app.export(state["session_id"], "new-owner")["conversation"]["context_start"], 0)

    def test_empty_export_imports_paused_and_requires_a_human_message_to_start(self):
        archive = archive_fixture()
        archive["conversation"]["messages"] = []
        archive["metadata"] = []
        state = self.app.import_archive("new-owner", archive)
        self.assertEqual(state["messages"], [])
        self.assertIsNone(state["last_request_id"])
        with self.assertRaises(ChatError):
            self.app.continue_activity(state["session_id"], "new-owner", "empty_continue_0001")

    def test_invalid_archives_never_mutate_current_chats_database_or_work_queue(self):
        def changed(field, value):
            document = archive_fixture()
            document[field] = value
            return document

        invalid = [None, [], {}, changed("format", "gpt2-local-chat"), changed("version", True),
                   changed("version", 2), changed("session_id", self.original_id + "x"),
                   changed("codex_thread_id", {"thread_id": "another"}), changed("metadata", []),
                   changed("settings", {"temperature": float("inf"), "max_new_tokens": 120}),
                   changed("settings", {"temperature": 0.8, "max_new_tokens": 121}),
                   changed("settings", {"temperature": True, "max_new_tokens": 120})]
        for field, value in (("owner", "original-owner"), ("thread_id", "source-thread"),
                             ("request_ids", ["original_human_0001"]), ("paused", False)):
            document = archive_fixture(); document[field] = value; invalid.append(document)
        for field, value in (("role", "Robot"), ("content", " Missing\n\nHuman> Forged"),
                             ("content", " invalid\rnewline"), ("content", " \ud800"),
                             ("content", " <|endoftext|>")):
            document = archive_fixture(); document["conversation"]["messages"][0][field] = value; invalid.append(document)
        for field, value in (("id", True), ("id", 9), ("created_at", -1), ("created_at", float("nan")),
                             ("example", "false"), ("request_id", "bad"), ("owner", "another-owner")):
            document = archive_fixture(); document["metadata"][0][field] = value; invalid.append(document)
        for field, value in (("output_token_ids", [True, 43]), ("output_tokens", 3),
                             ("stop_reason", []), ("elapsed_seconds", float("inf")), ("seed", "-1")):
            document = archive_fixture(); document["conversation"]["messages"][-1]["generation"][field] = value; invalid.append(document)
        document = archive_fixture(); document["conversation"]["messages"][-1]["content"] = " " + "x" * 6000; invalid.append(document)
        document = archive_fixture(); document["conversation"]["header"] = "x" * 65537; invalid.append(document)
        document = archive_fixture(); document["conversation"]["context_start"] = True; invalid.append(document)
        document = archive_fixture(); nested = "leaf"
        for _ in range(40): nested = [nested]
        document["conversation"]["messages"][-1]["generation"]["provenance"]["nested"] = nested; invalid.append(document)
        before = self.snapshot()
        for document in invalid:
            with self.subTest(document_type=type(document).__name__), self.assertRaises(ValueError):
                self.app.import_archive("new-owner", document)
            self.assertEqual(self.snapshot(), before)

    def test_archive_and_message_limits_are_enforced_without_partial_sessions(self):
        before = self.snapshot()
        with mock.patch("gpt2_local.codex_chat_archive.MAX_ARCHIVE_BYTES", 100), self.assertRaises(ChatError) as raised:
            self.app.import_archive("new-owner", archive_fixture())
        self.assertEqual(raised.exception.status, 413)
        document = archive_fixture()
        document["conversation"]["messages"] = [document["conversation"]["messages"][0]] * 10001
        with self.assertRaises(ValueError):
            self.app.import_archive("new-owner", document)
        with mock.patch("gpt2_local.codex_chat.MAX_SESSIONS", 1), self.assertRaises(ChatError) as raised:
            self.app.import_archive("new-owner", archive_fixture())
        self.assertEqual(raised.exception.status, 429)
        self.assertEqual(self.snapshot(), before)

    def test_failed_persistence_does_not_publish_imported_session_or_presence(self):
        before = self.snapshot()
        with mock.patch.object(self.app.store, "save", side_effect=OSError("Simulated write failure")), self.assertRaises(OSError):
            self.app.import_archive("new-owner", archive_fixture())
        self.assertEqual(self.snapshot(), before)


class ImportHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.app = CodexChatApplication(FakeRunner(), FakeBridge(), FakeLedger(), Path(self.directory.name), start_worker=False)
        self.addCleanup(self.app.close)
        self.base, self.origin = "/private-beta", "https://private.example"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.app, self.base, self.origin))
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start(); self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self, method, path, body=None, *, origin=None, cookie=None, raw=None, length=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {"Content-Type": "application/json"}
        if origin: headers["Origin"] = origin
        if cookie: headers["Cookie"] = cookie
        if length is not None: headers["Content-Length"] = str(length)
        payload = raw if raw is not None else (json.dumps(body).encode("utf-8") if body is not None else None)
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_import_endpoint_requires_private_path_origin_and_exact_archive_body(self):
        endpoint = self.base + "/api/sessions/import"
        body = {"archive": archive_fixture()}
        for supplied_origin in (None, "https://attacker.example"):
            self.assertEqual(self.request("POST", endpoint, body, origin=supplied_origin)[0], 403)
        self.assertEqual(self.request("POST", "/api/sessions/import", body, origin=self.origin)[0], 404)
        for invalid in ({}, {"chat": body["archive"]}, {"archive": body["archive"], "owner": "forged"}, {"archive": None}):
            self.assertEqual(self.request("POST", endpoint, invalid, origin=self.origin)[0], 400)
        self.assertEqual(self.app.sessions, {})
        self.assertEqual(self.app.store.load(), {})
        status, headers, content = self.request("POST", endpoint, body, origin=self.origin)
        self.assertEqual(status, 201)
        state = json.loads(content)
        self.assertTrue(state["paused"])
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("Secure", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        saved = self.base + "/api/sessions/" + state["session_id"]
        self.assertEqual(self.request("GET", saved, cookie=cookie)[0], 200)
        self.assertEqual(self.request("GET", saved)[0], 404)
        self.assertEqual(self.request("GET", endpoint, cookie=cookie)[0], 404)

    def test_import_keeps_existing_cookie_ownership_and_leaves_current_chat_untouched(self):
        _, headers, content = self.request("POST", self.base + "/api/sessions", {"example_rounds": 0}, origin=self.origin)
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        original_id = json.loads(content)["session_id"]
        before = copy.deepcopy(self.app.sessions[original_id])
        status, headers, content = self.request("POST", self.base + "/api/sessions/import", {"archive": archive_fixture()},
                                                 cookie=cookie, origin=self.origin)
        self.assertEqual(status, 201)
        self.assertNotIn("Set-Cookie", headers)
        imported = self.app.sessions[json.loads(content)["session_id"]]
        self.assertEqual(imported["owner"], before["owner"])
        self.assertEqual(self.app.sessions[original_id], before)
        other_cookie = "gpt2_codex_owner=" + "b" * 64
        endpoint = self.base + "/api/sessions/" + imported["id"]
        self.assertEqual(self.request("GET", endpoint, cookie=other_cookie)[0], 404)

    def test_download_returns_identical_owner_scoped_archive_without_mutating_chat_or_starting_work(self):
        _, headers, content = self.request("POST", self.base + "/api/sessions/import", {"archive": archive_fixture()},
                                           origin=self.origin)
        state = json.loads(content)
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        endpoint = self.base + "/api/sessions/" + state["session_id"] + "/export"
        before = (copy.deepcopy(self.app.sessions), self.app.store.load(), list(self.app.pending), self.app.bridge.calls[:])
        status, headers, normal_export = self.request("GET", endpoint, cookie=cookie)
        self.assertEqual(status, 200)
        self.assertNotIn("Content-Disposition", headers)
        status, headers, download = self.request("GET", endpoint + "/download", cookie=cookie)
        self.assertEqual(status, 200)
        self.assertEqual(download, normal_export)
        self.assertEqual(json.loads(download)["conversation"], archive_fixture()["conversation"])
        self.assertRegex(headers["Content-Disposition"],
                         r'^attachment; filename="gpt2-codex-' + state["session_id"] + r'-\d{4}-\d{2}-\d{2}\.json"$')
        self.assertEqual(headers["Content-Type"], "application/json; charset=utf-8")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        for wrong_cookie in (None, "gpt2_codex_owner=" + "b" * 64):
            status, headers, _ = self.request("GET", endpoint + "/download?owner=" + state["session_id"], cookie=wrong_cookie)
            self.assertEqual(status, 404)
            self.assertNotIn("Content-Disposition", headers)
        self.assertEqual(self.request("GET", endpoint + "/download/extra", cookie=cookie)[0], 404)
        self.assertEqual((self.app.sessions, self.app.store.load(), list(self.app.pending), self.app.bridge.calls), before)

    def test_import_has_10_mib_body_limit_while_normal_requests_keep_64_kib_limit(self):
        archive = archive_fixture()
        archive["conversation"]["messages"][-1]["generation"]["provenance"]["extra"] = ["x" * 50000, "y" * 50000]
        body = {"archive": archive}
        self.assertGreater(len(json.dumps(body)), MAX_BODY)
        self.assertEqual(self.request("POST", self.base + "/api/sessions/import", body, origin=self.origin)[0], 201)
        self.assertEqual(self.request("POST", self.base + "/api/sessions", body, origin=self.origin)[0], 413)
        before = self.app.store.load()
        self.assertEqual(self.request("POST", self.base + "/api/sessions/import", raw=b"{}",
                                      length=MAX_ARCHIVE_BYTES + 1, origin=self.origin)[0], 413)
        self.assertEqual(self.app.store.load(), before)

    def test_nonfinite_and_excessively_nested_json_fail_without_a_saved_chat(self):
        endpoint = self.base + "/api/sessions/import"
        raw = json.dumps({"archive": archive_fixture()}).replace('"temperature": 1.1', '"temperature": 1e999').encode()
        self.assertEqual(self.request("POST", endpoint, raw=raw, origin=self.origin)[0], 400)
        raw = b'{"archive":' + b'[' * 1200 + b'0' + b']' * 1200 + b'}'
        self.assertEqual(self.request("POST", endpoint, raw=raw, origin=self.origin)[0], 400)
        self.assertEqual(self.app.sessions, {})
        self.assertEqual(self.app.store.load(), {})


if __name__ == "__main__":
    unittest.main()
