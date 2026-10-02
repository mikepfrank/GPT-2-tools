from __future__ import annotations

import copy
import json
import threading
import unittest
from http.client import HTTPConnection
from unittest import mock

from gpt2_local.chat import (
    HUMAN_MARKER,
    MODEL_MARKER,
    MESSAGE_DELIMITER_PATTERN,
    ChatApplication,
    ChatServer,
    ChatSession,
    Round,
)
from gpt2_local.chat_archive import UNKNOWN_GENERATION_NOTE, export_chat
from gpt2_local.runtime import GenerationResult, GenerationSettings


class TransferRunner:
    """No checkpoint or tensor library is loaded by these integration tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, GenerationSettings, str | None]] = []
        self.failure: Exception | None = None
        self.text = " A synthetic answer.\nIts trailing whitespace stays.  \n"

    @staticmethod
    def count_prompt_tokens(text: str) -> int:
        return (len(text) + 3) // 4

    def generate_result(
        self, prompt: str, settings: GenerationSettings, *, stop_pattern: str | None = None,
    ) -> GenerationResult:
        self.calls.append((prompt, settings, stop_pattern))
        if self.failure is not None:
            raise self.failure
        tokens = self.count_prompt_tokens(prompt)
        return GenerationResult(
            text=self.text,
            original_input_tokens=tokens,
            input_tokens=tokens,
            output_tokens=1,
            output_token_ids=(42,),
            stop_reason="max_new_tokens",
            elapsed_seconds=0.125,
            tokens_per_second=8,
            prompt_truncated=False,
            cold_start_included=len(self.calls) == 1,
        )


class ChatTransferIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = TransferRunner()
        self.app = ChatApplication(self.runner)  # type: ignore[arg-type]
        self.server = ChatServer(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(
            target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True,
        )
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def request(
        self, method: str, path: str, body: object | None = None,
        *, headers: dict[str, str] | None = None, raw: bytes | None = None,
    ) -> tuple[int, dict[str, object]]:
        request_headers = dict(headers or {})
        if method == "POST":
            request_headers.setdefault("Content-Type", "application/json")
        payload = raw if raw is not None else json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=payload, headers=request_headers)
            response = connection.getresponse()
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def fixture(self) -> ChatSession:
        """All text is synthetic; no saved user conversation is used here."""
        transcript = [
            Round("Synthetic example?", " A synthetic example reply.", example=True),
            Round("An old question.", " An old answer.  "),
            Round("A question\nwith <script>quoted text</script> and \u03a9.", " A retained reply.\nTwo lines.  \n"),
        ]
        session = ChatSession(
            "synthetic-session",
            "Context: Synthetic transfer fixture.\nDate: January 1, 2019.\n",
            1,
            rounds=list(transcript[2:]),
            transcript=list(transcript),
            dropped_rounds=2,
            settings=GenerationSettings(max_new_tokens=80, temperature=0.7, seed=123),
            notes=["This synthetic historical fixture has no generation measurements.", UNKNOWN_GENERATION_NOTE],
        )
        self.app.sessions[session.session_id] = session
        return session

    def state(self, session: ChatSession) -> dict[str, object]:
        status, state = self.request("GET", "/api/state?session_id=" + session.session_id)
        self.assertEqual(status, 200)
        return state

    def export(self, session_id: str, settings: dict[str, object] | None = None) -> dict[str, object]:
        body: dict[str, object] = {"session_id": session_id}
        if settings is not None:
            body["settings"] = settings
        status, archive = self.request("POST", "/api/export", body)
        self.assertEqual(status, 200)
        return archive

    def assert_exact_blocks(self, state: dict[str, object]) -> None:
        blocks = state["context_blocks"]
        self.assertEqual("".join(item["text"] for item in blocks), state["context_text"])
        self.assertEqual(blocks[0]["role"], "context")
        self.assertTrue(all(item["role"] in ("context", "human", "model") for item in blocks))

    def test_http_roundtrip_keeps_full_transcript_retained_suffix_and_unknown_metadata(self) -> None:
        session = self.fixture()
        initial = self.state(session)
        archive = self.export(session.session_id)
        self.assertEqual(len(archive["blocks"]), 6)
        self.assertEqual(archive["context"]["first_retained_round"], 2)
        self.assertEqual(archive["context"]["text"], initial["context_text"])
        status, imported = self.request("POST", "/api/import", {"chat": archive, "previous_session_id": session.session_id})
        self.assertEqual(status, 200)
        self.assertNotEqual(imported["session_id"], session.session_id)
        for key in ("context_text", "context_tokens", "turns", "example_rounds", "dropped_rounds", "settings", "notes", "last_generation"):
            self.assertEqual(imported[key], initial[key])
        self.assertEqual(imported["last_generation"], None)
        self.assertTrue(all(item["generation"] is None for item in imported["turns"]))
        self.assertIn(UNKNOWN_GENERATION_NOTE, imported["notes"])
        self.assertIn("January 1, 2019", imported["context_text"])
        self.assert_exact_blocks(imported)
        status, _ = self.request("GET", "/api/state?session_id=" + session.session_id)
        self.assertEqual(status, 404)
        restored_archive = self.export(imported["session_id"])
        archive.pop("exported_at")
        restored_archive.pop("exported_at")
        self.assertEqual(restored_archive, archive)
        self.assertEqual(self.runner.calls, [])

    def test_invalid_http_import_does_not_remove_previous_or_install_partial_session(self) -> None:
        session = self.fixture()
        initial = self.state(session)
        archive = self.export(session.session_id)
        bad_context = copy.deepcopy(archive)
        bad_context["context"]["text"] += " altered"
        bad_settings = copy.deepcopy(archive)
        bad_settings["settings"]["seed"] = 7
        for document, previous in (({}, session.session_id), (bad_context, session.session_id), (bad_settings, session.session_id), (archive, 5)):
            with self.subTest(previous=previous, document=document):
                status, error = self.request("POST", "/api/import", {"chat": document, "previous_session_id": previous})
                self.assertEqual(status, 400)
                self.assertIn("error", error)
                self.assertEqual(self.state(session), initial)
                self.assertEqual(set(self.app.sessions), {session.session_id})
        self.assertEqual(self.runner.calls, [])

    def test_export_controls_persist_for_reload_and_import_without_changing_text(self) -> None:
        session = self.fixture()
        before = self.state(session)
        controls = {"temperature": 0.35, "max_new_tokens": 67, "seed": str(2**63 - 1)}
        archive = self.export(session.session_id, controls)
        state = self.state(session)
        self.assertEqual(state["context_text"], before["context_text"])
        self.assertEqual(state["turns"], before["turns"])
        for key, value in controls.items():
            self.assertEqual(state["settings"][key], value)
            self.assertEqual(archive["settings"][key], value)
        status, imported = self.request("POST", "/api/import", {"chat": archive, "previous_session_id": session.session_id})
        self.assertEqual(status, 200)
        self.assertEqual(imported["settings"], state["settings"])
        self.assertEqual(self.runner.calls, [])

    def test_invalid_export_controls_preserve_existing_settings_and_context(self) -> None:
        session = self.fixture()
        before = self.state(session)
        status, error = self.request("POST", "/api/export", {"session_id": session.session_id, "settings": {"seed": 123}})
        self.assertEqual(status, 400)
        self.assertIn("error", error)
        self.assertEqual(self.state(session), before)
        self.assertEqual(self.runner.calls, [])

    def test_context_color_segments_preserve_exact_text_before_and_during_generation(self) -> None:
        session = self.fixture()
        initial = self.state(session)
        self.assert_exact_blocks(initial)
        self.assertEqual([item["role"] for item in initial["context_blocks"]], ["context", "human", "model"])
        self.assertEqual(initial["context_blocks"][1]["text"], HUMAN_MARKER + " " + session.rounds[0].human)
        self.assertEqual(initial["context_blocks"][2]["text"], MODEL_MARKER + session.rounds[0].assistant)
        message = "Pending first line.\r\nPending second line."
        status, preview = self.request("POST", "/api/preview", {"session_id": session.session_id, "message": message, "settings": {"max_new_tokens": 40}})
        self.assertEqual(status, 200)
        self.assert_exact_blocks(preview)
        self.assertEqual(preview["context_blocks"][-2], {"role": "human", "text": HUMAN_MARKER + " Pending first line.\nPending second line."})
        self.assertEqual(preview["context_blocks"][-1], {"role": "model", "text": MODEL_MARKER})
        self.assertEqual(self.state(session), initial)
        self.assertEqual(self.runner.calls, [])

    def test_blank_initial_seed_is_chosen_once_and_reused_by_replies_and_exports(self) -> None:
        for temperature in (0, 0.8):
            with self.subTest(temperature=temperature):
                seed = 2**63 - 1
                with mock.patch("gpt2_local.chat.secrets.randbelow", return_value=seed) as choose:
                    controls = {"temperature": temperature, "max_new_tokens": 40, "seed": None}
                    status, initial = self.request("POST", "/api/new", {"example_rounds": 0, "date": "2026-10-01", "settings": controls})
                    self.assertEqual(status, 200)
                    self.assertEqual(initial["settings"]["seed"], str(seed))
                    self.assertIsNone(initial["last_generation"])
                    self.assertEqual(initial["turns"], [])
                    initial_archive = self.export(initial["session_id"], controls)
                    self.assertEqual(initial_archive["settings"]["seed"], str(seed))
                    for message in ("First synthetic question?", "Second synthetic question?"):
                        status, state = self.request("POST", "/api/message", {"session_id": initial["session_id"], "message": message, "settings": controls})
                        self.assertEqual(status, 200)
                        prompt, effective, stop_pattern = self.runner.calls[-1]
                        self.assertEqual(effective.seed, seed)
                        self.assertEqual(effective.temperature, temperature)
                        self.assertEqual(stop_pattern, MESSAGE_DELIMITER_PATTERN)
                        generation = state["last_generation"]
                        self.assertEqual(generation["seed"], str(seed))
                        self.assertIsNone(generation["requested_seed"])
                        self.assertEqual(state["settings"]["seed"], str(seed))
                        self.assertEqual(generation["prompt_text"], prompt)
                        self.assertEqual(generation["output_token_ids"], [42])
                        self.assertEqual(state["turns"][-1]["generation"], generation)
                        archive = self.export(initial["session_id"], controls)
                        self.assertEqual(archive["blocks"][-1]["generation"], generation)
                        self.assertEqual(archive["settings"]["seed"], str(seed))
                    choose.assert_called_once_with(2**63)

    def test_each_new_chat_with_blank_seed_gets_its_own_recorded_seed(self) -> None:
        with mock.patch("gpt2_local.chat.secrets.randbelow", side_effect=[123, 456]) as choose:
            first = self.app.new_session(0, "2026-10-01")
            second = self.app.new_session(0, "2026-10-01")
        self.assertEqual(first.settings.seed, 123)
        self.assertEqual(second.settings.seed, 456)
        self.assertEqual(choose.call_args_list, [mock.call(2**63), mock.call(2**63)])
        self.assertEqual(self.runner.calls, [])

    def test_legacy_import_chooses_only_future_seed_and_keeps_historical_metadata_unknown(self) -> None:
        session = self.fixture()
        session.settings = GenerationSettings(max_new_tokens=80, temperature=0.7)
        archive = export_chat(session)
        original = copy.deepcopy(archive)
        with mock.patch("gpt2_local.chat.secrets.randbelow", return_value=789) as choose:
            status, imported = self.request("POST", "/api/import", {"chat": archive, "previous_session_id": session.session_id})
            self.assertEqual(status, 200)
            self.assertEqual(imported["settings"]["seed"], "789")
            self.assertEqual(imported["context_text"], original["context"]["text"])
            self.assertEqual(imported["turns"], [item.record() for item in session.transcript])
            self.assertIsNone(imported["last_generation"])
            self.assertTrue(all(item["generation"] is None for item in imported["turns"]))
            self.assertIn(UNKNOWN_GENERATION_NOTE, imported["notes"])
            self.assertGreater(len(imported["notes"]), len(original["notes"]))
            restored_archive = self.export(imported["session_id"], {"temperature": 0.7, "max_new_tokens": 80, "seed": None})
            self.assertEqual(restored_archive["settings"]["seed"], "789")
            self.assertEqual(restored_archive["blocks"], original["blocks"])
            status, resumed = self.request("POST", "/api/message", {"session_id": imported["session_id"], "message": "A new synthetic question?", "settings": {"temperature": 0.7, "max_new_tokens": 80, "seed": None}})
            self.assertEqual(status, 200)
            self.assertEqual(resumed["last_generation"]["seed"], "789")
            self.assertTrue(all(item["generation"] is None for item in resumed["turns"][:-1]))
            choose.assert_called_once_with(2**63)
        self.assertEqual(archive, original)

    def test_direct_legacy_session_resolves_seed_once_on_first_successful_reply(self) -> None:
        session = self.fixture()
        session.settings = GenerationSettings(max_new_tokens=80, temperature=0.7)
        with mock.patch("gpt2_local.chat.secrets.randbelow", return_value=456) as choose:
            for message in ("First synthetic question?", "Second synthetic question?"):
                status, state = self.request("POST", "/api/message", {"session_id": session.session_id, "message": message, "settings": {"max_new_tokens": 40, "seed": None}})
                self.assertEqual(status, 200)
                self.assertEqual(state["settings"]["seed"], "456")
                self.assertEqual(state["last_generation"]["seed"], "456")
            choose.assert_called_once_with(2**63)

    def test_explicit_63_bit_replay_seed_survives_export_import_and_next_controls(self) -> None:
        seed = str(2**63 - 1)
        controls = {"temperature": 0.8, "max_new_tokens": 40, "seed": seed}
        with mock.patch("gpt2_local.chat.secrets.randbelow", side_effect=AssertionError("Explicit seeds must not be replaced")):
            session = self.app.new_session(0, "2026-10-01", settings=GenerationSettings(max_new_tokens=40, seed=0))
            self.assertEqual(session.settings.seed, 0)
            status, generated = self.request("POST", "/api/message", {"session_id": session.session_id, "message": "First synthetic question?", "settings": controls})
            self.assertEqual(status, 200)
            archive = self.export(session.session_id)
            status, imported = self.request("POST", "/api/import", {"chat": archive, "previous_session_id": session.session_id})
            self.assertEqual(status, 200)
            self.assertEqual(imported["settings"]["seed"], seed)
            self.assertEqual(imported["last_generation"], generated["last_generation"])
            self.assertEqual(len(self.runner.calls), 1)
            replay_controls = {key: imported["settings"][key] for key in ("temperature", "max_new_tokens", "seed")}
            status, resumed = self.request("POST", "/api/message", {"session_id": imported["session_id"], "message": "Second synthetic question?", "settings": replay_controls})
        self.assertEqual(status, 200)
        self.assertEqual(self.runner.calls[-1][1].seed, 2**63 - 1)
        self.assertEqual(resumed["last_generation"]["requested_seed"], seed)
        self.assertEqual(resumed["last_generation"]["seed"], seed)
        self.assert_exact_blocks(resumed)

    def test_generation_failure_does_not_change_saved_controls_or_historical_metadata(self) -> None:
        session = self.fixture()
        session.settings = GenerationSettings(max_new_tokens=80, temperature=0.7, seed=123)
        before = self.state(session)
        self.runner.failure = RuntimeError("synthetic generation failure")
        with mock.patch("gpt2_local.chat.secrets.randbelow", side_effect=AssertionError("A saved chat seed must be reused")):
            for requested, effective in ((None, 123), ("456", 456)):
                with self.subTest(requested_seed=requested):
                    status, error = self.request("POST", "/api/message", {"session_id": session.session_id, "message": "Unsent question?", "settings": {"temperature": 0.2, "max_new_tokens": 30, "seed": requested}})
                    self.assertEqual(status, 500)
                    self.assertIn("preserved", error["error"])
                    self.assertEqual(self.runner.calls[-1][1].seed, effective)
                    self.assertEqual(self.state(session), before)

    def test_import_accepts_large_full_history_above_normal_request_limit(self) -> None:
        session = self.fixture()
        transcript = [Round(f"Synthetic old round {index}: " + "h" * 700, " " + "a" * 700) for index in range(60)]
        session.example_rounds = 0
        session.transcript = transcript
        session.rounds = list(transcript[-1:])
        session.dropped_rounds = len(transcript) - 1
        archive = self.export(session.session_id)
        payload = {"chat": archive, "previous_session_id": session.session_id}
        encoded = json.dumps(payload).encode("utf-8")
        self.assertGreater(len(encoded), 65536)
        self.assertLess(len(encoded), 4 * 1024 * 1024)
        status, imported = self.request("POST", "/api/import", raw=encoded)
        self.assertEqual(status, 200)
        self.assertEqual(len(imported["turns"]), len(transcript))
        self.assertEqual(imported["dropped_rounds"], len(transcript) - 1)
        self.assertEqual(imported["context_text"], session.context_text())
        self.assertEqual(self.runner.calls, [])

    def test_transfer_body_size_errors_preserve_old_session(self) -> None:
        session = self.fixture()
        before = self.state(session)
        for path, maximum in (("/api/import", 4 * 1024 * 1024), ("/api/export", 65536)):
            with self.subTest(path=path):
                status, error = self.request("POST", path, raw=b"{}", headers={"Content-Length": str(maximum + 1)})
                self.assertEqual(status, 400)
                self.assertIn(str(maximum), error["error"])
                self.assertEqual(self.state(session), before)
        self.assertEqual(self.runner.calls, [])

    def test_transfer_while_model_busy_returns_409_without_replacing_session(self) -> None:
        session = self.fixture()
        before = self.state(session)
        archive = self.export(session.session_id)
        with self.app.lock:
            for path, body in (
                ("/api/export", {"session_id": session.session_id}),
                ("/api/import", {"chat": archive, "previous_session_id": session.session_id}),
            ):
                with self.subTest(path=path):
                    status, error = self.request("POST", path, body)
                    self.assertEqual(status, 409)
                    self.assertIn("busy", error["error"])
        self.assertEqual(self.state(session), before)
        self.assertEqual(self.runner.calls, [])


if __name__ == "__main__":
    unittest.main()
