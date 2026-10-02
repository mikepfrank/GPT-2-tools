from __future__ import annotations

import copy
import json
import threading
import unittest
from http.client import HTTPConnection

from gpt2_local.chat import (
    HUMAN_MARKER,
    MESSAGE_DELIMITER_PATTERN,
    MODEL_MARKER,
    OMISSION_MARKER,
    ChatApplication,
    ChatServer,
    ChatSession,
    Round,
)
from gpt2_local.chat_archive import export_chat, import_chat
from gpt2_local.runtime import MODEL_CONTEXT_TOKENS, GenerationResult, GenerationSettings


class RegenerateRunner:
    """Synthetic inference with character counts; never maps a model."""

    def __init__(self) -> None:
        self.text = " First synthetic answer."
        self.failure: Exception | None = None
        self.input_token_delta = 0
        self.prompt_truncated = False
        self.calls: list[tuple[str, GenerationSettings, str | None]] = []

    @staticmethod
    def count_prompt_tokens(text: str) -> int:
        return len(text)

    def generate_result(
        self, prompt: str, settings: GenerationSettings, *, stop_pattern: str | None = None,
    ) -> GenerationResult:
        self.calls.append((prompt, settings, stop_pattern))
        if self.failure is not None:
            raise self.failure
        count = len(prompt)
        return GenerationResult(
            text=self.text,
            original_input_tokens=count,
            input_tokens=count + self.input_token_delta,
            output_tokens=3,
            output_token_ids=(11, 12, 13),
            stop_reason="max_new_tokens",
            elapsed_seconds=0.02,
            tokens_per_second=150,
            prompt_truncated=self.prompt_truncated,
            cold_start_included=len(self.calls) == 1,
        )


class ChatRegenerateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = RegenerateRunner()

    def session(self, seed: int = 123) -> ChatSession:
        return ChatSession(
            "synthetic-regenerate", "Synthetic fixed header.", 0,
            settings=GenerationSettings(max_new_tokens=60, temperature=0.8, seed=seed),
        )

    def completed(self, seed: int = 123) -> ChatSession:
        session = self.session(seed)
        session.reply("A synthetic question?", session.settings, self.runner)  # type: ignore[arg-type]
        return session

    def test_regeneration_replaces_only_last_answer_and_records_rejected_response(self) -> None:
        session = self.completed()
        first_round = session.rounds[-1]
        original = copy.deepcopy(session.state(len))
        original_prompt = self.runner.calls[0][0]
        self.runner.text = " A different answer.\nSecond line.  "
        session.regenerate(GenerationSettings(max_new_tokens=60, seed=None), self.runner)  # type: ignore[arg-type]

        self.assertEqual(self.runner.calls[-1][0], original_prompt)
        self.assertEqual(self.runner.calls[-1][1].seed, 124)
        self.assertEqual(self.runner.calls[-1][2], MESSAGE_DELIMITER_PATTERN)
        self.assertEqual(len(session.rounds), 1)
        self.assertEqual(len(session.transcript), 1)
        self.assertIs(session.rounds[-1], session.transcript[-1])
        self.assertEqual(session.rounds[-1].human, first_round.human)
        self.assertEqual(session.rounds[-1].assistant, self.runner.text)
        self.assertEqual(first_round.assistant, original["turns"][-1]["assistant"])
        self.assertEqual(first_round.generation, original["last_generation"])
        self.assertEqual(first_round.previous_responses, [])
        self.assertEqual(session.rounds[-1].previous_responses, [{
            "text": first_round.assistant, "generation": original["last_generation"],
        }])
        self.assertEqual(session.last_generation["seed"], "124")
        self.assertIsNone(session.last_generation["requested_seed"])
        self.assertEqual(session.settings.seed, 124)
        self.assertEqual(session.state(len)["turns"][-1]["previous_responses"], session.rounds[-1].previous_responses)
        self.assertEqual(session.context_text(), session.header + session.rounds[-1].text())
        self.assertNotIn(first_round.assistant, session.context_text())

    def test_repeated_regeneration_keeps_prompt_and_advances_seed_without_new_human_turns(self) -> None:
        session = self.completed()
        initial_prompt = self.runner.calls[-1][0]
        for index, expected_seed in enumerate((124, 125, 126), start=1):
            self.runner.text = f" Synthetic candidate {index}."
            session.regenerate(GenerationSettings(max_new_tokens=60, seed=None), self.runner)  # type: ignore[arg-type]
            self.assertEqual(self.runner.calls[-1][0], initial_prompt)
            self.assertEqual(self.runner.calls[-1][1].seed, expected_seed)
            self.assertEqual(len(session.rounds), 1)
            self.assertEqual(len(session.transcript), 1)
            self.assertEqual(len(session.rounds[-1].previous_responses), index)
        self.assertEqual([item["generation"]["seed"] for item in session.rounds[-1].previous_responses], ["123", "124", "125"])
        session.reply("Next synthetic question?", GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        self.assertEqual(len(session.transcript), 2)
        self.assertEqual(session.transcript[-1].previous_responses, [])
        self.assertEqual(session.last_generation["seed"], "126")
        self.assertIn(" Synthetic candidate 3.", self.runner.calls[-1][0])
        self.assertNotIn(" First synthetic answer.", self.runner.calls[-1][0])

    def test_ui_seed_is_incremented_and_63_bit_limit_wraps_to_zero(self) -> None:
        for requested, expected in ((0, 1), (9007199254740993, 9007199254740994), (2**63 - 1, 0)):
            with self.subTest(requested=requested):
                session = self.completed(17)
                session.regenerate(GenerationSettings(max_new_tokens=60, seed=requested), self.runner)  # type: ignore[arg-type]
                self.assertEqual(self.runner.calls[-1][1].seed, expected)
                self.assertEqual(session.settings.seed, expected)
                self.assertEqual(session.last_generation["requested_seed"], str(requested))
                self.assertEqual(session.last_generation["seed"], str(expected))
                self.assertEqual(session.state(len)["settings"]["seed"], str(expected))

    def test_preview_uses_saved_prompt_and_upcoming_seed_without_mutation_or_sampling(self) -> None:
        session = self.completed(2**63 - 1)
        before = copy.deepcopy(session.state(len))
        preview = session.regeneration_preview(GenerationSettings(max_new_tokens=40, seed=None), len)
        self.assertEqual(preview["context_text"], session.last_generation["prompt_text"])
        self.assertEqual(preview["context_tokens"], len(preview["context_text"]))
        self.assertEqual(preview["settings"]["seed"], "0")
        self.assertEqual("".join(item["text"] for item in preview["context_blocks"]), preview["context_text"])
        self.assertEqual(preview["context_blocks"][-2], {"role": "human", "text": HUMAN_MARKER + " " + session.rounds[-1].human})
        self.assertEqual(preview["context_blocks"][-1], {"role": "model", "text": MODEL_MARKER})
        self.assertEqual(session.state(len), before)
        self.assertEqual(len(self.runner.calls), 1)

    def test_exact_prompt_is_reused_even_after_original_reply_evicted_an_older_round(self) -> None:
        oldest = Round("Old synthetic question?", " Old synthetic reply.")
        recent = Round("Recent synthetic question?", " Recent synthetic reply.")
        session = self.session(31)
        session.rounds = [oldest, recent]
        session.transcript = [oldest, recent]
        question = "Latest synthetic question?"
        expected_prompt = session.header + OMISSION_MARKER + recent.text() + HUMAN_MARKER + " " + question + MODEL_MARKER
        settings = GenerationSettings(max_new_tokens=MODEL_CONTEXT_TOKENS - len(expected_prompt), seed=31)
        session.reply(question, settings, self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.dropped_rounds, 1)
        self.assertEqual(self.runner.calls[-1][0], expected_prompt)
        self.runner.text = " Another synthetic reply."
        for expected_seed in (32, 33):
            session.regenerate(GenerationSettings(max_new_tokens=settings.max_new_tokens), self.runner)  # type: ignore[arg-type]
            self.assertEqual(self.runner.calls[-1][0], expected_prompt)
            self.assertEqual(self.runner.calls[-1][1].seed, expected_seed)
            self.assertEqual(session.dropped_rounds, 1)
            self.assertEqual(session.rounds[:-1], [recent])
            self.assertEqual(session.transcript[:2], [oldest, recent])
            self.assertEqual(len(session.transcript), 3)
            self.assertEqual(session.rounds, session.transcript[1:])

    def test_larger_reply_limit_cannot_silently_evict_or_truncate_the_saved_prompt(self) -> None:
        session = self.completed()
        before = copy.deepcopy(session.state(len))
        too_large = GenerationSettings(max_new_tokens=MODEL_CONTEXT_TOKENS - len(session.last_generation["prompt_text"]) + 1)
        for operation in (
            lambda: session.regeneration_preview(too_large, len),
            lambda: session.regenerate(too_large, self.runner),
        ):
            with self.assertRaises(ValueError):
                operation()
            self.assertEqual(session.state(len), before)
            self.assertEqual(len(self.runner.calls), 1)

    def test_archived_prompt_must_match_current_header_and_retained_turns(self) -> None:
        session = self.completed()
        archive = export_chat(session)
        generation = archive["blocks"][-1]["generation"]
        generation["prompt_text"] = "Synthetic unrelated prompt" + MODEL_MARKER
        generation["input_tokens"] = len(generation["prompt_text"])
        restored = import_chat(archive, len)
        before = copy.deepcopy(restored.state(len))
        for operation in (
            lambda: restored.regeneration_preview(GenerationSettings(max_new_tokens=60), len),
            lambda: restored.regenerate(GenerationSettings(max_new_tokens=60), self.runner),
        ):
            with self.assertRaisesRegex(ValueError, "recorded|match"):
                operation()
            self.assertEqual(restored.state(len), before)
            self.assertEqual(len(self.runner.calls), 1)

    def test_regeneration_preserves_generic_delimiter_stopping_and_raw_generation_record(self) -> None:
        session = self.completed()
        original = copy.deepcopy(session.last_generation)
        self.runner.text = " A replacement with a trailing space. \n\nAI> simulated extra participant"
        session.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.rounds[-1].assistant, " A replacement with a trailing space. ")
        self.assertEqual(session.last_generation["stop_reason"], "stop_sequence")
        self.assertEqual(session.last_generation["output_tokens"], 3)
        self.assertEqual(session.last_generation["output_token_ids"], [11, 12, 13])
        self.assertEqual(session.rounds[-1].previous_responses[0]["generation"], original)
        self.assertNotIn("simulated extra participant", session.context_text())

    def test_greedy_regeneration_is_rejected_before_preview_or_inference(self) -> None:
        session = self.completed()
        before = copy.deepcopy(session.state(len))
        settings = GenerationSettings(max_new_tokens=60, temperature=0, seed=555)
        for operation in (
            lambda: session.regeneration_preview(settings, len),
            lambda: session.regenerate(settings, self.runner),
        ):
            with self.assertRaisesRegex(ValueError, "[Tt]emperature|greedy"):
                operation()
            self.assertEqual(session.state(len), before)
            self.assertEqual(len(self.runner.calls), 1)

    def test_empty_example_only_unknown_metadata_and_unretained_last_turn_are_ineligible(self) -> None:
        empty = self.session()
        example = Round("Example question?", " Example reply.", example=True)
        examples = self.session()
        examples.example_rounds = 1
        examples.rounds = [example]
        examples.transcript = [example]
        historical = self.session()
        historical.rounds = [Round("Historical question?", " Historical reply.")]
        historical.transcript = list(historical.rounds)
        evicted = self.completed()
        evicted.dropped_rounds = len(evicted.transcript)
        evicted.rounds = []
        for session in (empty, examples, historical, evicted):
            with self.subTest(session=session):
                before = copy.deepcopy(session.state(len))
                self.assertFalse(session.can_regenerate())
                self.assertFalse(before["can_regenerate"])
                with self.assertRaises(ValueError):
                    session.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
                self.assertEqual(session.state(len), before)
        self.assertEqual(len(self.runner.calls), 1)

    def test_failures_leave_old_answer_seed_settings_history_and_metadata_intact(self) -> None:
        cases = ((RuntimeError("Synthetic inference failure"), 0, False), (None, 1, False), (None, 0, True))
        for failure, delta, truncated in cases:
            with self.subTest(failure=failure, delta=delta, truncated=truncated):
                self.runner = RegenerateRunner()
                session = self.completed()
                before = copy.deepcopy(session.state(len))
                self.runner.failure = failure
                self.runner.input_token_delta = delta
                self.runner.prompt_truncated = truncated
                with self.assertRaises(RuntimeError):
                    session.regenerate(GenerationSettings(max_new_tokens=50, temperature=0.5, seed=456), self.runner)  # type: ignore[arg-type]
                self.assertEqual(session.state(len), before)
                self.assertEqual(self.runner.calls[-1][1].seed, 457)

    def test_discarded_response_metadata_and_earlier_candidates_are_deep_copied(self) -> None:
        session = self.completed()
        original_round = session.rounds[-1]
        original_metadata = copy.deepcopy(original_round.generation)
        session.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        first_replacement = session.rounds[-1]
        session.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        original_round.generation["output_token_ids"].append(99)
        first_replacement.previous_responses[0]["generation"]["seed"] = "999"
        self.assertEqual(session.rounds[-1].previous_responses[0]["generation"], original_metadata)
        self.assertEqual(session.rounds[-1].previous_responses[1]["generation"]["seed"], "124")

    def test_candidate_limit_rejects_generation_without_mutating_or_sampling(self) -> None:
        session = self.completed()
        item = session.rounds[-1]
        candidate = {"text": item.assistant, "generation": copy.deepcopy(item.generation)}
        full = Round(item.human, item.assistant, generation=item.generation, previous_responses=[copy.deepcopy(candidate) for _ in range(1000)])
        session.rounds[-1] = full
        session.transcript[-1] = full
        before = copy.deepcopy(session.state(len))
        with self.assertRaises(ValueError):
            session.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.state(len), before)
        self.assertEqual(len(self.runner.calls), 1)

    def test_regenerated_archive_restores_eligibility_and_future_retries_without_losing_candidates(self) -> None:
        session = self.completed()
        self.runner.text = " A replacement answer."
        session.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        archive = export_chat(session)
        self.assertEqual(archive["schema_version"], 2)
        restored = import_chat(archive, len)
        self.assertTrue(restored.can_regenerate())
        self.assertEqual(restored.context_text(), session.context_text())
        self.assertEqual(restored.rounds[-1].previous_responses, session.rounds[-1].previous_responses)
        self.assertEqual(restored.last_generation, session.last_generation)
        self.runner.text = " A post-import replacement."
        restored.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
        self.assertEqual(restored.last_generation["seed"], "125")
        self.assertEqual(self.runner.calls[-1][0], session.last_generation["prompt_text"])
        self.assertEqual(len(restored.rounds[-1].previous_responses), 2)


class ChatRegenerateHttpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = RegenerateRunner()
        self.app = ChatApplication(self.runner)  # type: ignore[arg-type]
        self.server = ChatServer(("127.0.0.1", 0), self.app)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()
        self.session = ChatSession(
            "synthetic-http-regenerate", "Synthetic header.", 0,
            settings=GenerationSettings(max_new_tokens=60, seed=2**63 - 1),
        )
        self.app.sessions[self.session.session_id] = self.session

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def request(self, path: str, body: object, *, method: str = "POST") -> tuple[int, dict]:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        try:
            connection.request(method, path, body=json.dumps(body).encode("utf-8") if method == "POST" else None, headers={"Content-Type": "application/json"})
            response = connection.getresponse()
            self.assertEqual(response.getheader("Cache-Control"), "no-store")
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def state(self) -> dict:
        status, state = self.request("/api/state?session_id=" + self.session.session_id, {}, method="GET")
        self.assertEqual(status, 200)
        return state

    def complete(self) -> dict:
        status, state = self.request("/api/message", {
            "session_id": self.session.session_id,
            "message": "Synthetic HTTP question?",
            "settings": {"max_new_tokens": 60, "seed": str(2**63 - 1)},
        })
        self.assertEqual(status, 200)
        return state

    def test_http_preview_then_regenerate_wraps_seed_and_keeps_a_single_turn(self) -> None:
        first = self.complete()
        self.assertTrue(first["can_regenerate"])
        request = {"session_id": self.session.session_id, "settings": {"max_new_tokens": 60, "seed": str(2**63 - 1)}}
        status, preview = self.request("/api/regenerate-preview", request)
        self.assertEqual(status, 200)
        self.assertEqual(preview["settings"]["seed"], "0")
        self.assertEqual(preview["context_text"], first["last_generation"]["prompt_text"])
        self.assertEqual(self.state(), first)
        self.assertEqual(len(self.runner.calls), 1)
        self.runner.text = " A replacement HTTP answer."
        status, state = self.request("/api/regenerate", request)
        self.assertEqual(status, 200)
        self.assertEqual(state["settings"]["seed"], "0")
        self.assertEqual(state["last_generation"]["seed"], "0")
        self.assertEqual(state["last_generation"]["requested_seed"], str(2**63 - 1))
        self.assertEqual(len(state["turns"]), 1)
        self.assertEqual(state["turns"][-1]["human"], first["turns"][-1]["human"])
        self.assertEqual(state["turns"][-1]["previous_responses"], [{"text": first["turns"][-1]["assistant"], "generation": first["last_generation"]}])
        self.assertEqual("".join(item["text"] for item in state["context_blocks"]), state["context_text"])

    def test_http_no_eligible_reply_bad_seed_and_greedy_settings_preserve_session(self) -> None:
        empty = self.state()
        self.assertFalse(empty["can_regenerate"])
        status, error = self.request("/api/regenerate", {"session_id": self.session.session_id, "settings": {}})
        self.assertEqual(status, 400)
        self.assertIn("error", error)
        self.assertEqual(self.state(), empty)
        self.assertEqual(self.runner.calls, [])
        first = self.complete()
        for settings in ({"seed": 123}, {"seed": str(2**63)}, {"temperature": 0}, {"max_new_tokens": 1024}):
            for path in ("/api/regenerate-preview", "/api/regenerate"):
                with self.subTest(path=path, settings=settings):
                    status, error = self.request(path, {"session_id": self.session.session_id, "settings": settings})
                    self.assertEqual(status, 400)
                    self.assertIn("error", error)
                    self.assertEqual(self.state(), first)
        self.assertEqual(len(self.runner.calls), 1)

    def test_http_failed_generation_and_busy_requests_preserve_original_answer(self) -> None:
        first = self.complete()
        body = {"session_id": self.session.session_id, "settings": {"max_new_tokens": 60, "seed": "456"}}
        self.runner.failure = RuntimeError("Synthetic regeneration failure")
        status, error = self.request("/api/regenerate", body)
        self.assertEqual(status, 500)
        self.assertIn("preserved", error["error"])
        self.assertEqual(self.state(), first)
        with self.app.lock:
            for path in ("/api/regenerate-preview", "/api/regenerate"):
                status, error = self.request(path, body)
                self.assertEqual(status, 409)
                self.assertIn("busy", error["error"])
        self.assertEqual(self.state(), first)
        self.assertEqual(len(self.runner.calls), 2)

    def test_http_export_import_retains_candidates_and_regeneration_capability(self) -> None:
        first = self.complete()
        self.runner.text = " Synthetic HTTP candidate."
        status, regenerated = self.request("/api/regenerate", {"session_id": self.session.session_id, "settings": {"max_new_tokens": 60}})
        self.assertEqual(status, 200)
        status, archive = self.request("/api/export", {"session_id": self.session.session_id})
        self.assertEqual(status, 200)
        status, restored = self.request("/api/import", {"chat": archive, "previous_session_id": self.session.session_id})
        self.assertEqual(status, 200)
        self.assertTrue(restored["can_regenerate"])
        self.assertEqual(restored["turns"], regenerated["turns"])
        self.assertEqual(restored["settings"], regenerated["settings"])
        self.assertEqual(restored["last_generation"], regenerated["last_generation"])
        self.assertEqual(restored["turns"][-1]["previous_responses"][0]["generation"], first["last_generation"])
        self.assertEqual(len(self.runner.calls), 2)


if __name__ == "__main__":
    unittest.main()
