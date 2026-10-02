from __future__ import annotations

import copy
import unittest
from datetime import date

from gpt2_local.chat import (
    HUMAN_MARKER,
    MODEL_MARKER,
    OMISSION_MARKER,
    ChatApplication,
    ChatSession,
    Round,
    prompt_header,
)
from gpt2_local.chat_archive import export_chat, import_chat
from gpt2_local.runtime import MODEL_CONTEXT_TOKENS, GenerationResult, GenerationSettings


class OmissionRunner:
    """Count characters as synthetic token units without loading a model."""

    def __init__(self) -> None:
        self.text = " A synthetic answer."
        self.failure: Exception | None = None
        self.input_token_delta = 0
        self.prompt_truncated = False
        self.calls: list[tuple[str, GenerationSettings]] = []

    @staticmethod
    def count_prompt_tokens(text: str) -> int:
        return len(text)

    def generate_result(
        self, prompt: str, settings: GenerationSettings, *, stop_pattern: str | None = None,
    ) -> GenerationResult:
        self.calls.append((prompt, settings))
        if self.failure is not None:
            raise self.failure
        count = self.count_prompt_tokens(prompt)
        return GenerationResult(
            text=self.text,
            original_input_tokens=count,
            input_tokens=count + self.input_token_delta,
            output_tokens=2,
            output_token_ids=(11, 12),
            stop_reason="max_new_tokens",
            elapsed_seconds=0.02,
            tokens_per_second=100,
            prompt_truncated=self.prompt_truncated,
            cold_start_included=len(self.calls) == 1,
        )


class ChatOmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = OmissionRunner()

    @staticmethod
    def session(rounds: list[Round] | None = None) -> ChatSession:
        initial = list(rounds or [])
        return ChatSession(
            "synthetic-omission", "Synthetic fixed header.",
            sum(item.example for item in initial), list(initial), list(initial),
            settings=GenerationSettings(max_new_tokens=60, seed=123),
        )

    @staticmethod
    def input_text(header: str, rounds: list[Round], message: str) -> str:
        return header + "".join(item.text() for item in rounds) + HUMAN_MARKER + " " + message + MODEL_MARKER

    @staticmethod
    def reserve_for(prompt: str) -> GenerationSettings:
        return GenerationSettings(max_new_tokens=MODEL_CONTEXT_TOKENS - len(prompt), seed=123)

    def eviction_case(self) -> tuple[ChatSession, Round, Round, str, str, GenerationSettings]:
        old = Round("Old question?", " Old answer.")
        recent = Round("Recent question?", " Recent answer.")
        session = self.session([old, recent])
        message = "Latest question?"
        expected = self.input_text(session.header + OMISSION_MARKER, [recent], message)
        return session, old, recent, message, expected, self.reserve_for(expected)

    def test_no_eviction_preserves_the_original_header_and_prompt(self) -> None:
        session = self.session([Round("Earlier question?", " Earlier answer.")])
        original_header = session.header
        expected = self.input_text(original_header, session.rounds, "Question with ... in it?")
        preview = session.preview("Question with ... in it?", session.settings, len)
        self.assertEqual(preview["context_text"], expected)
        session.reply("Question with ... in it?", session.settings, self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.header, original_header)
        self.assertEqual(session.dropped_rounds, 0)
        self.assertEqual(self.runner.calls[-1][0], expected)
        self.assertNotIn(OMISSION_MARKER, session.header)

    def test_preview_shows_the_proposed_marked_header_without_committing_it(self) -> None:
        session, old, recent, message, expected, settings = self.eviction_case()
        before = copy.deepcopy(session.state(len))
        prepared = session.prepare(message, settings, len)
        preview = session.preview(message, settings, len)
        self.assertEqual(OMISSION_MARKER, "\n\n...\n\n")
        self.assertEqual(prepared.header, session.header + OMISSION_MARKER)
        self.assertEqual(prepared.retained_rounds, [recent])
        self.assertEqual(preview["context_text"], expected)
        self.assertEqual(preview["context_tokens"], len(expected))
        self.assertEqual(preview["context_blocks"][0], {"role": "context", "text": prepared.header})
        self.assertEqual("".join(block["text"] for block in preview["context_blocks"]), expected)
        self.assertEqual(preview["dropped_rounds"], 1)
        self.assertEqual(session.state(len), before)
        self.assertEqual(session.transcript, [old, recent])
        self.assertEqual(self.runner.calls, [])

    def test_first_successful_eviction_commits_marker_and_preserves_full_transcript(self) -> None:
        session, old, recent, message, expected, settings = self.eviction_case()
        old_header = session.header
        session.reply(message, settings, self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.header, old_header + OMISSION_MARKER)
        self.assertEqual(self.runner.calls[-1][0], expected)
        self.assertEqual(session.last_generation["prompt_text"], expected)
        self.assertEqual(session.last_generation["input_tokens"] + settings.max_new_tokens, MODEL_CONTEXT_TOKENS)
        self.assertEqual(session.rounds[:-1], [recent])
        self.assertEqual(session.transcript[:2], [old, recent])
        self.assertEqual(session.dropped_rounds, 1)
        state = session.state(len)
        self.assertEqual(state["context_blocks"][0]["text"], session.header)
        self.assertEqual("".join(block["text"] for block in state["context_blocks"]), state["context_text"])

    def test_repeated_eviction_keeps_one_marker_and_increments_only_drop_count(self) -> None:
        session, _, _, message, _, settings = self.eviction_case()
        session.reply(message, settings, self.runner)  # type: ignore[arg-type]
        marked_header = session.header
        retained = session.rounds[-1]
        expected = self.input_text(marked_header, [retained], "Another question?")
        session.reply("Another question?", self.reserve_for(expected), self.runner)  # type: ignore[arg-type]
        self.assertEqual(self.runner.calls[-1][0], expected)
        self.assertEqual(session.header, marked_header)
        self.assertEqual(session.header.count(OMISSION_MARKER), 1)
        self.assertEqual(session.dropped_rounds, 2)
        self.assertEqual(session.rounds[:-1], [retained])
        self.assertEqual(len(session.transcript), 4)

    def test_marker_token_cost_can_force_an_additional_old_round_out(self) -> None:
        session, _, recent, message, _, _ = self.eviction_case()
        unmarked_budget_prompt = self.input_text(session.header, [recent], message)
        settings = self.reserve_for(unmarked_budget_prompt)
        expected = self.input_text(session.header + OMISSION_MARKER, [], message)
        preview = session.preview(message, settings, len)
        self.assertEqual(preview["context_text"], expected)
        self.assertEqual(preview["dropped_rounds"], 2)
        session.reply(message, settings, self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.dropped_rounds, 2)
        self.assertEqual(len(session.rounds), 1)
        self.assertEqual(self.runner.calls[-1][0], expected)
        self.assertLessEqual(len(expected) + settings.max_new_tokens, MODEL_CONTEXT_TOKENS)

    def test_exact_marker_budget_is_accepted_and_one_token_short_is_rejected(self) -> None:
        old = Round("Old question?", " Old answer.")
        message = "Latest question?"
        session = self.session([old])
        expected = self.input_text(session.header + OMISSION_MARKER, [], message)
        session.reply(message, self.reserve_for(expected), self.runner)  # type: ignore[arg-type]
        self.assertEqual(self.runner.calls[-1][0], expected)
        rejecting = self.session([old])
        before = copy.deepcopy(rejecting.state(len))
        settings = GenerationSettings(max_new_tokens=MODEL_CONTEXT_TOKENS - len(expected) + 1, seed=123)
        for operation in (
            lambda: rejecting.preview(message, settings, len),
            lambda: rejecting.reply(message, settings, self.runner),
        ):
            with self.assertRaisesRegex(ValueError, "Shorten the message|reduce"):
                operation()
            self.assertEqual(rejecting.state(len), before)
        self.assertEqual(len(self.runner.calls), 1)

    def test_inference_and_token_accounting_failures_preserve_the_unmarked_chat(self) -> None:
        for failure, delta, truncated in (
            (RuntimeError("Synthetic sampling failure"), 0, False),
            (None, 1, False),
            (None, 0, True),
        ):
            with self.subTest(failure=failure, delta=delta, truncated=truncated):
                self.runner = OmissionRunner()
                self.runner.failure = failure
                self.runner.input_token_delta = delta
                self.runner.prompt_truncated = truncated
                session, _, _, message, expected, settings = self.eviction_case()
                before = copy.deepcopy(session.state(len))
                with self.assertRaises(RuntimeError):
                    session.reply(message, settings, self.runner)  # type: ignore[arg-type]
                self.assertEqual(self.runner.calls[-1][0], expected)
                self.assertEqual(session.state(len), before)
                self.assertNotIn(OMISSION_MARKER, session.header)

    def test_first_eviction_of_a_seeded_example_also_marks_the_header(self) -> None:
        example = Round("Example question?", " Example answer.", example=True)
        recent = Round("Real question?", " Real answer.")
        session = self.session([example, recent])
        expected = self.input_text(session.header + OMISSION_MARKER, [recent], "Latest question?")
        preview = session.preview("Latest question?", self.reserve_for(expected), len)
        self.assertEqual(preview["retained_example_rounds"], 0)
        session.reply("Latest question?", self.reserve_for(expected), self.runner)  # type: ignore[arg-type]
        self.assertEqual(session.header.count(OMISSION_MARKER), 1)
        self.assertEqual(session.dropped_rounds, 1)
        self.assertTrue(session.transcript[0].example)
        self.assertEqual(session.example_rounds, 1)
        self.assertEqual(session.state(len)["retained_example_rounds"], 0)

    def legacy_archive(self, version: int = 2) -> dict:
        old = Round("Omitted historical question?", " Historical answer.")
        recent = Round("Retained historical question?", " Retained answer.")
        header = "Synthetic historical header."
        prompt = self.input_text(header, [recent], "Latest legacy question?")
        generation = {
            "input_tokens": len(prompt), "output_tokens": 2, "output_token_ids": [11, 12],
            "stop_reason": "max_new_tokens", "elapsed_seconds": 0.02, "cold_start_included": False,
            "seed": "123", "requested_seed": "123", "temperature": 0.8,
            "max_new_tokens": 60, "prompt_text": prompt,
        }
        last = Round("Latest legacy question?", " Legacy answer.", generation=generation)
        session = ChatSession(
            "synthetic-legacy", header, 0, [recent, last], [old, recent, last],
            dropped_rounds=1, last_generation=copy.deepcopy(generation),
            settings=GenerationSettings(max_new_tokens=60, seed=123),
        )
        archive = export_chat(session)
        if version == 1:
            archive["schema_version"] = 1
            for block in archive["blocks"]:
                block.pop("previous_responses", None)
        return archive

    def test_legacy_dropped_import_and_regeneration_keep_the_exact_original_prompt(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version):
                archive = self.legacy_archive(version)
                restored = import_chat(archive, len)
                original_header = restored.header
                original_prompt = restored.last_generation["prompt_text"]
                self.assertEqual(restored.context_text(), archive["context"]["text"])
                self.assertEqual(restored.dropped_rounds, 1)
                self.assertNotIn(OMISSION_MARKER, restored.header)
                preview = restored.regeneration_preview(GenerationSettings(max_new_tokens=60), len)
                self.assertEqual(preview["context_text"], original_prompt)
                restored.regenerate(GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
                self.assertEqual(self.runner.calls[-1][0], original_prompt)
                self.assertEqual(restored.header, original_header)
                self.assertEqual(restored.last_generation["seed"], "124")
                self.assertEqual(restored.dropped_rounds, 1)

    def test_first_new_reply_to_legacy_dropped_chat_adds_marker_without_more_eviction(self) -> None:
        for version in (1, 2):
            with self.subTest(version=version):
                restored = import_chat(self.legacy_archive(version), len)
                original_header = restored.header
                retained = list(restored.rounds)
                before = copy.deepcopy(restored.state(len))
                expected = self.input_text(original_header + OMISSION_MARKER, retained, "Continue?")
                preview = restored.preview("Continue?", GenerationSettings(max_new_tokens=60), len)
                self.assertEqual(preview["context_text"], expected)
                self.assertEqual(preview["dropped_rounds"], 1)
                self.assertEqual(restored.state(len), before)
                restored.reply("Continue?", GenerationSettings(max_new_tokens=60), self.runner)  # type: ignore[arg-type]
                self.assertEqual(self.runner.calls[-1][0], expected)
                self.assertEqual(restored.header, original_header + OMISSION_MARKER)
                self.assertEqual(restored.dropped_rounds, 1)
                self.assertEqual(restored.rounds[:-1], retained)

    def test_legacy_marker_overflow_preserves_header_and_history_without_sampling(self) -> None:
        restored = import_chat(self.legacy_archive(), len)
        restored.rounds = []
        restored.dropped_rounds = len(restored.transcript)
        message = "Continue?"
        unmarked = self.input_text(restored.header, [], message)
        before = copy.deepcopy(restored.state(len))
        with self.assertRaises(ValueError):
            restored.reply(message, self.reserve_for(unmarked), self.runner)  # type: ignore[arg-type]
        self.assertEqual(restored.state(len), before)
        self.assertEqual(self.runner.calls, [])

    def test_existing_marked_header_is_never_appended_again(self) -> None:
        for previous_drops in (0, 1):
            with self.subTest(previous_drops=previous_drops):
                session = self.session([Round("Retained question?", " Retained answer.")])
                session.header += OMISSION_MARKER
                session.dropped_rounds = previous_drops
                marked_header = session.header
                expected = self.input_text(marked_header, [], "Latest question?")
                session.reply("Latest question?", self.reserve_for(expected), self.runner)  # type: ignore[arg-type]
                self.assertEqual(self.runner.calls[-1][0], expected)
                self.assertEqual(session.header, marked_header)
                self.assertEqual(session.header.count(OMISSION_MARKER), 1)
                self.assertEqual(session.dropped_rounds, previous_drops + 1)

    def test_v2_roundtrip_preserves_marker_and_regeneration_after_actual_eviction(self) -> None:
        session, _, _, message, expected, settings = self.eviction_case()
        session.reply(message, settings, self.runner)  # type: ignore[arg-type]
        session.regenerate(settings, self.runner)  # type: ignore[arg-type]
        document = export_chat(session)
        self.assertEqual(document["schema_version"], 2)
        self.assertEqual(document["prompt_header"], session.header)
        self.assertEqual(document["context"]["first_retained_round"], 1)
        restored = import_chat(document, len)
        self.assertEqual(restored.context_text(), session.context_text())
        self.assertEqual(restored.header, session.header)
        self.assertEqual(restored.rounds[-1].previous_responses, session.rounds[-1].previous_responses)
        restored.regenerate(settings, self.runner)  # type: ignore[arg-type]
        self.assertEqual(self.runner.calls[-1][0], expected)
        self.assertEqual(restored.header.count(OMISSION_MARKER), 1)
        self.assertEqual(restored.dropped_rounds, 1)
        self.assertEqual(len(restored.rounds[-1].previous_responses), 2)

    def test_new_chat_starts_with_unmarked_header_and_no_removed_rounds(self) -> None:
        app = ChatApplication(self.runner)  # type: ignore[arg-type]
        previous = app.new_session(0, "2026-10-02", settings=GenerationSettings(max_new_tokens=60, seed=123))
        previous.header += OMISSION_MARKER
        previous.dropped_rounds = 1
        fresh = app.new_session(0, "2026-10-02", previous.session_id, GenerationSettings(max_new_tokens=60, seed=123))
        self.assertEqual(fresh.header, prompt_header(date(2026, 10, 2)))
        self.assertNotIn(OMISSION_MARKER, fresh.header)
        self.assertEqual(fresh.dropped_rounds, 0)
        self.assertEqual(fresh.rounds, [])
        self.assertEqual(fresh.transcript, [])


if __name__ == "__main__":
    unittest.main()
