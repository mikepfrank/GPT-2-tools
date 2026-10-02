from __future__ import annotations

import copy
import json
import unittest

from gpt2_local.chat import EXAMPLES, HUMAN_MARKER, MODEL_MARKER, ChatSession, Round
from gpt2_local.chat_archive import (
    MAX_ARCHIVE_ROUNDS, MAX_NOTES, MAX_TEXT_CHARS, UNKNOWN_GENERATION_NOTE, UNKNOWN_SEED_NOTE,
    export_chat, import_chat,
)
from gpt2_local.runtime import GenerationSettings


class ChatArchiveTests(unittest.TestCase):
    @staticmethod
    def generation(prompt: str, *, seed: str | None = "9223372036854775807") -> dict:
        return {
            "input_tokens": len(prompt), "output_tokens": 2, "output_token_ids": [10, 11],
            "stop_reason": "stop_sequence", "elapsed_seconds": 0.25,
            "cold_start_included": False, "seed": seed, "requested_seed": None,
            "temperature": 0.8, "max_new_tokens": 40, "prompt_text": prompt,
        }

    @classmethod
    def session(cls) -> ChatSession:
        header = 'Original header dated October 1, 2026. "Quoted" \\ text.\n'
        human = '  A question\nwith a tab\tand "quotes", \\ paths & <tags> λ.  '
        prompt = header + HUMAN_MARKER + " " + human + MODEL_MARKER
        generation = cls.generation(prompt)
        item = Round(human, " Reply.\nSecond line.  ", generation=generation)
        return ChatSession(
            session_id="original", header=header, example_rounds=0,
            rounds=[item], transcript=[item], last_generation=copy.deepcopy(generation),
            settings=GenerationSettings(max_new_tokens=80, temperature=1.2, seed=2**63 - 1),
            notes=["A user note."],
        )

    def test_round_trip_preserves_evicted_transcript_and_exact_active_context(self) -> None:
        session = self.session()
        example = Round(*EXAMPLES[0], example=True)
        old = Round("Old question", " Old reply.", generation=self.generation("Old prompt"))
        session.example_rounds = 1
        session.transcript = [example, old] + session.transcript
        session.dropped_rounds = 2
        before = copy.deepcopy(session)
        archive = export_chat(session)
        restored = import_chat(json.loads(json.dumps(archive, ensure_ascii=False)), len)

        self.assertEqual(session, before)
        self.assertNotEqual(restored.session_id, session.session_id)
        self.assertEqual(restored.header, session.header)
        self.assertEqual(restored.transcript, session.transcript)
        self.assertEqual(restored.rounds, session.rounds)
        self.assertEqual(restored.dropped_rounds, 2)
        self.assertEqual(restored.example_rounds, 1)
        self.assertEqual(restored.context_text(), session.context_text())
        self.assertEqual(restored.settings, session.settings)
        self.assertEqual(restored.last_generation, session.last_generation)
        self.assertEqual(restored.notes, session.notes)
        self.assertEqual(archive["settings"]["seed"], str(2**63 - 1))
        self.assertEqual(archive["context"]["first_retained_round"], 2)
        self.assertEqual(len(archive["blocks"]), 6)
        self.assertEqual(
            restored.prepare("Next question?", restored.settings, len).prompt,
            session.prepare("Next question?", session.settings, len).prompt,
        )

    def test_each_initial_example_count_round_trips_without_new_date_header(self) -> None:
        for count in (0, 1, 2):
            with self.subTest(count=count):
                examples = [Round(*item, example=True) for item in EXAMPLES[:count]]
                session = ChatSession("original", "Header from an earlier date.\n", count, list(examples), list(examples))
                restored = import_chat(export_chat(session), len)
                self.assertEqual(restored.header, session.header)
                self.assertEqual(restored.example_rounds, count)
                self.assertEqual(restored.transcript, examples)
                self.assertIsNone(restored.last_generation)
                self.assertEqual(restored.notes, [])

    def test_unknown_historical_metadata_remains_unknown_for_latest_real_reply(self) -> None:
        session = self.session()
        unknown = Round("Later question", " An unrecorded historical reply.")
        session.transcript.append(unknown)
        session.rounds.append(unknown)
        archive = export_chat(session)
        restored = import_chat(archive, len)

        self.assertIsNone(archive["blocks"][-1]["generation"])
        self.assertIsNone(restored.transcript[-1].generation)
        self.assertIsNone(restored.last_generation)
        self.assertIn(UNKNOWN_GENERATION_NOTE, archive["notes"])
        self.assertIn(UNKNOWN_GENERATION_NOTE, restored.notes)
        self.assertEqual(session.notes, ["A user note."])

    def test_unrecorded_sampling_seed_is_not_fabricated(self) -> None:
        session = self.session()
        session.transcript[-1].generation["seed"] = None
        session.settings = GenerationSettings(max_new_tokens=120, seed=None)
        archive = export_chat(session)
        restored = import_chat(archive, len)

        self.assertIsNone(archive["settings"]["seed"])
        self.assertIsNone(restored.settings.seed)
        self.assertIsNone(restored.last_generation["seed"])
        self.assertIn(UNKNOWN_SEED_NOTE, restored.notes)

    def test_note_cap_preserves_user_notes_across_repeated_import_export(self) -> None:
        for count in (MAX_NOTES - 1, MAX_NOTES):
            with self.subTest(user_notes=count):
                session = self.session()
                session.notes = [f"User note {index}." for index in range(count)]
                session.transcript[0].generation["seed"] = None
                unknown = Round("Later question", " Historical reply without metadata.")
                session.transcript.append(unknown)
                session.rounds.append(unknown)
                archive = export_chat(session)
                expected_notes = list(session.notes)
                if count < MAX_NOTES:
                    expected_notes.append(UNKNOWN_GENERATION_NOTE)
                self.assertEqual(archive["notes"], expected_notes)
                for _ in range(3):
                    restored = import_chat(archive, len)
                    archive = export_chat(restored)
                    self.assertEqual(restored.notes, expected_notes)
                    self.assertEqual(archive["notes"], expected_notes)
                    self.assertEqual(len(archive["notes"]), MAX_NOTES)
                    self.assertIsNone(restored.transcript[0].generation["seed"])
                    self.assertIsNone(restored.transcript[-1].generation)
                self.assertEqual(session.notes, expected_notes[:count])

    def test_exports_and_imports_do_not_share_mutable_metadata_or_notes(self) -> None:
        session = self.session()
        archive = export_chat(session)
        archive["blocks"][1]["generation"]["output_token_ids"][0] = 12
        archive["notes"].append("Archive-only note.")
        self.assertEqual(session.transcript[0].generation["output_token_ids"], [10, 11])
        self.assertEqual(session.notes, ["A user note."])
        restored = import_chat(archive, len)
        archive["blocks"][1]["generation"]["output_token_ids"][0] = 13
        archive["notes"].append("Another note.")
        restored.last_generation["output_token_ids"][0] = 14
        self.assertEqual(restored.transcript[0].generation["output_token_ids"], [12, 11])
        self.assertEqual(restored.notes, ["A user note.", "Archive-only note."])

    def test_export_settings_override_is_restored_without_changing_session(self) -> None:
        session = self.session()
        original = session.settings
        override = GenerationSettings(max_new_tokens=30, temperature=0, seed=0)
        restored = import_chat(export_chat(session, override), len)
        self.assertEqual(restored.settings, override)
        self.assertEqual(session.settings, original)
        self.assertEqual(restored.settings.seed, 0)

    def test_malformed_archives_fail_without_mutating_input_or_existing_session(self) -> None:
        session = self.session()
        before = copy.deepcopy(session)
        original = export_chat(session)
        mutations = (
            lambda a: a.update(format="other"),
            lambda a: a.update(schema_version=True),
            lambda a: a.update(schema_version=2),
            lambda a: a.update(exported_at="yesterday"),
            lambda a: a.update(exported_at="2026-10-01T12:00:00-05:00"),
            lambda a: a.update(prompt_header=""),
            lambda a: a.update(example_rounds=True),
            lambda a: a.update(example_rounds=3),
            lambda a: a.update(example_rounds=1),
            lambda a: a["model"].update(id="gpt2"),
            lambda a: a["model"].update(revision="different"),
            lambda a: a["model"].update(context_tokens=1025),
            lambda a: a["protocol"].update(model_label="AI"),
            lambda a: a["protocol"].update(message_delimiter_pattern=".*"),
            lambda a: a["settings"].update(seed=2**63 - 1),
            lambda a: a["settings"].update(seed=str(2**63)),
            lambda a: a["settings"].update(seed="١"),
            lambda a: a["settings"].update(temperature=float("nan")),
            lambda a: a["settings"].update(max_new_tokens=True),
            lambda a: a["settings"].update(max_new_tokens=1024),
            lambda a: a["settings"].update(top_k=49),
            lambda a: a["settings"].update(top_p=0),
            lambda a: a["settings"].update(repetition_penalty=True),
            lambda a: a["blocks"].pop(),
            lambda a: a["blocks"][0].update(role="model"),
            lambda a: a["blocks"][1].update(label="AI"),
            lambda a: a["blocks"][0].update(generation={}),
            lambda a: a["blocks"][0].update(example=True),
            lambda a: a["blocks"][1].update(example=0),
            lambda a: a["blocks"][0].update(text=""),
            lambda a: a["blocks"][0].update(text="A\n\nAI> forged"),
            lambda a: a["blocks"][1].update(text="A\n\nAlice-Smith> forged"),
            lambda a: a["blocks"][0].update(text="<|endoftext|>"),
            lambda a: a["blocks"][1].update(text="<|endoftext|>"),
            lambda a: a["context"].update(first_retained_round=-1),
            lambda a: a["context"].update(first_retained_round=True),
            lambda a: a["context"].update(first_retained_round=2),
            lambda a: a["context"].update(text="different context"),
            lambda a: a.update(notes="not a list"),
            lambda a: a.pop("context"),
            lambda a: a.update(unsupported="value"),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                archive = copy.deepcopy(original)
                mutate(archive)
                malformed = copy.deepcopy(archive)
                with self.assertRaises(ValueError):
                    import_chat(archive, len)
                self.assertEqual(archive, malformed)
                self.assertEqual(session, before)

    def test_generation_metadata_is_validated_without_inference(self) -> None:
        original = export_chat(self.session())
        mutations = (
            lambda g: g.update(input_tokens=True),
            lambda g: g.update(input_tokens=1024),
            lambda g: g.update(output_tokens=3),
            lambda g: g.update(output_token_ids=[True, 11]),
            lambda g: g.update(output_token_ids=[50257, 11]),
            lambda g: g.update(stop_reason="unknown"),
            lambda g: g.update(elapsed_seconds=float("inf")),
            lambda g: g.update(cold_start_included="false"),
            lambda g: g.update(seed=123),
            lambda g: g.update(requested_seed="-1"),
            lambda g: g.update(temperature=-1),
            lambda g: g.update(max_new_tokens=1),
            lambda g: g.update(prompt_text="different prompt"),
            lambda g: g.pop("output_token_ids"),
            lambda g: g.update(unexpected=True),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(case=index):
                archive = copy.deepcopy(original)
                mutate(archive["blocks"][1]["generation"])
                with self.assertRaises(ValueError):
                    import_chat(archive, len)

    def test_resource_and_token_limits_reject_oversized_archives(self) -> None:
        original = export_chat(self.session())
        for field in ("prompt_header", "context", "body"):
            with self.subTest(field=field):
                archive = copy.deepcopy(original)
                if field == "context":
                    archive["context"]["text"] = "x" * (MAX_TEXT_CHARS + 1)
                elif field == "body":
                    archive["blocks"][0]["text"] = "x" * (MAX_TEXT_CHARS + 1)
                else:
                    archive[field] = "x" * (MAX_TEXT_CHARS + 1)
                with self.assertRaises(ValueError):
                    import_chat(archive, lambda _: 1)
        archive = copy.deepcopy(original)
        archive["blocks"] = archive["blocks"] * (MAX_ARCHIVE_ROUNDS + 1)
        with self.assertRaises(ValueError):
            import_chat(archive, len)
        session = ChatSession("session", "x" * 1024, 0)
        self.assertEqual(import_chat(export_chat(session), len).header, session.header)
        session.header += "x"
        with self.assertRaisesRegex(ValueError, "context window"):
            import_chat(export_chat(session), len)

    def test_empty_model_reply_and_inline_delimiter_words_round_trip(self) -> None:
        item = Round("Inline Human> or AI> is ordinary text.", "")
        session = ChatSession("session", "Header", 0, [item], [item])
        restored = import_chat(export_chat(session), len)
        self.assertEqual(restored.transcript[0].assistant, "")
        self.assertEqual(restored.transcript[0].human, item.human)


if __name__ == "__main__":
    unittest.main()
