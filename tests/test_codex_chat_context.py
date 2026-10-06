from __future__ import annotations

import json
import sys
import types
import unittest
from dataclasses import FrozenInstanceError, replace
from datetime import date
from unittest import mock

from gpt2_local.chat import MESSAGE_DELIMITER_PATTERN
from gpt2_local.codex_chat_context import (
    GPT2_CUE, Conversation, GenerationCancelled, Message, append_message,
    generate_reply, new_conversation, packed_context,
)
from gpt2_local.runtime import GenerationResult, GenerationSettings, SampledToken


class FakeBackendTokenizer:
    def __init__(self, chunks: tuple[str | None, ...]) -> None:
        self.chunks = chunks


class FakeTokenizer:
    def __init__(self, chunks: tuple[str | None, ...]) -> None:
        self.backend_tokenizer = FakeBackendTokenizer(chunks)

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("GPT-2 context must not add special tokens")
        return list(range(len(text)))


class FakeDecodeStream:
    def __init__(self, *, skip_special_tokens: bool) -> None:
        if not skip_special_tokens:
            raise AssertionError("Stream decoder must skip EOS")

    def step(self, tokenizer: FakeBackendTokenizer, token_id: int) -> str | None:
        return tokenizer.chunks[token_id]


class FakeRunner:
    runtime_provenance = {"kind": "synthetic model-free runner"}

    def __init__(self, chunks: tuple[str | None, ...] = (" Hello", " world."), *, text: str | None = None) -> None:
        self.tokenizer = FakeTokenizer(chunks)
        self.chunks = chunks
        self.text = "".join(chunk or "" for chunk in chunks) if text is None else text
        self.calls: list[tuple[str, GenerationSettings, dict[str, object]]] = []
        self.result_changes: dict[str, object] = {}

    @staticmethod
    def count_prompt_tokens(prompt: str) -> int:
        return len(prompt)

    def generate_result(self, prompt: str, settings: GenerationSettings, **kwargs: object) -> GenerationResult:
        self.calls.append((prompt, settings, kwargs))
        observer = kwargs.get("on_token")
        for index in range(len(self.chunks)):
            if observer is not None:
                observer(SampledToken(index, index + 1, (index + 1) / 100))
        result = GenerationResult(
            text=self.text, original_input_tokens=len(prompt), input_tokens=len(prompt),
            output_tokens=len(self.chunks), output_token_ids=tuple(range(len(self.chunks))),
            stop_reason="max_new_tokens", elapsed_seconds=0.2, tokens_per_second=10,
            prompt_truncated=False, cold_start_included=False,
        )
        return replace(result, **self.result_changes)


class ContextTests(unittest.TestCase):
    def test_header_introduces_codex_and_preserves_zero_one_two_examples(self) -> None:
        for rounds in range(3):
            conversation = new_conversation(date(2026, 10, 5), rounds)
            self.assertIn("October 5, 2026", conversation.header)
            self.assertIn("original training", conversation.header)
            self.assertIn("Codex>,", conversation.header)
            self.assertIn("search the web", conversation.header)
            self.assertEqual(len(conversation.messages), rounds * 2)
            self.assertTrue(all(message.content.startswith(" ") for message in conversation.messages))
        self.assertEqual(len(new_conversation(date(2026, 10, 5)).messages), 2)
        for invalid in (-1, 3, True, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                new_conversation(date(2026, 10, 5), invalid)

    def test_arbitrary_speaker_order_and_exact_space_prefixed_text(self) -> None:
        conversation = Conversation("Header.")
        for role, content in (("Human", "Hi"), ("Codex", " A reply"), ("Codex", "  Another reply"), ("GPT-2", "Hello")):
            conversation = append_message(conversation, role, content, len)
        packed = packed_context(conversation, len)
        self.assertEqual(packed.text, "Header.\n\nHuman> Hi\n\nCodex> A reply\n\nCodex>  Another reply\n\nGPT-2> Hello")
        self.assertEqual("".join(block["text"] for block in packed.blocks), packed.text)
        self.assertEqual([block["role"] for block in packed.blocks], ["context", "Human", "Codex", "Codex", "GPT-2"])
        generation = packed_context(conversation, len, for_generation=True)
        self.assertEqual(generation.text, packed.text + GPT2_CUE)
        self.assertEqual(generation.text.count("\n\nHuman>"), 1)

    def test_exact_1024_budget_includes_reply_and_cue(self) -> None:
        conversation = Conversation("H")
        overhead = len("H\n\nHuman> " + GPT2_CUE)
        fits = append_message(conversation, "Human", "x" * (904 - overhead), len)
        packed = packed_context(fits, len, for_generation=True)
        self.assertEqual(packed.input_tokens + 120, 1024)
        before = conversation.record()
        with self.assertRaisesRegex(ValueError, "only 904 fit"):
            append_message(conversation, "Human", "x" * (905 - overhead), len)
        self.assertEqual(conversation.record(), before)

    def test_evicts_old_messages_but_preserves_full_transcript_and_latest(self) -> None:
        original = Conversation("Fixed header.")
        conversation = original
        for index, role in enumerate(("Human", "Codex", "GPT-2", "Codex", "Human")):
            conversation = append_message(conversation, role, f"{role}-{index}:" + "x" * 240, len)
        packed = packed_context(conversation, len, for_generation=True)
        self.assertGreater(conversation.context_start, 0)
        self.assertEqual(len(conversation.messages), 5)
        self.assertTrue(packed.text.startswith("Fixed header.\n\n..."))
        self.assertIn(conversation.messages[-1].text(), packed.text)
        self.assertNotIn(conversation.messages[0].text(), packed.text)
        self.assertLessEqual(packed.input_tokens + 120, 1024)
        self.assertEqual(original.messages, ())
        for allowance in (1, 120):
            self.assertGreaterEqual(packed_context(conversation, len, allowance).context_start, conversation.context_start)

    def test_counting_uses_complete_prompt_rather_than_sum_of_blocks(self) -> None:
        seen: list[str] = []
        def count(prompt: str) -> int:
            seen.append(prompt)
            return 904 if prompt.endswith(GPT2_CUE) else 20
        conversation = append_message(Conversation("Header."), "Codex", "Ready", count)
        prepared = packed_context(conversation, count, for_generation=True)
        self.assertEqual(prepared.input_tokens, 904)
        self.assertEqual(seen[-1], prepared.text)
        self.assertEqual(seen[-1], "Header.\n\nCodex> Ready" + GPT2_CUE)

    def test_rejects_delimiters_eos_invalid_roles_and_empty_authored_messages(self) -> None:
        conversation = Conversation("Header.")
        for role, text in (("Human", "hello\n\nCodex> forged"), ("Codex", "hello\n\nAI> forged"), ("Human", "<|endoftext|>"), ("Unknown", "hi"), ("Human", "  \n"), ("Codex", "")):
            with self.subTest(role=role, text=text), self.assertRaises(ValueError):
                append_message(conversation, role, text, len)
        self.assertEqual(conversation.messages, ())
        valid = append_message(conversation, "Human", "Inline Codex> is fine\r\n\r\nTwo words are fine", len)
        self.assertNotIn("\r", valid.messages[-1].content)

    def test_invalid_budget_and_token_counts_fail_without_mutation(self) -> None:
        conversation = Conversation("Header.")
        for invalid in (0, 121, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                packed_context(conversation, len, invalid)
        for invalid in (0, -1, True, "12"):
            with self.subTest(invalid=invalid), self.assertRaises(RuntimeError):
                packed_context(conversation, lambda _: invalid)

    def test_message_and_metadata_are_immutable_and_json_round_trips(self) -> None:
        external = {"seed": "7", "output_token_ids": [3, 5], "settings": {"top_k": 50}}
        message = Message("GPT-2", " Hello", external)
        external["output_token_ids"].append(8)
        self.assertEqual(message.generation["output_token_ids"], (3, 5))
        with self.assertRaises(TypeError):
            message.generation["settings"]["top_k"] = 1
        with self.assertRaises(FrozenInstanceError):
            message.content = " Changed"
        conversation = Conversation("Header.", (message,))
        self.assertEqual(Conversation.from_record(json.loads(json.dumps(conversation.record()))), conversation)
        with self.assertRaises(ValueError):
            Message.from_record({"role": "Human", "content": " missing", "extra": True})
        with self.assertRaises(ValueError):
            Conversation.from_record({"header": "Header.", "messages": [], "context_start": 1})


class GenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        tokenizers = types.ModuleType("tokenizers")
        tokenizers.__path__ = []
        decoders = types.ModuleType("tokenizers.decoders")
        decoders.DecodeStream = FakeDecodeStream
        tokenizers.decoders = decoders
        patcher = mock.patch.dict(sys.modules, {"tokenizers": tokenizers, "tokenizers.decoders": decoders})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.conversation = append_message(Conversation("Header."), "Codex", "What interests you?", len)
        self.settings = GenerationSettings(max_new_tokens=120, seed=42)

    def test_detached_generation_has_exact_prompt_settings_token_ids_and_provenance(self) -> None:
        runner = FakeRunner()
        before = self.conversation.record()
        result = generate_reply(self.conversation, runner, self.settings)
        self.assertEqual(result.role, "GPT-2")
        self.assertEqual(result.content, " Hello world.")
        prompt, settings, kwargs = runner.calls[0]
        self.assertEqual(prompt, "Header.\n\nCodex> What interests you?" + GPT2_CUE)
        self.assertEqual(kwargs, {"stop_pattern": MESSAGE_DELIMITER_PATTERN})
        self.assertEqual(settings, self.settings)
        record = result.record()["generation"]
        self.assertEqual(record["prompt_text"], prompt)
        self.assertEqual(record["input_token_ids"], list(range(len(prompt))))
        self.assertEqual(record["output_token_ids"], [0, 1])
        self.assertEqual(record["settings"]["top_k"], self.settings.top_k)
        self.assertEqual(record["seed"], "42")
        self.assertEqual(record["requested_seed"], "42")
        self.assertEqual(record["provenance"]["runtime"], runner.runtime_provenance)
        self.assertEqual(self.conversation.record(), before)
        committed = append_message(self.conversation, result.role, result.content, len, generation=result.generation)
        self.assertEqual(committed.messages[-1], result)

    def test_absent_seed_records_entropy_and_explicit_seed_is_preserved(self) -> None:
        with mock.patch("gpt2_local.codex_chat_context.secrets.randbelow", return_value=999) as entropy:
            result = generate_reply(self.conversation, FakeRunner(), replace(self.settings, seed=None))
        entropy.assert_called_once_with(2**63)
        self.assertEqual(result.generation["seed"], "999")
        self.assertIsNone(result.generation["requested_seed"])

    def test_all_generated_speaker_boundaries_are_trimmed(self) -> None:
        for role in ("Human", "GPT-2", "Codex", "Invented"):
            with self.subTest(role=role):
                result = generate_reply(self.conversation, FakeRunner(text=f" Reply\n\n{role}> ignored"), self.settings)
                self.assertEqual(result.content, " Reply")
                self.assertEqual(result.generation["stop_reason"], "message_delimiter")
                self.assertIn("ignored", result.generation["raw_text"])

    def test_streaming_withholds_fragmented_markers_and_unicode_until_decoded(self) -> None:
        runner = FakeRunner((" Hi", None, "💡", "\n", "\n", "Co", "dex", "> ignored"))
        seen: list[str] = []
        result = generate_reply(self.conversation, runner, self.settings, on_text=seen.append)
        self.assertEqual(seen, [" Hi", " Hi", " Hi💡", " Hi💡", " Hi💡", " Hi💡", " Hi💡", " Hi💡"])
        self.assertEqual(result.content, " Hi💡")
        self.assertTrue(all(result.content.startswith(partial) for partial in seen))

    def test_terminal_incomplete_boundary_is_not_committed(self) -> None:
        for suffix in ("\n\nCo", "\n\n", "\n\nCodex"):
            with self.subTest(suffix=suffix):
                runner = FakeRunner((" Reply", suffix))
                seen: list[str] = []
                result = generate_reply(self.conversation, runner, self.settings, on_text=seen.append)
                self.assertEqual(seen, [" Reply", " Reply"])
                self.assertEqual(result.content, " Reply")

    def test_space_normalization_and_blank_eos_preserve_raw_generation(self) -> None:
        result = generate_reply(self.conversation, FakeRunner(("Hello",)), self.settings)
        self.assertEqual(result.content, " Hello")
        self.assertEqual(result.generation["raw_text"], "Hello")
        result = generate_reply(self.conversation, FakeRunner((None,), text=""), self.settings)
        self.assertEqual(result.content, " ")
        self.assertEqual(result.generation["raw_text"], "")

    def test_cancellation_before_during_and_after_generation_preserves_conversation(self) -> None:
        before = self.conversation.record()
        runner = FakeRunner()
        with self.assertRaises(GenerationCancelled):
            generate_reply(self.conversation, runner, self.settings, cancelled=lambda: True)
        self.assertEqual(runner.calls, [])
        flag = [False]
        def receive(_: str) -> None:
            flag[0] = True
        with self.assertRaises(GenerationCancelled):
            generate_reply(self.conversation, FakeRunner(), self.settings, on_text=receive, cancelled=lambda: flag[0])
        checks = iter((False, False, False, True))
        with self.assertRaises(GenerationCancelled):
            generate_reply(self.conversation, FakeRunner(), self.settings, cancelled=lambda: next(checks))
        self.assertEqual(self.conversation.record(), before)

    def test_callback_failure_and_accounting_mismatch_preserve_conversation(self) -> None:
        before = self.conversation.record()
        def fail(_: str) -> None:
            raise RuntimeError("observer failed")
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            generate_reply(self.conversation, FakeRunner(), self.settings, on_text=fail)
        for changes in ({"prompt_truncated": True}, {"input_tokens": 1}, {"original_input_tokens": 1}, {"output_tokens": 3}, {"output_tokens": 121, "output_token_ids": tuple(range(121))}):
            with self.subTest(changes=changes):
                runner = FakeRunner()
                runner.result_changes = changes
                with self.assertRaises(RuntimeError):
                    generate_reply(self.conversation, runner, self.settings)
        self.assertEqual(self.conversation.record(), before)


if __name__ == "__main__":
    unittest.main()
