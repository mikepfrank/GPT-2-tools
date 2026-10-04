from __future__ import annotations

import unittest
from contextlib import nullcontext
from dataclasses import FrozenInstanceError, replace
from unittest import mock

from gpt2_local.runtime import GenerationResult, GenerationSettings, Gpt2Runner, SampledToken


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_generation_settings_reject_out_of_range_seeds(self) -> None:
        for seed in (-1, 2**63, True):
            with self.subTest(seed=seed):
                with self.assertRaisesRegex(ValueError, "seed must be"):
                    GenerationSettings(seed=seed).validate()

    def test_generate_remains_a_text_only_wrapper(self) -> None:
        runner = object.__new__(Gpt2Runner)
        expected = GenerationResult(
            text="text only",
            original_input_tokens=1,
            input_tokens=1,
            output_tokens=1,
            output_token_ids=(1,),
            stop_reason="max_new_tokens",
            elapsed_seconds=1.0,
            tokens_per_second=1.0,
            prompt_truncated=False,
            cold_start_included=False,
        )
        calls: list[tuple[str, GenerationSettings, bool]] = []

        def fake_generate_result(
            prompt: str,
            settings: GenerationSettings,
            *,
            include_prompt: bool = False,
        ) -> GenerationResult:
            calls.append((prompt, settings, include_prompt))
            return expected

        runner.generate_result = fake_generate_result  # type: ignore[method-assign]
        settings = GenerationSettings(max_new_tokens=1, temperature=0)

        self.assertEqual(
            runner.generate("prompt", settings, include_prompt=True),
            "text only",
        )
        self.assertEqual(calls, [("prompt", settings, True)])

    def test_structured_result_tracks_truncation_tokens_eos_and_cold_start(self) -> None:
        import torch

        class FakeTokenizer:
            eos_token_id = 99

            def __call__(
                self, prompt: str, *, return_tensors: str, add_special_tokens: bool
            ) -> dict[str, torch.Tensor]:
                del prompt, return_tensors, add_special_tokens
                return {
                    "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]]),
                    "attention_mask": torch.ones((1, 6), dtype=torch.long),
                }

            def encode(self, prompt: str, *, add_special_tokens: bool) -> list[int]:
                del prompt, add_special_tokens
                return [1, 2, 3, 4, 5, 6]

            def decode(self, tokens: torch.Tensor, *, skip_special_tokens: bool) -> str:
                del skip_special_tokens
                return ",".join(str(token) for token in tokens.tolist())

        class FakeModel:
            def generate(self, input_ids: torch.Tensor, **_: object) -> torch.Tensor:
                generated = torch.tensor([[7, 99]])
                return torch.cat((input_ids, generated), dim=1)

        runner = object.__new__(Gpt2Runner)
        runner._torch = mock.Mock(
            manual_seed=torch.manual_seed,
            inference_mode=lambda: nullcontext(),
        )
        runner.tokenizer = FakeTokenizer()
        runner.model = FakeModel()
        runner._generation_count = 0

        result = runner.generate_result(
            "six tokens",
            GenerationSettings(max_new_tokens=1020, temperature=0),
        )

        self.assertEqual(runner.count_prompt_tokens("six tokens"), 6)
        self.assertEqual(result.original_input_tokens, 6)
        self.assertEqual(result.input_tokens, 4)
        self.assertTrue(result.prompt_truncated)
        self.assertEqual(result.output_token_ids, (7, 99))
        self.assertEqual(result.output_tokens, 2)
        self.assertEqual(result.stop_reason, "eos_token")
        self.assertEqual(result.text, "7,99")
        self.assertTrue(result.cold_start_included)

    def test_recorded_seed_replays_after_an_intervening_sample(self) -> None:
        import torch

        class FakeTokenizer:
            eos_token_id = 99

            def __call__(self, *_: object, **__: object) -> dict[str, torch.Tensor]:
                return {
                    "input_ids": torch.tensor([[1, 2]]),
                    "attention_mask": torch.ones((1, 2), dtype=torch.long),
                }

            def decode(self, tokens: torch.Tensor, **_: object) -> str:
                return ",".join(str(token) for token in tokens.tolist())

        class RandomModel:
            def generate(
                self,
                input_ids: torch.Tensor,
                *,
                max_new_tokens: int,
                **_: object,
            ) -> torch.Tensor:
                generated = torch.randint(0, 50, (1, max_new_tokens))
                return torch.cat((input_ids, generated), dim=1)

        runner = object.__new__(Gpt2Runner)
        runner._torch = torch
        runner.tokenizer = FakeTokenizer()
        runner.model = RandomModel()
        runner._generation_count = 0

        settings_a = GenerationSettings(max_new_tokens=4, temperature=0.8, seed=123)
        settings_b = GenerationSettings(max_new_tokens=4, temperature=0.8, seed=456)
        first = runner.generate_result("prompt", settings_a)
        runner.generate_result("prompt", settings_b)
        replay = runner.generate_result("prompt", settings_a)

        self.assertEqual(first.output_token_ids, replay.output_token_ids)
        self.assertEqual(first.text, replay.text)


class RuntimeStopSequenceTests(unittest.TestCase):
    @staticmethod
    def _runner(
        fragments: tuple[str, ...],
        *,
        honor_stops: bool = True,
        include_eos: bool = False,
    ) -> Gpt2Runner:
        """Exercise real criteria with tiny fake token fragments; load no model."""
        import torch

        class FragmentTokenizer:
            eos_token_id = 99
            prompt = ""

            def __call__(self, prompt: str, **_: object) -> dict[str, torch.Tensor]:
                self.prompt = prompt
                return {
                    "input_ids": torch.tensor([[1]]),
                    "attention_mask": torch.ones((1, 1), dtype=torch.long),
                }

            def decode(self, tokens: torch.Tensor, **_: object) -> str:
                pieces = {1: self.prompt, self.eos_token_id: ""}
                pieces.update({10 + index: text for index, text in enumerate(fragments)})
                return "".join(pieces[token_id] for token_id in tokens.tolist())

        class FragmentModel:
            seen_criteria = None
            checks: list[bool]

            def __init__(self) -> None:
                self.checks = []

            def generate(
                self,
                input_ids: torch.Tensor,
                *,
                stopping_criteria=None,
                **_: object,
            ) -> torch.Tensor:
                self.seen_criteria = stopping_criteria
                output = input_ids
                for index in range(len(fragments)):
                    output = torch.cat((output, torch.tensor([[10 + index]])), dim=1)
                    if stopping_criteria is not None:
                        done = bool(stopping_criteria(output, None)[0].item())
                        self.checks.append(done)
                        if done and honor_stops:
                            return output
                if include_eos:
                    output = torch.cat((output, torch.tensor([[99]])), dim=1)
                return output

        runner = object.__new__(Gpt2Runner)
        runner._torch = torch
        runner.tokenizer = FragmentTokenizer()
        runner.model = FragmentModel()
        runner._generation_count = 0
        return runner

    def test_stops_only_after_complete_delimiter_across_fragments(self) -> None:
        runner = self._runner(("Hello", "\n", "\nHu", "man", ">", "simulated human"))
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
            stop_sequences=("\n\nHuman>", "\n\nGPT-2>"),
        )

        self.assertEqual(result.text, "Hello")
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.output_tokens, 5)
        self.assertEqual(result.output_token_ids, (10, 11, 12, 13, 14))
        self.assertEqual(runner.model.checks, [False, False, False, False, True])

    def test_delimiter_at_start_produces_empty_reply(self) -> None:
        runner = self._runner(("\n\nGPT-2>", "simulated assistant"))
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
            stop_sequences=("\n\nHuman>", "\n\nGPT-2>"),
        )

        self.assertEqual(result.text, "")
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.output_token_ids, (10,))

    def test_eos_reason_is_preserved_with_configured_stops(self) -> None:
        runner = self._runner(("Answer",), include_eos=True)
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
            stop_sequences=("\n\nHuman>", "\n\nGPT-2>"),
        )

        self.assertEqual(result.text, "Answer")
        self.assertEqual(result.stop_reason, "eos_token")
        self.assertEqual(result.output_token_ids, (10, 99))
        self.assertEqual(result.output_tokens, 2)

    def test_ordinary_newlines_and_partial_or_inline_markers_do_not_stop(self) -> None:
        fragments = ("Human> is a label.", "\n", "Second line.", "\n\nHuman", "like prose")
        runner = self._runner(fragments)
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
            stop_sequences=("\n\nHuman>", "\n\nGPT-2>"),
        )

        self.assertEqual(result.text, "".join(fragments))
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertFalse(any(runner.model.checks))

    def test_prompt_delimiters_and_prompt_boundary_are_excluded(self) -> None:
        for prompt, continuation in (
            ("Header\n\nHuman> question\n\nGPT-2>", "Answer"),
            ("Header\n\nHu", "man>"),
        ):
            with self.subTest(prompt=prompt):
                runner = self._runner((continuation,))
                result = runner.generate_result(
                    prompt,
                    GenerationSettings(max_new_tokens=10, temperature=0),
                    include_prompt=True,
                    stop_sequences=("\n\nHuman>", "\n\nGPT-2>"),
                )

                self.assertEqual(result.text, prompt + continuation)
                self.assertEqual(result.stop_reason, "max_new_tokens")
                self.assertEqual(runner.model.checks, [False])

    def test_defensive_trim_uses_first_sequence_even_if_generation_ignores_stop(self) -> None:
        fragments = ("Answer\n\nGPT-2> simulated", "\n\nHuman> second")
        runner = self._runner(fragments, honor_stops=False)
        result = runner.generate_result(
            "Header\n\nHuman> question\n\nGPT-2>",
            GenerationSettings(max_new_tokens=10, temperature=0),
            include_prompt=True,
            stop_sequences=("\n\nHuman>", "\n\nGPT-2>"),
        )

        self.assertEqual(result.text, "Header\n\nHuman> question\n\nGPT-2>Answer")
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.output_token_ids, (10, 11))
        self.assertEqual(result.output_tokens, 2)

    def test_no_stops_preserves_existing_output_and_omits_criteria(self) -> None:
        runner = self._runner(("Answer\n\nHuman>", " simulated"))
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
        )

        self.assertEqual(result.text, "Answer\n\nHuman> simulated")
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertIsNone(runner.model.seen_criteria)

    def test_empty_or_non_text_stop_sequences_fail_before_generation(self) -> None:
        runner = object.__new__(Gpt2Runner)
        for sequences in (("",), ("\n\nHuman>", ""), (None,), "Human>"):
            with self.subTest(sequences=sequences):
                with self.assertRaisesRegex(ValueError, "non-empty strings"):
                    runner.generate_result(
                        "prompt",
                        GenerationSettings(max_new_tokens=10, temperature=0),
                        stop_sequences=sequences,
                    )

    def test_regex_stops_fragmented_hallucinated_speaker_labels(self) -> None:
        for label in ("AI", "Assistant", "assistant", "Mike", "other-AI", "GPT-2"):
            with self.subTest(label=label):
                middle = len(label) // 2
                runner = self._runner(
                    ("Answer", "\n", "\n", label[:middle], label[middle:], ">", "simulated")
                )
                result = runner.generate_result(
                    "prompt",
                    GenerationSettings(max_new_tokens=10, temperature=0),
                    stop_pattern=r"\n\n[^\s>]+>",
                )

                self.assertEqual(result.text, "Answer")
                self.assertEqual(result.stop_reason, "stop_sequence")
                self.assertEqual(result.output_tokens, 6)
                self.assertEqual(result.output_token_ids, (10, 11, 12, 13, 14, 15))
                self.assertEqual(runner.model.checks, [False] * 5 + [True])

    def test_regex_and_literal_stops_trim_at_earliest_match(self) -> None:
        for text in (
            "Answer\n\nAI> simulated[END]",
            "Answer[END]\n\nAI> simulated",
        ):
            with self.subTest(text=text):
                runner = self._runner((text, "ignored tail"), honor_stops=False)
                result = runner.generate_result(
                    "prompt",
                    GenerationSettings(max_new_tokens=10, temperature=0),
                    stop_sequences=("[END]",),
                    stop_pattern=r"\n\n[^\s>]+>",
                )

                self.assertEqual(result.text, "Answer")
                self.assertEqual(result.stop_reason, "stop_sequence")
                self.assertEqual(result.output_token_ids, (10, 11))
                self.assertEqual(result.output_tokens, 2)

    def test_regex_allows_blank_lines_inline_labels_and_partial_delimiters(self) -> None:
        fragments = (
            "AI> is an inline label.",
            "\n\n",
            "A new paragraph.",
            "\nAI> still only one newline.",
            "\n\nTwo words> is not a speaker label.",
            "\n\nAI",
        )
        runner = self._runner(fragments)
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
            stop_pattern=r"\n\n[^\s>]+>",
        )

        self.assertEqual(result.text, "".join(fragments))
        self.assertEqual(result.stop_reason, "max_new_tokens")
        self.assertFalse(any(runner.model.checks))

    def test_regex_does_not_match_prompt_or_join_across_prompt_boundary(self) -> None:
        for prompt, continuation in (
            ("Header\n\nAI> example\n\nGPT-2>", "Answer"),
            ("Header\n\nA", "I>"),
        ):
            with self.subTest(prompt=prompt):
                runner = self._runner((continuation,))
                result = runner.generate_result(
                    prompt,
                    GenerationSettings(max_new_tokens=10, temperature=0),
                    include_prompt=True,
                    stop_pattern=r"\n\n[^\s>]+>",
                )

                self.assertEqual(result.text, prompt + continuation)
                self.assertEqual(result.stop_reason, "max_new_tokens")
                self.assertEqual(runner.model.checks, [False])

    def test_eos_reason_is_preserved_with_regex_stop(self) -> None:
        runner = self._runner(("Answer",), include_eos=True)
        result = runner.generate_result(
            "prompt",
            GenerationSettings(max_new_tokens=10, temperature=0),
            stop_pattern=r"\n\n[^\s>]+>",
        )

        self.assertEqual(result.text, "Answer")
        self.assertEqual(result.stop_reason, "eos_token")
        self.assertEqual(result.output_token_ids, (10, 99))

    def test_invalid_or_empty_matching_regex_fails_before_generation(self) -> None:
        runner = object.__new__(Gpt2Runner)
        for pattern in ("[", "(", r"\x", "", "a*", r"(?=)", 123):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, "stop_pattern"):
                    runner.generate_result(
                        "prompt",
                        GenerationSettings(max_new_tokens=10, temperature=0),
                        stop_pattern=pattern,
                    )

    def test_zero_width_regex_on_generated_text_is_rejected(self) -> None:
        runner = self._runner(("Answer",))
        with self.assertRaisesRegex(ValueError, "zero-width"):
            runner.generate_result(
                "prompt",
                GenerationSettings(max_new_tokens=10, temperature=0),
                stop_pattern=r"(?=A)",
            )


class RuntimeObserverTests(unittest.TestCase):
    @staticmethod
    def runner(
        token_ids: tuple[int, ...] = (10, 11, 99), *, random_tokens: bool = False,
        failure_before_callback: int | None = None,
    ) -> Gpt2Runner:
        import torch

        class Tokenizer:
            eos_token_id = 99

            def __call__(self, *_: object, **__: object) -> dict[str, torch.Tensor]:
                return {"input_ids": torch.tensor([[1, 2]])}

            def decode(self, tokens: torch.Tensor, **_: object) -> str:
                pieces = {1: "prompt", 2: " remainder", 10: " Answer", 11: "!", 99: ""}
                return "".join(pieces.get(token, f"[{token}]") for token in tokens.tolist())

        class Model:
            def __init__(self) -> None:
                self.arguments: dict[str, object] = {}
                self.end_called = False

            def generate(self, input_ids: torch.Tensor, **kwargs: object) -> torch.Tensor:
                self.arguments = dict(kwargs)
                streamer = kwargs.get("streamer")
                criteria = kwargs.get("stopping_criteria")
                if streamer is not None:
                    streamer.put(input_ids.cpu())
                output = input_ids
                ids = (
                    torch.randint(10, 60, (kwargs["max_new_tokens"],)).tolist()
                    if random_tokens else token_ids
                )
                for index, token_id in enumerate(ids):
                    next_token = torch.tensor([token_id])
                    output = torch.cat((output, next_token[:, None]), dim=1)
                    if index == failure_before_callback:
                        raise RuntimeError("synthetic deadline before callback")
                    done = bool(criteria(output, None)[0].item()) if criteria is not None else False
                    # Match the inspected Transformers 5.18.0 CPU ordering.
                    if streamer is not None:
                        streamer.put(next_token.cpu())
                    if done or token_id == Tokenizer.eos_token_id:
                        break
                if streamer is not None:
                    streamer.end()
                    self.end_called = True
                return output

        runner = object.__new__(Gpt2Runner)
        runner._torch = torch
        runner.tokenizer = Tokenizer()
        runner.model = Model()
        runner._generation_count = 0
        return runner

    def test_prompt_is_skipped_and_eos_is_observed_with_frozen_events(self) -> None:
        runner = self.runner()
        observed: list[SampledToken] = []
        result = runner.generate_result(
            "prompt", GenerationSettings(max_new_tokens=3, temperature=0),
            on_token=observed.append,
        )
        self.assertEqual([event.token_id for event in observed], [10, 11, 99])
        self.assertEqual([event.output_tokens for event in observed], [1, 2, 3])
        self.assertEqual(tuple(event.token_id for event in observed), result.output_token_ids)
        self.assertEqual(result.output_tokens, 3)
        self.assertEqual(result.text, " Answer!")
        self.assertEqual(result.stop_reason, "eos_token")
        self.assertTrue(runner.model.end_called)
        with self.assertRaises(FrozenInstanceError):
            observed[0].token_id = 123

    def test_elapsed_time_starts_before_prefill_and_excludes_prompt_notification(self) -> None:
        runner = self.runner((10, 11))
        observed: list[SampledToken] = []
        with mock.patch("gpt2_local.runtime.time.perf_counter", side_effect=[100, 100.8, 101.1, 102]):
            result = runner.generate_result(
                "prompt", GenerationSettings(max_new_tokens=2, temperature=0),
                on_token=observed.append,
            )
        self.assertAlmostEqual(observed[0].elapsed_seconds, 0.8)
        self.assertAlmostEqual(observed[1].elapsed_seconds, 1.1)
        self.assertEqual(result.elapsed_seconds, 2)

    def test_observation_preserves_seeded_token_ids_text_and_generate_arguments(self) -> None:
        for temperature in (0, 0.8):
            with self.subTest(temperature=temperature):
                runner = self.runner(random_tokens=True)
                settings = GenerationSettings(max_new_tokens=6, temperature=temperature, seed=42)
                plain = runner.generate_result("prompt", settings)
                plain_arguments = dict(runner.model.arguments)
                self.assertNotIn("streamer", plain_arguments)
                observed: list[SampledToken] = []
                streamed = runner.generate_result("prompt", settings, on_token=observed.append)
                streamed_arguments = dict(runner.model.arguments)
                self.assertIn("streamer", streamed_arguments)
                del streamed_arguments["streamer"]
                self.assertEqual(streamed_arguments, plain_arguments)
                self.assertEqual(
                    replace(plain, elapsed_seconds=0, tokens_per_second=0, cold_start_included=False),
                    replace(streamed, elapsed_seconds=0, tokens_per_second=0, cold_start_included=False),
                )
                self.assertEqual(tuple(event.token_id for event in observed), plain.output_token_ids)

    def test_token_completing_a_stop_sequence_is_still_observed(self) -> None:
        runner = self.runner((10, 11, 99))
        observed: list[SampledToken] = []
        result = runner.generate_result(
            "prompt", GenerationSettings(max_new_tokens=3, temperature=0),
            stop_sequences=("!",), on_token=observed.append,
        )
        self.assertEqual([event.token_id for event in observed], [10, 11])
        self.assertEqual(result.output_token_ids, (10, 11))
        self.assertEqual(result.text, " Answer")
        self.assertEqual(result.stop_reason, "stop_sequence")

    def test_callback_failure_propagates_without_recording_completed_generation(self) -> None:
        runner = self.runner()
        def fail(_: SampledToken) -> None:
            raise RuntimeError("observer failed")
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            runner.generate_result("prompt", GenerationSettings(), on_token=fail)
        self.assertEqual(runner._generation_count, 0)
        self.assertFalse(runner.model.end_called)

    def test_exception_before_latest_callback_does_not_publish_that_token_or_end(self) -> None:
        runner = self.runner(failure_before_callback=1)
        observed: list[SampledToken] = []
        with self.assertRaisesRegex(RuntimeError, "synthetic deadline"):
            runner.generate_result("prompt", GenerationSettings(), on_token=observed.append)
        self.assertEqual([event.token_id for event in observed], [10])
        self.assertEqual(runner._generation_count, 0)
        self.assertFalse(runner.model.end_called)

    def test_non_callable_observer_is_rejected_before_tokenization_or_seed(self) -> None:
        runner = object.__new__(Gpt2Runner)
        with self.assertRaisesRegex(ValueError, "on_token must be callable"):
            runner.generate_result("prompt", GenerationSettings(seed=42), on_token=123)


if __name__ == "__main__":
    unittest.main()
