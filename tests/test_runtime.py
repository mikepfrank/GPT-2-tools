from __future__ import annotations

import unittest
from contextlib import nullcontext
from unittest import mock

from gpt2_local.runtime import GenerationResult, GenerationSettings, Gpt2Runner


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


if __name__ == "__main__":
    unittest.main()
