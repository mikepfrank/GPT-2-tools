from __future__ import annotations

import copy
import unittest
from typing import Callable

from gpt2_local.chat import ChatSession, MESSAGE_DELIMITER_PATTERN
from gpt2_local.runtime import GenerationResult, GenerationSettings, SampledToken


class ObserverRunner:
    """Emit synthetic progress before returning the canonical result."""

    text = " Canonical reply."

    def __init__(self) -> None:
        self.calls: list[tuple[str, GenerationSettings, dict[str, object]]] = []

    @staticmethod
    def count_prompt_tokens(prompt: str) -> int:
        return len(prompt)

    def generate_result(self, prompt: str, settings: GenerationSettings, **kwargs: object) -> GenerationResult:
        self.calls.append((prompt, settings, kwargs))
        observer: Callable[[SampledToken], None] | None = kwargs.get("on_token")
        if observer is not None:
            observer(SampledToken(11, 1, 0.01))
            observer(SampledToken(12, 2, 0.02))
        return GenerationResult(
            text=self.text, original_input_tokens=len(prompt), input_tokens=len(prompt),
            output_tokens=2, output_token_ids=(11, 12), stop_reason="max_new_tokens",
            elapsed_seconds=0.03, tokens_per_second=2 / 0.03,
            prompt_truncated=False, cold_start_included=len(self.calls) == 1,
        )


class ChatObserverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = ObserverRunner()
        self.session = ChatSession("synthetic", "Fixed header.", 0)
        self.settings = GenerationSettings(max_new_tokens=32, temperature=0.8, seed=42)

    def test_reply_passes_observer_and_commits_only_canonical_text_after_callbacks(self) -> None:
        before = copy.deepcopy(self.session.state(len))
        observed: list[SampledToken] = []
        def observe(event: SampledToken) -> None:
            self.assertEqual(self.session.state(len), before)
            observed.append(event)
        self.session.reply("Question?", self.settings, self.runner, on_token=observe)
        self.assertEqual(len(observed), 2)
        self.assertIs(self.runner.calls[-1][2]["on_token"], observe)
        self.assertEqual(self.runner.calls[-1][2]["stop_pattern"], MESSAGE_DELIMITER_PATTERN)
        self.assertEqual(self.session.rounds[-1].assistant, self.runner.text)
        self.assertEqual(self.session.last_generation["output_token_ids"], [11, 12])

    def test_none_observer_is_not_forwarded_to_runner(self) -> None:
        self.session.reply("Question?", self.settings, self.runner, on_token=None)
        self.assertEqual(self.runner.calls[-1][2], {"stop_pattern": MESSAGE_DELIMITER_PATTERN})
        self.session.regenerate(self.settings, self.runner, on_token=None)
        self.assertEqual(self.runner.calls[-1][2], {"stop_pattern": MESSAGE_DELIMITER_PATTERN})

    def test_regeneration_keeps_old_candidate_until_success_and_advances_seed_once(self) -> None:
        self.session.reply("Question?", self.settings, self.runner)
        before = copy.deepcopy(self.session.state(len))
        prompt = self.runner.calls[-1][0]
        self.runner.text = " Replacement."
        def observe(_: SampledToken) -> None:
            self.assertEqual(self.session.state(len), before)
        self.session.regenerate(self.settings, self.runner, on_token=observe)
        self.assertEqual(self.runner.calls[-1][0], prompt)
        self.assertEqual(self.runner.calls[-1][1].seed, 43)
        self.assertEqual(self.session.settings.seed, 43)
        self.assertEqual(self.session.rounds[-1].previous_responses[0]["text"], before["turns"][-1]["assistant"])
        self.assertEqual(self.session.rounds[-1].assistant, self.runner.text)
        self.assertEqual(len(self.session.transcript), 1)

    def test_observer_failure_preserves_reply_and_regeneration_state(self) -> None:
        def fail(_: SampledToken) -> None:
            raise RuntimeError("observer failed")
        before = copy.deepcopy(self.session.state(len))
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            self.session.reply("Question?", self.settings, self.runner, on_token=fail)
        self.assertEqual(self.session.state(len), before)
        self.session.reply("Question?", self.settings, self.runner)
        before = copy.deepcopy(self.session.state(len))
        with self.assertRaisesRegex(RuntimeError, "observer failed"):
            self.session.regenerate(self.settings, self.runner, on_token=fail)
        self.assertEqual(self.session.state(len), before)


if __name__ == "__main__":
    unittest.main()
