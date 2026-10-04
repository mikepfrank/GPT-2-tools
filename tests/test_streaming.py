from __future__ import annotations

import os
import re
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from gpt2_local.chat import MESSAGE_DELIMITER_PATTERN
from gpt2_local.runtime import MODEL_ID, MODEL_REVISION
from gpt2_local.streaming import ReplyStreamDecoder


class FakeBackendTokenizer:
    """Script incremental decoder chunks without importing model libraries."""

    def __init__(self, chunks: tuple[str | None, ...]) -> None:
        self.chunks = chunks
        self.step_calls: list[int] = []
        self.decode_calls: list[tuple[tuple[int, ...], bool]] = []

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.decode_calls.append((tuple(token_ids), skip_special_tokens))
        return "".join(self.chunks[token_id] or "" for token_id in token_ids)


class FakeDecodeStream:
    def __init__(self, *, skip_special_tokens: bool) -> None:
        if skip_special_tokens is not True:
            raise AssertionError("streamed chat must skip special tokens")

    def step(self, tokenizer: FakeBackendTokenizer, token_id: int) -> str | None:
        tokenizer.step_calls.append(token_id)
        return tokenizer.chunks[token_id]


def accepted_reply(text: str) -> str:
    boundary = re.search(MESSAGE_DELIMITER_PATTERN, text)
    return text[:boundary.start()] if boundary is not None else text


class ReplyStreamDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        # The constructor imports Tokenizers lazily. These tests need only stdlib.
        tokenizers = types.ModuleType("tokenizers")
        tokenizers.__path__ = []  # type: ignore[attr-defined]
        decoders = types.ModuleType("tokenizers.decoders")
        decoders.DecodeStream = FakeDecodeStream  # type: ignore[attr-defined]
        tokenizers.decoders = decoders  # type: ignore[attr-defined]
        patcher = mock.patch.dict(
            sys.modules, {"tokenizers": tokenizers, "tokenizers.decoders": decoders}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def snapshots(
        self, chunks: tuple[str | None, ...]
    ) -> tuple[list[str], FakeBackendTokenizer]:
        tokenizer = FakeBackendTokenizer(chunks)
        decoder = ReplyStreamDecoder(tokenizer)
        previews = [decoder.push(token_id) for token_id in range(len(chunks))]
        final = accepted_reply(tokenizer.decode(list(range(len(chunks))), skip_special_tokens=True))
        previous = ""
        for preview in previews:
            self.assertTrue(preview.startswith(previous), (previous, preview))
            self.assertTrue(final.startswith(preview), (final, preview))
            previous = preview
        return previews, tokenizer

    def test_plain_text_preserves_spaces_punctuation_and_single_newlines(self) -> None:
        chunks = (" Hello", ",", " world", "!", "\n", "Line two.", "  ")
        previews, tokenizer = self.snapshots(chunks)
        self.assertEqual(previews, [
            " Hello", " Hello,", " Hello, world", " Hello, world!",
            " Hello, world!", " Hello, world!\nLine two.",
            " Hello, world!\nLine two.  ",
        ])
        self.assertEqual(tokenizer.step_calls, list(range(len(chunks))))

    def test_fragmented_speaker_marker_never_appears_in_a_preview(self) -> None:
        previews, _ = self.snapshots(("Reply", "\n", "\n", "Hu", "man", ">", " ignored"))
        self.assertEqual(previews, ["Reply"] * 7)

    def test_boundary_at_every_two_chunk_split(self) -> None:
        for label in ("Human", "GPT-2", "Other_Speaker", "東京💡", "λ:participant"):
            text = "Reply\n\n" + label + "> discarded"
            for split in range(len(text) + 1):
                with self.subTest(label=label, split=split):
                    previews, _ = self.snapshots((text[:split], text[split:]))
                    self.assertEqual(previews[-1], "Reply")

    def test_long_fragmented_label_is_withheld_until_its_boundary_completes(self) -> None:
        label = "L" * 8192
        chunks = ("Reply", "\n\n") + tuple(
            label[start:start + 64] for start in range(0, len(label), 64)
        ) + ("> discarded", " more discarded")
        previews, _ = self.snapshots(chunks)
        self.assertEqual(previews, ["Reply"] * len(chunks))

    def test_overlapping_newlines_preserve_text_before_the_first_boundary(self) -> None:
        for count in (2, 3, 4, 7):
            with self.subTest(newlines=count):
                chunks = ("Reply",) + ("\n",) * count + ("Human", "> ignored")
                previews, _ = self.snapshots(chunks)
                self.assertEqual(previews[-1], "Reply" + "\n" * (count - 2))
                self.assertEqual(previews[count], "Reply" + "\n" * (count - 2))

    def test_whitespace_invalidates_a_label_and_releases_ordinary_reply_text(self) -> None:
        for whitespace in (" ", "\t", "\r", "\v", "\f", "\u2003"):
            with self.subTest(whitespace=repr(whitespace)):
                previews, _ = self.snapshots(("Reply", "\n\nHuman", whitespace, "> ordinary"))
                self.assertEqual(previews, [
                    "Reply", "Reply", "Reply\n\nHuman" + whitespace,
                    "Reply\n\nHuman" + whitespace + "> ordinary",
                ])

    def test_newline_invalidates_a_label_but_can_start_a_later_boundary(self) -> None:
        previews, _ = self.snapshots(("Reply", "\n\nHuman", "\n", "\n", "Other", ">"))
        self.assertEqual(previews, [
            "Reply", "Reply", "Reply\n\nHuman", "Reply\n\nHuman",
            "Reply\n\nHuman", "Reply\n\nHuman",
        ])

    def test_empty_label_does_not_form_a_boundary(self) -> None:
        previews, _ = self.snapshots(("Reply", "\n\n", ">", " more"))
        self.assertEqual(previews, ["Reply", "Reply", "Reply\n\n>", "Reply\n\n> more"])

    def test_complete_boundary_discards_aftertext_in_the_same_chunk(self) -> None:
        previews, _ = self.snapshots(("Reply\n\nHuman> first ignored text", "\n\nGPT-2> more"))
        self.assertEqual(previews, ["Reply", "Reply"])

    def test_first_complete_boundary_wins_after_an_invalidated_candidate(self) -> None:
        previews, _ = self.snapshots(("Reply\n\nHuman words\n\nOther> ignored", " later"))
        self.assertEqual(previews, ["Reply\n\nHuman words"] * 2)

    def test_tokens_after_a_complete_boundary_are_permanently_ignored(self) -> None:
        tokenizer = FakeBackendTokenizer(("Reply\n\nHuman>",))
        decoder = ReplyStreamDecoder(tokenizer)
        self.assertEqual(decoder.push(0), "Reply")
        # These IDs are outside the fake vocabulary: a completed decoder must
        # return its accepted prefix without attempting any further decoding.
        for token_id in (1, 99, 1000):
            self.assertEqual(decoder.push(token_id), "Reply")
        self.assertEqual(tokenizer.step_calls, [0])

    def test_partial_unicode_and_special_token_chunks_do_not_add_replacement_text(self) -> None:
        chunks = (" A", None, None, "💡", None, "B", " e", None, "\u0301", " café 東京")
        previews, tokenizer = self.snapshots(chunks)
        self.assertEqual(previews, [
            " A", " A", " A", " A💡", " A💡", " A💡B", " A💡B e",
            " A💡B e", " A💡B e\u0301", " A💡B e\u0301 café 東京",
        ])
        self.assertEqual(tokenizer.step_calls, list(range(len(chunks))))
        self.assertTrue(all("\ufffd" not in preview for preview in previews))

    def test_none_chunks_do_not_release_an_incomplete_boundary(self) -> None:
        previews, _ = self.snapshots((None, "Reply\n\nHu", None, "man", None, ">", None))
        self.assertEqual(previews, ["", "Reply", "Reply", "Reply", "Reply", "Reply", "Reply"])

    def test_empty_chunks_do_not_release_an_incomplete_boundary(self) -> None:
        previews, _ = self.snapshots(("Reply\n\nHu", "", "", "man", ">"))
        self.assertEqual(previews, ["Reply"] * 5)

    def test_terminal_incomplete_markers_wait_for_canonical_result_reconciliation(self) -> None:
        for suffix in ("\n", "\n\n", "\n\nHu", "\n\n" + "L" * 8192):
            with self.subTest(suffix=suffix[:12]):
                tokenizer = FakeBackendTokenizer(("Reply", suffix))
                decoder = ReplyStreamDecoder(tokenizer)
                self.assertEqual(decoder.push(0), "Reply")
                self.assertEqual(decoder.push(1), "Reply")
                # No flush is required of the stream helper. The final generation
                # decoder remains authoritative and includes this ordinary tail.
                canonical = tokenizer.decode([0, 1], skip_special_tokens=True)
                self.assertEqual(canonical, "Reply" + suffix)
                self.assertEqual(tokenizer.decode_calls, [((0, 1), True)])

    def test_invalid_token_ids_are_rejected_before_decode(self) -> None:
        tokenizer = FakeBackendTokenizer(("Reply",))
        decoder = ReplyStreamDecoder(tokenizer)
        for token_id in (True, False, -1, 1.5, "0", None):
            with self.subTest(token_id=token_id), self.assertRaises(ValueError):
                decoder.push(token_id)  # type: ignore[arg-type]
        self.assertEqual(tokenizer.step_calls, [])


class CachedPinnedTokenizerStreamTests(unittest.TestCase):
    """Optional byte-level checks; never download files or load model weights."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import tokenizers
        except ImportError as error:
            raise unittest.SkipTest("Tokenizers is not installed") from error
        if tokenizers.__version__ != "0.23.2":
            raise unittest.SkipTest("requires pinned Tokenizers 0.23.2")
        cache_roots: list[Path] = []
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if os.environ.get(name):
                cache_roots.append(Path(os.environ[name]))
        if os.environ.get("HF_HOME"):
            cache_roots.append(Path(os.environ["HF_HOME"]) / "hub")
        if os.environ.get("XDG_CACHE_HOME"):
            cache_roots.append(Path(os.environ["XDG_CACHE_HOME"]) / "huggingface" / "hub")
        cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
        relative = Path("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION / "tokenizer.json"
        cached = next((root / relative for root in cache_roots if (root / relative).is_file()), None)
        if cached is None:
            raise unittest.SkipTest("pinned GPT-2 XL tokenizer.json is not cached locally")
        cls.tokenizer = tokenizers.Tokenizer.from_file(str(cached))

    def test_cached_decoder_has_stable_prefixes_and_matches_final_text(self) -> None:
        for text in (
            " Hello, world!", " A💡B", " e\u0301 café 東京", "\n\nHuman> next",
            " Reply💡\n\n東京> ignored", " Paragraph one.\n\nOrdinary paragraph two.",
        ):
            with self.subTest(text=text):
                token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
                canonical = accepted_reply(self.tokenizer.decode(token_ids, skip_special_tokens=True))
                decoder = ReplyStreamDecoder(self.tokenizer)
                previous = ""
                for token_id in token_ids:
                    preview = decoder.push(token_id)
                    self.assertTrue(preview.startswith(previous), (previous, preview))
                    self.assertTrue(canonical.startswith(preview), (canonical, preview))
                    previous = preview
                self.assertEqual(previous, canonical)

    def test_cached_decoder_waits_for_split_utf8_and_skips_eos(self) -> None:
        decoder = ReplyStreamDecoder(self.tokenizer)
        # These pinned GPT-2 IDs split the UTF-8 bytes for the light-bulb emoji.
        self.assertEqual(decoder.push(8582), "")
        self.assertEqual(decoder.push(240), "")
        self.assertEqual(decoder.push(94), "💡")
        self.assertEqual(decoder.push(50256), "💡")
        self.assertEqual(self.tokenizer.decode([8582, 240, 94, 50256], skip_special_tokens=True), "💡")

    def test_incomplete_utf8_tail_is_left_to_canonical_final_decode(self) -> None:
        decoder = ReplyStreamDecoder(self.tokenizer)
        self.assertEqual(decoder.push(8582), "")
        self.assertEqual(self.tokenizer.decode([8582], skip_special_tokens=True), "\ufffd")


if __name__ == "__main__":
    unittest.main()
