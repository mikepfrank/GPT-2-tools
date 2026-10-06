"""Immutable, three-participant context and generation for Codex/GPT-2 chat.

This module owns no sessions, threads, tools, or persistence. A caller commits
the returned conversation/message only after its own transaction succeeds.
"""
from __future__ import annotations

import platform
import re
import secrets
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import date
from types import MappingProxyType

from .chat import EXAMPLES, MESSAGE_DELIMITER_PATTERN, MESSAGE_DELIMITER_RE
from .runtime import (
    MODEL_CONTEXT_TOKENS, MODEL_ID, MODEL_REVISION, MODEL_SHA256,
    GenerationSettings, Gpt2Runner, SampledToken, runtime_summary,
)
from .streaming import ReplyStreamDecoder

ROLES = ("Human", "GPT-2", "Codex")
GPT2_REPLY_TOKENS = 120
GPT2_CUE = "\n\nGPT-2>"
OMISSION_MARKER = "\n\n..."
_INCOMPLETE_BOUNDARY = re.compile(r"\n\n[^\s>]*\Z")


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _validate_content(role: str, content: str) -> None:
    if role not in ROLES:
        raise ValueError("Message role must be Human, GPT-2, or Codex")
    if not isinstance(content, str) or not content.startswith(" "):
        raise ValueError("Message content must begin with a space")
    if role != "GPT-2" and not content.strip():
        raise ValueError("Enter a message before sending")
    if "\r" in content:
        raise ValueError("Message content must use LF newlines")
    if MESSAGE_DELIMITER_RE.search(content):
        raise ValueError("A blank line followed by a speaker label and > is reserved for message boundaries")
    if "<|endoftext|>" in content:
        raise ValueError("<|endoftext|> is reserved by the GPT-2 tokenizer")


def _normalize_content(text: object) -> str:
    if not isinstance(text, str):
        raise ValueError("Message must be text")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text if text.startswith(" ") else " " + text


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    generation: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _validate_content(self.role, self.content)
        if self.generation is not None:
            if not isinstance(self.generation, Mapping):
                raise ValueError("Generation metadata must be an object")
            object.__setattr__(self, "generation", _freeze(self.generation))

    def text(self) -> str:
        return "\n\n" + self.role + ">" + self.content

    def record(self) -> dict[str, object]:
        return {"role": self.role, "content": self.content, "generation": _thaw(self.generation)}

    @classmethod
    def from_record(cls, record: object) -> Message:
        if not isinstance(record, Mapping) or set(record) - {"role", "content", "generation"}:
            raise ValueError("Invalid message record")
        return cls(record.get("role"), record.get("content"), record.get("generation"))


@dataclass(frozen=True)
class Conversation:
    header: str
    messages: tuple[Message, ...] = ()
    context_start: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.header, str) or not self.header.strip():
            raise ValueError("Conversation header must be nonempty text")
        if "<|endoftext|>" in self.header or MESSAGE_DELIMITER_RE.search(self.header):
            raise ValueError("Conversation header contains a reserved marker")
        messages = tuple(self.messages)
        if any(not isinstance(message, Message) for message in messages):
            raise ValueError("Conversation messages must be Message instances")
        if isinstance(self.context_start, bool) or not isinstance(self.context_start, int) or not 0 <= self.context_start <= len(messages):
            raise ValueError("Invalid retained-context position")
        object.__setattr__(self, "messages", messages)

    def record(self) -> dict[str, object]:
        return {"header": self.header, "messages": [message.record() for message in self.messages], "context_start": self.context_start}

    @classmethod
    def from_record(cls, record: object) -> Conversation:
        if not isinstance(record, Mapping) or set(record) - {"header", "messages", "context_start"}:
            raise ValueError("Invalid conversation record")
        messages = record.get("messages", [])
        if not isinstance(messages, (list, tuple)):
            raise ValueError("Conversation messages must be an array")
        return cls(record.get("header"), tuple(Message.from_record(item) for item in messages), record.get("context_start", 0))


@dataclass(frozen=True)
class PreparedContext:
    text: str
    input_tokens: int
    context_start: int
    blocks: tuple[dict[str, str], ...]


def new_conversation(chat_date: date, example_rounds: int = 1) -> Conversation:
    if not isinstance(chat_date, date):
        raise ValueError("Chat date must be a date")
    if isinstance(example_rounds, bool) or not isinstance(example_rounds, int) or example_rounds not in (0, 1, 2):
        raise ValueError("Example rounds must be 0, 1, or 2")
    date_text = f"{chat_date.strftime('%B')} {chat_date.day}, {chat_date.year}"
    header = (
        "Context: This is a chat experiment with GPT-2 XL, OpenAI's original "
        "1.56-billion-parameter language model released in 2019. "
        f"Today's date is {date_text}. "
        "GPT-2 retains its original training. Codex is a newer AI participant "
        "that can use tools, search the web, and run code. Human is a person.\n\n"
        "Messages begin with Human>, GPT-2>, or Codex>, preceded by a blank line. "
        "Continue the latest GPT-2 message with a short conversational reply. "
        "Stop before beginning another message."
    )
    examples = tuple(
        message for human, model in EXAMPLES[:example_rounds]
        for message in (Message("Human", " " + human), Message("GPT-2", model))
    )
    return Conversation(header, examples)


def _reply_allowance(max_reply_tokens: int) -> None:
    if isinstance(max_reply_tokens, bool) or not isinstance(max_reply_tokens, int) or not 1 <= max_reply_tokens <= GPT2_REPLY_TOKENS:
        raise ValueError("GPT-2 reply allowance must be between 1 and 120 tokens")


def packed_context(
    conversation: Conversation, count_tokens: Callable[[str], int],
    max_reply_tokens: int = GPT2_REPLY_TOKENS, *, for_generation: bool = False,
) -> PreparedContext:
    """Pack a retained suffix, keeping the header and latest message intact.

    Earlier private transcript messages are never deleted or resurrected. The
    omission hint participates in token accounting. A generation prompt adds
    only a GPT-2 cue; the preceding speaker can be any of the three roles.
    """
    _reply_allowance(max_reply_tokens)
    start = conversation.context_start
    # Preserve the latest message even if a loaded record has an empty suffix.
    if conversation.messages:
        start = min(start, len(conversation.messages) - 1)
    while True:
        header = conversation.header
        if start and not header.endswith(OMISSION_MARKER):
            header += OMISSION_MARKER
        blocks = [{"role": "context", "text": header}]
        blocks.extend({"role": message.role, "text": message.text()} for message in conversation.messages[start:])
        if for_generation:
            blocks.append({"role": "GPT-2", "text": GPT2_CUE})
        text = "".join(block["text"] for block in blocks)
        tokens = count_tokens(text)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
            raise RuntimeError("Tokenizer returned an invalid input token count")
        if tokens + max_reply_tokens <= MODEL_CONTEXT_TOKENS:
            return PreparedContext(text, tokens, start, tuple(blocks))
        if start >= len(conversation.messages) - 1:
            raise ValueError(
                f"The header and latest message need {tokens} input tokens; "
                f"only {MODEL_CONTEXT_TOKENS - max_reply_tokens} fit with a {max_reply_tokens}-token reply allowance"
            )
        start += 1


def append_message(
    conversation: Conversation, role: str, text: str,
    count_tokens: Callable[[str], int], max_reply_tokens: int = GPT2_REPLY_TOKENS,
    *, generation: Mapping[str, object] | None = None,
) -> Conversation:
    """Return a validated new snapshot; oversized input leaves the old one intact."""
    message = Message(role, _normalize_content(text), generation)
    candidate = replace(conversation, messages=conversation.messages + (message,))
    # Reserve the next GPT-2 cue as well as the reply, even after a Codex turn.
    prepared = packed_context(candidate, count_tokens, max_reply_tokens, for_generation=True)
    return replace(candidate, context_start=prepared.context_start)


class GenerationCancelled(RuntimeError):
    """The caller interrupted generation; no message should be committed."""


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise GenerationCancelled("GPT-2 generation was cancelled")


def _provenance(runner: Gpt2Runner) -> dict[str, object]:
    supplied = getattr(runner, "runtime_provenance", None)
    if isinstance(supplied, Mapping):
        runtime = dict(supplied)
    else:
        try:
            runtime = runtime_summary()
        except (RuntimeError, AttributeError):
            runtime = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    return {
        "model": MODEL_ID, "revision": MODEL_REVISION,
        "expected_checkpoint_sha256": MODEL_SHA256,
        "context_tokens": MODEL_CONTEXT_TOKENS,
        "runner_class": type(runner).__qualname__, "runtime": runtime,
    }


def generate_reply(
    conversation: Conversation, runner: Gpt2Runner, settings: GenerationSettings,
    *, on_text: Callable[[str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Message:
    """Generate a detached, canonical GPT-2 message with exact replay evidence.

    ``on_text`` receives provisional cumulative continuations, never delimiters
    or their incomplete suffixes. It does not commit the conversation. A fresh
    entropy seed is recorded when no explicit seed is supplied.
    """
    settings.validate()
    _reply_allowance(settings.max_new_tokens)
    if on_text is not None and not callable(on_text):
        raise ValueError("on_text must be callable or None")
    if cancelled is not None and not callable(cancelled):
        raise ValueError("cancelled must be callable or None")
    _check_cancelled(cancelled)
    prepared = packed_context(conversation, runner.count_prompt_tokens, settings.max_new_tokens, for_generation=True)
    effective = replace(settings, seed=secrets.randbelow(2**63)) if settings.seed is None else settings
    tokenizer = getattr(runner, "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(getattr(runner, "_runner", None), "tokenizer", None)
    input_ids = list(tokenizer.encode(prepared.text, add_special_tokens=False)) if tokenizer is not None and hasattr(tokenizer, "encode") else None
    if input_ids is not None and len(input_ids) != prepared.input_tokens:
        raise RuntimeError("Prompt token IDs disagreed with input token accounting")
    decoder = None
    if on_text is not None:
        if tokenizer is None or not hasattr(tokenizer, "backend_tokenizer"):
            raise RuntimeError("Streaming requires the GPT-2 backend tokenizer")
        decoder = ReplyStreamDecoder(tokenizer.backend_tokenizer)
    observed_count = 0

    def observe(event: SampledToken) -> None:
        nonlocal observed_count
        _check_cancelled(cancelled)
        if not observed_count < event.output_tokens <= effective.max_new_tokens:
            raise RuntimeError("Invalid streamed output token accounting")
        observed_count = event.output_tokens
        if decoder is not None and on_text is not None:
            visible = decoder.push(event.token_id)
            on_text(_normalize_content(visible) if visible else "")

    kwargs = {"stop_pattern": MESSAGE_DELIMITER_PATTERN}
    if on_text is not None or cancelled is not None:
        kwargs["on_token"] = observe
    result = runner.generate_result(prepared.text, effective, **kwargs)
    _check_cancelled(cancelled)
    if result.prompt_truncated or result.input_tokens != prepared.input_tokens or result.original_input_tokens != prepared.input_tokens:
        raise RuntimeError("Chat token accounting disagreed with the runtime; history was preserved")
    if result.output_tokens != len(result.output_token_ids) or result.output_tokens > effective.max_new_tokens:
        raise RuntimeError("Invalid generated output token accounting")
    text = result.text
    boundary = MESSAGE_DELIMITER_RE.search(text)
    if boundary is not None:
        text = text[:boundary.start()]
    # A terminal partial speaker marker must not become a committed message.
    incomplete = _INCOMPLETE_BOUNDARY.search(text)
    if incomplete is not None:
        text = text[:incomplete.start()]
    generation = {
        "prompt_text": prepared.text, "input_tokens": result.input_tokens,
        "input_token_ids": input_ids, "output_tokens": result.output_tokens,
        "output_token_ids": list(result.output_token_ids), "raw_text": result.text,
        "stop_reason": "message_delimiter" if text != result.text else result.stop_reason,
        "elapsed_seconds": result.elapsed_seconds, "tokens_per_second": result.tokens_per_second,
        "cold_start_included": result.cold_start_included,
        "seed": str(effective.seed), "requested_seed": str(settings.seed) if settings.seed is not None else None,
        "settings": {**asdict(effective), "seed": str(effective.seed)},
        "temperature": effective.temperature, "max_new_tokens": effective.max_new_tokens,
        "context_start": prepared.context_start, "provenance": _provenance(runner),
    }
    return Message("GPT-2", _normalize_content(text), generation)
