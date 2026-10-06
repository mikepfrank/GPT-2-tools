"""Validate portable, visible-history archives for the three-participant chat."""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from typing import Callable

from .codex_chat_context import Conversation, packed_context

MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_TEXT_CHARS = 65536
_REQUEST_ID = re.compile(r"[A-Za-z0-9_-]{8,80}\Z")
_SESSION_ID = re.compile(r"[a-f0-9]{32}\Z")
_THREAD_ID = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


class ArchiveTooLarge(ValueError):
    pass


def _json_copy(value, depth=0):
    """Copy bounded JSON without retaining caller-owned mutable objects."""
    if depth > MAX_JSON_DEPTH:
        raise ValueError("Chat archive nesting is too deep")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        if len(value) > MAX_TEXT_CHARS:
            raise ValueError("Chat archive text exceeds 65,536 characters")
        try:
            value.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Chat archive contains invalid Unicode text") from exc
        return value
    if isinstance(value, int):
        if not -(2**63) <= value < 2**63:
            raise ValueError("Chat archive integers must fit in 64 bits")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Chat archive numbers must be finite")
        return value
    if isinstance(value, list):
        return [_json_copy(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Chat archive object keys must be text")
        return {_json_copy(key, depth + 1): _json_copy(item, depth + 1) for key, item in value.items()}
    raise ValueError("Chat archive must contain only JSON values")


def _object(value, name, required, optional=()):
    if not isinstance(value, dict) or not required <= set(value) or set(value) - required - set(optional):
        raise ValueError(f"{name} has missing or unsupported fields")
    return value


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return value


def _number(value, name, low=0, high=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value < low
            or (high is not None and value > high)):
        raise ValueError(f"{name} must be a finite number in the allowed range")
    return value


def _generation(value, role):
    if value is None:
        return
    if not isinstance(value, dict) or role != "GPT-2":
        raise ValueError("Generation evidence belongs to a GPT-2 message and must be an object")
    # Historical evidence may have fewer fields or additional provenance. Keep
    # it intact, while validating known measurements without running a model.
    for field, low, high in (("input_tokens", 1, 1024), ("output_tokens", 0, 120),
                             ("max_new_tokens", 1, 120), ("context_start", 0, 10000)):
        if field in value:
            _integer(value[field], f"generation.{field}", low, high)
    for field, count, limit in (("input_token_ids", "input_tokens", 1024),
                                ("output_token_ids", "output_tokens", 120)):
        if field not in value or (field == "input_token_ids" and value[field] is None):
            continue
        ids = value[field]
        if not isinstance(ids, list) or len(ids) > limit:
            raise ValueError(f"generation.{field} must be a bounded token array")
        for token in ids:
            _integer(token, f"generation.{field} entry", 0, 50256)
        if count in value and len(ids) != value[count]:
            raise ValueError(f"generation.{field} does not match its token count")
    if "max_new_tokens" in value and "output_tokens" in value and value["output_tokens"] > value["max_new_tokens"]:
        raise ValueError("Generation output exceeds its recorded allowance")
    if "input_tokens" in value and "max_new_tokens" in value and value["input_tokens"] + value["max_new_tokens"] > 1024:
        raise ValueError("Generation evidence exceeds the GPT-2 context window")
    for field in ("elapsed_seconds", "tokens_per_second", "temperature"):
        if field in value:
            _number(value[field], f"generation.{field}")
    for field in ("prompt_text", "raw_text"):
        if field in value and not isinstance(value[field], str):
            raise ValueError(f"generation.{field} must be text")
    if "cold_start_included" in value and type(value["cold_start_included"]) is not bool:
        raise ValueError("generation.cold_start_included must be a boolean")
    if "stop_reason" in value and (not isinstance(value["stop_reason"], str) or value["stop_reason"] not in {
            "eos_token", "max_new_tokens", "stop_sequence", "message_delimiter"}):
        raise ValueError("Generation stop reason is unsupported")
    for field in ("settings", "provenance"):
        if field in value and not isinstance(value[field], dict):
            raise ValueError(f"generation.{field} must be an object")
    for field in ("seed", "requested_seed"):
        if field in value and value[field] is not None:
            seed = value[field]
            if (not isinstance(seed, str) or not seed.isascii() or not seed.isdecimal()
                    or len(seed) > 19 or int(seed) > 2**63 - 1):
                raise ValueError(f"generation.{field} must be a decimal seed string or null")


def validate_archive(document: object, count_tokens: Callable[[str], int], *, max_messages: int):
    """Validate completely and return detached history, metadata and settings.

    Exported session/thread IDs and historical request IDs describe the source;
    callers must create fresh operational IDs rather than resume those handles.
    """
    archive = _json_copy(document)
    if len(json.dumps(archive, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")) > MAX_ARCHIVE_BYTES:
        raise ArchiveTooLarge("Chat archive exceeds the 10 MiB import limit")
    _object(archive, "Chat archive", {"format", "version", "session_id", "conversation", "metadata", "settings", "codex_thread_id"})
    if archive["format"] != "gpt2-codex-chat" or type(archive["version"]) is not int or archive["version"] != 1:
        raise ValueError("Import a gpt2-codex-chat version 1 export")
    if not isinstance(archive["session_id"], str) or not _SESSION_ID.fullmatch(archive["session_id"]):
        raise ValueError("Archive source session identifier is invalid")
    thread = archive["codex_thread_id"]
    if thread is not None and (not isinstance(thread, str) or not _THREAD_ID.fullmatch(thread)):
        raise ValueError("Archive source Codex thread identifier is invalid")
    raw = _object(archive["conversation"], "Conversation", {"header", "messages", "context_start"})
    if not isinstance(raw["messages"], list) or len(raw["messages"]) > max_messages:
        raise ValueError(f"A chat archive supports at most {max_messages} messages")
    if isinstance(raw["header"], str) and "\r" in raw["header"]:
        raise ValueError("Conversation header must use LF newlines")
    for message in raw["messages"]:
        _object(message, "Message", {"role", "content", "generation"})
        _generation(message["generation"], message["role"])
    conversation = Conversation.from_record(raw)
    metadata = archive["metadata"]
    if not isinstance(metadata, list) or len(metadata) != len(conversation.messages):
        raise ValueError("Archive metadata must match the complete message history")
    historical_requests = set()
    for index, item in enumerate(metadata):
        _object(item, "Message metadata", {"id", "created_at", "example"}, {"request_id"})
        if _integer(item["id"], "Message metadata id", 1, max_messages) != index + 1:
            raise ValueError("Message metadata IDs must follow the archived message order")
        _number(item["created_at"], "Message timestamp")
        if type(item["example"]) is not bool:
            raise ValueError("Message example flag must be a boolean")
        if "request_id" in item:
            request = item["request_id"]
            if (not isinstance(request, str) or not _REQUEST_ID.fullmatch(request)
                    or conversation.messages[index].role != "Human" or item["example"]
                    or request in historical_requests):
                raise ValueError("Historical Human request identifier is invalid")
            historical_requests.add(request)
    settings = _object(archive["settings"], "Settings", {"max_new_tokens", "temperature"})
    if _integer(settings["max_new_tokens"], "GPT-2 reply allowance", 120, 120) != 120:
        raise ValueError("GPT-2 replies are limited to 120 tokens")
    _number(settings["temperature"], "Temperature", high=2)
    # Reproduce the ordinary displayed window first. Reserving the generation
    # cue checks that a future reply is possible, but must not evict an extra
    # message from the imported browser context ahead of that future query.
    displayed = packed_context(conversation, count_tokens)
    packed_context(conversation, count_tokens, for_generation=True)
    conversation = replace(conversation, context_start=displayed.context_start)
    return conversation, metadata, settings
