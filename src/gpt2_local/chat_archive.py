"""Portable, validated archives for the local GPT-2 chat transcript."""
from __future__ import annotations

import copy
import math
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable

from .runtime import MODEL_CONTEXT_TOKENS, MODEL_ID, MODEL_REVISION, GenerationSettings

if TYPE_CHECKING:
    from .chat import ChatSession

ARCHIVE_FORMAT = "gpt2-local-chat"
ARCHIVE_SCHEMA_VERSION = 1
MAX_ARCHIVE_ROUNDS = 10000
MAX_TEXT_CHARS = 65536
MAX_NOTES = 100
GENERATION_FIELDS = {
    "input_tokens", "output_tokens", "output_token_ids", "stop_reason",
    "elapsed_seconds", "cold_start_included", "seed", "requested_seed",
    "temperature", "max_new_tokens", "prompt_text",
}
UNKNOWN_GENERATION_NOTE = (
    "Generation metadata is unavailable for one or more historical model replies; "
    "their sampling seeds are unknown."
)
UNKNOWN_SEED_NOTE = "One or more sampled model replies have no recorded sampling seed."


def _object(value: object, name: str, required: set[str], optional: set[str] | None = None) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    keys = set(value)
    if not required <= keys or keys - required - (optional or set()):
        raise ValueError(f"{name} has missing or unsupported fields")
    return value


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer from {minimum} through {maximum}")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _text(value: object, name: str, *, nonempty: bool = False, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or len(value) > limit or (nonempty and not value):
        raise ValueError(f"{name} must be {'nonempty ' if nonempty else ''}text of at most {limit} characters")
    return value


def _seed(value: object, name: str) -> int | None:
    if value is None:
        return None
    if (
        not isinstance(value, str) or not value.isascii() or not value.isdecimal()
        or len(value) > 19
    ):
        raise ValueError(f"{name} must be a decimal seed string or null")
    return _integer(int(value), name, 0, 2**63 - 1)


def _settings(raw: object) -> GenerationSettings:
    values = _object(raw, "settings", {
        "temperature", "max_new_tokens", "seed", "top_k", "top_p", "repetition_penalty",
    })
    if (
        _integer(values["top_k"], "settings.top_k", 0, 2**31 - 1) != 50
        or _number(values["top_p"], "settings.top_p") != 0.95
        or _number(values["repetition_penalty"], "settings.repetition_penalty") != 1.0
    ):
        raise ValueError("settings must retain top_k=50, top_p=0.95, and repetition_penalty=1")
    settings = GenerationSettings(
        max_new_tokens=_integer(values["max_new_tokens"], "settings.max_new_tokens", 1, 1023),
        temperature=_number(values["temperature"], "settings.temperature"),
        seed=_seed(values["seed"], "settings.seed"),
    )
    settings.validate()
    return settings


def _settings_record(settings: GenerationSettings) -> dict[str, object]:
    result = {
        "temperature": settings.temperature,
        "max_new_tokens": settings.max_new_tokens,
        "seed": str(settings.seed) if settings.seed is not None else None,
        "top_k": settings.top_k,
        "top_p": settings.top_p,
        "repetition_penalty": settings.repetition_penalty,
    }
    _settings(result)
    return result


def _notes(value: object) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_NOTES:
        raise ValueError(f"notes must be a list of at most {MAX_NOTES} text entries")
    return [_text(note, "notes entry", limit=4096) for note in value]


def _body(value: object, name: str, *, human: bool) -> str:
    from .chat import MESSAGE_DELIMITER_RE

    text = _text(value, name)
    if human and not text.strip():
        raise ValueError(f"{name} must contain a human message")
    if MESSAGE_DELIMITER_RE.search(text) or "<|endoftext|>" in text:
        raise ValueError(f"{name} contains a reserved message delimiter or end-of-text marker")
    return text


def _generation(value: object, count_tokens: Callable[[str], int]) -> dict[str, object] | None:
    if value is None:
        return None
    record = _object(value, "generation", GENERATION_FIELDS)
    maximum = _integer(record["max_new_tokens"], "generation.max_new_tokens", 1, 1023)
    input_tokens = _integer(record["input_tokens"], "generation.input_tokens", 1, 1023)
    output_tokens = _integer(record["output_tokens"], "generation.output_tokens", 0, maximum)
    if input_tokens + maximum > MODEL_CONTEXT_TOKENS:
        raise ValueError("generation exceeds GPT-2's context window")
    ids = record["output_token_ids"]
    if not isinstance(ids, list) or len(ids) != output_tokens:
        raise ValueError("generation.output_token_ids must match the sampled token count")
    for token in ids:
        _integer(token, "generation token ID", 0, 50256)
    if record["stop_reason"] not in ("eos_token", "max_new_tokens", "stop_sequence"):
        raise ValueError("generation.stop_reason is unsupported")
    _number(record["elapsed_seconds"], "generation.elapsed_seconds")
    if not isinstance(record["cold_start_included"], bool):
        raise ValueError("generation.cold_start_included must be a boolean")
    _seed(record["seed"], "generation.seed")
    _seed(record["requested_seed"], "generation.requested_seed")
    _number(record["temperature"], "generation.temperature")
    prompt = _text(record["prompt_text"], "generation.prompt_text", nonempty=True)
    if count_tokens(prompt) != input_tokens:
        raise ValueError("generation.prompt_text does not match its recorded input token count")
    return copy.deepcopy(record)


def _add_unknown_notes(notes: list[str], rounds: list) -> list[str]:
    result = list(notes)
    real = [item for item in rounds if not item.example]
    if (
        len(result) < MAX_NOTES
        and any(item.generation is None for item in real)
        and UNKNOWN_GENERATION_NOTE not in result
    ):
        result.append(UNKNOWN_GENERATION_NOTE)
    if any(
        item.generation is not None and item.generation["temperature"] > 0
        and item.generation["seed"] is None
        for item in real
    ) and UNKNOWN_SEED_NOTE not in result and len(result) < MAX_NOTES:
        result.append(UNKNOWN_SEED_NOTE)
    return _notes(result)


def export_chat(session: ChatSession, settings: GenerationSettings | None = None) -> dict[str, object]:
    """Export exact text and retained-context position without changing a session."""
    from .chat import MESSAGE_DELIMITER_PATTERN

    effective_settings = settings if settings is not None else session.settings
    if len(session.transcript) > MAX_ARCHIVE_ROUNDS:
        raise ValueError(f"A chat archive supports at most {MAX_ARCHIVE_ROUNDS} rounds")
    if (
        not 0 <= session.dropped_rounds <= len(session.transcript)
        or session.rounds != session.transcript[session.dropped_rounds:]
    ):
        raise ValueError("The retained chat context is not the archived transcript suffix")
    blocks = []
    for item in session.transcript:
        blocks.extend((
            {"role": "human", "label": "Human", "text": item.human, "example": item.example, "generation": None},
            {"role": "model", "label": "GPT-2", "text": item.assistant, "example": item.example,
             "generation": copy.deepcopy(item.generation)},
        ))
    return {
        "format": ARCHIVE_FORMAT,
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "context_tokens": MODEL_CONTEXT_TOKENS},
        "protocol": {"human_label": "Human", "model_label": "GPT-2", "message_delimiter_pattern": MESSAGE_DELIMITER_PATTERN},
        "prompt_header": session.header,
        "example_rounds": session.example_rounds,
        "settings": _settings_record(effective_settings),
        "blocks": blocks,
        "context": {"first_retained_round": session.dropped_rounds, "text": session.context_text()},
        "notes": _add_unknown_notes(_notes(session.notes), session.transcript),
    }


def import_chat(data: object, count_tokens: Callable[[str], int]) -> ChatSession:
    """Validate an archive fully, then construct a fresh resumable chat session."""
    from .chat import ChatSession, MESSAGE_DELIMITER_PATTERN, Round

    archive = _object(data, "archive", {
        "format", "schema_version", "exported_at", "model", "protocol", "prompt_header",
        "example_rounds", "settings", "blocks", "context",
    }, {"notes"})
    if archive["format"] != ARCHIVE_FORMAT:
        raise ValueError("Unsupported chat archive format")
    if _integer(archive["schema_version"], "schema_version", 1, 1) != ARCHIVE_SCHEMA_VERSION:
        raise ValueError("Unsupported chat archive schema version")
    timestamp = _text(archive["exported_at"], "exported_at", nonempty=True, limit=64)
    try:
        exported_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("exported_at must be a UTC timestamp") from exc
    if exported_at.tzinfo is None or exported_at.utcoffset() != timedelta(0):
        raise ValueError("exported_at must be a UTC timestamp")
    model = _object(archive["model"], "model", {"id", "revision", "context_tokens"})
    if (
        model["id"] != MODEL_ID or model["revision"] != MODEL_REVISION
        or _integer(model["context_tokens"], "model.context_tokens", 1024, 1024) != MODEL_CONTEXT_TOKENS
    ):
        raise ValueError("The archive uses a different GPT-2 checkpoint or context window")
    protocol = _object(archive["protocol"], "protocol", {
        "human_label", "model_label", "message_delimiter_pattern",
    })
    if protocol != {
        "human_label": "Human", "model_label": "GPT-2",
        "message_delimiter_pattern": MESSAGE_DELIMITER_PATTERN,
    }:
        raise ValueError("The archive uses an unsupported conversation protocol")
    header = _text(archive["prompt_header"], "prompt_header", nonempty=True)
    examples = _integer(archive["example_rounds"], "example_rounds", 0, 2)
    settings = _settings(archive["settings"])
    notes = _notes(archive.get("notes", []))
    blocks = archive["blocks"]
    if not isinstance(blocks, list) or len(blocks) % 2 or len(blocks) > MAX_ARCHIVE_ROUNDS * 2:
        raise ValueError(f"blocks must contain complete human/model pairs, at most {MAX_ARCHIVE_ROUNDS} rounds")
    if len(blocks) // 2 < examples:
        raise ValueError("The archive is missing its initial example rounds")
    rounds = []
    for index in range(0, len(blocks), 2):
        human = _object(blocks[index], "human block", {"role", "label", "text", "example", "generation"})
        model_block = _object(blocks[index + 1], "model block", {"role", "label", "text", "example", "generation"})
        if human["role"] != "human" or human["label"] != "Human" or human["generation"] is not None:
            raise ValueError("Human blocks must use the Human label and have no generation metadata")
        if model_block["role"] != "model" or model_block["label"] != "GPT-2":
            raise ValueError("Model blocks must use the GPT-2 label")
        expected_example = index // 2 < examples
        if (
            not isinstance(human["example"], bool) or not isinstance(model_block["example"], bool)
            or human["example"] != model_block["example"] or human["example"] != expected_example
        ):
            raise ValueError("Example flags must be paired and match the initial example-round prefix")
        generation = _generation(model_block["generation"], count_tokens)
        if expected_example and generation is not None:
            raise ValueError("Example replies must not claim generated-token metadata")
        rounds.append(Round(
            human=_body(human["text"], "human text", human=True),
            assistant=_body(model_block["text"], "model text", human=False),
            example=expected_example,
            generation=generation,
        ))
    context = _object(archive["context"], "context", {"first_retained_round", "text"})
    first_retained = _integer(context["first_retained_round"], "context.first_retained_round", 0, len(rounds))
    text = _text(context["text"], "context.text", nonempty=True)
    retained = rounds[first_retained:]
    expected_context = header + "".join(item.text() for item in retained)
    if text != expected_context:
        raise ValueError("context.text does not exactly match the header and retained transcript suffix")
    for value, name in ((header, "prompt_header"), (text, "context.text")):
        tokens = count_tokens(value)
        if isinstance(tokens, bool) or not isinstance(tokens, int) or not 1 <= tokens <= MODEL_CONTEXT_TOKENS:
            raise ValueError(f"{name} must fit GPT-2's 1024-token context window")
    real_rounds = [item for item in rounds if not item.example]
    return ChatSession(
        session_id=secrets.token_urlsafe(24),
        header=header,
        example_rounds=examples,
        rounds=retained,
        transcript=rounds,
        dropped_rounds=first_retained,
        last_generation=copy.deepcopy(real_rounds[-1].generation) if real_rounds else None,
        settings=settings,
        notes=_add_unknown_notes(notes, rounds),
    )
