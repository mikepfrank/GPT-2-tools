"""A small local browser chat using the historical GPT-2 XL checkpoint."""
from __future__ import annotations

import argparse
import copy
import json
import re
import secrets
import sys
import threading
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from .runtime import MODEL_CONTEXT_TOKENS, GenerationSettings, Gpt2Runner

HUMAN_MARKER = "\n\nHuman>"
MODEL_MARKER = "\n\nGPT-2>"
OMISSION_MARKER = "\n\n...\n\n"
MESSAGE_DELIMITER_PATTERN = r"\n\n[^\s>]+>"
MESSAGE_DELIMITER_RE = re.compile(MESSAGE_DELIMITER_PATTERN)
EXAMPLES = (
    (
        "Hello, who are you?",
        " I'm GPT-2, a language model. I generate text from the words you give me. "
        "This conversation is a little unfamiliar, but I can try to reply.",
    ),
    (
        "What would you like to talk about?",
        " Perhaps books, science, or everyday life. What interests you?",
    ),
)


def prompt_header(chat_date: date) -> str:
    date_text = f"{chat_date.strftime('%B')} {chat_date.day}, {chat_date.year}"
    return (
        "Context: This is a local chat experiment with GPT-2 XL, OpenAI's original "
        "1.56-billion-parameter language model released in 2019. "
        f"Today's date is {date_text}. "
        "Since 2019, language models have become more capable at conversation, "
        "coding, and reasoning; newer systems can also use tools and process images "
        "and audio. This GPT-2 checkpoint retains its original training.\n\n"
        "Messages begin with Human> or GPT-2>, preceded by a blank line. "
        "Continue the latest GPT-2 message with a short conversational reply. "
        "Stop before beginning another message."
    )


@dataclass(frozen=True)
class Round:
    human: str
    assistant: str
    example: bool = False
    generation: dict[str, object] | None = None
    previous_responses: list[dict[str, object]] = field(default_factory=list)

    def text(self) -> str:
        # Assistant text is the exact continuation, including its leading space.
        return HUMAN_MARKER + " " + self.human + MODEL_MARKER + self.assistant

    def record(self) -> dict[str, object]:
        return {"human": self.human, "assistant": self.assistant, "example": self.example, "generation": self.generation,
                "previous_responses": copy.deepcopy(self.previous_responses)}


@dataclass(frozen=True)
class PreparedInput:
    message: str
    prompt: str
    input_tokens: int
    retained_rounds: list[Round]
    header: str | None = None


def _trim_reply(text: str) -> str:
    match = MESSAGE_DELIMITER_RE.search(text)
    return text[: match.start()] if match else text


def parse_settings(raw: object) -> GenerationSettings:
    if not isinstance(raw, dict):
        raise ValueError("settings must be an object")
    if set(raw) - {"max_new_tokens", "temperature", "seed"}:
        raise ValueError("Unknown generation setting")
    maximum = raw.get("max_new_tokens", 120)
    temperature = raw.get("temperature", 0.8)
    seed = raw.get("seed")
    if isinstance(maximum, bool) or not isinstance(maximum, int):
        raise ValueError("Reply length must be an integer")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("Temperature must be a number")
    try:
        temperature = float(temperature)
    except OverflowError as exc:
        raise ValueError("Temperature must be finite") from exc
    if seed is not None:
        if not isinstance(seed, str) or not seed.isascii() or not seed.isdecimal():
            raise ValueError("Seed must be a decimal integer, or left blank")
        seed = int(seed)
    settings = GenerationSettings(max_new_tokens=maximum, temperature=temperature, seed=seed)
    settings.validate()
    return settings


def settings_record(settings: GenerationSettings) -> dict[str, object]:
    record = asdict(settings)
    record["seed"] = str(settings.seed) if settings.seed is not None else None
    return record


def resolve_seed(settings: GenerationSettings, fallback_seed: int | None = None) -> GenerationSettings:
    """Use a requested/stored seed, choosing one only when neither exists."""
    seed = settings.seed if settings.seed is not None else fallback_seed
    if seed is None:
        seed = secrets.randbelow(2**63)
    return replace(settings, seed=seed)


def _generate_reply(
    prepared: PreparedInput, requested: GenerationSettings,
    effective: GenerationSettings, runner: Gpt2Runner,
) -> tuple[str, dict[str, object]]:
    result = runner.generate_result(prepared.prompt, effective, stop_pattern=MESSAGE_DELIMITER_PATTERN)
    if result.prompt_truncated or result.input_tokens != prepared.input_tokens:
        raise RuntimeError("Chat token accounting disagreed with the runtime; history was preserved")
    reply_text = _trim_reply(result.text)
    generation = {
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "stop_reason": "stop_sequence" if reply_text != result.text else result.stop_reason,
        "elapsed_seconds": result.elapsed_seconds,
        "seed": str(effective.seed),
        "requested_seed": str(requested.seed) if requested.seed is not None else None,
        "temperature": effective.temperature,
        "max_new_tokens": effective.max_new_tokens,
        "prompt_text": prepared.prompt,
        "output_token_ids": list(result.output_token_ids),
        "cold_start_included": result.cold_start_included,
    }
    return reply_text, generation


@dataclass
class ChatSession:
    session_id: str
    header: str
    example_rounds: int
    rounds: list[Round] = field(default_factory=list)
    transcript: list[Round] = field(default_factory=list)
    dropped_rounds: int = 0
    last_generation: dict[str, object] | None = None
    settings: GenerationSettings = field(default_factory=lambda: GenerationSettings(max_new_tokens=120))
    notes: list[str] = field(default_factory=list)

    def context_text(self) -> str:
        return self.header + "".join(item.text() for item in self.rounds)

    def context_blocks(
        self, rounds: list[Round] | None = None, pending: str | None = None,
        *, header: str | None = None,
    ) -> list[dict[str, str]]:
        blocks = [{"role": "context", "text": self.header if header is None else header}]
        for item in self.rounds if rounds is None else rounds:
            blocks.append({"role": "human", "text": HUMAN_MARKER + " " + item.human})
            blocks.append({"role": "model", "text": MODEL_MARKER + item.assistant})
        if pending is not None:
            blocks.append({"role": "human", "text": HUMAN_MARKER + " " + pending})
            blocks.append({"role": "model", "text": MODEL_MARKER})
        return blocks

    def state(self, count_tokens: Callable[[str], int]) -> dict[str, object]:
        context = self.context_text()
        return {
            "session_id": self.session_id,
            "context_text": context,
            "context_tokens": count_tokens(context),
            "example_rounds": self.example_rounds,
            "retained_example_rounds": sum(item.example for item in self.rounds),
            "dropped_rounds": self.dropped_rounds,
            "turns": [item.record() for item in self.transcript],
            "last_generation": self.last_generation,
            "context_blocks": self.context_blocks(),
            "settings": settings_record(self.settings),
            "notes": list(self.notes),
            "can_regenerate": self.can_regenerate(),
        }

    def prepare(
        self, message: object, settings: GenerationSettings,
        count_tokens: Callable[[str], int],
    ) -> PreparedInput:
        if not isinstance(message, str):
            raise ValueError("Message must be text")
        message = message.replace("\r\n", "\n").replace("\r", "\n")
        if not message.strip():
            raise ValueError("Enter a message before sending")
        if MESSAGE_DELIMITER_RE.search(message):
            raise ValueError("A blank line followed by a non-whitespace speaker label and > is reserved for message boundaries")
        if "<|endoftext|>" in message:
            raise ValueError("<|endoftext|> is reserved by the GPT-2 tokenizer")
        settings.validate()
        retained = list(self.rounds)
        budget = MODEL_CONTEXT_TOKENS - settings.max_new_tokens
        suffix = HUMAN_MARKER + " " + message + MODEL_MARKER
        header = self.header
        while True:
            if (self.dropped_rounds or len(retained) < len(self.rounds)) and not header.endswith(OMISSION_MARKER):
                header += OMISSION_MARKER
            prompt = header + "".join(item.text() for item in retained) + suffix
            prompt_tokens = count_tokens(prompt)
            if prompt_tokens <= budget:
                break
            if not retained:
                raise ValueError(
                    f"This message needs {prompt_tokens} input tokens including the header; "
                    f"only {budget} fit with a {settings.max_new_tokens}-token reply allowance. "
                    "Shorten the message or reduce the reply length."
                )
            retained.pop(0)
        return PreparedInput(message, prompt, prompt_tokens, retained, header)

    def preview(self, message: object, settings: GenerationSettings, count_tokens: Callable[[str], int]) -> dict[str, object]:
        prepared = self.prepare(message, settings, count_tokens)
        state = self.state(count_tokens)
        state.update(
            context_text=prepared.prompt,
            context_tokens=prepared.input_tokens,
            retained_example_rounds=sum(item.example for item in prepared.retained_rounds),
            dropped_rounds=self.dropped_rounds + len(self.rounds) - len(prepared.retained_rounds),
            context_blocks=self.context_blocks(prepared.retained_rounds, prepared.message, header=prepared.header),
        )
        return state

    def reply(self, message: object, settings: GenerationSettings, runner: Gpt2Runner) -> None:
        prepared = self.prepare(message, settings, runner.count_prompt_tokens)
        effective_settings = resolve_seed(settings, self.settings.seed)
        reply_text, generation = _generate_reply(prepared, settings, effective_settings, runner)
        new_round = Round(human=prepared.message, assistant=reply_text, generation=generation)
        # Commit history and metadata only after successful inference and validation.
        if prepared.header is not None:
            self.header = prepared.header
        self.dropped_rounds += len(self.rounds) - len(prepared.retained_rounds)
        self.rounds = prepared.retained_rounds + [new_round]
        self.transcript.append(new_round)
        self.last_generation = generation
        self.settings = effective_settings

    def can_regenerate(self) -> bool:
        return bool(
            self.rounds and self.transcript
            and self.rounds[-1] == self.transcript[-1]
            and not self.rounds[-1].example
            and self.rounds[-1].generation is not None
            and isinstance(self.rounds[-1].generation.get("prompt_text"), str)
        )

    def _prepare_regeneration(
        self, settings: GenerationSettings, count_tokens: Callable[[str], int],
    ) -> tuple[PreparedInput, GenerationSettings]:
        from .chat_archive import MAX_PREVIOUS_RESPONSES

        settings.validate()
        if not self.can_regenerate():
            raise ValueError("There is no generated reply with a recorded input prompt to regenerate")
        if settings.temperature == 0:
            raise ValueError("Regenerate requires temperature greater than 0; changing a seed does not change greedy decoding")
        last = self.rounds[-1]
        if len(last.previous_responses) >= MAX_PREVIOUS_RESPONSES:
            raise ValueError(f"A reply can retain at most {MAX_PREVIOUS_RESPONSES} previous responses")
        prompt = last.generation["prompt_text"]
        expected = self.header + "".join(item.text() for item in self.rounds[:-1]) + HUMAN_MARKER + " " + last.human + MODEL_MARKER
        if prompt != expected:
            raise ValueError("The recorded last prompt does not match the retained chat; regeneration was cancelled")
        input_tokens = count_tokens(prompt)
        if input_tokens + settings.max_new_tokens > MODEL_CONTEXT_TOKENS:
            raise ValueError(
                f"The original prompt needs {input_tokens} tokens; reduce the reply limit "
                f"to at most {MODEL_CONTEXT_TOKENS - input_tokens} to regenerate it unchanged"
            )
        starting = resolve_seed(settings, self.settings.seed)
        effective = replace(starting, seed=(starting.seed + 1) % (2**63))
        return PreparedInput(last.human, prompt, input_tokens, list(self.rounds[:-1]), self.header), effective

    def regeneration_preview(self, settings: GenerationSettings, count_tokens: Callable[[str], int]) -> dict[str, object]:
        prepared, effective = self._prepare_regeneration(settings, count_tokens)
        state = self.state(count_tokens)
        state.update(
            context_text=prepared.prompt,
            context_tokens=prepared.input_tokens,
            context_blocks=self.context_blocks(prepared.retained_rounds, prepared.message, header=prepared.header),
            settings=settings_record(effective),
        )
        return state

    def regenerate(self, settings: GenerationSettings, runner: Gpt2Runner) -> None:
        prepared, effective = self._prepare_regeneration(settings, runner.count_prompt_tokens)
        reply_text, generation = _generate_reply(prepared, settings, effective, runner)
        last = self.rounds[-1]
        previous = copy.deepcopy(last.previous_responses)
        previous.append({"text": last.assistant, "generation": copy.deepcopy(last.generation)})
        replacement = Round(last.human, reply_text, generation=generation, previous_responses=previous)
        # Replace only after inference succeeds; no extra human turn or eviction.
        self.rounds[-1] = replacement
        self.transcript[-1] = replacement
        self.last_generation = generation
        self.settings = effective


class ChatApplication:
    def __init__(self, runner: Gpt2Runner) -> None:
        self.runner = runner
        self.sessions: dict[str, ChatSession] = {}
        # Torch's RNG and the model are shared; serialize inference and state updates.
        self.lock = threading.Lock()

    def new_session(self, example_rounds: object, date_text: object, previous_session_id: object = None, settings: GenerationSettings | None = None) -> ChatSession:
        if isinstance(example_rounds, bool) or not isinstance(example_rounds, int) or example_rounds not in (0, 1, 2):
            raise ValueError("Choose 0, 1, or 2 example rounds")
        if not isinstance(date_text, str):
            raise ValueError("Date must use YYYY-MM-DD")
        try:
            chat_date = date.fromisoformat(date_text)
        except ValueError as exc:
            raise ValueError("Date must use YYYY-MM-DD") from exc
        if chat_date.isoformat() != date_text:
            raise ValueError("Date must use YYYY-MM-DD")
        if previous_session_id is not None and not isinstance(previous_session_id, str):
            raise ValueError("Previous session identifier must be text")
        settings = settings if settings is not None else GenerationSettings(max_new_tokens=120)
        settings.validate()
        settings = resolve_seed(settings)
        examples = [Round(human, assistant, example=True) for human, assistant in EXAMPLES[:example_rounds]]
        session = ChatSession(secrets.token_urlsafe(24), prompt_header(chat_date), example_rounds, list(examples), list(examples), settings=settings)
        self.sessions[session.session_id] = session
        if previous_session_id is not None:
            self.sessions.pop(previous_session_id, None)
        return session

    def import_session(self, document: object, previous_session_id: object = None) -> ChatSession:
        from .chat_archive import MAX_NOTES, import_chat

        if previous_session_id is not None and not isinstance(previous_session_id, str):
            raise ValueError("Previous session identifier must be text")
        session = import_chat(document, self.runner.count_prompt_tokens)
        if session.settings.seed is None:
            session.settings = resolve_seed(session.settings)
            if len(session.notes) < MAX_NOTES:
                session.notes.append(
                    "This archive had no active seed. A new seed was selected for future replies; "
                    "historical seeds remain as recorded."
                )
        # Full validation precedes replacing any existing session.
        self.sessions[session.session_id] = session
        if previous_session_id is not None:
            self.sessions.pop(previous_session_id, None)
        return session

    def session(self, session_id: object) -> ChatSession:
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise KeyError("Chat session expired or was not found; start a new chat")
        return self.sessions[session_id]


class ChatServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], application: ChatApplication) -> None:
        self.application = application
        super().__init__(address, ChatHandler)


class ChatHandler(BaseHTTPRequestHandler):
    server: ChatServer

    def log_message(self, format: str, *args: object) -> None:
        # Requests contain no chat text in paths; do not log session identifiers.
        pass

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: int, value: object) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _local_request(self) -> bool:
        port = self.server.server_port
        allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        host = self.headers.get("Host", "")
        origin = self.headers.get("Origin")
        if host not in allowed_hosts or (origin is not None and origin != f"http://{host}"):
            self._json(403, {"error": "Use the local chat page to access this server"})
            return False
        return True

    def do_GET(self) -> None:
        if not self._local_request():
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/":
            page = Path(__file__).with_name("web").joinpath("chat.html").read_bytes()
            self._send(200, page, "text/html; charset=utf-8")
        elif parsed.path == "/api/state":
            with self.server.application.lock:
                try:
                    session_id = parse_qs(parsed.query).get("session_id", [None])[0]
                    session = self.server.application.session(session_id)
                    self._json(200, session.state(self.server.application.runner.count_prompt_tokens))
                except KeyError as exc:
                    self._json(404, {"error": exc.args[0]})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if not self._local_request():
            return
        if self.headers.get_content_type() != "application/json":
            self._json(415, {"error": "Send application/json"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            maximum = 4 * 1024 * 1024 if self.path == "/api/import" else 65536
            if not 0 < length <= maximum:
                raise ValueError(f"Request body must be between 1 and {maximum} bytes")
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object")
        except (ValueError, UnicodeError) as exc:
            self._json(400, {"error": str(exc)})
            return
        app = self.server.application
        if not app.lock.acquire(blocking=False):
            self._json(409, {"error": "The model is busy; try again after the current reply"})
            return
        try:
            if self.path == "/api/new":
                session = app.new_session(body.get("example_rounds", 2), body.get("date"), body.get("previous_session_id"), parse_settings(body.get("settings", {})))
            elif self.path == "/api/message":
                session = app.session(body.get("session_id"))
                settings = parse_settings(body.get("settings", {}))
                session.reply(body.get("message"), settings, app.runner)
            elif self.path == "/api/preview":
                session = app.session(body.get("session_id"))
                settings = parse_settings(body.get("settings", {}))
                self._json(200, session.preview(body.get("message"), settings, app.runner.count_prompt_tokens))
                return
            elif self.path == "/api/regenerate-preview":
                session = app.session(body.get("session_id"))
                settings = parse_settings(body.get("settings", {}))
                self._json(200, session.regeneration_preview(settings, app.runner.count_prompt_tokens))
                return
            elif self.path == "/api/regenerate":
                session = app.session(body.get("session_id"))
                settings = parse_settings(body.get("settings", {}))
                session.regenerate(settings, app.runner)
            elif self.path == "/api/export":
                from .chat_archive import export_chat

                session = app.session(body.get("session_id"))
                export_settings = parse_settings(body["settings"]) if "settings" in body else session.settings
                export_settings = resolve_seed(export_settings, session.settings.seed)
                self._json(200, export_chat(session, export_settings))
                session.settings = export_settings
                return
            elif self.path == "/api/import":
                session = app.import_session(body.get("chat"), body.get("previous_session_id"))
            else:
                self._json(404, {"error": "Not found"})
                return
            self._json(200, session.state(app.runner.count_prompt_tokens))
        except KeyError as exc:
            self._json(404, {"error": exc.args[0]})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:
            print(f"Chat inference failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            self._json(500, {"error": "Generation failed; your message and chat history were preserved. Check the server terminal."})
        finally:
            app.lock.release()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--restore-chat", type=Path, help="Restore a local exported chat before serving")
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if arguments.threads is not None and arguments.threads < 1:
        parser.error("threads must be at least 1")
    # Bind before mapping the large checkpoint, so port conflicts fail quickly.
    server = ChatServer(("127.0.0.1", arguments.port), ChatApplication(None))  # type: ignore[arg-type]
    try:
        server.application.runner = Gpt2Runner(offline=arguments.offline, threads=arguments.threads)
        if arguments.restore_chat is not None:
            if arguments.restore_chat.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("Chat file exceeds the 4 MiB limit")
            with arguments.restore_chat.open(encoding="utf-8") as chat_file:
                restored = server.application.import_session(json.load(chat_file))
            print(f"Restored chat: http://localhost:{server.server_port}/?session_id={restored.session_id}", flush=True)
        print(f"GPT-2 chat ready: http://localhost:{server.server_port}/\nPress Ctrl+C to stop.", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nChat stopped.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
