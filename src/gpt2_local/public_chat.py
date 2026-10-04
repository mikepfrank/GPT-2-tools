"""A bounded, anonymous GPT-2 service with browser-owned chat archives."""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import secrets
import socket
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

from .chat import ChatApplication, ChatSession, parse_settings, resolve_seed
from .chat_archive import export_chat
from .runtime import MODEL_CONTEXT_TOKENS, MODEL_ID, MODEL_REVISION, GenerationSettings, Gpt2Runner, SampledToken
from .streaming import ReplyStreamDecoder

MAX_ARCHIVE_BYTES = 4 * 1024 * 1024
MAX_BODY_BYTES = MAX_ARCHIVE_BYTES + 65536
MAX_STREAM_FRAME_BYTES = 1024 * 1024


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class GenerationTimeout(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    max_reply_tokens: int = 256
    max_queue: int = 8
    max_jobs: int = 32
    result_ttl_seconds: float = 600
    queue_ttl_seconds: float = 300
    generation_timeout_seconds: float = 300
    max_active_jobs_per_client: int = 2
    generations_per_minute: int = 6
    requests_per_minute: int = 240
    max_rate_clients: int = 1024

    def __post_init__(self) -> None:
        if not 1 <= self.max_reply_tokens < MODEL_CONTEXT_TOKENS:
            raise ValueError("max_reply_tokens must be between 1 and 1023")
        for field in ("max_queue", "max_jobs", "max_active_jobs_per_client", "generations_per_minute", "requests_per_minute", "max_rate_clients"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        for field in ("result_ttl_seconds", "queue_ttl_seconds", "generation_timeout_seconds"):
            value = getattr(self, field)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be positive and finite")


class _LockedTokenizer:
    """Serialize short tokenizer calls without locking an entire inference."""
    def __init__(self, tokenizer: object, lock: threading.RLock) -> None:
        self._tokenizer = tokenizer
        self._lock = lock

    def __call__(self, *args: object, **kwargs: object) -> object:
        with self._lock:
            return self._tokenizer(*args, **kwargs)  # type: ignore[operator]

    def __getattr__(self, name: str) -> object:
        attribute = getattr(self._tokenizer, name)
        if not callable(attribute):
            return attribute
        def call(*args: object, **kwargs: object) -> object:
            with self._lock:
                return attribute(*args, **kwargs)
        return call


class _DeadlineModel:
    """Add a deadline at decoding boundaries; never alter the local runner."""
    def __init__(self, model: object, deadlines: threading.local, clock: Callable[[], float]) -> None:
        self._model = model
        self._deadlines = deadlines
        self._clock = clock

    def __getattr__(self, name: str) -> object:
        return getattr(self._model, name)

    def generate(self, **kwargs: object) -> object:
        deadline = getattr(self._deadlines, "value", None)
        if deadline is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            clock = self._clock
            class Deadline(StoppingCriteria):
                def __call__(self, input_ids: object, scores: object, **unused: object) -> bool:
                    if clock() >= deadline:
                        raise GenerationTimeout("Generation exceeded the service time limit")
                    return False
            kwargs["stopping_criteria"] = StoppingCriteriaList([
                *list(kwargs.get("stopping_criteria") or []), Deadline(),
            ])
        return self._model.generate(**kwargs)  # type: ignore[attr-defined]


class _PublicRunner:
    def __init__(self, runner: Gpt2Runner, timeout: float, clock: Callable[[], float]) -> None:
        self._runner = runner
        self._timeout = timeout
        self._clock = clock
        self._tokenizer_lock = threading.RLock()
        self._deadlines = threading.local()
        if hasattr(runner, "tokenizer"):
            runner.tokenizer = _LockedTokenizer(runner.tokenizer, self._tokenizer_lock)  # type: ignore[assignment]
        if hasattr(runner, "model"):
            runner.model = _DeadlineModel(runner.model, self._deadlines, clock)  # type: ignore[assignment]

    def count_prompt_tokens(self, prompt: str) -> int:
        with self._tokenizer_lock:
            return self._runner.count_prompt_tokens(prompt)

    def generate_result(self, *args: object, **kwargs: object) -> object:
        deadline = self._clock() + self._timeout
        self._deadlines.value = deadline
        try:
            result = self._runner.generate_result(*args, **kwargs)
            if self._clock() >= deadline:
                raise GenerationTimeout("Generation exceeded the service time limit")
            return result
        finally:
            del self._deadlines.value


@dataclass
class _Job:
    identifier: str
    client: str
    created: float
    session: ChatSession | None
    settings: GenerationSettings
    message: object
    regenerate: bool
    status: str = "queued"
    finished: float | None = None
    state: dict[str, object] | None = None
    error: str | None = None
    seq: int = 0
    progress_status: str = ""
    progress_position: int = -1
    reply_text: str = ""
    generated_tokens: int = 0
    started: float | None = None
    first_token_seconds: float | None = None
    first_visible_seconds: float | None = None
    last_token_seconds: float | None = None


class PublicChatApplication:
    """Only queued/running jobs and short-lived results live on the server."""
    def __init__(self, runner: Gpt2Runner, limits: Limits | None = None, *, clock: Callable[[], float] = time.monotonic, enable_streaming: bool = False) -> None:
        self.limits = limits or Limits()
        self.clock = clock
        self.enable_streaming = enable_streaming
        self.runner = _PublicRunner(runner, self.limits.generation_timeout_seconds, clock)
        self._condition = threading.Condition()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, _Job] = {}
        self._rates: dict[str, tuple[deque[float], deque[float]]] = {}
        self._running: str | None = None
        self._closed = False
        self._worker = threading.Thread(target=self._work, name="gpt2-public-inference", daemon=True)
        self._worker.start()

    def settings(self, raw: object) -> GenerationSettings:
        settings = parse_settings(raw)
        if settings.max_new_tokens > self.limits.max_reply_tokens:
            raise ValueError(f"This service allows at most {self.limits.max_reply_tokens} reply tokens")
        return settings

    def _session(self, document: object) -> ChatSession:
        # The baseline application exists only for this operation; it retains no
        # browser sessions after returning. Its importer preserves legacy seeds.
        application = ChatApplication(self.runner)  # type: ignore[arg-type]
        return application.import_session(document)

    @staticmethod
    def _check_archive_size(document: object) -> None:
        if len(json.dumps(document, ensure_ascii=False, allow_nan=False).encode("utf-8")) > MAX_ARCHIVE_BYTES:
            raise ValueError("Chat archive exceeds the 4 MiB service limit; export it and start a new chat")

    def _state(self, session: ChatSession) -> dict[str, object]:
        archive = export_chat(session)
        self._check_archive_size(archive)
        return {**session.state(self.runner.count_prompt_tokens), "chat": archive}

    def new(self, body: dict[str, object]) -> dict[str, object]:
        application = ChatApplication(self.runner)  # type: ignore[arg-type]
        session = application.new_session(body.get("example_rounds", 2), body.get("date"), settings=self.settings(body.get("settings", {})))
        return self._state(session)

    def imported(self, body: dict[str, object]) -> dict[str, object]:
        self._check_archive_size(body.get("chat"))
        return self._state(self._session(body.get("chat")))

    def preview(self, body: dict[str, object], *, regenerate: bool = False) -> dict[str, object]:
        session = self._session(body.get("chat"))
        settings = self.settings(body.get("settings", {}))
        if regenerate:
            return session.regeneration_preview(settings, self.runner.count_prompt_tokens)
        return session.preview(body.get("message"), settings, self.runner.count_prompt_tokens)

    def exported(self, body: dict[str, object]) -> dict[str, object]:
        session = self._session(body.get("chat"))
        settings = self.settings(body["settings"]) if "settings" in body else session.settings
        document = export_chat(session, resolve_seed(settings, session.settings.seed))
        self._check_archive_size(document)
        return document

    def _cleanup(self, now: float) -> None:
        for identifier, job in list(self._jobs.items()):
            if job.finished is not None and now - job.finished >= self.limits.result_ttl_seconds:
                del self._jobs[identifier]
            elif job.status == "queued" and now - job.created >= self.limits.queue_ttl_seconds:
                job.status, job.finished, job.error, job.session = "failed", now, "The queued reply expired; send it again when the service is less busy", None
                job.message = None
                self._condition.notify_all()
        self._pending = deque(identifier for identifier in self._pending if identifier in self._jobs and self._jobs[identifier].status == "queued")
        for client, (requests, generations) in list(self._rates.items()):
            for values in (requests, generations):
                while values and now - values[0] >= 60:
                    values.popleft()
            if not requests and not generations:
                del self._rates[client]

    def rate_request(self, client: str, *, generation: bool = False) -> None:
        with self._condition:
            now = self.clock()
            self._cleanup(now)
            if client not in self._rates:
                if len(self._rates) >= self.limits.max_rate_clients:
                    raise ServiceError(429, "The service is busy; try again in a minute")
                self._rates[client] = (deque(), deque())
            requests, generations = self._rates[client]
            values, maximum = (generations, self.limits.generations_per_minute) if generation else (requests, self.limits.requests_per_minute)
            if len(values) >= maximum:
                raise ServiceError(429, "Too many requests; try again in a minute")
            values.append(now)

    def submit(self, body: dict[str, object], client: str, *, regenerate: bool = False) -> dict[str, object]:
        self.rate_request(client, generation=True)
        self._check_archive_size(body.get("chat"))
        session = self._session(body.get("chat"))
        settings = self.settings(body.get("settings", {}))
        # Reject an impossible generation before it occupies the queue.
        if regenerate:
            session.regeneration_preview(settings, self.runner.count_prompt_tokens)
        else:
            session.preview(body.get("message"), settings, self.runner.count_prompt_tokens)
        with self._condition:
            self._cleanup(self.clock())
            if self._closed:
                raise ServiceError(503, "The service is stopping; try again shortly")
            active = sum(job.client == client and job.status in ("queued", "running") for job in self._jobs.values())
            if active >= self.limits.max_active_jobs_per_client:
                raise ServiceError(429, "You already have replies waiting; wait for them to finish")
            if len(self._pending) >= self.limits.max_queue or len(self._jobs) >= self.limits.max_jobs:
                raise ServiceError(503, "The reply queue is full; try again shortly")
            identifier = secrets.token_urlsafe(24)
            job = _Job(identifier, client, self.clock(), session, settings, body.get("message"), regenerate)
            self._jobs[identifier] = job
            self._pending.append(identifier)
            result = self._job_state(job)
            self._condition.notify_all()
            return result

    def _progress(self, job: _Job) -> dict[str, object]:
        """Call under the condition; keep only one cumulative snapshot per job."""
        position = list(self._pending).index(job.identifier) + 1 if job.identifier in self._pending else 0
        if (job.progress_status, job.progress_position) != (job.status, position):
            job.seq += 1
            job.progress_status, job.progress_position = job.status, position
        return {"version": 1, "seq": job.seq, "status": job.status, "position": position,
                "generated_tokens": job.generated_tokens, "reply_text": job.reply_text}

    def wait_progress(self, identifier: str, after: int | None = None, timeout: float = 0) -> dict[str, object]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                self._cleanup(self.clock())
                if identifier not in self._jobs:
                    raise ServiceError(404, "Reply job expired or was not found; your saved chat is unchanged")
                progress = self._progress(self._jobs[identifier])
                remaining = deadline - time.monotonic()
                if after is None or progress["seq"] != after or progress["status"] in ("complete", "failed") or self._closed or remaining <= 0:
                    return progress
                self._condition.wait(timeout=remaining)

    def _observer(self, job: _Job) -> Callable[[SampledToken], None]:
        with self.runner._tokenizer_lock:
            decoder = ReplyStreamDecoder(self.runner._runner.tokenizer.backend_tokenizer)
        def observe(event: SampledToken) -> None:
            callback_started = time.perf_counter()
            with self.runner._tokenizer_lock:
                text = decoder.push(event.token_id)
            with self._condition:
                if not job.generated_tokens < event.output_tokens <= job.settings.max_new_tokens:
                    raise ValueError("Invalid streamed token accounting")
                job.reply_text, job.generated_tokens = text, event.output_tokens
                if job.first_token_seconds is None:
                    job.first_token_seconds = event.elapsed_seconds
                if text and job.first_visible_seconds is None:
                    job.first_visible_seconds = event.elapsed_seconds + time.perf_counter() - callback_started
                job.last_token_seconds = event.elapsed_seconds
                job.seq += 1
                self._condition.notify_all()
        return observe

    def _job_state(self, job: _Job) -> dict[str, object]:
        position = list(self._pending).index(job.identifier) + 1 if job.identifier in self._pending else 0
        result: dict[str, object] = {"job_id": job.identifier, "status": job.status, "position": position}
        if self.enable_streaming:
            progress = self._progress(job)
            if job.status in ("queued", "running"):
                result["progress"] = progress
            duration = (job.last_token_seconds - job.first_token_seconds
                        if job.first_token_seconds is not None and job.last_token_seconds is not None else 0)
            result["delivery"] = {"queue_seconds": job.started - job.created if job.started is not None else None,
                                  "first_token_seconds": job.first_token_seconds,
                                  "first_visible_seconds": job.first_visible_seconds,
                                  "decoding_tokens_per_second": (job.generated_tokens - 1) / duration if job.generated_tokens >= 2 and duration > 0 else None}
        if job.state is not None:
            result["state"] = job.state
        if job.error is not None:
            result["error"] = job.error
        return result

    def job(self, identifier: str) -> dict[str, object]:
        # An unguessable job ID is a temporary bearer capability. Never put it
        # in access logs, analytics, external links, or referrer headers.
        with self._condition:
            self._cleanup(self.clock())
            if identifier not in self._jobs:
                raise ServiceError(404, "Reply job expired or was not found; your saved chat is unchanged")
            return self._job_state(self._jobs[identifier])

    def health(self) -> dict[str, object]:
        with self._condition:
            self._cleanup(self.clock())
            return {
                "status": "stopping" if self._closed else "ready",
                "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "context_tokens": MODEL_CONTEXT_TOKENS, "precision": "FP32", "device": "cpu"},
                "limits": {"max_reply_tokens": self.limits.max_reply_tokens, "archive_bytes": MAX_ARCHIVE_BYTES, "request_bytes": MAX_BODY_BYTES,
                           "max_queue": self.limits.max_queue, "result_ttl_seconds": self.limits.result_ttl_seconds,
                           "generation_timeout_seconds": self.limits.generation_timeout_seconds},
                "queue": {"waiting": len(self._pending), "running": self._running is not None},
                "capabilities": {"streaming": self.enable_streaming},
            }

    def _work(self) -> None:
        while True:
            with self._condition:
                self._cleanup(self.clock())
                while not self._pending and not self._closed:
                    self._condition.wait(timeout=min(30, self.limits.queue_ttl_seconds))
                    self._cleanup(self.clock())
                if self._closed:
                    return
                identifier = self._pending.popleft()
                job = self._jobs[identifier]
                job.status = "running"
                job.started = self.clock()
                self._running = identifier
                self._progress(job)
                self._condition.notify_all()
            try:
                assert job.session is not None
                observed = {"on_token": self._observer(job)} if self.enable_streaming else {}
                if job.regenerate:
                    job.session.regenerate(job.settings, self.runner, **observed)  # type: ignore[arg-type]
                else:
                    job.session.reply(job.message, job.settings, self.runner, **observed)  # type: ignore[arg-type]
                state = self._state(job.session)
                error = None
            except GenerationTimeout:
                state, error = None, "Generation exceeded the service time limit; your saved chat is unchanged"
            except Exception:
                # Do not print exceptions: model/tokenizer errors may echo a
                # request's text. Operators receive only aggregate health data.
                state, error = None, "Generation failed; your saved chat is unchanged. Try again later"
            with self._condition:
                job.state, job.error = state, error
                job.status = "complete" if error is None else "failed"
                job.finished, job.session = self.clock(), None
                job.message = None
                self._running = None
                self._progress(job)
                self._condition.notify_all()

    def close(self, timeout: float = 5) -> None:
        with self._condition:
            self._closed = True
            for identifier in self._pending:
                job = self._jobs[identifier]
                job.status, job.finished, job.error, job.session, job.message = "failed", self.clock(), "The service stopped; your saved chat is unchanged", None, None
            self._pending.clear()
            self._condition.notify_all()
        self._worker.join(timeout=timeout)


def normalize_origin(value: str) -> str:
    parsed = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Public origins must use a valid port") from exc
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username is not None or parsed.password is not None or parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("Public origins must be HTTP(S) origins, without credentials, paths, queries, or fragments")
    hostname = parsed.hostname.lower()
    try:
        hostname.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Use an ASCII hostname in the public origin") from exc
    if any(character.isspace() for character in hostname) or "\\" in hostname:
        raise ValueError("Invalid public origin hostname")
    authority = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and port != (443 if parsed.scheme == "https" else 80):
        authority += f":{port}"
    return f"{parsed.scheme}://{authority}"


class PublicChatServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 16

    def __init__(self, address: tuple[str, int], application: PublicChatApplication | None, *, public_origins: tuple[str, ...] = (), trusted_proxies: tuple[str, ...] = (), max_handlers: int = 16, request_timeout: float = 15, max_streams: int = 4, max_streams_per_client: int = 2, stream_heartbeat: float = 5, stream_lease: float = 60) -> None:
        if not ipaddress.ip_address(address[0]).is_loopback:
            raise ValueError("The Python service must bind a loopback address behind a reverse proxy")
        self.application = application
        self.public_origins = {normalize_origin(origin) for origin in public_origins}
        self.trusted_proxies = {str(ipaddress.ip_address(peer)) for peer in trusted_proxies}
        if any(not ipaddress.ip_address(peer).is_loopback for peer in self.trusted_proxies):
            raise ValueError("Trusted reverse proxies must use loopback addresses")
        self.request_timeout = request_timeout
        self._handler_slots = threading.BoundedSemaphore(max_handlers)
        self._stream_lock = threading.Lock()
        self._stream_clients: dict[str, int] = {}
        self._max_streams = min(max_streams, max(0, max_handlers - 1))
        self._max_streams_per_client = max_streams_per_client
        self.stream_heartbeat, self.stream_lease = stream_heartbeat, stream_lease
        if max_streams < 1 or max_streams_per_client < 1 or not all(math.isfinite(value) and value > 0 for value in (stream_heartbeat, stream_lease)):
            raise ValueError("Streaming limits must be positive and finite")
        super().__init__(address, PublicChatHandler)
        self.local_origins = {f"http://127.0.0.1:{self.server_port}", f"http://localhost:{self.server_port}"}

    def acquire_stream(self, client: str) -> None:
        with self._stream_lock:
            if sum(self._stream_clients.values()) >= self._max_streams or self._stream_clients.get(client, 0) >= self._max_streams_per_client:
                raise ServiceError(503, "Streaming connections are busy; check the same reply with polling")
            self._stream_clients[client] = self._stream_clients.get(client, 0) + 1

    def release_stream(self, client: str) -> None:
        with self._stream_lock:
            remaining = self._stream_clients[client] - 1
            if remaining:
                self._stream_clients[client] = remaining
            else:
                del self._stream_clients[client]

    def process_request(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        if not self._handler_slots.acquire(blocking=False):
            # Do not read a saturated connection or create another worker.
            self.shutdown_request(request)
            return
        try:
            request.settimeout(self.request_timeout)
            super().process_request(request, client_address)
        except BaseException:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()

    def handle_error(self, request: socket.socket, client_address: tuple[str, int]) -> None:
        # Suppress BaseServer's traceback, which can include chat/error text.
        pass

    def server_close(self) -> None:
        super().server_close()
        if self.application is not None:
            self.application.close()


class PublicChatHandler(BaseHTTPRequestHandler):
    server: PublicChatServer
    server_version = "GPT2"
    sys_version = ""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
        self.send_header("Connection", "close")
        if status in (429, 503):
            self.send_header("Retry-After", "60" if status == 429 else "5")
        self.end_headers()
        self.wfile.write(content)
        self.close_connection = True

    def _json(self, status: int, value: object) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8")

    def _client(self, *, post: bool = False) -> str:
        peer = str(ipaddress.ip_address(self.client_address[0]))
        if not ipaddress.ip_address(peer).is_loopback:
            raise ServiceError(403, "Access this service through its configured reverse proxy")
        origins = self.server.public_origins or self.server.local_origins
        hosts = {urlsplit(origin).netloc for origin in origins}
        if len(self.headers.get_all("Host", [])) != 1 or self.headers.get("Host", "").lower() not in hosts:
            raise ServiceError(403, "This host is not configured for the service")
        origin = self.headers.get("Origin")
        if post and origin is None:
            raise ServiceError(403, "Use the configured chat page to submit requests")
        if origin is not None:
            try:
                normalized = normalize_origin(origin)
            except ValueError:
                normalized = ""
            if len(self.headers.get_all("Origin", [])) != 1 or normalized not in origins:
                raise ServiceError(403, "This origin is not configured for the service")
        # X-Forwarded-For and Forwarded are never trusted. Only a configured
        # local proxy may supply one validated X-Real-IP for anonymous limits.
        if peer in self.server.trusted_proxies and "X-Real-IP" in self.headers:
            try:
                if len(self.headers.get_all("X-Real-IP", [])) != 1:
                    raise ValueError("duplicate client header")
                peer = str(ipaddress.ip_address(self.headers["X-Real-IP"]))
            except ValueError as exc:
                raise ServiceError(400, "Invalid proxy client address") from exc
        if self.server.application is None:
            raise ServiceError(503, "The model is starting; try again shortly")
        self.server.application.rate_request(peer)
        return peer

    def _events(self, identifier: str, client: str) -> None:
        application = self.server.application
        assert application is not None
        if not application.enable_streaming:
            raise ServiceError(404, "Streaming is not enabled")
        cursors = self.headers.get_all("Last-Event-ID", [])
        if len(cursors) > 1 or (cursors and (not cursors[0].isascii() or not cursors[0].isdigit() or len(cursors[0]) > 16 or int(cursors[0]) > 2**53 - 1)):
            raise ServiceError(400, "Invalid streaming cursor")
        progress = application.wait_progress(identifier)
        self.server.acquire_stream(client)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            deadline = time.monotonic() + self.server.stream_lease
            after = None
            while True:
                status = progress["status"]
                if status in ("complete", "failed"):
                    payload = {"version": 1, "seq": progress["seq"]}
                    event = status
                elif progress["seq"] != after:
                    payload, event = progress, "progress"
                else:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    payload, event = None, None
                if payload is not None:
                    encoded = (f'id: {progress["seq"]}\nevent: {event}\ndata: ' + json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n\n').encode("utf-8")
                    if len(encoded) > MAX_STREAM_FRAME_BYTES:
                        return
                    self.wfile.write(encoded)
                    self.wfile.flush()
                if status in ("complete", "failed") or application._closed:
                    return
                after = progress["seq"]
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                progress = application.wait_progress(identifier, after, min(self.server.stream_heartbeat, remaining))
        except (ServiceError, BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            # A started stream cannot be replaced by a JSON HTTP response.
            pass
        except Exception:
            pass
        finally:
            self.close_connection = True
            self.server.release_stream(client)

    def do_GET(self) -> None:
        try:
            client = self._client()
            assert self.server.application is not None
            parsed = urlsplit(self.path)
            if parsed.query or parsed.fragment:
                raise ServiceError(404, "Not found")
            if parsed.path == "/":
                page = Path(__file__).with_name("web").joinpath("public.html").read_bytes()
                self._send(200, page, "text/html; charset=utf-8")
            elif parsed.path == "/api/health":
                self._json(200, self.server.application.health())
            elif parsed.path.startswith("/api/jobs/"):
                identifier = parsed.path.removeprefix("/api/jobs/")
                streaming = identifier.endswith("/events")
                if streaming:
                    identifier = identifier.removesuffix("/events")
                if not identifier or len(identifier) > 64 or not all(character.isascii() and (character.isalnum() or character in "_-") for character in identifier):
                    raise ServiceError(404, "Not found")
                if streaming:
                    self._events(identifier, client)
                else:
                    self._json(200, self.server.application.job(identifier))
            else:
                raise ServiceError(404, "Not found")
        except ServiceError as exc:
            self._json(exc.status, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception:
            self._json(500, {"error": "The service could not process this request"})

    def do_POST(self) -> None:
        try:
            client = self._client(post=True)
            assert self.server.application is not None
            allowed = {"/api/new", "/api/import", "/api/preview", "/api/regenerate-preview", "/api/export", "/api/message", "/api/regenerate"}
            if self.path not in allowed:
                raise ServiceError(404, "Not found")
            if self.headers.get_content_type() != "application/json":
                raise ServiceError(415, "Send application/json")
            if "Transfer-Encoding" in self.headers or len(self.headers.get_all("Content-Length", [])) != 1:
                raise ServiceError(400, "Send one Content-Length header without Transfer-Encoding")
            try:
                length = int(self.headers["Content-Length"])
            except (ValueError, TypeError) as exc:
                raise ServiceError(400, "Invalid request length") from exc
            maximum = 65536 if self.path == "/api/new" else MAX_BODY_BYTES
            if not 0 < length <= maximum:
                raise ServiceError(413, f"Request must be between 1 and {maximum} bytes")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ServiceError(400, "Incomplete request body")
            body = json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON number")))
            if not isinstance(body, dict):
                raise ValueError("Request must be a JSON object")
            app = self.server.application
            if self.path == "/api/new":
                result = app.new(body)
            elif self.path == "/api/import":
                result = app.imported(body)
            elif self.path in ("/api/preview", "/api/regenerate-preview"):
                result = app.preview(body, regenerate=self.path == "/api/regenerate-preview")
            elif self.path == "/api/export":
                result = app.exported(body)
            else:
                result = app.submit(body, client, regenerate=self.path == "/api/regenerate")
                self._json(202, result)
                return
            self._json(200, result)
        except ServiceError as exc:
            # These failures occur before a job is enqueued. An explicit
            # acknowledgement lets the browser safely offer another attempt.
            # Network failures and unexpected server failures remain uncertain.
            self._json(exc.status, {"error": str(exc), "accepted": False})
        except (UnicodeError, RecursionError):
            self._json(400, {"error": "Invalid chat archive, message, or settings. Check the reply limit and import format"})
        except ValueError as exc:
            # The caller needs packing, reply-cap, and regeneration explanations.
            # These bounded validation messages are returned only to that caller;
            # neither errors nor request bodies are logged.
            self._json(400, {"error": str(exc)[:512] or "Invalid chat archive, message, or settings"})
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except Exception:
            self._json(500, {"error": "The service could not process this request; your saved chat is unchanged"})


def available_memory_gib(path: Path = Path("/proc/meminfo"), *, cgroup_root: Path = Path("/sys/fs/cgroup"), cgroup_membership: Path = Path("/proc/self/cgroup")) -> float:
    available_bytes = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) != 3 or fields[2] != "kB" or int(fields[1]) < 0:
                break
            available_bytes = int(fields[1]) * 1024
            break
    if available_bytes is None:
        raise ValueError("MemAvailable is unavailable; memory capacity has not been verified")
    # A large host does not imply room inside a service's MemoryMax cgroup.
    # Check the current group and its ancestors, including namespace roots.
    candidates: set[tuple[Path, str, str]] = {(cgroup_root, "memory.max", "memory.current")}
    if cgroup_membership.exists():
        for line in cgroup_membership.read_text(encoding="ascii").splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3 or not parts[2].startswith("/") or ".." in parts[2].split("/"):
                continue
            if parts[0] == "0" and not parts[1]:
                root, maximum, usage = cgroup_root, "memory.max", "memory.current"
            elif "memory" in parts[1].split(","):
                root, maximum, usage = cgroup_root / "memory", "memory.limit_in_bytes", "memory.usage_in_bytes"
            else:
                continue
            group = root.joinpath(*parts[2].strip("/").split("/"))
            while True:
                candidates.add((group, maximum, usage))
                if group == root:
                    break
                group = group.parent
    for group, maximum_name, usage_name in candidates:
        maximum_path = group / maximum_name
        if not maximum_path.exists():
            continue
        maximum_text = maximum_path.read_text(encoding="ascii").strip()
        if maximum_text == "max":
            continue
        maximum = int(maximum_text)
        if maximum >= 2**60:  # cgroup v1's effectively-unlimited sentinel
            continue
        current = int((group / usage_name).read_text(encoding="ascii").strip())
        if maximum < 0 or current < 0:
            raise ValueError("Invalid cgroup memory capacity")
        available_bytes = min(available_bytes, max(0, maximum - current))
    return available_bytes / (1024**3)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Loopback listen address; expose only through a local reverse proxy")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--enable-streaming", action="store_true", help="Enable provisional reply progress and SSE subscriptions")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--public-origin", action="append", default=[], help="Allowed external HTTP(S) origin; repeat for aliases")
    parser.add_argument("--trusted-proxy", action="append", default=[], help="Loopback proxy allowed to set X-Real-IP; repeat as needed")
    parser.add_argument("--max-reply-tokens", type=int, default=256)
    parser.add_argument("--max-generation-seconds", type=float, default=300, help="Cooperative decoding deadline; a single model forward cannot be interrupted")
    parser.add_argument("--minimum-available-memory-gib", type=float, default=8)
    arguments = parser.parse_args(argv)
    if not 1 <= arguments.port <= 65535 or (arguments.threads is not None and arguments.threads < 1):
        parser.error("port must be between 1 and 65535; threads must be positive")
    if not math.isfinite(arguments.minimum_available_memory_gib) or arguments.minimum_available_memory_gib < 0:
        parser.error("minimum available memory must be finite and nonnegative")
    try:
        limits = Limits(max_reply_tokens=arguments.max_reply_tokens, generation_timeout_seconds=arguments.max_generation_seconds)
        # Guard before binding, downloading, or mapping any model weights.
        if arguments.minimum_available_memory_gib:
            available = available_memory_gib()
            if available < arguments.minimum_available_memory_gib:
                parser.error(f"Only {available:.2f} GiB RAM is available; this service requires {arguments.minimum_available_memory_gib:.2f} GiB before model loading. Review capacity first")
        server = PublicChatServer((arguments.host, arguments.port), None, public_origins=tuple(arguments.public_origin), trusted_proxies=tuple(arguments.trusted_proxy))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    try:
        runner = Gpt2Runner(offline=arguments.offline, threads=arguments.threads)
        server.application = PublicChatApplication(runner, limits, enable_streaming=arguments.enable_streaming)
        print(f"GPT-2 public service ready on {arguments.host}:{server.server_port}; chat state stays in browsers", flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        print("GPT-2 public service stopped", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
