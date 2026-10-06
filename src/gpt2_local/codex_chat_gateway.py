"""Scoped Responses-only proxy: the real API key never enters a Codex sandbox.

The caller must give the container network access only to this proxy. A scoped
token is not sufficient isolation if arbitrary outbound networking is possible.
Every parent and subagent request uses the same durable daily admission ledger.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import secrets
import socket
import threading
import time

from .codex_chat_budget import DailyBudgetExceeded, MAX_OUTPUT_TOKENS, MAX_REQUEST_BYTES

MODEL = "gpt-6.1-sol"
ALLOWED_FIELDS = frozenset({
    "model", "input", "instructions", "tools", "tool_choice", "parallel_tool_calls",
    "reasoning", "text", "stream", "store", "include", "metadata", "prompt_cache_key",
    "prompt_cache_retention", "max_output_tokens", "max_tool_calls", "service_tier",
    "temperature", "top_p", "truncation", "safety_identifier", "stream_options", "client_metadata"})
FORBIDDEN_INPUT_TYPES = frozenset({"input_image", "input_audio", "input_file", "image",
    "audio", "file", "item_reference", "computer_call", "computer_call_output"})


class GatewayPolicyError(ValueError):
    pass


def _validate_local_tool(tool, depth=0):
    if depth > 4 or not isinstance(tool, dict):
        raise GatewayPolicyError("Invalid local tool definition.")
    kind = tool.get("type")
    if not isinstance(kind, str):
        raise GatewayPolicyError("Invalid local tool type.")
    if kind in {"function", "custom"}:
        if not isinstance(tool.get("name"), str) or not tool["name"] or len(tool["name"]) > 160:
            raise GatewayPolicyError("Invalid local tool name.")
        return
    if kind == "namespace":
        children = tool.get("tools")
        if not isinstance(tool.get("name"), str) or not isinstance(children, list) or len(children) > 96:
            raise GatewayPolicyError("Invalid local tool namespace.")
        for child in children:
            _validate_local_tool(child, depth+1)
        return
    raise GatewayPolicyError("Hosted tools are disabled; use the application's local tools.")


def _validate_input(value, depth=0):
    if depth > 40:
        raise GatewayPolicyError("Request nesting exceeds the allowed depth.")
    if isinstance(value, dict):
        if isinstance(value.get("type"), str) and value["type"] in FORBIDDEN_INPUT_TYPES:
            raise GatewayPolicyError("Only inline text and local tool results are permitted.")
        for child in value.values():
            _validate_input(child, depth+1)
    elif isinstance(value, list):
        for child in value:
            _validate_input(child, depth+1)


def validate_request(raw):
    """Validate and normalize before reservation; reject hidden server context."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_REQUEST_BYTES:
        raise GatewayPolicyError("Request body exceeds the allowed size.")
    try:
        body = json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise GatewayPolicyError("Request must be UTF-8 JSON.") from exc
    if not isinstance(body, dict):
        raise GatewayPolicyError("Unsupported Responses request fields.")
    if set(body)-ALLOWED_FIELDS:
        # Field names are protocol metadata; never include values or prompts.
        unknown = sorted(set(body)-ALLOWED_FIELDS)
        raise GatewayPolicyError("Unsupported Responses request fields: "+", ".join(unknown[:12])[:250]+".")
    # Codex's client-side telemetry is not part of the forwarded Responses
    # contract. Discard it wholesale; no metadata value can supply provider
    # context, credentials, routing, or a different pricing policy.
    body.pop("client_metadata", None)
    if body.get("model") != MODEL:
        raise GatewayPolicyError("Only gpt-6.1-sol is enabled.")
    if body.get("service_tier", "default") != "default":
        raise GatewayPolicyError("Only standard API pricing is enabled.")
    output = body.get("max_output_tokens", MAX_OUTPUT_TOKENS)
    if type(output) is not int or not 1 <= output <= MAX_OUTPUT_TOKENS:
        raise GatewayPolicyError("Output allowance exceeds the permitted limit.")
    tools = body.get("tools", [])
    if not isinstance(tools, list) or len(tools) > 96:
        raise GatewayPolicyError("Invalid local tool definitions.")
    for tool in tools:
        _validate_local_tool(tool)
    _validate_input(body.get("input"))
    effort = body.get("reasoning", {}).get("effort", "medium") if isinstance(body.get("reasoning", {}), dict) else None
    if not isinstance(effort, str) or effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise GatewayPolicyError("Unsupported reasoning effort.")
    for key in ("stream", "store", "parallel_tool_calls"):
        if key in body and type(body[key]) is not bool:
            raise GatewayPolicyError("Invalid boolean request field.")
    if body.get("store", False):
        raise GatewayPolicyError("Provider-side conversation storage is disabled.")
    calls = body.get("max_tool_calls", 4)
    if type(calls) is not int or not 1 <= calls <= 4:
        raise GatewayPolicyError("Tool call allowance exceeds the permitted limit.")
    body.update(max_output_tokens=output, max_tool_calls=calls, store=False, service_tier="default")
    try:
        encoded = json.dumps(body, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, UnicodeError) as exc:
        raise GatewayPolicyError("Request contains invalid numeric or Unicode data.") from exc
    if len(encoded) > MAX_REQUEST_BYTES:
        raise GatewayPolicyError("Normalized request body exceeds the allowed size.")
    return body, encoded


@dataclass
class _Scope:
    session_id: str
    expires: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    closers: set = field(default_factory=set)
    timer: object = None
    failure_code: str | None = None


class _UsageReader:
    """Keep authoritative completed-response usage, never interim SSE usage.

    The complete usage mapping is preserved for the ledger's own validation of
    totals, cached reads and cache writes. No request or transcript is persisted.
    """
    def __init__(self, streaming):
        self.streaming = streaming
        self.buffer = bytearray()
        self.usage = None

    def feed(self, data):
        self.buffer.extend(data)
        if len(self.buffer) > 2_000_000:
            self.buffer.clear()
            return
        if self.streaming:
            while b"\n" in self.buffer:
                line, _, rest = self.buffer.partition(b"\n")
                self.buffer = bytearray(rest)
                if line.startswith(b"data: "):
                    self._parse(line[6:])

    def _parse(self, data):
        try:
            value = json.loads(data)
            if isinstance(value, dict):
                if self.streaming:
                    if value.get("type") != "response.completed":
                        return
                    response = value.get("response")
                    if not isinstance(response, dict) or response.get("status", "completed") != "completed":
                        return
                else:
                    response = value
                    if response.get("status") != "completed":
                        return
                usage = response.get("usage")
                if isinstance(usage, dict):
                    self.usage = usage
        except (ValueError, UnicodeError, AttributeError):
            pass

    def finish(self):
        if not self.streaming:
            self._parse(self.buffer)
        return self.usage


class CodexAPIGateway:
    def __init__(self, api_key, ledger, host="127.0.0.1", port=0, *, connection_factory=None):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("The gateway requires the operator's API key.")
        self._api_key = api_key
        self.ledger = ledger
        self.host, self.port = host, port
        self._scopes = {}
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(4)
        self._connection_factory = connection_factory or (
            lambda: http.client.HTTPSConnection("api.openai.com", timeout=30))
        self._server = self._thread = None

    @property
    def base_url(self):
        return f"http://{self.host}:{self.port}/v1"

    def activate(self, session_id, timeout_seconds=180):
        if not 0 < timeout_seconds <= 600:
            raise ValueError("Invalid gateway action deadline.")
        token = secrets.token_urlsafe(32)
        scope = _Scope(str(session_id), time.monotonic()+timeout_seconds)
        scope.timer = threading.Timer(timeout_seconds, self.revoke, args=(token,))
        scope.timer.daemon = True
        with self._lock:
            self._scopes[token] = scope
        scope.timer.start()
        return token

    def revoke(self, token):
        with self._lock:
            scope = self._scopes.pop(token, None)
            if not scope:
                return
            scope.cancelled.set()
            scope.timer.cancel()
            closers = tuple(scope.closers)
        for closer in closers:
            try:
                closer()
            except OSError:
                pass

    def action_failure(self, token):
        """Trusted action failure metadata, available until the scope is revoked."""
        with self._lock:
            scope = self._scopes.get(token)
            if scope and scope.failure_code:
                return {"code": scope.failure_code}
        return None

    def _scope(self, authorization):
        if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
            return None
        token = authorization[7:]
        with self._lock:
            scope = self._scopes.get(token)
            if scope and not scope.cancelled.is_set() and time.monotonic() < scope.expires:
                return scope
        return None

    def start(self):
        if self._server:
            return self
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args):
                pass  # Authorization, prompts and raw upstream errors stay private.

            def _json(self, status, message):
                # Rejected authentication/routes can leave a small body unread.
                # Drain it before closing so Windows does not reset the socket
                # before the client receives the complete JSON error response.
                if not getattr(self, "_body_read", False):
                    try:
                        size = int(self.headers.get("Content-Length", "0"))
                        if 0 < size <= MAX_REQUEST_BYTES and not self.headers.get("Transfer-Encoding"):
                            self.connection.settimeout(1)
                            self.rfile.read(size)
                    except (ValueError, OSError):
                        pass
                    self._body_read = True
                payload = json.dumps(message).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(payload)
                self.wfile.flush()

            def do_GET(self):
                if not gateway._scope(self.headers.get("Authorization")):
                    return self._json(401, {"error": {"message": "Expired or invalid action token."}})
                if self.path != "/v1/models":
                    return self._json(404, {"error": {"message": "Unsupported gateway route."}})
                self._json(200, {"object": "list", "data": [
                    {"id": MODEL, "object": "model", "created": 0, "owned_by": "openai"}]})

            def do_POST(self):
                scope = gateway._scope(self.headers.get("Authorization"))
                if not scope:
                    return self._json(401, {"error": {"message": "Expired or invalid action token."}})
                if self.path != "/v1/responses":
                    return self._json(404, {"error": {"message": "Unsupported gateway route."}})
                if self.headers.get("Transfer-Encoding"):
                    return self._json(400, {"error": {"message": "Chunked uploads are disabled."}})
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= MAX_REQUEST_BYTES:
                        raise GatewayPolicyError("Request body exceeds the allowed size.")
                    self.connection.settimeout(10)
                    raw = self.rfile.read(size)
                    self._body_read = True
                    if len(raw) != size:
                        raise GatewayPolicyError("Incomplete request body.")
                    body, encoded = validate_request(raw)
                except (ValueError, OSError) as exc:
                    message = str(exc) if isinstance(exc, GatewayPolicyError) else "Invalid request body."
                    return self._json(400, {"error": {"message": message}})
                if not gateway._slots.acquire(blocking=False):
                    return self._json(429, {"error": {"message": "All gateway request slots are occupied."}})
                admission = connection = None
                abort = None
                sent = False
                outcome = "network_error"
                reader = _UsageReader(body.get("stream", False))
                try:
                    if scope.cancelled.is_set():
                        return self._json(401, {"error": {"message": "Action cancelled."}})
                    admission = gateway.ledger.reserve(scope.session_id, len(encoded), body["max_output_tokens"])
                    connection = gateway._connection_factory()

                    def abort(active_connection=connection):
                        # Capture the admitted connection, not a mutable outer
                        # binding. Shutdown wakes the read without racing
                        # HTTPResponse.close; only the HTTP worker closes it.
                        sock = getattr(active_connection, "sock", None)
                        if sock:
                            try:
                                sock.shutdown(socket.SHUT_RDWR)
                            except OSError:
                                pass

                    with gateway._lock:
                        if scope.cancelled.is_set():
                            raise OSError("Cancelled")
                        scope.closers.add(abort)
                    if scope.cancelled.is_set():
                        raise OSError("Cancelled before connection")
                    connection.connect()
                    # If revocation raced TLS setup, do not send an API request.
                    # Later revocation shuts this established socket down, so
                    # request() cannot silently reconnect after a close race.
                    if scope.cancelled.is_set():
                        raise OSError("Cancelled during connection")
                    connection.request("POST", "/v1/responses", body=encoded, headers={
                        "Authorization": "Bearer "+gateway._api_key,
                        "Content-Type": "application/json", "Accept": "text/event-stream" if body.get("stream") else "application/json"})
                    response = connection.getresponse()
                    if response.status >= 400:
                        outcome = "upstream_error"
                        return self._json(502, {"error": {"message": "The API request was rejected; its admission remains charged."}})
                    self.send_response(response.status)
                    self.send_header("Content-Type", "text/event-stream" if body.get("stream") else "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    sent = True
                    while not scope.cancelled.is_set():
                        data = response.read1(65_536)
                        if not data:
                            outcome = "completed"
                            break
                        reader.feed(data)
                        self.wfile.write(data)
                        self.wfile.flush()
                    if scope.cancelled.is_set():
                        outcome = "cancelled"
                except DailyBudgetExceeded:
                    with gateway._lock:
                        scope.failure_code = "daily_allowance_exhausted"
                    self._json(429, {"error": {"message": "The UTC daily Codex allowance is exhausted."}})
                except (OSError, http.client.HTTPException):
                    if scope.cancelled.is_set():
                        outcome = "cancelled"
                    if not sent:
                        self._json(502, {"error": {"message": "API connection interrupted; any admission remains charged."}})
                finally:
                    if connection:
                        with gateway._lock:
                            if abort:
                                scope.closers.discard(abort)
                        connection.close()
                    if admission:
                        # Provider completion is authoritative even if the
                        # client cancels or a socket fails before trailing EOF.
                        # The reader never returns interim/incomplete usage.
                        settled_usage = reader.finish()
                        settled_outcome = "completed" if settled_usage is not None else outcome
                        gateway.ledger.record_result(admission["id"], status=settled_outcome, usage=settled_usage)
                    gateway._slots.release()

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def close(self):
        with self._lock:
            tokens = tuple(self._scopes)
        for token in tokens:
            self.revoke(token)
        if self._server:
            self._server.shutdown()
            self._server.server_close()
            self._thread.join(timeout=2)
            self._server = self._thread = None
