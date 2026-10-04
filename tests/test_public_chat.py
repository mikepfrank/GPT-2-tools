from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import types
import unittest
from http.client import HTTPConnection
from pathlib import Path
from unittest.mock import patch

from gpt2_local.public_chat import (
    MAX_BODY_BYTES, GenerationTimeout, Limits, PublicChatApplication, PublicChatServer, PublicChatHandler,
    ServiceError, _DeadlineModel, _PublicRunner, available_memory_gib, main, normalize_origin,
)
from gpt2_local.runtime import GenerationResult, GenerationSettings, SampledToken


class FakeRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls: list[tuple[str, GenerationSettings]] = []
        self.failure: Exception | None = None
        self.active = 0
        self.max_active = 0

    @staticmethod
    def count_prompt_tokens(prompt: str) -> int:
        return len(prompt)

    def generate_result(self, prompt: str, settings: GenerationSettings, **unused: object) -> GenerationResult:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append((prompt, settings))
        self.started.set()
        try:
            if not self.release.wait(timeout=5):
                raise RuntimeError("fake wait expired")
            if self.failure:
                raise self.failure
            return GenerationResult(
                text=" A deterministic reply.", original_input_tokens=len(prompt),
                input_tokens=len(prompt), output_tokens=1, output_token_ids=(1,),
                stop_reason="max_new_tokens", elapsed_seconds=0.01,
                tokens_per_second=100, prompt_truncated=False, cold_start_included=False,
            )
        finally:
            self.active -= 1


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class PublicApplicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = FakeRunner()
        self.app = self.make_app(self.runner)
        self.settings = {"max_new_tokens": 32, "temperature": 0.8, "seed": "42"}
        self.state = self.app.new({"example_rounds": 0, "date": "2026-10-02", "settings": self.settings})
        self.archive = self.state["chat"]

    def make_app(self, runner: FakeRunner, limits: Limits | None = None, **kwargs: object) -> PublicChatApplication:
        app = PublicChatApplication(runner, limits, **kwargs)  # type: ignore[arg-type]
        self.addCleanup(app.close)
        self.addCleanup(runner.release.set)
        return app

    def wait_done(self, job: dict[str, object], app: PublicChatApplication | None = None) -> dict[str, object]:
        app = app or self.app
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = app.job(str(job["job_id"]))
            if result["status"] in ("complete", "failed"):
                return result
            time.sleep(0.005)
        self.fail("fake job did not complete")

    def submit(self, message: str = "Hello", client: str = "client", app: PublicChatApplication | None = None, archive: object = None) -> dict[str, object]:
        return (app or self.app).submit({"chat": archive or self.archive, "message": message, "settings": self.settings}, client)

    def test_browser_archive_restores_across_service_restart(self) -> None:
        result = self.wait_done(self.submit())
        self.assertEqual(result["status"], "complete")
        saved = result["state"]["chat"]
        self.app.close()
        fresh = self.make_app(FakeRunner())
        restored = fresh.imported({"chat": saved})
        self.assertEqual(restored["turns"], result["state"]["turns"])
        self.assertEqual(restored["context_text"], result["state"]["context_text"])
        self.assertEqual(restored["settings"]["seed"], "42")
        self.assertNotEqual(restored["session_id"], result["state"]["session_id"])
        with self.assertRaises(ServiceError):
            fresh.job(str(result["job_id"]))

    def test_independent_browsers_do_not_share_transcript_mutations(self) -> None:
        before = copy.deepcopy(self.archive)
        first = self.wait_done(self.submit("First browser", "a"))
        second = self.wait_done(self.submit("Second browser", "b"))
        self.assertEqual(first["state"]["turns"][-1]["human"], "First browser")
        self.assertEqual(second["state"]["turns"][-1]["human"], "Second browser")
        self.assertEqual(len(first["state"]["turns"]), 1)
        self.assertEqual(len(second["state"]["turns"]), 1)
        self.assertEqual(self.archive, before)
        self.assertFalse(hasattr(self.app, "sessions"))

    def test_fifo_has_one_inference_and_preview_remains_available(self) -> None:
        self.runner.release.clear()
        first = self.submit("One", "a")
        self.assertTrue(self.runner.started.wait(timeout=1))
        second = self.submit("Two", "b")
        third = self.submit("Three", "c")
        self.assertEqual(self.app.job(str(first["job_id"]))["status"], "running")
        self.assertEqual(self.app.job(str(second["job_id"]))["position"], 1)
        self.assertEqual(self.app.job(str(third["job_id"]))["position"], 2)
        finished = threading.Event()
        previews = []
        def preview() -> None:
            previews.append(self.app.preview({"chat": self.archive, "message": "Preview", "settings": self.settings}))
            finished.set()
        thread = threading.Thread(target=preview)
        thread.start()
        self.assertTrue(finished.wait(timeout=1), "preview blocked on model inference")
        thread.join()
        self.assertIn("Preview", previews[0]["context_text"])
        self.assertEqual(self.app.health()["queue"], {"waiting": 2, "running": True})
        self.runner.release.set()
        for job in (first, second, third):
            self.assertEqual(self.wait_done(job)["status"], "complete")
        self.assertEqual(self.runner.max_active, 1)
        for (prompt, _), message in zip(self.runner.calls, ("One", "Two", "Three")):
            self.assertIn(f"Human> {message}", prompt)

    def test_queue_and_per_client_caps_reject_without_mutating_archive(self) -> None:
        app = self.make_app(self.runner, Limits(max_queue=1, max_active_jobs_per_client=1))
        self.runner.release.clear()
        before = copy.deepcopy(self.archive)
        first = self.submit(client="a", app=app)
        self.assertTrue(self.runner.started.wait(timeout=1))
        with self.assertRaises(ServiceError) as repeated:
            self.submit(client="a", app=app)
        self.assertEqual(repeated.exception.status, 429)
        second = self.submit(client="b", app=app)
        with self.assertRaises(ServiceError) as full:
            self.submit(client="c", app=app)
        self.assertEqual(full.exception.status, 503)
        self.assertEqual(self.archive, before)
        self.runner.release.set()
        self.wait_done(first, app)
        self.wait_done(second, app)

    def test_retained_job_cap_and_result_expiry(self) -> None:
        clock = Clock()
        app = self.make_app(FakeRunner(), Limits(max_jobs=1, result_ttl_seconds=10), clock=clock)
        first = self.wait_done(self.submit(app=app), app)
        with self.assertRaises(ServiceError) as full:
            self.submit(client="other", app=app)
        self.assertEqual(full.exception.status, 503)
        clock.now += 11
        with self.assertRaises(ServiceError) as expired:
            app.job(str(first["job_id"]))
        self.assertEqual(expired.exception.status, 404)
        self.assertEqual(self.wait_done(self.submit(client="other", app=app), app)["status"], "complete")

    def test_queued_jobs_expire_and_release_archive_memory(self) -> None:
        clock = Clock()
        app = self.make_app(self.runner, Limits(queue_ttl_seconds=10), clock=clock)
        self.runner.release.clear()
        self.submit(client="a", app=app)
        self.assertTrue(self.runner.started.wait(timeout=1))
        queued = self.submit(client="b", app=app)
        clock.now += 11
        result = app.job(str(queued["job_id"]))
        self.assertEqual(result["status"], "failed")
        self.assertIn("expired", result["error"])
        self.assertIsNone(app._jobs[str(queued["job_id"])].session)
        self.assertEqual(app.health()["queue"]["waiting"], 0)

    def test_generation_failure_preserves_browser_archive_and_worker_recovers(self) -> None:
        before = copy.deepcopy(self.archive)
        self.runner.failure = RuntimeError("private chat text must never be logged")
        with patch("sys.stderr") as stderr:
            failed = self.wait_done(self.submit())
        self.assertEqual(failed["status"], "failed")
        self.assertNotIn("private", failed["error"])
        stderr.write.assert_not_called()
        self.assertEqual(self.archive, before)
        self.runner.failure = None
        self.assertEqual(self.wait_done(self.submit())["status"], "complete")

    def test_preview_and_regeneration_preserve_original_and_candidate_history(self) -> None:
        first = self.wait_done(self.submit())
        saved = first["state"]["chat"]
        before = copy.deepcopy(saved)
        preview = self.app.preview({"chat": saved, "settings": self.settings}, regenerate=True)
        self.assertEqual(saved, before)
        job = self.app.submit({"chat": saved, "settings": self.settings}, "client", regenerate=True)
        replacement = self.wait_done(job)["state"]
        self.assertEqual(saved, before)
        self.assertEqual(replacement["last_generation"]["prompt_text"], first["state"]["last_generation"]["prompt_text"])
        self.assertEqual(replacement["last_generation"]["seed"], "43")
        self.assertEqual(len(replacement["turns"][-1]["previous_responses"]), 1)
        self.assertEqual(preview["context_text"], replacement["last_generation"]["prompt_text"])

    def test_regeneration_greedy_and_excessive_cap_fail_before_queue(self) -> None:
        saved = self.wait_done(self.submit())["state"]["chat"]
        for settings in ({**self.settings, "temperature": 0}, {**self.settings, "max_new_tokens": 257}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                self.app.submit({"chat": saved, "settings": settings}, "client", regenerate=True)
        self.assertEqual(len(self.runner.calls), 1)

    def test_max_reply_tokens_and_invalid_archive_are_enforced(self) -> None:
        with self.assertRaises(ValueError):
            self.app.new({"date": "2026-10-02", "settings": {"max_new_tokens": 257}})
        malformed = copy.deepcopy(self.archive)
        malformed["context"]["text"] = "Does not match transcript"
        with self.assertRaises(ValueError):
            self.app.submit({"chat": malformed, "message": "Hi", "settings": self.settings}, "client")
        self.assertFalse(self.runner.calls)

    def test_export_updates_settings_without_changing_input_archive(self) -> None:
        before = copy.deepcopy(self.archive)
        exported = self.app.exported({"chat": self.archive, "settings": {**self.settings, "seed": "44"}})
        self.assertEqual(exported["settings"]["seed"], "44")
        self.assertEqual(self.archive, before)

    def test_generation_and_general_request_rate_windows_are_bounded(self) -> None:
        clock = Clock()
        app = self.make_app(FakeRunner(), Limits(generations_per_minute=1, requests_per_minute=2, max_rate_clients=2), clock=clock)
        app.rate_request("a")
        app.rate_request("a")
        with self.assertRaises(ServiceError) as limited:
            app.rate_request("a")
        self.assertEqual(limited.exception.status, 429)
        app.rate_request("a", generation=True)
        with self.assertRaises(ServiceError):
            app.rate_request("a", generation=True)
        app.rate_request("b")
        with self.assertRaises(ServiceError):
            app.rate_request("c")
        clock.now += 60
        app.rate_request("c")
        self.assertEqual(len(app._rates), 1)

    def test_shutdown_is_bounded_and_queued_jobs_fail(self) -> None:
        self.runner.release.clear()
        self.submit("Running", "a")
        self.assertTrue(self.runner.started.wait(timeout=1))
        queued = self.submit("Queued", "b")
        started = time.monotonic()
        self.app.close(timeout=0.01)
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(self.app.job(str(queued["job_id"]))["status"], "failed")
        with self.assertRaises(ServiceError) as stopping:
            self.submit("Later", "c")
        self.assertEqual(stopping.exception.status, 503)

    def test_late_generation_result_does_not_commit(self) -> None:
        clock = Clock()
        app = self.make_app(self.runner, Limits(generation_timeout_seconds=1), clock=clock)
        self.runner.release.clear()
        before = copy.deepcopy(self.archive)
        job = self.submit(app=app)
        self.assertTrue(self.runner.started.wait(timeout=1))
        clock.now += 2
        self.runner.release.set()
        result = self.wait_done(job, app)
        self.assertEqual(result["status"], "failed")
        self.assertIn("time limit", result["error"])
        self.assertNotIn("state", result)
        self.assertEqual(self.archive, before)


class PublicHTTPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = FakeRunner()
        self.app = PublicChatApplication(self.runner)  # type: ignore[arg-type]
        self.server = PublicChatServer(("127.0.0.1", 0), self.app, public_origins=("https://gpt.example",), trusted_proxies=("127.0.0.1",))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self.cleanup)

    def cleanup(self) -> None:
        self.runner.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def request(self, path: str, body: object = None, *, headers: dict[str, str] | None = None, raw: bytes | None = None) -> tuple[int, object, dict[str, str]]:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        request_headers = {"Host": "gpt.example", "Origin": "https://gpt.example", "X-Real-IP": "192.0.2.1"}
        if headers is not None:
            request_headers.update(headers)
        data = raw if raw is not None else json.dumps(body).encode() if body is not None else None
        method = "POST" if data is not None else "GET"
        if data is not None:
            request_headers["Content-Type"] = "application/json"
        connection.request(method, path, data, request_headers)
        response = connection.getresponse()
        payload, response_headers = response.read(), dict(response.getheaders())
        status = response.status
        connection.close()
        return status, json.loads(payload), response_headers

    def test_complete_http_flow_and_health_contains_no_chat(self) -> None:
        status, state, headers = self.request("/api/new", {"example_rounds": 0, "date": "2026-10-02", "settings": {"max_new_tokens": 32, "seed": "42"}})
        self.assertEqual(status, 200)
        self.assertIn("chat", state)
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        status, job, _ = self.request("/api/message", {"chat": state["chat"], "message": "Private message", "settings": {"max_new_tokens": 32, "seed": "42"}})
        self.assertEqual(status, 202)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            status, job, _ = self.request(f"/api/jobs/{job['job_id']}")
            if job["status"] == "complete":
                break
            time.sleep(0.005)
        self.assertEqual(job["state"]["turns"][-1]["human"], "Private message")
        status, health, _ = self.request("/api/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["limits"]["max_reply_tokens"], 256)
        self.assertNotIn("Private", json.dumps(health))

    def test_full_queue_post_acknowledges_rejection_without_creating_job(self) -> None:
        self.app.limits = Limits(max_queue=1)
        _, state, _ = self.request("/api/new", {"example_rounds": 0, "date": "2026-10-02", "settings": {"max_new_tokens": 32}})
        body = {"chat": state["chat"], "message": "Hello", "settings": {"max_new_tokens": 32}}
        self.runner.release.clear()
        self.assertEqual(self.request("/api/message", body)[0], 202)
        self.assertTrue(self.runner.started.wait(timeout=1))
        self.assertEqual(self.request("/api/message", body, headers={"X-Real-IP": "192.0.2.2"})[0], 202)
        jobs_before = set(self.app._jobs)
        status, result, _ = self.request("/api/message", body, headers={"X-Real-IP": "192.0.2.3"})
        self.assertEqual(status, 503)
        self.assertIs(result["accepted"], False)
        self.assertIn("queue is full", result["error"])
        self.assertNotIn("job_id", result)
        self.assertEqual(set(self.app._jobs), jobs_before)
        self.assertEqual(self.app.health()["queue"]["waiting"], 1)

    def test_wrong_host_and_cross_origin_requests_are_rejected(self) -> None:
        for headers in ({"Host": "evil.example"}, {"Origin": "https://evil.example"}, {"Origin": "http://gpt.example"}, {"Origin": "null"}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request("/api/health", headers=headers)[0], 403)

    def test_post_without_origin_is_rejected(self) -> None:
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        connection.request("POST", "/api/new", headers={"Host": "gpt.example", "Content-Type": "application/json", "Content-Length": "0"})
        response = connection.getresponse()
        self.assertEqual(response.status, 403)
        response.read()
        connection.close()

    def test_only_explicit_proxy_real_ip_is_used_for_limits(self) -> None:
        self.request("/api/health", headers={"X-Real-IP": "192.0.2.99", "X-Forwarded-For": "203.0.113.9"})
        self.assertIn("192.0.2.99", self.app._rates)
        self.assertNotIn("203.0.113.9", self.app._rates)
        self.server.trusted_proxies.clear()
        self.request("/api/health", headers={"X-Real-IP": "192.0.2.100"})
        self.assertIn("127.0.0.1", self.app._rates)
        self.assertNotIn("192.0.2.100", self.app._rates)

    def test_malformed_proxy_ip_is_rejected(self) -> None:
        self.assertEqual(self.request("/api/health", headers={"X-Real-IP": "192.0.2.1, 203.0.113.2"})[0], 400)

    def test_oversized_and_nonfinite_json_requests_are_rejected(self) -> None:
        # An excessive declared length is rejected without waiting for/reading
        # that body. Avoid sending megabytes into a connection already closed.
        status, _, _ = self.request("/api/import", raw=b"{}", headers={"Content-Length": str(MAX_BODY_BYTES + 1)})
        self.assertEqual(status, 413)
        status, _, _ = self.request("/api/new", raw=b'{"settings":{"temperature":NaN}}')
        self.assertEqual(status, 400)

    def test_generation_validation_returns_helpful_bounded_errors(self) -> None:
        status, error, _ = self.request("/api/new", {"date": "2026-10-02", "settings": {"max_new_tokens": 257}})
        self.assertEqual(status, 400)
        self.assertIn("256 reply tokens", error["error"])
        _, state, _ = self.request("/api/new", {"example_rounds": 0, "date": "2026-10-02", "settings": {"max_new_tokens": 32}})
        status, error, _ = self.request("/api/preview", {"chat": state["chat"], "message": "x" * 1024, "settings": {"max_new_tokens": 32}})
        self.assertEqual(status, 400)
        self.assertIn("Shorten the message", error["error"])
        self.assertLessEqual(len(error["error"]), 512)

    def test_import_preview_and_export_work_while_model_runs(self) -> None:
        _, state, _ = self.request("/api/new", {"example_rounds": 0, "date": "2026-10-02", "settings": {"max_new_tokens": 32}})
        self.runner.release.clear()
        self.request("/api/message", {"chat": state["chat"], "message": "Running", "settings": {"max_new_tokens": 32}})
        self.assertTrue(self.runner.started.wait(timeout=1))
        for path, body in (
            ("/api/import", {"chat": state["chat"]}),
            ("/api/preview", {"chat": state["chat"], "message": "Preview", "settings": {"max_new_tokens": 32}}),
            ("/api/export", {"chat": state["chat"]}),
        ):
            with self.subTest(path=path):
                self.assertEqual(self.request(path, body)[0], 200)

    def test_http_handlers_are_bounded_under_incomplete_requests(self) -> None:
        # No worker is spawned when the semaphore is saturated.
        self.assertTrue(self.server._handler_slots.acquire(blocking=False))
        held = 1
        while self.server._handler_slots.acquire(blocking=False):
            held += 1
        try:
            connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
            connection.request("GET", "/api/health", headers={"Host": "gpt.example"})
            with self.assertRaises((ConnectionError, OSError)):
                connection.getresponse()
            connection.close()
        finally:
            for _ in range(held):
                self.server._handler_slots.release()


class FakeProgressDecoder:
    def __init__(self, backend: object) -> None:
        self.text = ""

    def push(self, token_id: int) -> str:
        self.text += {1: " A", 2: " deterministic reply."}[token_id]
        return self.text


class FakeStreamingRunner(FakeRunner):
    def __init__(self) -> None:
        super().__init__()
        self.tokenizer = types.SimpleNamespace(backend_tokenizer=object())
        self.observed = threading.Event()
        self.finish = threading.Event()

    def generate_result(self, prompt: str, settings: GenerationSettings, **options: object) -> GenerationResult:
        self.calls.append((prompt, settings))
        self.started.set()
        if not self.release.wait(3):
            raise RuntimeError("Fake prefill expired")
        observer = options.get("on_token")
        if observer:
            observer(SampledToken(1, 1, 0.01))
        self.observed.set()
        if not self.finish.wait(3):
            raise RuntimeError("Fake decoding expired")
        if self.failure:
            raise self.failure
        if observer:
            observer(SampledToken(2, 2, 0.02))
        return GenerationResult(text=" A deterministic reply.", original_input_tokens=len(prompt),
                                input_tokens=len(prompt), output_tokens=2, output_token_ids=(1, 2),
                                stop_reason="max_new_tokens", elapsed_seconds=0.03,
                                tokens_per_second=2 / 0.03, prompt_truncated=False, cold_start_included=False)


class StreamingHTTPTests(unittest.TestCase):
    request = PublicHTTPTests.request

    def setUp(self) -> None:
        decoder = patch("gpt2_local.public_chat.ReplyStreamDecoder", FakeProgressDecoder)
        decoder.start()
        self.addCleanup(decoder.stop)
        self.runner = FakeStreamingRunner()
        self.app = PublicChatApplication(self.runner, enable_streaming=True)  # type: ignore[arg-type]
        self.server = PublicChatServer(("127.0.0.1", 0), self.app, public_origins=("https://gpt.example",),
                                      trusted_proxies=("127.0.0.1",), stream_heartbeat=0.03, stream_lease=1)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.streams = []
        self.addCleanup(self.cleanup)
        self.settings = {"max_new_tokens": 32, "temperature": 0.8, "seed": "42"}
        self.archive = self.app.new({"example_rounds": 0, "date": "2026-10-03", "settings": self.settings})["chat"]

    def cleanup(self) -> None:
        self.runner.release.set()
        self.runner.finish.set()
        for connection, response in self.streams:
            response.close()
            connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def submit(self, client="192.0.2.1") -> dict:
        return self.app.submit({"chat": self.archive, "message": "Hello", "settings": self.settings}, client)

    def stream(self, identifier: str, client="192.0.2.1", cursor=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=2)
        headers = {"Host": "gpt.example", "Origin": "https://gpt.example", "X-Real-IP": client}
        if cursor is not None:
            headers["Last-Event-ID"] = str(cursor)
        connection.request("GET", f"/api/jobs/{identifier}/events", headers=headers)
        response = connection.getresponse()
        self.streams.append((connection, response))
        return response

    @staticmethod
    def frame(response):
        lines = []
        while True:
            line = response.readline()
            if line in (b"\n", b"\r\n", b""):
                return b"".join(lines).decode("utf-8")
            lines.append(line)

    def progress(self, response):
        while True:
            frame = self.frame(response)
            if frame.startswith("id:"):
                payload = json.loads(frame.split("data: ", 1)[1])
                return frame, payload
            self.assertTrue(frame, "stream ended before an event")

    def test_early_flush_heartbeat_progress_reconnect_and_final_archive(self) -> None:
        self.runner.release.clear()
        before = copy.deepcopy(self.archive)
        job = self.submit()
        identifier = job["job_id"]
        self.assertTrue(self.runner.started.wait(1))
        stream = self.stream(identifier)
        self.assertEqual(stream.status, 200)
        self.assertEqual(stream.getheader("Content-Type"), "text/event-stream; charset=utf-8")
        self.assertIsNone(stream.getheader("Content-Length"))
        frame, initial = self.progress(stream)
        self.assertEqual(initial["generated_tokens"], 0)
        self.assertEqual(self.frame(stream), ": heartbeat\n")
        self.runner.release.set()
        self.assertTrue(self.runner.observed.wait(1))
        frame, partial = self.progress(stream)
        self.assertEqual(partial["reply_text"], " A")
        self.assertGreater(partial["seq"], initial["seq"])
        self.assertNotIn("state", self.app.job(identifier))
        self.assertEqual(self.archive, before)
        self.assertEqual(self.request("/api/health")[0], 200)
        resumed = self.stream(identifier, cursor=partial["seq"])
        _, latest = self.progress(resumed)
        self.assertEqual(latest, partial)
        self.runner.finish.set()
        while True:
            terminal, payload = self.progress(resumed)
            if "event: complete\n" in terminal:
                break
        self.assertEqual(set(payload), {"version", "seq"})
        result = self.app.job(identifier)
        self.assertEqual(result["state"]["turns"][-1]["assistant"], " A deterministic reply.")
        self.assertEqual(len(self.runner.calls), 1)
        self.assertEqual(result["delivery"]["decoding_tokens_per_second"], 100)
        self.assertLessEqual(result["delivery"]["first_token_seconds"], result["delivery"]["first_visible_seconds"])

    def test_subscriber_caps_leave_health_available_and_release_slots(self) -> None:
        job = self.submit()
        identifier = job["job_id"]
        self.assertTrue(self.runner.observed.wait(1))
        first = self.stream(identifier)
        second = self.stream(identifier)
        self.progress(first)
        self.progress(second)
        self.assertEqual(self.request(f"/api/jobs/{identifier}/events")[0], 503)
        third = self.stream(identifier, "192.0.2.2")
        fourth = self.stream(identifier, "192.0.2.3")
        self.progress(third)
        self.progress(fourth)
        self.assertEqual(self.request(f"/api/jobs/{identifier}/events", headers={"X-Real-IP": "192.0.2.4"})[0], 503)
        self.assertEqual(self.request("/api/health")[0], 200)
        self.runner.finish.set()
        for response in (first, second, third, fourth):
            response.read()
        deadline = time.monotonic() + 1
        while self.server._stream_clients and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(self.server._stream_clients)

    def test_failed_generation_notifies_without_committing_partial_archive(self) -> None:
        before = copy.deepcopy(self.archive)
        job = self.submit()
        self.assertTrue(self.runner.observed.wait(1))
        response = self.stream(job["job_id"])
        _, partial = self.progress(response)
        self.assertEqual(partial["reply_text"], " A")
        self.runner.failure = RuntimeError("private callback failure")
        self.runner.finish.set()
        terminal, unused = self.progress(response)
        self.assertIn("event: failed\n", terminal)
        result = self.app.job(job["job_id"])
        self.assertNotIn("state", result)
        self.assertNotIn("private", result["error"])
        self.assertEqual(self.archive, before)

    def test_lease_and_disconnect_do_not_cancel_or_resubmit_generation(self) -> None:
        self.server.stream_lease = 0.08
        job = self.submit()
        self.assertTrue(self.runner.observed.wait(1))
        response = self.stream(job["job_id"])
        response.read()
        self.assertEqual(self.app.job(job["job_id"])["status"], "running")
        self.assertEqual(len(self.runner.calls), 1)
        self.assertFalse(self.server._stream_clients)

    def test_cursor_validation_missing_job_and_disabled_capability(self) -> None:
        job = self.submit()
        for cursor in ("-1", "1.5", "9" * 30, "\N{ARABIC-INDIC DIGIT ONE}"):
            if not cursor.isascii():
                # HTTP headers are Latin-1; test the application-facing ASCII
                # guard with a representable non-ASCII numeral instead.
                cursor = "\xb2"
            response = self.stream(job["job_id"], cursor=cursor)
            self.assertEqual(response.status, 400)
            response.read()
        self.assertEqual(self.request("/api/jobs/missing/events")[0], 404)
        self.app.enable_streaming = False
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/events")[0], 404)
        self.assertFalse(self.app.health()["capabilities"]["streaming"])
        self.assertEqual(self.request(f"/api/jobs/{job['job_id']}/events", headers={"Origin": "https://wrong.example"})[0], 403)

    def test_queue_position_changes_get_new_revisions(self) -> None:
        first = self.submit()
        self.assertTrue(self.runner.observed.wait(1))
        second = self.submit("192.0.2.2")
        third = self.submit("192.0.2.3")
        original = self.app.wait_progress(third["job_id"])
        self.assertEqual(original["position"], 2)
        self.runner.finish.set()
        deadline = time.monotonic() + 1
        while self.app.job(first["job_id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.005)
        changed = self.app.wait_progress(third["job_id"])
        self.assertGreater(changed["seq"], original["seq"])

    def test_stalled_socket_writer_does_not_block_inference_or_health(self) -> None:
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        class SlowHandler(PublicChatHandler):
            def _events(handler, identifier, client):
                original = handler.wfile
                class Writer:
                    def write(writer, content):
                        if content.startswith(b"id:"):
                            entered.set()
                            release.wait(2)
                            raise TimeoutError("Simulated stalled socket")
                        return original.write(content)
                    def __getattr__(writer, name):
                        return getattr(original, name)
                handler.wfile = Writer()
                try:
                    super()._events(identifier, client)
                finally:
                    handler.wfile = original
        self.server.RequestHandlerClass = SlowHandler
        job = self.submit()
        self.assertTrue(self.runner.observed.wait(1))
        response = self.stream(job["job_id"])
        self.assertTrue(entered.wait(1))
        self.runner.finish.set()
        deadline = time.monotonic() + 1
        while self.app.job(job["job_id"])["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(self.app.job(job["job_id"])["status"], "complete")
        self.assertEqual(self.request("/api/health")[0], 200)
        release.set()
        self.assertEqual(response.read(), b"")
        self.assertFalse(self.server._stream_clients)

    def test_oversized_frame_closes_subscription_but_generation_continues(self) -> None:
        job = self.submit()
        self.assertTrue(self.runner.observed.wait(1))
        with patch("gpt2_local.public_chat.MAX_STREAM_FRAME_BYTES", 32):
            response = self.stream(job["job_id"])
            self.assertEqual(response.read(), b"")
        self.assertFalse(self.server._stream_clients)
        self.assertEqual(self.app.job(job["job_id"])["status"], "running")
        self.runner.finish.set()

    def test_queued_expiry_and_shutdown_release_waiting_subscribers(self) -> None:
        clock = Clock()
        clock.now = time.monotonic()
        self.app.clock = clock
        self.app.limits = Limits(queue_ttl_seconds=10)
        first = self.submit()
        self.assertTrue(self.runner.observed.wait(1))
        queued = self.submit("192.0.2.2")
        response = self.stream(queued["job_id"], "192.0.2.2")
        self.progress(response)
        with self.app._condition:
            clock.now += 11
            self.app._condition.notify_all()
        frame, unused = self.progress(response)
        self.assertIn("event: failed\n", frame)
        self.assertEqual(self.app.job(queued["job_id"])["status"], "failed")
        running = self.stream(first["job_id"])
        self.progress(running)
        self.app.close(timeout=0)
        running.read()
        deadline = time.monotonic() + 1
        while self.server._stream_clients and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertFalse(self.server._stream_clients)
        self.assertEqual(len(self.runner.calls), 1)


class PublicConfigurationTests(unittest.TestCase):
    def test_timeout_stopping_criterion_preserves_existing_stops(self) -> None:
        clock, deadlines = Clock(), threading.local()
        deadlines.value = clock.now + 1
        class Model:
            def generate(self, **kwargs: object) -> object:
                return kwargs["stopping_criteria"]
        original = object()
        transformers = types.SimpleNamespace(StoppingCriteria=type("StoppingCriteria", (), {}), StoppingCriteriaList=list)
        with patch.dict("sys.modules", {"transformers": transformers}):
            criteria = _DeadlineModel(Model(), deadlines, clock).generate(stopping_criteria=[original])
        self.assertIs(criteria[0], original)
        self.assertFalse(criteria[1](None, None))
        clock.now += 1
        with self.assertRaises(GenerationTimeout):
            criteria[1](None, None)

    def test_tokenizer_calls_serialize_without_holding_inference_lock(self) -> None:
        entered, release, generation_done = threading.Event(), threading.Event(), threading.Event()
        class Tokenizer:
            def encode(self, prompt: str) -> list[int]:
                entered.set()
                release.wait(timeout=2)
                return [1]
            def __call__(self, prompt: str) -> str:
                return prompt
        class Runner:
            tokenizer = Tokenizer()
            def count_prompt_tokens(self, prompt: str) -> int:
                return len(self.tokenizer.encode(prompt))
            def generate_result(self, prompt: str) -> str:
                result = self.tokenizer(prompt)
                generation_done.set()
                return result
        runner = _PublicRunner(Runner(), 10, time.monotonic)  # type: ignore[arg-type]
        count = threading.Thread(target=runner.count_prompt_tokens, args=("a",))
        generate = threading.Thread(target=runner.generate_result, args=("b",))
        count.start()
        self.assertTrue(entered.wait(timeout=1))
        generate.start()
        try:
            self.assertFalse(generation_done.wait(timeout=0.05))
        finally:
            release.set()
            count.join(timeout=1)
            generate.join(timeout=1)
        self.assertTrue(generation_done.is_set())

    def test_origin_normalization_and_invalid_origins(self) -> None:
        self.assertEqual(normalize_origin("https://GPT.EXAMPLE:443/"), "https://gpt.example")
        self.assertEqual(normalize_origin("http://127.0.0.1:9090"), "http://127.0.0.1:9090")
        for origin in ("gpt.example", "ftp://gpt.example", "https://user:pass@gpt.example", "https://gpt.example/path", "https://gpt.example?x=1", "https://gpt.example:bad"):
            with self.subTest(origin=origin), self.assertRaises(ValueError):
                normalize_origin(origin)

    def test_nonloopback_bind_and_proxy_are_rejected(self) -> None:
        for address, proxies in ((("0.0.0.0", 0), ()), (("127.0.0.1", 0), ("192.0.2.1",))):
            with self.subTest(address=address, proxies=proxies), self.assertRaises(ValueError):
                PublicChatServer(address, None, trusted_proxies=proxies)

    def test_available_memory_honors_host_and_cgroup_ancestor_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meminfo, membership, cgroups = root / "meminfo", root / "cgroup", root / "groups"
            cgroups.mkdir()
            meminfo.write_text("MemAvailable: 16777216 kB\n", encoding="ascii")
            membership.write_text("0::/service/worker\n", encoding="ascii")
            group = cgroups / "service"
            group.mkdir()
            (group / "worker").mkdir()
            (cgroups / "memory.max").write_text("max", encoding="ascii")
            (group / "memory.max").write_text(str(9 * 1024**3), encoding="ascii")
            (group / "memory.current").write_text(str(2 * 1024**3), encoding="ascii")
            self.assertEqual(available_memory_gib(meminfo, cgroup_root=cgroups, cgroup_membership=membership), 7)
            (group / "worker" / "memory.max").write_text(str(5 * 1024**3), encoding="ascii")
            (group / "worker" / "memory.current").write_text(str(1024**3), encoding="ascii")
            self.assertEqual(available_memory_gib(meminfo, cgroup_root=cgroups, cgroup_membership=membership), 4)

    def test_memory_guard_fails_before_constructing_real_runner(self) -> None:
        with patch("gpt2_local.public_chat.available_memory_gib", return_value=4.8), patch("gpt2_local.public_chat.Gpt2Runner") as runner, patch("sys.stderr"):
            with self.assertRaises(SystemExit) as stopped:
                main(["--offline"])
        self.assertEqual(stopped.exception.code, 2)
        runner.assert_not_called()

    def test_available_memory_honors_cgroup_v1_limit(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meminfo, membership, cgroups = root / "meminfo", root / "cgroup", root / "groups"
            group = cgroups / "memory" / "service"
            group.mkdir(parents=True)
            meminfo.write_text("MemAvailable: 16777216 kB\n", encoding="ascii")
            membership.write_text("5:memory:/service\n", encoding="ascii")
            (group / "memory.limit_in_bytes").write_text(str(8 * 1024**3), encoding="ascii")
            (group / "memory.usage_in_bytes").write_text(str(3 * 1024**3), encoding="ascii")
            self.assertEqual(available_memory_gib(meminfo, cgroup_root=cgroups, cgroup_membership=membership), 5)

    def test_missing_memavailable_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "meminfo"
            path.write_text("MemFree: 1000 kB\n", encoding="ascii")
            with self.assertRaises(ValueError):
                available_memory_gib(path)


if __name__ == "__main__":
    unittest.main()
