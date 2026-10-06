from http.client import HTTPConnection
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from gpt2_local.codex_chat_budget import DailyBudgetLedger, MAX_OUTPUT_TOKENS
from gpt2_local.codex_chat_gateway import (CodexAPIGateway, GatewayPolicyError, MAX_ACTION_SECONDS,
                                         UPSTREAM_IDLE_TIMEOUT_SECONDS, _UsageReader, validate_request)


class FakeResponse:
    status = 200
    def __init__(self, streaming=False):
        self.parts = ([b'data: {"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":20,"output_tokens":3}}}\n\n']
                      if streaming else [b'{"status":"completed","usage":{"input_tokens":20,"output_tokens":3},"output":[]}'])
    def read1(self, size):
        return self.parts.pop(0) if self.parts else b""


class FakeConnection:
    sock = None
    def __init__(self, calls, streaming=False, failure=False):
        self.calls, self.streaming, self.failure = calls, streaming, failure
    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self.failure:
            raise OSError("network failure")
    def connect(self):
        pass
    def getresponse(self):
        return FakeResponse(self.streaming)
    def close(self):
        pass


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.calls = []
        self.ledger = DailyBudgetLedger(Path(self.temp.name)/"budget.sqlite3")
        self.gateway = CodexAPIGateway("operator-test-secret", self.ledger,
            connection_factory=lambda: FakeConnection(self.calls)).start()
        self.addCleanup(self.gateway.close)
        self.token = self.gateway.activate("chat")

    def request(self, body=None, *, token=None, path="/v1/responses", method="POST"):
        connection = HTTPConnection(self.gateway.host,self.gateway.port, timeout=5)
        try:
            connection.request(method, path, body=json.dumps(body or {"model":"gpt-6.1-sol","input":"Hello"}),
                headers={"Authorization":"Bearer "+(token or self.token),"Content-Type":"application/json"})
            response = connection.getresponse()
            return response.status,response.read()
        finally:
            connection.close()

    def test_upstream_receives_real_key_and_bounded_normalized_body(self):
        status, _ = self.request()
        self.assertEqual(status, 200)
        args,kwargs = self.calls[0]
        self.assertEqual(args, ("POST","/v1/responses"))
        self.assertEqual(kwargs["headers"]["Authorization"],"Bearer operator-test-secret")
        body = json.loads(kwargs["body"])
        self.assertEqual(body["max_output_tokens"],MAX_OUTPUT_TOKENS)
        self.assertFalse(body["store"])
        self.assertEqual(self.ledger.snapshot()["requests"],1)
        self.assertEqual(self.ledger.snapshot()["unknown_usage_requests"],0)

    def test_default_upstream_connection_allows_180_second_read_gaps_without_connecting(self):
        self.assertEqual(UPSTREAM_IDLE_TIMEOUT_SECONDS, 180)
        with mock.patch("gpt2_local.codex_chat_gateway.http.client.HTTPSConnection") as connection:
            gateway = CodexAPIGateway("private-test-credential", self.ledger)
            self.assertIs(gateway._connection_factory(), connection.return_value)
            connection.assert_called_once_with("api.openai.com", timeout=180)
            connection.return_value.connect.assert_not_called()
            connection.return_value.request.assert_not_called()

    def test_default_scope_last_for_600_seconds_and_expires_at_the_bound(self):
        self.assertEqual(MAX_ACTION_SECONDS, 600)
        with mock.patch("gpt2_local.codex_chat_gateway.time.monotonic", return_value=1000) as clock, \
                mock.patch("gpt2_local.codex_chat_gateway.threading.Timer") as timer:
            token = self.gateway.activate("default-deadline")
            try:
                scope = self.gateway._scopes[token]
                self.assertEqual(scope.expires, 1600)
                timer.assert_called_once_with(600, self.gateway.revoke, args=(token,))
                timer.return_value.start.assert_called_once_with()
                clock.return_value = 1599.99
                self.assertIs(self.gateway._scope("Bearer " + token), scope)
                clock.return_value = 1600
                self.assertIsNone(self.gateway._scope("Bearer " + token))
                with self.assertRaises(ValueError):
                    self.gateway.activate("over-bound", timeout_seconds=601)
            finally:
                self.gateway.revoke(token)
            timer.return_value.cancel.assert_called_once_with()

    def test_streaming_usage_is_extracted_without_altering_sse(self):
        self.gateway._connection_factory = lambda: FakeConnection(self.calls, streaming=True)
        status, payload = self.request({"model":"gpt-6.1-sol","input":"Hello","stream":True})
        self.assertEqual(status,200)
        self.assertIn(b'response.completed',payload)
        self.assertEqual(self.ledger.snapshot()["reported_estimate_usd"],.000080)

    def test_invalid_and_revoked_tokens_never_forward(self):
        self.assertEqual(self.request(token="not-a-token")[0],401)
        self.gateway.revoke(self.token)
        self.assertEqual(self.request()[0],401)
        self.assertEqual(self.calls,[])

    def test_unsupported_routes_and_hosted_tools_never_admit(self):
        self.assertEqual(self.request(path="/v1/responses/123")[0],404)
        for tool in ({"type":"web_search"},{"type":"code_interpreter"},{"type":"image_generation"}):
            self.assertEqual(self.request({"model":"gpt-6.1-sol","input":"x","tools":[tool]})[0],400)
        self.assertEqual(self.calls,[])
        self.assertEqual(self.ledger.snapshot()["requests"],0)

    def test_network_ambiguity_remains_charged(self):
        self.gateway._connection_factory = lambda: FakeConnection(self.calls, failure=True)
        self.assertEqual(self.request()[0],502)
        self.assertGreater(self.ledger.snapshot()["admitted_usd"],0)
        self.assertEqual(self.ledger.snapshot()["unknown_usage_requests"],1)

    def test_catalog_is_local_and_requires_scoped_token(self):
        status,payload = self.request(method="GET",path="/v1/models")
        self.assertEqual(status,200)
        self.assertEqual(json.loads(payload)["data"][0]["id"],"gpt-6.1-sol")
        self.assertEqual(self.calls,[])

    def test_local_namespaces_are_validated_recursively(self):
        body = {"model":"gpt-6.1-sol","input":"x","tools":[{"type":"namespace","name":"local","tools":[
            {"type":"function","name":"emit_message","parameters":{"type":"object"}},
            {"type":"custom","name":"exec"}]}]}
        validate_request(json.dumps(body).encode())
        body["tools"][0]["tools"].append({"type":"web_search"})
        with self.assertRaises(GatewayPolicyError):
            validate_request(json.dumps(body).encode())

    def test_hidden_context_modalities_prices_and_output_are_rejected(self):
        invalid = [{"previous_response_id":"opaque"},{"conversation":"opaque"},
                   {"model":"gpt-6-astra"},{"service_tier":"priority"},
                   {"max_output_tokens":MAX_OUTPUT_TOKENS+1},{"max_output_tokens":True},
                   {"tools":[{"type":[]}]},{"reasoning":{"effort":[]}},
                   {"input":[{"type":"input_image","image_url":"https://example.com/a.png"}]},
                   {"store":True},{"background":True}]
        for change in invalid:
            with self.subTest(change=change),self.assertRaises(GatewayPolicyError):
                validate_request(json.dumps({"model":"gpt-6.1-sol","input":"x",**change}).encode())

    def test_codex_client_metadata_is_stripped_before_forwarding(self):
        body={"model":"gpt-6.1-sol","input":[{"role":"user","content":[{"type":"input_text","text":"Hello"}]}],
              "reasoning":{"effort":"medium","summary":"none"},"stream":True,"store":False,
              "include":["reasoning.encrypted_content"],
              "tools":[{"type":"function","name":"emit_message","parameters":{"type":"object"}}],
              "client_metadata":{"turn_id":"client-turn","private_diagnostic":"DO_NOT_FORWARD",
                                 "previous_response_id":"opaque","service_tier":"priority"}}
        normalized,encoded=validate_request(json.dumps(body).encode())
        self.assertNotIn("client_metadata",normalized)
        self.assertNotIn(b"DO_NOT_FORWARD",encoded)
        self.assertEqual(normalized["input"],body["input"])
        self.assertEqual(normalized["tools"],body["tools"])
        self.assertEqual(normalized["reasoning"],body["reasoning"])
        self.assertEqual(normalized["max_output_tokens"],MAX_OUTPUT_TOKENS)
        self.assertEqual(normalized["service_tier"],"default")
        self.assertEqual(self.request(body)[0],200)
        self.assertNotIn("client_metadata",json.loads(self.calls[0][1]["body"]))

    def test_client_metadata_cannot_bypass_request_policy(self):
        metadata={"model":"gpt-6.1-sol","max_output_tokens":MAX_OUTPUT_TOKENS,"service_tier":"default"}
        for change in ({"previous_response_id":"opaque"},{"model":"gpt-6-astra"},
                       {"max_output_tokens":MAX_OUTPUT_TOKENS+1},{"service_tier":"priority"},
                       {"tools":[{"type":"web_search"}]}):
            body={"model":"gpt-6.1-sol","input":"x","client_metadata":metadata,**change}
            with self.subTest(change=change),self.assertRaises(GatewayPolicyError):
                validate_request(json.dumps(body).encode())

    def test_expanded_output_limit_is_forwarded_reserved_and_enforced_without_reset(self):
        self.assertEqual(MAX_OUTPUT_TOKENS, 16384)
        for output in (4096, MAX_OUTPUT_TOKENS):
            with self.subTest(output=output):
                self.assertEqual(self.request({"model":"gpt-6.1-sol","input":"Hello",
                                              "max_output_tokens":output})[0], 200)
                encoded = self.calls[-1][1]["body"]
                self.assertEqual(json.loads(encoded)["max_output_tokens"], output)
                with closing(self.ledger._connect()) as connection:
                    row = connection.execute("SELECT max_output_tokens,reserved_micros FROM api_admissions ORDER BY rowid DESC LIMIT 1").fetchone()
                self.assertEqual(row["max_output_tokens"], output)
                self.assertEqual(row["reserved_micros"], self.ledger.estimate_micros(len(encoded), output))
        before = self.ledger.snapshot()
        self.assertEqual(before["daily_limit_usd"], 10)
        self.assertEqual(before["requests"], 2)
        self.assertEqual(self.request({"model":"gpt-6.1-sol","input":"Hello",
                                      "max_output_tokens":MAX_OUTPUT_TOKENS+1})[0], 400)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.ledger.snapshot(), before)
        with closing(self.ledger._connect()) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM budget_resets").fetchone()[0], 0)

    def test_cancellation_shuts_down_socket_and_worker_alone_closes(self):
        entered=threading.Event()
        closed=threading.Event()
        lock=threading.Lock()
        counts={"active":0,"maximum":0,"closes":0}
        class BlockingResponse:
            status=200
            def read1(self,_size):
                entered.set()
                if not closed.wait(timeout=5):
                    raise OSError("Test timed out waiting for cancellation")
                return b""
        class BlockingConnection(FakeConnection):
            class Sock:
                def shutdown(self,_how):
                    closed.set()
            sock=Sock()
            def getresponse(self):
                return BlockingResponse()
            def close(self):
                with lock:
                    counts["active"]+=1
                    counts["maximum"]=max(counts["maximum"],counts["active"])
                    counts["closes"]+=1
                closed.set()
                time.sleep(.03)
                with lock:
                    counts["active"]-=1
        upstream=BlockingConnection(self.calls)
        self.gateway._connection_factory=lambda:upstream
        errors=[]
        self.gateway._server.handle_error=lambda *_:errors.append("worker-error")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.request)
            self.assertTrue(entered.wait(timeout=5))
            with self.gateway._lock:
                closer=next(iter(self.gateway._scopes[self.token].closers))
            self.assertIs(closer.__defaults__[0],upstream)
            self.gateway.revoke(self.token)
            self.assertEqual(pending.result(timeout=5)[0],200)
        self.assertEqual(errors,[])
        self.assertEqual(counts["maximum"],1)
        self.assertEqual(counts["closes"],1)
        self.assertGreater(self.ledger.snapshot()["admitted_usd"],0)
        self.assertEqual(self.request()[0],401)

    def test_only_ledger_rejection_sets_typed_action_failure(self):
        self.gateway.ledger=DailyBudgetLedger(Path(self.temp.name)/"tiny.sqlite3",daily_limit_usd=.01)
        self.assertEqual(self.request()[0],429)
        self.assertEqual(self.gateway.action_failure(self.token),{"code":"daily_allowance_exhausted"})
        self.assertEqual(self.calls,[])
        self.gateway.revoke(self.token)
        self.assertIsNone(self.gateway.action_failure(self.token))

    def test_provider_and_slot_limits_do_not_set_budget_failure(self):
        class RejectedConnection(FakeConnection):
            def getresponse(self):
                response=FakeResponse()
                response.status=429
                return response
        self.gateway._connection_factory=lambda:RejectedConnection(self.calls)
        self.assertEqual(self.request()[0],502)
        self.assertIsNone(self.gateway.action_failure(self.token))
        for _ in range(4):
            self.gateway._slots.acquire()
        try:
            self.assertEqual(self.request()[0],429)
            self.assertIsNone(self.gateway.action_failure(self.token))
        finally:
            for _ in range(4):
                self.gateway._slots.release()

    def test_cancellation_during_connect_never_sends_post(self):
        entered=threading.Event()
        proceed=threading.Event()
        class PendingConnection(FakeConnection):
            def connect(self):
                entered.set()
                if not proceed.wait(timeout=5):
                    raise OSError("Test timed out")
        upstream=PendingConnection(self.calls)
        self.gateway._connection_factory=lambda:upstream
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.request)
            self.assertTrue(entered.wait(timeout=5))
            self.gateway.revoke(self.token)
            proceed.set()
            self.assertEqual(pending.result(timeout=5)[0],502)
        self.assertEqual(self.calls,[])
        self.assertGreater(self.ledger.snapshot()["admitted_usd"],0)

    def test_interim_or_incomplete_sse_usage_does_not_settle_reservation(self):
        for kind in ("response.in_progress","response.incomplete","response.failed"):
            event={"type":kind,"response":{"status":"incomplete","usage":{"input_tokens":1,"output_tokens":1}}}
            class PartialConnection(FakeConnection):
                def getresponse(self):
                    response=FakeResponse(True)
                    response.parts=[b"data: "+json.dumps(event).encode()+b"\n\n"]
                    return response
            self.gateway._connection_factory=lambda:PartialConnection(self.calls)
            with self.subTest(kind=kind):
                self.assertEqual(self.request({"model":"gpt-6.1-sol","input":"x","stream":True})[0],200)
        state=self.ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"],3)
        self.assertEqual(state["charged_usd"],state["admitted_usd"])
        self.assertEqual(state["reported_estimate_usd"],0)

    def test_incomplete_json_usage_does_not_settle_reservation(self):
        class IncompleteConnection(FakeConnection):
            def getresponse(self):
                response=FakeResponse()
                response.parts=[b'{"status":"incomplete","usage":{"input_tokens":1,"output_tokens":1}}']
                return response
        self.gateway._connection_factory=lambda:IncompleteConnection(self.calls)
        self.assertEqual(self.request()[0],200)
        state=self.ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"],1)
        self.assertEqual(state["charged_usd"],state["admitted_usd"])

    def test_completed_sse_overrides_interim_usage_and_preserves_cache_details(self):
        reader=_UsageReader(True)
        reader.feed(b'data: {"type":"response.in_progress","response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n')
        self.assertIsNone(reader.finish())
        final={"input_tokens":120,"output_tokens":40,
               "input_tokens_details":{"cached_tokens":70,"cache_write_tokens":15},
               "cache_write_tokens":15}
        event={"type":"response.completed","response":{"status":"completed","usage":final}}
        encoded=b"data: "+json.dumps(event).encode()+b"\n\n"
        reader.feed(encoded[:25])
        self.assertIsNone(reader.finish())
        reader.feed(encoded[25:])
        self.assertEqual(reader.finish(),final)

    def test_completed_type_with_incomplete_status_or_missing_final_marker_is_unknown(self):
        reader=_UsageReader(True)
        reader.feed(b'data: {"type":"response.completed","response":{"status":"incomplete","usage":{"input_tokens":1,"output_tokens":1}}}\n\n')
        reader.feed(b'data: {"usage":{"input_tokens":1,"output_tokens":1}}\n\n')
        self.assertIsNone(reader.finish())

    def test_cancelled_partial_stream_keeps_full_reservation(self):
        waiting=threading.Event()
        cancelled=threading.Event()
        class PartialResponse:
            status=200
            first=True
            def read1(self,_size):
                if self.first:
                    self.first=False
                    return b'data: {"type":"response.in_progress","response":{"usage":{"input_tokens":1,"output_tokens":1}}}\n\n'
                waiting.set()
                if not cancelled.wait(timeout=5):
                    raise OSError("Test timed out")
                raise OSError("Cancelled")
        class PartialConnection(FakeConnection):
            class Sock:
                def shutdown(self,_how):
                    cancelled.set()
            sock=Sock()
            def getresponse(self):
                return PartialResponse()
        self.gateway._connection_factory=lambda:PartialConnection(self.calls)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.request,{"model":"gpt-6.1-sol","input":"x","stream":True})
            self.assertTrue(waiting.wait(timeout=5))
            self.gateway.revoke(self.token)
            self.assertEqual(pending.result(timeout=5)[0],200)
        state=self.ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"],1)
        self.assertEqual(state["charged_usd"],state["admitted_usd"])
        self.assertEqual(state["reported_estimate_usd"],0)

    def test_cancellation_after_authoritative_completion_still_settles(self):
        waiting=threading.Event()
        cancelled=threading.Event()
        class CompletedResponse(FakeResponse):
            def __init__(self):
                super().__init__(True)
            def read1(self,size):
                if self.parts:
                    return super().read1(size)
                waiting.set()
                if not cancelled.wait(timeout=5):
                    raise OSError("Test timed out")
                raise OSError("Cancelled after completion before EOF")
        class CompletedConnection(FakeConnection):
            class Sock:
                def shutdown(self,_how):
                    cancelled.set()
            sock=Sock()
            def getresponse(self):
                return CompletedResponse()
        self.gateway._connection_factory=lambda:CompletedConnection(self.calls)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(self.request,{"model":"gpt-6.1-sol","input":"x","stream":True})
            self.assertTrue(waiting.wait(timeout=5))
            self.gateway.revoke(self.token)
            status,payload=pending.result(timeout=5)
            self.assertEqual(status,200)
            self.assertIn(b"response.completed",payload)
        state=self.ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"],0)
        self.assertEqual(state["charged_usd"],.000080)
        self.assertLess(state["charged_usd"],state["admitted_usd"])
        with closing(self.ledger._connect()) as connection:
            self.assertEqual(connection.execute("SELECT status FROM api_admissions").fetchone()[0],"completed")

    def test_upstream_failure_after_authoritative_completion_still_settles(self):
        class CompletedResponse(FakeResponse):
            def __init__(self):
                super().__init__(True)
            def read1(self,size):
                if self.parts:
                    return super().read1(size)
                raise OSError("Transport lost after final provider usage")
        class CompletedConnection(FakeConnection):
            def getresponse(self):
                return CompletedResponse()
        self.gateway._connection_factory=lambda:CompletedConnection(self.calls)
        self.assertEqual(self.request({"model":"gpt-6.1-sol","input":"x","stream":True})[0],200)
        state=self.ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"],0)
        self.assertEqual(state["charged_usd"],.000080)

    def test_client_broken_pipe_after_reading_final_usage_still_settles(self):
        original=self.gateway._server.RequestHandlerClass
        class BrokenClient(original):
            def end_headers(self):
                super().end_headers()
                writer=self.wfile
                class BrokenWriter:
                    def write(self,_data):
                        raise BrokenPipeError("Client closed after receiving headers")
                    def __getattr__(self,name):
                        return getattr(writer,name)
                self.wfile=BrokenWriter()
        self.gateway._server.RequestHandlerClass=BrokenClient
        self.gateway._connection_factory=lambda:FakeConnection(self.calls,streaming=True)
        status,payload=self.request({"model":"gpt-6.1-sol","input":"x","stream":True})
        self.assertEqual(status,200)
        self.assertEqual(payload,b"")
        state=self.ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"],0)
        self.assertEqual(state["charged_usd"],.000080)


if __name__ == "__main__":
    unittest.main()
