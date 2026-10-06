from http.client import HTTPConnection
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from gpt2_local.codex_chat_budget import DailyBudgetLedger
from gpt2_local.codex_chat_gateway import CodexAPIGateway, GatewayPolicyError, validate_request


class FakeResponse:
    status = 200
    def __init__(self, streaming=False):
        self.parts = ([b'data: {"type":"response.completed","response":{"usage":{"input_tokens":20,"output_tokens":3}}}\n\n']
                      if streaming else [b'{"usage":{"input_tokens":20,"output_tokens":3},"output":[]}'])
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
        self.assertEqual(body["max_output_tokens"],4096)
        self.assertFalse(body["store"])
        self.assertEqual(self.ledger.snapshot()["requests"],1)
        self.assertEqual(self.ledger.snapshot()["unknown_usage_requests"],0)

    def test_streaming_usage_is_extracted_without_altering_sse(self):
        self.gateway._connection_factory = lambda: FakeConnection(self.calls, streaming=True)
        status, payload = self.request({"model":"gpt-6.1-sol","input":"Hello","stream":True})
        self.assertEqual(status,200)
        self.assertIn(b'response.completed',payload)
        self.assertEqual(self.ledger.snapshot()["reported_estimate_usd"],.000145)

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
                   {"max_output_tokens":4097},{"max_output_tokens":True},
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
        self.assertEqual(normalized["max_output_tokens"],4096)
        self.assertEqual(normalized["service_tier"],"default")
        self.assertEqual(self.request(body)[0],200)
        self.assertNotIn("client_metadata",json.loads(self.calls[0][1]["body"]))

    def test_client_metadata_cannot_bypass_request_policy(self):
        metadata={"model":"gpt-6.1-sol","max_output_tokens":4096,"service_tier":"default"}
        for change in ({"previous_response_id":"opaque"},{"model":"gpt-6-astra"},
                       {"max_output_tokens":4097},{"service_tier":"priority"},
                       {"tools":[{"type":"web_search"}]}):
            body={"model":"gpt-6.1-sol","input":"x","client_metadata":metadata,**change}
            with self.subTest(change=change),self.assertRaises(GatewayPolicyError):
                validate_request(json.dumps(body).encode())

    def test_cancellation_captures_connection_and_serializes_teardown(self):
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
        self.assertEqual(counts["closes"],2)
        self.assertGreater(self.ledger.snapshot()["admitted_usd"],0)
        self.assertEqual(self.request()[0],401)


if __name__ == "__main__":
    unittest.main()
