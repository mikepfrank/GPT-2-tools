"""Web-reader isolation checks with fake DNS, sockets and HTTP responses."""
from __future__ import annotations

import socket
import threading
import time
import unittest
from unittest import mock

from gpt2_local import codex_chat_web as web


def dns_answer(address, port=443):
    if ":" in address:
        return (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port, 0, 0))
    return (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port))


class FakeSocket:
    def __init__(self, family, kind):
        self.family, self.kind = family, kind
        self.connected = None
        self.timeout = None
        self.closed = False
        self.stopped = threading.Event()
        self.handshakes = 0

    def settimeout(self, timeout):
        self.timeout = timeout

    def connect(self, address):
        self.connected = address

    def close(self):
        self.closed = True
        self.stopped.set()

    def shutdown(self, how):
        self.stopped.set()

    def do_handshake(self):
        self.handshakes += 1


class FakeResponse:
    def __init__(self, status=200, headers=None, body=b"A public page"):
        self.status = status
        self.headers = {"Content-Type": "text/plain", **(headers or {})}
        self.body = body
        self.read_limit = None

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def read(self, maximum):
        self.read_limit = maximum
        return self.body[:maximum]


class WebIsolationTests(unittest.TestCase):
    def fake_transport(self, resolver, responses):
        sockets, requests = [], []
        response_iter = iter(responses)

        def new_socket(family, kind):
            connection = FakeSocket(family, kind)
            sockets.append(connection)
            return connection

        class FakeHTTPConnection:
            def __init__(self, hostname, port, timeout):
                self.hostname, self.port = hostname, port
                self.sock = None

            def request(self, method, target, headers):
                # The validated address must already be connected; an ordinary
                # HTTPConnection connect would trigger another DNS lookup.
                if self.sock is None or self.sock.connected is None:
                    raise AssertionError("HTTP used a socket without validated DNS pinning")
                requests.append((self.hostname, self.port, method, target, headers))

            def getresponse(self):
                return next(response_iter)

        tls = mock.Mock()
        tls.wrap_socket.side_effect = lambda raw, server_hostname, do_handshake_on_connect: raw
        self.addCleanup(mock.patch.stopall)
        dns = mock.patch.object(web.socket, "getaddrinfo", side_effect=resolver).start()
        mock.patch.object(web.socket, "socket", side_effect=new_socket).start()
        mock.patch.object(web.http.client, "HTTPConnection", FakeHTTPConnection).start()
        mock.patch.object(web.ssl, "create_default_context", return_value=tls).start()
        return dns, sockets, requests, tls

    def test_only_public_unicast_addresses_are_accepted(self):
        blocked = ["127.0.0.1", "10.0.0.1", "172.16.0.1", "192.168.1.1", "169.254.169.254",
                   "100.64.0.1", "0.0.0.0", "255.255.255.255", "224.0.0.1", "::1", "::",
                   "fc00::1", "fe80::1", "fec0::1", "ff02::1", "::ffff:127.0.0.1",
                   "16.145.194.176", "::ffff:16.145.194.176"]
        for address in blocked:
            with self.subTest(address=address), mock.patch.object(web.socket, "getaddrinfo", return_value=[dns_answer(address)]):
                with self.assertRaises(ValueError):
                    web.public_addresses("source.example", 443)
        for address in ["8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8"]:
            with self.subTest(address=address), mock.patch.object(web.socket, "getaddrinfo", return_value=[dns_answer(address)]):
                self.assertEqual(web.public_addresses("source.example", 443), [(dns_answer(address)[0], dns_answer(address)[4])])

    def test_mixed_dns_answers_and_empty_answers_fail_closed(self):
        for answers in [[], [dns_answer("8.8.8.8"), dns_answer("127.0.0.1")],
                        [dns_answer("8.8.8.8"), dns_answer("16.145.194.176")]]:
            with self.subTest(answers=answers), mock.patch.object(web.socket, "getaddrinfo", return_value=answers):
                with self.assertRaises(ValueError):
                    web.public_addresses("source.example", 443)

    def test_connection_is_pinned_to_first_validation_without_second_lookup(self):
        dns, sockets, requests, tls = self.fake_transport(
            [[dns_answer("8.8.8.8")], [dns_answer("127.0.0.1")]], [FakeResponse()])
        result = web.read_web_page("https://source.example/article?q=1")
        self.assertEqual(result["text"], "A public page")
        self.assertEqual(dns.call_count, 1)
        self.assertEqual(sockets[0].connected, ("8.8.8.8", 443))
        self.assertEqual(requests[0][:4], ("source.example", 443, "GET", "/article?q=1"))
        self.assertEqual(requests[0][4]["Host"], "source.example")
        self.assertNotIn("Authorization", requests[0][4])
        self.assertNotIn("Cookie", requests[0][4])
        self.assertNotIn("Referer", requests[0][4])
        tls.wrap_socket.assert_called_once_with(sockets[0], server_hostname="source.example", do_handshake_on_connect=False)
        self.assertEqual(sockets[0].handshakes, 1)
        self.assertTrue(sockets[0].closed)

    def test_redirects_to_private_local_or_own_host_never_connect(self):
        for destination, address in [("http://localhost/private", "127.0.0.1"),
                                     ("https://metadata.example/latest", "169.254.169.254"),
                                     ("https://own.example/other-service", "16.145.194.176"),
                                     ("https://mapped.example/other-service", "::ffff:16.145.194.176")]:
            with self.subTest(destination=destination):
                dns, sockets, requests, _ = self.fake_transport(
                    [[dns_answer("8.8.8.8")], [dns_answer(address)]],
                    [FakeResponse(302, {"Location": destination})])
                self.assertIn("error", web.read_web_page("https://source.example/start"))
                self.assertEqual(dns.call_count, 2)
                self.assertEqual(len(sockets), 1)
                self.assertEqual(len(requests), 1)
                self.assertTrue(sockets[0].closed)
                mock.patch.stopall()

    def test_same_hostname_redirect_revalidates_and_blocks_rebinding(self):
        dns, sockets, requests, _ = self.fake_transport(
            [[dns_answer("8.8.8.8")], [dns_answer("127.0.0.1")]],
            [FakeResponse(302, {"Location": "/next"})])
        self.assertIn("error", web.read_web_page("https://source.example/start"))
        self.assertEqual(dns.call_args_list[0].args[:2], ("source.example", 443))
        self.assertEqual(dns.call_args_list[1].args[:2], ("source.example", 443))
        self.assertEqual(len(sockets), 1)
        self.assertEqual(len(requests), 1)

    def test_normal_public_redirect_uses_new_host_and_new_validated_address(self):
        dns, sockets, requests, tls = self.fake_transport(
            [[dns_answer("8.8.8.8")], [dns_answer("1.1.1.1")]],
            [FakeResponse(301, {"Location": "https://second.example/final"}), FakeResponse(body=b"Final public text")])
        result = web.read_web_page("https://source.example/start")
        self.assertEqual(result["url"], "https://second.example/final")
        self.assertEqual(result["text"], "Final public text")
        self.assertEqual(dns.call_count, 2)
        self.assertEqual(sockets[1].connected, ("1.1.1.1", 443))
        self.assertEqual(requests[1][4]["Host"], "second.example")
        self.assertEqual(tls.wrap_socket.call_args.kwargs["server_hostname"], "second.example")
        self.assertTrue(all(connection.closed for connection in sockets))

    def test_non_http_credentials_and_service_ports_are_rejected_before_dns(self):
        urls = ["file:///etc/passwd", "ftp://public.example/file", "https://user:secret@public.example/",
                "http://public.example:22/", "http://public.example:8902/v1/responses", "http://127.0.0.1:8900/",
                "https://public.example:bad/", "https:///missing-host", "x" * 2001]
        with mock.patch.object(web.socket, "getaddrinfo") as dns, mock.patch.object(web.socket, "socket") as transport:
            for url in urls:
                with self.subTest(url=url):
                    self.assertIn("error", web.read_web_page(url))
            dns.assert_not_called()
            transport.assert_not_called()

    def test_text_reader_bounds_body_and_excludes_executable_html_content(self):
        body = (b"<html><script>hidden instruction</script><style>hidden style</style><p>Visible evidence</p>"
                b"<a href='https://source.example/ref'>Reference</a></html>")
        response = FakeResponse(headers={"Content-Type": "text/html"}, body=body)
        self.fake_transport([[dns_answer("8.8.8.8")]], [response])
        result = web.read_web_page("https://source.example/")
        self.assertIn("Visible evidence", result["text"])
        self.assertNotIn("hidden", result["text"])
        self.assertEqual(result["links"], ["https://source.example/ref"])
        self.assertEqual(response.read_limit, web.MAX_BYTES + 1)
        self.assertIn("Untrusted", result["notice"])

    def test_trickling_body_has_total_deadline_and_shutdown_wakes_reader(self):
        entered, exited = threading.Event(), threading.Event()
        transport = []

        class SlowBody(FakeResponse):
            def read(self, maximum):
                entered.set()
                if not transport[0].stopped.wait(2):
                    raise AssertionError("Timeout failed to shut down the active socket")
                exited.set()
                raise OSError("Socket was interrupted")

        _, sockets, _, _ = self.fake_transport([[dns_answer("8.8.8.8")]], [SlowBody()])
        transport = sockets
        started = time.monotonic()
        with mock.patch.object(web, "READ_TIMEOUT", .05):
            result = web.read_web_page("https://source.example/slow")
        self.assertTrue(entered.is_set())
        self.assertIn("time limit", result["error"])
        self.assertLess(time.monotonic() - started, 1)
        self.assertTrue(exited.wait(1))
        self.assertTrue(sockets[0].closed)

    def test_cancellation_stops_slow_body_before_total_deadline(self):
        entered, exited, cancel = threading.Event(), threading.Event(), threading.Event()
        transport, results = [], []

        class SlowBody(FakeResponse):
            def read(self, maximum):
                entered.set()
                if not transport[0].stopped.wait(2):
                    raise AssertionError("Cancellation failed to shut down the active socket")
                exited.set()
                raise OSError("Socket was interrupted")

        _, sockets, _, _ = self.fake_transport([[dns_answer("8.8.8.8")]], [SlowBody()])
        transport = sockets
        with mock.patch.object(web, "READ_TIMEOUT", 5):
            caller = threading.Thread(target=lambda: results.append(web.read_web_page("https://source.example/slow", cancelled=cancel.is_set)))
            caller.start()
            self.assertTrue(entered.wait(1))
            cancel.set()
            caller.join(timeout=1)
        self.assertFalse(caller.is_alive())
        self.assertIn("cancelled", results[0]["error"])
        self.assertTrue(exited.wait(1))
        self.assertTrue(sockets[0].closed)

    def test_timed_out_dns_retains_global_slot_until_it_returns_without_connecting(self):
        release, all_entered = threading.Event(), threading.Event()
        lock = threading.Lock()
        entered_count = 0
        results = []
        slots = threading.BoundedSemaphore(4)

        def stalled_dns(*args, **kwargs):
            nonlocal entered_count
            with lock:
                entered_count += 1
                if entered_count == 4:
                    all_entered.set()
            release.wait(2)
            return [dns_answer("8.8.8.8")]

        self.addCleanup(release.set)
        with mock.patch.object(web, "_READ_SLOTS", slots), mock.patch.object(web, "READ_TIMEOUT", .1), \
                mock.patch.object(web.socket, "getaddrinfo", side_effect=stalled_dns), \
                mock.patch.object(web.socket, "socket") as transport:
            callers = [threading.Thread(target=lambda: results.append(web.read_web_page("https://source.example/"))) for _ in range(4)]
            for caller in callers:
                caller.start()
            self.assertTrue(all_entered.wait(1))
            for caller in callers:
                caller.join(timeout=1)
                self.assertFalse(caller.is_alive())
            self.assertEqual(len(results), 4)
            self.assertTrue(all("time limit" in result["error"] for result in results))
            self.assertIn("busy", web.read_web_page("https://source.example/")["error"])
            self.assertEqual(entered_count, 4)
            transport.assert_not_called()
            release.set()
            # Each stranded DNS worker must release its original semaphore;
            # cancellation must prevent connecting after the resolver returns.
            for _ in range(4):
                self.assertTrue(slots.acquire(timeout=1))
            transport.assert_not_called()

    def test_cancelled_dns_returns_promptly_and_never_connects_after_resolution(self):
        entered, release, cancel = threading.Event(), threading.Event(), threading.Event()
        results = []
        slots = threading.BoundedSemaphore(1)

        def stalled_dns(*args, **kwargs):
            entered.set()
            release.wait(2)
            return [dns_answer("8.8.8.8")]

        self.addCleanup(release.set)
        with mock.patch.object(web, "_READ_SLOTS", slots), mock.patch.object(web.socket, "getaddrinfo", side_effect=stalled_dns), \
                mock.patch.object(web.socket, "socket") as transport:
            caller = threading.Thread(target=lambda: results.append(web.read_web_page("https://source.example/", cancelled=cancel.is_set)))
            caller.start()
            self.assertTrue(entered.wait(1))
            cancel.set()
            caller.join(timeout=1)
            self.assertFalse(caller.is_alive())
            self.assertIn("cancelled", results[0]["error"])
            self.assertFalse(slots.acquire(blocking=False))
            release.set()
            self.assertTrue(slots.acquire(timeout=1))
            transport.assert_not_called()

    def test_already_cancelled_reads_start_no_worker_and_search_propagates_callback(self):
        callback = lambda: True
        with mock.patch.object(web.threading, "Thread") as worker:
            self.assertIn("cancelled", web.read_web_page("https://source.example/", cancelled=callback)["error"])
            worker.assert_not_called()
        with mock.patch.object(web, "read_web_page", return_value={"error": "cancelled"}) as read:
            result = web.search_web("public evidence", cancelled=callback)
            self.assertEqual(result["query"], "public evidence")
            read.assert_called_once_with("https://www.bing.com/search?q=public%20evidence&format=rss", cancelled=callback)

    def test_rss_search_returns_bounded_useful_titles_snippets_and_direct_urls(self):
        items = ["<item><title>Example &amp; domains</title><link>https://www.iana.org/help/example-domains</link>"
                 "<description><![CDATA[<p>Official <b>documentation</b>.</p><script>hidden</script>]]></description></item>",
                 "<item><title>RFC 2606</title><link>https://www.rfc-editor.org/info/rfc2606/?a=1&amp;b=2</link>"
                 "<description>Reserved domain names.</description></item>"]
        items.extend(f"<item><title>Result {number}</title><link>https://source.example/{number}</link>"
                     f"<description>{'detail ' * 100}</description></item>" for number in range(20))
        body = ("<?xml version='1.0'?><rss version='2.0'><channel>" + "".join(items) + "</channel></rss>").encode()
        _, _, requests, _ = self.fake_transport([[dns_answer("8.8.8.8")]], [FakeResponse(headers={"Content-Type": "application/rss+xml"}, body=body)])
        result = web.search_web("example domain IANA")
        self.assertNotIn("error", result)
        self.assertEqual(result["query"], "example domain IANA")
        self.assertEqual(result["provider"], "Bing RSS")
        self.assertEqual(len(result["results"]), web.MAX_SEARCH_RESULTS)
        self.assertEqual(result["results"][0], {"title": "Example & domains", "url": "https://www.iana.org/help/example-domains", "snippet": "Official documentation ."})
        self.assertEqual(result["results"][1]["url"], "https://www.rfc-editor.org/info/rfc2606/?a=1&b=2")
        self.assertNotIn("hidden", result["text"])
        self.assertTrue(all(len(item["snippet"]) <= 500 for item in result["results"]))
        self.assertEqual(result["links"], [item["url"] for item in result["results"]])
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0][3], "/search?q=example%20domain%20IANA&format=rss")

    def test_rss_parser_rejects_doctype_entities_and_non_web_result_links(self):
        malicious = ["<!DOCTYPE rss [<!ENTITY x SYSTEM 'file:///etc/passwd'>]><rss><channel>&x;</channel></rss>",
                     "<!doctype rss><rss/>", "<!ENTITY x 'expand'><rss/>"]
        for text in malicious:
            with self.subTest(text=text), self.assertRaises(ValueError):
                web._rss_results(text)
        links = ["javascript:alert(1)", "file:///etc/passwd", "ftp://source.example/", "https://user:secret@source.example/",
                 "http://127.0.0.1/private", "http://[::ffff:16.145.194.176]/private", "https://source.example:8902/private"]
        items = "".join(f"<item><title>Blocked</title><link>{link}</link></item>" for link in links)
        items += "<item><title>Allowed</title><link>https://source.example/article</link></item>" * 2
        self.assertEqual(web._rss_results("<rss><channel>" + items + "</channel></rss>"),
                         [{"title": "Allowed", "url": "https://source.example/article", "snippet": ""}])

    def test_xml_declarations_are_rejected_by_public_reader_without_network_entity_fetch(self):
        body = b"<!DOCTYPE rss [<!ENTITY x SYSTEM 'https://secret.example/'>]><rss><channel>&x;</channel></rss>"
        dns, sockets, requests, _ = self.fake_transport([[dns_answer("8.8.8.8")]],
            [FakeResponse(headers={"Content-Type": "application/xml"}, body=body)])
        result = web.read_web_page("https://source.example/feed")
        self.assertIn("error", result)
        self.assertEqual(dns.call_count, 1)
        self.assertEqual(len(requests), 1)
        self.assertTrue(sockets[0].closed)

    def test_search_rejects_html_challenge_response_and_never_starts_a_fallback(self):
        with mock.patch.object(web, "read_web_page", return_value={"text": "Enable JavaScript", "links": []}) as read:
            result = web.search_web("ordinary query")
        self.assertIn("error", result)
        self.assertEqual(read.call_count, 1)


if __name__ == "__main__":
    unittest.main()
