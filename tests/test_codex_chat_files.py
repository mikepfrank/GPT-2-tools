from __future__ import annotations

import copy
from http.client import HTTPConnection, IncompleteRead
from http.server import ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from unittest import mock
from urllib.parse import quote

from gpt2_local.codex_chat import CodexChatApplication, make_handler
from gpt2_local.codex_chat_files import (
    CHUNK_BYTES, MAX_DOWNLOAD_BYTES, PAGE_SIZE, WorkspaceDownload, WorkspaceFiles,
    WorkspaceFilesError, content_disposition, files_query, relative_parts, secure_descriptors_supported,
)


class QueryTests(unittest.TestCase):
    def test_paths_queries_and_unicode_disposition_are_unambiguous(self):
        self.assertEqual(files_query(""), ("", 0))
        self.assertEqual(files_query("path=dir%20name%2F%CE%B1.txt&offset=200"), ("dir name/α.txt", 200))
        self.assertEqual(files_query("path=a%2Bb.txt", download=True), ("a+b.txt", 0))
        self.assertEqual(relative_parts("space dir/α.bin"), ["space dir", "α.bin"])
        disposition = content_disposition('α "report".txt')
        self.assertTrue(disposition.startswith('attachment; filename="workspace-file"; filename*=UTF-8\'\''))
        self.assertIn("%CE%B1%20%22report%22.txt", disposition)
        self.assertTrue(disposition.isascii())
        for query in ("path=../secret", "path=%2e%2e%2fsecret", "path=%2Fetc%2Fpasswd", "path=x%5Cy",
                      "path=x%00y", "path=%FF", "path=%GG", "path=a&path=b", "offset=0&offset=1",
                      "owner=x", "offset=-1", "offset=1.5", "offset=10001", "path", "path=a//b"):
            with self.subTest(query=query), self.assertRaises(WorkspaceFilesError):
                files_query(query)
        for query in ("", "offset=0", "path=a&offset=0"):
            with self.assertRaises(WorkspaceFilesError):
                files_query(query, download=True)
        for path in ("a/./b", "/", "a/", "a\nfile", "\ud800", "a" * 256, "/".join(["x"] * 65)):
            with self.assertRaises(WorkspaceFilesError):
                relative_parts(path)

    def test_unsupported_platform_fails_closed_before_opening_paths(self):
        store = WorkspaceFiles(Path("unavailable"))
        with mock.patch("gpt2_local.codex_chat_files.secure_descriptors_supported", return_value=False), \
                mock.patch("gpt2_local.codex_chat_files.os.open") as opener:
            with self.assertRaises(WorkspaceFilesError) as raised:
                store.listing("a" * 32)
            self.assertEqual(raised.exception.status, 503)
            opener.assert_not_called()


@unittest.skipUnless(secure_descriptors_supported(), "Requires Linux secure descriptor traversal")
class DescriptorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "data"; self.data.mkdir()
        self.identifier = "a" * 32
        self.workspace = self.data / "agents" / "players" / self.identifier / "workspace"
        self.workspace.mkdir(parents=True)
        self.store = WorkspaceFiles(self.data)
        self.secret = self.root / "outside.txt"; self.secret.write_bytes(b"PRIVATE OUTSIDE DATA")

    def test_sorted_pagination_parent_paths_and_missing_workspace_are_bounded(self):
        (self.workspace / "z folder").mkdir(); (self.workspace / "A folder").mkdir()
        for index in range(203):
            (self.workspace / f"file-{index:03}.txt").write_text(str(index))
        first = self.store.listing(self.identifier)
        self.assertEqual(len(first["entries"]), PAGE_SIZE)
        self.assertEqual([entry["kind"] for entry in first["entries"][:2]], ["directory", "directory"])
        self.assertEqual([entry["name"] for entry in first["entries"][:2]], ["A folder", "z folder"])
        self.assertEqual(first["next_offset"], 200)
        self.assertIsNone(first["parent"])
        second = self.store.listing(self.identifier, offset=200)
        self.assertEqual(len(second["entries"]), 5)
        self.assertIsNone(second["next_offset"])
        self.assertEqual(len({entry["path"] for entry in first["entries"] + second["entries"]}), 205)
        (self.workspace / "A folder" / "α.bin").write_bytes(b"abc")
        nested = self.store.listing(self.identifier, "A folder")
        self.assertEqual(nested["parent"], "")
        self.assertEqual(nested["entries"][0]["path"], "A folder/α.bin")
        self.assertEqual(nested["entries"][0]["size"], 3)
        self.assertIsInstance(nested["entries"][0]["modified"], float)
        self.assertEqual(self.store.listing("b" * 32)["entries"], [])
        with self.assertRaises(WorkspaceFilesError) as raised:
            self.store.listing("b" * 32, "missing")
        self.assertEqual(raised.exception.status, 404)

    def test_rejected_entries_count_toward_scan_limit(self):
        for index in range(4):
            (self.workspace / f"link-{index}").symlink_to(self.secret)
        with self.assertRaises(WorkspaceFilesError) as raised:
            WorkspaceFiles(self.data, max_scan=3).listing(self.identifier)
        self.assertEqual(raised.exception.status, 413)

    def test_symlink_scaffolding_and_nested_components_cannot_leave_the_workspace(self):
        for component in ("agents", "players", self.identifier, "workspace", "nested"):
            with self.subTest(component=component):
                case = self.root / component; case.mkdir()
                data = case / "data"; data.mkdir()
                workspace = data / "agents" / "players" / self.identifier / "workspace"
                workspace.mkdir(parents=True)
                (workspace / "nested").mkdir()
                ancestors = {"agents": data / "agents", "players": data / "agents" / "players",
                             self.identifier: workspace.parent, "workspace": workspace, "nested": workspace / "nested"}
                target = ancestors[component]; moved = target.with_name(target.name + "-original")
                target.rename(moved); target.symlink_to(self.root, target_is_directory=True)
                store = WorkspaceFiles(data)
                with self.assertRaises(WorkspaceFilesError) as raised:
                    store.listing(self.identifier, "nested" if component == "nested" else "")
                self.assertEqual(raised.exception.status, 403)

    def test_links_special_files_and_undecodable_names_are_hidden_and_not_downloadable(self):
        (self.workspace / "good.txt").write_bytes(b"good")
        (self.workspace / "symlink").symlink_to(self.secret)
        os.link(self.secret, self.workspace / "hardlink")
        os.mkfifo(self.workspace / "fifo")
        listener = socket.socket(socket.AF_UNIX); self.addCleanup(listener.close)
        listener.bind(str(self.workspace / "socket"))
        descriptor = os.open(os.fsencode(self.workspace) + b"/invalid-\xff", os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(descriptor)
        self.assertEqual([entry["name"] for entry in self.store.listing(self.identifier)["entries"]], ["good.txt"])
        for name in ("symlink", "hardlink", "fifo", "socket"):
            with self.subTest(name=name), self.assertRaises(WorkspaceFilesError) as raised:
                self.store.download(self.identifier, name)
            self.assertEqual(raised.exception.status, 403)
        with self.assertRaises(WorkspaceFilesError):
            self.store.download(self.identifier, "../outside.txt")

    def test_directory_swap_to_symlink_is_rejected_during_descriptor_traversal(self):
        folder = self.workspace / "folder"; folder.mkdir()
        actual_open = os.open
        def swapping_open(path, flags, *args, **kwargs):
            if path == "folder":
                folder.rename(self.workspace / "old-folder")
                folder.symlink_to(self.root, target_is_directory=True)
            return actual_open(path, flags, *args, **kwargs)
        with mock.patch("gpt2_local.codex_chat_files.secure_descriptors_supported", return_value=True), \
                mock.patch("gpt2_local.codex_chat_files.os.open", side_effect=swapping_open):
            with self.assertRaises(WorkspaceFilesError) as raised:
                self.store.listing(self.identifier, "folder")
        self.assertEqual(raised.exception.status, 403)

    def test_leaf_swap_to_fifo_is_nonblocking_and_rejected_after_open(self):
        file = self.workspace / "race.txt"; file.write_bytes(b"original")
        actual_open = os.open; used_flags = []
        def swapping_open(path, flags, *args, **kwargs):
            if path == "race.txt":
                used_flags.append(flags); file.unlink(); os.mkfifo(file)
            return actual_open(path, flags, *args, **kwargs)
        with mock.patch("gpt2_local.codex_chat_files.secure_descriptors_supported", return_value=True), \
                mock.patch("gpt2_local.codex_chat_files.os.open", side_effect=swapping_open):
            with self.assertRaises(WorkspaceFilesError) as raised:
                self.store.download(self.identifier, "race.txt")
        self.assertEqual(raised.exception.status, 409)
        self.assertTrue(used_flags[0] & os.O_NONBLOCK)

    def test_leaf_swap_to_symlink_never_opens_the_target(self):
        file = self.workspace / "race.txt"; file.write_bytes(b"original")
        actual_open = os.open
        def swapping_open(path, flags, *args, **kwargs):
            if path == "race.txt":
                file.unlink(); file.symlink_to(self.secret)
            return actual_open(path, flags, *args, **kwargs)
        with mock.patch("gpt2_local.codex_chat_files.secure_descriptors_supported", return_value=True), \
                mock.patch("gpt2_local.codex_chat_files.os.open", side_effect=swapping_open):
            with self.assertRaises(WorkspaceFilesError):
                self.store.download(self.identifier, "race.txt")

    def test_downloads_stream_exact_bounded_bytes_and_reject_growth_replacement_and_oversize(self):
        payload = b"xyz" * (CHUNK_BYTES + 19)
        file = self.workspace / "data α.bin"; file.write_bytes(payload)
        actual_read = os.read; read_sizes = []
        def limited_read(descriptor, size):
            read_sizes.append(size); return actual_read(descriptor, size)
        with self.store.download(self.identifier, file.name) as opened, \
                mock.patch("gpt2_local.codex_chat_files.os.read", side_effect=limited_read):
            self.assertEqual(b"".join(opened.chunks()), payload)
        self.assertGreater(len(read_sizes), 1)
        self.assertLessEqual(max(read_sizes), CHUNK_BYTES)
        with self.store.download(self.identifier, file.name) as opened:
            chunks = opened.chunks(); self.assertEqual(len(next(chunks)), CHUNK_BYTES)
            with file.open("ab") as writer: writer.write(b"new bytes")
            with self.assertRaises(WorkspaceFilesError): next(chunks)
        file.write_bytes(payload)
        with self.store.download(self.identifier, file.name) as opened:
            file.unlink(); file.write_bytes(b"replacement")
            with self.assertRaises(WorkspaceFilesError): b"".join(opened.chunks())
        large = self.workspace / "large.bin"
        with large.open("wb") as writer: writer.truncate(MAX_DOWNLOAD_BYTES + 1)
        with self.assertRaises(WorkspaceFilesError) as raised:
            self.store.download(self.identifier, large.name)
        self.assertEqual(raised.exception.status, 413)

    def test_descriptors_close_on_failed_traversal_and_download(self):
        start = len(os.listdir("/proc/self/fd"))
        for _ in range(20):
            for operation in (lambda: self.store.listing(self.identifier, "missing"),
                              lambda: self.store.download(self.identifier, "missing.txt")):
                with self.assertRaises(WorkspaceFilesError): operation()
        self.assertEqual(len(os.listdir("/proc/self/fd")), start)


class FakeRunner:
    def count_prompt_tokens(self, text): return (len(text) + 3) // 4

class FakeBridge:
    def run(self, *args, **kwargs): raise AssertionError("Files must never call a model")

class FakeLedger:
    def snapshot(self): return {"daily_limit_usd": 10, "charged_usd": 0}


@unittest.skipUnless(secure_descriptors_supported(), "Requires Linux secure descriptor traversal")
class FilesHTTPTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.addCleanup(self.directory.cleanup)
        self.data = Path(self.directory.name)
        self.app = CodexChatApplication(FakeRunner(), FakeBridge(), FakeLedger(), self.data, start_worker=False)
        self.addCleanup(self.app.close)
        self.base, self.origin = "/private-beta", "https://private.example"
        self.handler = make_handler(self.app, self.base, self.origin)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler); self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start(); self.addCleanup(self.stop_server)
        _, headers, body = self.request("POST", self.base + "/api/sessions", {"example_rounds": 0}, origin=self.origin)
        self.identifier = json.loads(body)["session_id"]
        self.cookie = headers["Set-Cookie"].split(";", 1)[0]
        self.workspace = self.data / "agents" / "players" / self.identifier / "workspace"
        self.workspace.mkdir(parents=True)
        self.endpoint = self.base + "/api/sessions/" + self.identifier + "/files"

    def stop_server(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(timeout=2)

    def request(self, method, path, body=None, *, origin=None, cookie=None):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        headers = {}
        if origin: headers["Origin"] = origin
        if cookie: headers["Cookie"] = cookie
        payload = None
        if body is not None: payload = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=payload, headers=headers)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def test_attachments_have_exact_bytes_unicode_names_and_private_headers_without_state_mutation(self):
        name = 'report α "draft".bin'; payload = b"binary\0data\xff\n" * 10000
        (self.workspace / name).write_bytes(payload)
        before = (copy.deepcopy(self.app.sessions), self.app.store.load(), list(self.app.pending), dict(self.app.live))
        status, _, body = self.request("GET", self.endpoint, cookie=self.cookie)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["entries"][0]["name"], name)
        status, headers, body = self.request("GET", self.endpoint + "/download?path=" + quote(name, safe=""), cookie=self.cookie)
        self.assertEqual(status, 200); self.assertEqual(body, payload)
        self.assertEqual(headers["Content-Type"], "application/octet-stream")
        self.assertEqual(headers["Content-Length"], str(len(payload)))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        self.assertEqual(headers["Content-Disposition"], content_disposition(name))
        self.assertIn("default-src 'none'", headers["Content-Security-Policy"])
        self.assertEqual((self.app.sessions, self.app.store.load(), list(self.app.pending), self.app.live), before)

    def test_owner_authorization_precedes_all_filesystem_operations(self):
        with mock.patch.object(self.app.file_store, "listing") as listing, mock.patch.object(self.app.file_store, "download") as download:
            for cookie in (None, "gpt2_codex_owner=" + "b" * 64):
                self.assertEqual(self.request("GET", self.endpoint, cookie=cookie)[0], 404)
                self.assertEqual(self.request("GET", self.endpoint + "/download?path=private.txt", cookie=cookie)[0], 404)
            absent = self.base + "/api/sessions/" + "f" * 32 + "/files"
            self.assertEqual(self.request("GET", absent, cookie=self.cookie)[0], 404)
            listing.assert_not_called(); download.assert_not_called()

    def test_encoded_traversal_duplicate_and_invalid_queries_are_rejected(self):
        for query in ("path=%2e%2e%2fsecret", "path=%2Fetc%2Fpasswd", "path=%FF", "path=%GG", "path=x&path=y",
                      "offset=1&offset=2", "owner=x", "path=x%00y", "offset=-1", "path=%5Cetc%5Cpasswd"):
            with self.subTest(query=query):
                self.assertEqual(self.request("GET", self.endpoint + "?" + query, cookie=self.cookie)[0], 400)
        self.assertEqual(self.request("GET", self.endpoint + "/download?path=x&offset=0", cookie=self.cookie)[0], 400)
        self.assertEqual(self.request("GET", self.endpoint + "/download", cookie=self.cookie)[0], 400)
        self.assertEqual(self.request("GET", self.endpoint + "/download?path=missing", cookie=self.cookie)[0], 404)

    def test_post_header_failure_closes_connection_without_a_second_json_response(self):
        file = self.workspace / "changing.txt"; file.write_bytes(b"0123456789")
        def interrupted_chunks(opened):
            yield b"01"
            raise WorkspaceFilesError(409, "file changed")
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        self.addCleanup(connection.close)
        with mock.patch.object(WorkspaceDownload, "chunks", interrupted_chunks):
            connection.request("GET", self.endpoint + "/download?path=changing.txt", headers={"Cookie": self.cookie})
            response = connection.getresponse(); self.assertEqual(response.status, 200)
            with self.assertRaises(IncompleteRead) as raised: response.read()
        self.assertEqual(raised.exception.partial, b"01")
        self.assertEqual(raised.exception.expected, 8)

    def test_preheader_change_is_a_controlled_error_and_broken_pipe_closes_cleanly(self):
        file = self.workspace / "file.txt"; file.write_bytes(b"data")
        with mock.patch.object(WorkspaceDownload, "verify", side_effect=WorkspaceFilesError(409, "file changed")):
            status, headers, body = self.request("GET", self.endpoint + "/download?path=file.txt", cookie=self.cookie)
        self.assertEqual(status, 409)
        self.assertNotIn("Content-Disposition", headers)
        self.assertEqual(json.loads(body), {"error": "file changed"})
        sender = self.handler.__new__(self.handler)
        sender.send_response = lambda status: None
        sender.send_header = lambda name, value: None
        sender.end_headers = lambda: None
        sender.wfile = mock.Mock(); sender.wfile.write.side_effect = BrokenPipeError()
        sender.close_connection = False
        with self.app.download_file(self.identifier, self.app.sessions[self.identifier]["owner"], "file.txt") as opened:
            sender._send_file(opened)
        self.assertTrue(sender.close_connection)
        self.assertIsNone(opened.descriptor)


if __name__ == "__main__":
    unittest.main()
