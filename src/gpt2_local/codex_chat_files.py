"""Read-only, descriptor-confined access to a chat's isolated workspace."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import errno
import math
import os
from pathlib import Path
import re
import stat
from urllib.parse import parse_qsl, quote

PAGE_SIZE = 200
MAX_SCAN_ENTRIES = 10000
MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
MAX_PATH_BYTES = 4096
MAX_PATH_PARTS = 64


class WorkspaceFilesError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def relative_parts(path, *, file=False):
    if not isinstance(path, str):
        raise WorkspaceFilesError(400, "Choose a workspace-relative path")
    try:
        encoded = path.encode("utf-8")
    except UnicodeError as exc:
        raise WorkspaceFilesError(400, "Workspace paths must use valid Unicode") from exc
    if (len(encoded) > MAX_PATH_BYTES or "\\" in path or
            any(ord(character) < 32 or ord(character) == 127 for character in path)):
        raise WorkspaceFilesError(400, "Invalid workspace-relative path")
    parts = path.split("/") if path else []
    if (len(parts) > MAX_PATH_PARTS or any(part in ("", ".", "..") or len(part.encode("utf-8")) > 255 for part in parts)
            or (file and not parts)):
        raise WorkspaceFilesError(400, "Invalid workspace-relative path")
    return parts


def files_query(query, *, download=False):
    """Decode once, rejecting ambiguous fields and malformed UTF-8 escapes."""
    if re.search(r"%(?![0-9A-Fa-f]{2})", query):
        raise WorkspaceFilesError(400, "Invalid workspace file query")
    try:
        pairs = parse_qsl(query, keep_blank_values=True, strict_parsing=True,
                          encoding="utf-8", errors="strict", max_num_fields=2)
    except (ValueError, UnicodeError) as exc:
        raise WorkspaceFilesError(400, "Invalid workspace file query") from exc
    allowed = {"path"} if download else {"path", "offset"}
    if any(key not in allowed for key, _ in pairs) or len({key for key, _ in pairs}) != len(pairs):
        raise WorkspaceFilesError(400, "Unsupported or repeated workspace file query field")
    values = dict(pairs)
    path = values.get("path", "")
    relative_parts(path, file=download)
    raw_offset = values.get("offset", "0")
    if not re.fullmatch(r"[0-9]{1,5}", raw_offset) or int(raw_offset) > MAX_SCAN_ENTRIES:
        raise WorkspaceFilesError(400, "Invalid workspace file page offset")
    return path, int(raw_offset)


def content_disposition(name):
    # The fixed fallback has no attacker-controlled header characters. Modern
    # browsers use the UTF-8 filename while older clients get a safe fallback.
    return "attachment; filename=\"workspace-file\"; filename*=UTF-8''" + quote(name, safe="", encoding="utf-8", errors="strict")


def secure_descriptors_supported():
    return (os.name == "posix" and all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK"))
            and os.open in os.supports_dir_fd and os.stat in os.supports_dir_fd
            and os.stat in os.supports_follow_symlinks and os.scandir in os.supports_fd)


def _filesystem_error(exc):
    if exc.errno == errno.ENOENT:
        return WorkspaceFilesError(404, "Workspace entry not found")
    if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EACCES, errno.EPERM):
        return WorkspaceFilesError(403, "This workspace entry cannot be accessed safely")
    return WorkspaceFilesError(503, "Workspace files are temporarily unavailable")


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


class WorkspaceDownload:
    def __init__(self, descriptor, name, info):
        self.descriptor, self.name, self.info = descriptor, name, info
        self.size = info.st_size

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None

    def verify(self):
        if self.descriptor is None:
            raise WorkspaceFilesError(409, "The workspace file is no longer open")
        try:
            current = os.fstat(self.descriptor)
        except OSError as exc:
            raise WorkspaceFilesError(409, "The workspace file changed; try downloading it again") from exc
        if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or _identity(current) != _identity(self.info):
            raise WorkspaceFilesError(409, "The workspace file changed; try downloading it again")

    def chunks(self):
        """Read only the verified initial size, aborting on concurrent writes."""
        remaining = self.size
        while remaining:
            self.verify()
            try:
                block = os.read(self.descriptor, min(CHUNK_BYTES, remaining))
            except OSError as exc:
                raise WorkspaceFilesError(409, "The workspace file could not finish downloading") from exc
            self.verify()
            if not block:
                raise WorkspaceFilesError(409, "The workspace file changed; try downloading it again")
            remaining -= len(block)
            yield block
        self.verify()


class WorkspaceFiles:
    def __init__(self, data_dir: Path, *, page_size=PAGE_SIZE, max_scan=MAX_SCAN_ENTRIES,
                 max_download=MAX_DOWNLOAD_BYTES):
        self.data_dir = Path(data_dir).absolute()
        self.page_size, self.max_scan, self.max_download = page_size, max_scan, max_download

    @contextmanager
    def _directory(self, identifier, parts, *, allow_missing=False):
        if not re.fullmatch(r"[a-f0-9]{32}", identifier):
            raise WorkspaceFilesError(400, "Invalid chat identifier")
        if not secure_descriptors_supported():
            raise WorkspaceFilesError(503, "Secure workspace file access is unavailable on this host")
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        with ExitStack() as descriptors:
            try:
                descriptor = os.open(self.data_dir, flags)
                descriptors.callback(os.close, descriptor)
                # Every descendant of the trusted data-directory anchor is
                # untrusted, including the server's per-chat scaffolding.
                for component in ("agents", "players", identifier, "workspace", *parts):
                    descriptor = os.open(component, flags, dir_fd=descriptor)
                    descriptors.callback(os.close, descriptor)
            except OSError as exc:
                if allow_missing and exc.errno == errno.ENOENT:
                    yield None
                    return
                raise _filesystem_error(exc) from exc
            yield descriptor

    def listing(self, identifier, path="", offset=0):
        parts = relative_parts(path)
        if type(offset) is not int or not 0 <= offset <= MAX_SCAN_ENTRIES:
            raise WorkspaceFilesError(400, "Invalid workspace file page offset")
        entries = []
        with self._directory(identifier, parts, allow_missing=not parts) as descriptor:
            if descriptor is not None:
                try:
                    with os.scandir(descriptor) as iterator:
                        for count, entry in enumerate(iterator, 1):
                            if count > self.max_scan:
                                raise WorkspaceFilesError(413, "This folder has too many entries to browse")
                            relative = "/".join((*parts, entry.name))
                            try:
                                relative_parts(relative)
                                info = os.stat(entry.name, dir_fd=descriptor, follow_symlinks=False)
                            except (WorkspaceFilesError, FileNotFoundError):
                                continue
                            if stat.S_ISDIR(info.st_mode):
                                kind, size = "directory", None
                            elif stat.S_ISREG(info.st_mode) and info.st_nlink == 1:
                                kind, size = "file", info.st_size
                            else:
                                continue
                            if not math.isfinite(info.st_mtime):
                                continue
                            entries.append(dict(name=entry.name, path=relative, kind=kind, size=size, modified=info.st_mtime))
                except OSError as exc:
                    raise _filesystem_error(exc) from exc
        entries.sort(key=lambda entry: (entry["kind"] != "directory", entry["name"].casefold(), entry["name"]))
        page = entries[offset:offset + self.page_size]
        next_offset = offset + len(page) if offset + len(page) < len(entries) else None
        return dict(session_id=identifier, path=path, parent="/".join(parts[:-1]) if parts else None,
                    entries=page, next_offset=next_offset)

    def download(self, identifier, path):
        parts = relative_parts(path, file=True)
        descriptor = None
        with self._directory(identifier, parts[:-1]) as parent:
            try:
                before = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise WorkspaceFilesError(403, "Only regular workspace files can be downloaded")
                if not 0 <= before.st_size <= self.max_download:
                    raise WorkspaceFilesError(413, "Workspace downloads are limited to 64 MiB")
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK | getattr(os, "O_NOCTTY", 0)
                descriptor = os.open(parts[-1], flags, dir_fd=parent)
                current = os.fstat(descriptor)
                if not stat.S_ISREG(current.st_mode) or current.st_nlink != 1 or _identity(current) != _identity(before):
                    raise WorkspaceFilesError(409, "The workspace file changed; try downloading it again")
                if not 0 <= current.st_size <= self.max_download:
                    raise WorkspaceFilesError(413, "Workspace downloads are limited to 64 MiB")
                result = WorkspaceDownload(descriptor, parts[-1], current)
                descriptor = None
                return result
            except OSError as exc:
                raise _filesystem_error(exc) from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
