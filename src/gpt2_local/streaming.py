"""Provisional display decoding for the chat protocol; no model or transport."""
from __future__ import annotations

import re

from .chat import MESSAGE_DELIMITER_RE


_POSSIBLE_BOUNDARY_SUFFIX = re.compile(r"\n\n[^\s>]*\Z")


class ReplyStreamDecoder:
    """Return a safe cumulative continuation while withholding speaker markers.

    The caller owns tokenizer synchronization and the generated-token cap.
    This helper keeps one decoded prefix and one incremental decoder, without
    event history. It does not flush at completion: the final GenerationResult
    remains authoritative for incomplete UTF-8 and unfinished marker prefixes.
    """

    def __init__(self, backend_tokenizer: object) -> None:
        from tokenizers.decoders import DecodeStream

        self._tokenizer = backend_tokenizer
        self._decoder = DecodeStream(skip_special_tokens=True)
        self._decoded = ""
        self._visible = ""
        self._stopped = False

    def push(self, token_id: int) -> str:
        """Observe a continuation ID and return the complete safe visible text."""
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError("token_id must be a non-negative integer")
        if self._stopped:
            return self._visible
        chunk = self._decoder.step(self._tokenizer, token_id)
        if chunk is None:
            return self._visible
        self._decoded += chunk
        boundary = MESSAGE_DELIMITER_RE.search(self._decoded)
        if boundary is not None:
            self._visible = self._decoded[:boundary.start()]
            self._decoded = self._visible
            self._stopped = True
            return self._visible

        possible_boundary = _POSSIBLE_BOUNDARY_SUFFIX.search(self._decoded)
        if possible_boundary is not None:
            self._visible = self._decoded[:possible_boundary.start()]
        elif self._decoded.endswith("\n"):
            self._visible = self._decoded[:-1]
        else:
            self._visible = self._decoded
        return self._visible
