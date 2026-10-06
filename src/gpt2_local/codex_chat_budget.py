"""Durable, conservative admission control for this application's API requests.

Reservations are charged before forwarding and never refunded, including after
crashes or ambiguous network failures. This is deliberately more restrictive
than observed billing. It is local admission control, not a provider spend cap.
Prices: GPT-6.1 Sol standard long-context cache writes $5/M input and $15/M
output, checked October 5, 2026: https://developers.openai.com/api/docs/pricing.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import sqlite3
import time
import uuid

MAX_REQUEST_BYTES = 750_000
MAX_OUTPUT_TOKENS = 4096
INPUT_OVERHEAD_TOKENS = 20_000


class DailyBudgetExceeded(RuntimeError):
    """The request cannot fit within the remaining UTC daily allowance."""


class DailyBudgetLedger:
    def __init__(self, path, daily_limit_usd=10, *, clock=time.time):
        try:
            if type(daily_limit_usd) is bool:
                raise ValueError("Boolean values are not monetary allowances.")
            allowance = Decimal(str(daily_limit_usd))
            if not allowance.is_finite() or allowance <= 0 or not math.isfinite(float(allowance)):
                raise ValueError("Invalid monetary allowance.")
            self.daily_limit_micros = int(allowance * 1_000_000)
            if self.daily_limit_micros <= 0:
                raise ValueError("Allowance is below the ledger's currency precision.")
        except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
            raise ValueError("Daily allowance must be a finite positive monetary amount.") from exc
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path.parent.chmod(0o700)
        self.clock = clock
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS api_admissions (
                id TEXT PRIMARY KEY, utc_day TEXT NOT NULL, session_id TEXT NOT NULL,
                created REAL NOT NULL, reserved_micros INTEGER NOT NULL,
                request_bytes INTEGER NOT NULL, max_output_tokens INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'reserved',
                reported_input INTEGER, reported_output INTEGER,
                reported_estimate_micros INTEGER)""")
            connection.execute("CREATE INDEX IF NOT EXISTS admission_day ON api_admissions(utc_day)")
        if self.path.exists():
            self.path.chmod(0o600)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _day(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def estimate_micros(request_bytes, max_output_tokens=MAX_OUTPUT_TOKENS):
        if type(request_bytes) is not int or not 0 < request_bytes <= MAX_REQUEST_BYTES:
            raise ValueError("Request body exceeds the allowed size.")
        if type(max_output_tokens) is not int or not 1 <= max_output_tokens <= MAX_OUTPUT_TOKENS:
            raise ValueError("Invalid output allowance.")
        # UTF-8 byte count upper-bounds ordinary text tokenization; additional
        # room accounts for message/tool framing and protocol instructions.
        return (request_bytes + INPUT_OVERHEAD_TOKENS) * 5 + max_output_tokens * 15

    def reserve(self, session_id, request_bytes, max_output_tokens=MAX_OUTPUT_TOKENS):
        amount = self.estimate_micros(request_bytes, max_output_tokens)
        admission_id = uuid.uuid4().hex
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                admitted_at = self.clock()
                day = datetime.fromtimestamp(admitted_at, timezone.utc).strftime("%Y-%m-%d")
                spent = connection.execute(
                    "SELECT COALESCE(SUM(reserved_micros),0) FROM api_admissions WHERE utc_day=?",
                    (day,)).fetchone()[0]
                if spent + amount > self.daily_limit_micros:
                    raise DailyBudgetExceeded("The UTC daily Codex allowance is exhausted.")
                connection.execute("""INSERT INTO api_admissions
                    (id,utc_day,session_id,created,reserved_micros,request_bytes,max_output_tokens)
                    VALUES (?,?,?,?,?,?,?)""",
                    (admission_id, day, str(session_id), admitted_at, amount,
                     request_bytes, max_output_tokens))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return {"id": admission_id, "utc_day": day, "reserved_micros": amount}

    def record_result(self, admission_id, *, status, usage=None):
        if status not in {"completed", "upstream_error", "network_error", "cancelled"}:
            raise ValueError("Invalid admission outcome.")
        usage = usage if isinstance(usage, dict) else {}
        incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
        if type(incoming) is not int or incoming < 0 or type(outgoing) is not int or outgoing < 0:
            incoming = outgoing = estimate = None
        else:
            # A conservative reported-usage estimate, not an invoice calculation:
            # cached reads, writes and short-context discounts are not inferred.
            estimate = incoming * 5 + outgoing * 15
        with closing(self._connect()) as connection:
            connection.execute("""UPDATE api_admissions SET status=?,reported_input=?,
                reported_output=?,reported_estimate_micros=? WHERE id=?""",
                (status, incoming, outgoing, estimate, admission_id))

    def snapshot(self):
        day = self._day()
        with closing(self._connect()) as connection:
            row = connection.execute("""SELECT COUNT(*) AS requests,
                COALESCE(SUM(reserved_micros),0) AS charged,
                COALESCE(SUM(reported_estimate_micros),0) AS reported,
                SUM(CASE WHEN reported_estimate_micros IS NULL THEN 1 ELSE 0 END) AS unknown
                FROM api_admissions WHERE utc_day=?""", (day,)).fetchone()
        return {"utc_day": day, "daily_limit_usd": self.daily_limit_micros / 1_000_000,
                "admitted_usd": row["charged"] / 1_000_000,
                "remaining_usd": max(0, self.daily_limit_micros-row["charged"]) / 1_000_000,
                "reported_estimate_usd": row["reported"] / 1_000_000,
                "requests": row["requests"], "unknown_usage_requests": row["unknown"] or 0}
