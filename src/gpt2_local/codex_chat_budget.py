"""Durable, conservative admission control for this application's API requests.

Requests reserve an allowance before forwarding. Once complete usage is known,
its conservative estimate replaces the reservation; unknown usage, crashes and
ambiguous failures retain the full reservation. Every admission stays in the
audit history. This is local admission control, not a provider spend cap.
GPT-6.1 Sol standard prices, checked October 5, 2026:
https://developers.openai.com/api/docs/models/gpt-6.1-sol
Up to 272,000 input tokens: $2/M ordinary input, $0.10/M cache reads,
$2.50/M cache writes and $10/M output. Longer requests double the input/cache
rates and multiply output by 1.5. Missing cache-write details use the higher
write rate for all noncached input. Reservations use the longest-context rates.
"""
from __future__ import annotations

from contextlib import closing
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import math
from pathlib import Path
import sqlite3
import time
import uuid

MAX_REQUEST_BYTES = 750_000
MAX_OUTPUT_TOKENS = 4096
INPUT_OVERHEAD_TOKENS = 20_000
SHORT_CONTEXT_INPUT_TOKENS = 272_000
USAGE_ESTIMATE_VERSION = 3


def _usage_estimate_micros(incoming, outgoing, cached=0, written=None):
    """Conservatively price complete token totals, rounding up once."""
    if type(cached) is not int or not 0 <= cached <= incoming:
        cached = 0
    if type(written) is not int or not 0 <= written <= incoming-cached:
        written = incoming-cached
    long_context = incoming > SHORT_CONTEXT_INPUT_TOKENS
    input_rate = Decimal("4") if long_context else Decimal("2")
    write_rate = Decimal("5") if long_context else Decimal("2.5")
    read_rate = Decimal("0.2") if long_context else Decimal("0.1")
    output_rate = Decimal("15") if long_context else Decimal("10")
    ordinary = incoming-cached-written
    estimate = ordinary*input_rate + written*write_rate + cached*read_rate + outgoing*output_rate
    return int(estimate.to_integral_value(rounding=ROUND_CEILING))


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
            # Switching a legacy database to WAL may contend with another
            # constructor before SQLite's transaction busy timeout applies.
            deadline = time.monotonic()+15
            while True:
                try:
                    connection.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if exc.sqlite_errorcode not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED} or time.monotonic() >= deadline:
                        raise
                    time.sleep(.025)
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("""CREATE TABLE IF NOT EXISTS api_admissions (
                    id TEXT PRIMARY KEY, utc_day TEXT NOT NULL, session_id TEXT NOT NULL,
                    created REAL NOT NULL, reserved_micros INTEGER NOT NULL,
                    request_bytes INTEGER NOT NULL, max_output_tokens INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'reserved',
                    reported_input INTEGER, reported_output INTEGER,
                    reported_estimate_micros INTEGER)""")
                columns = {row["name"] for row in connection.execute("PRAGMA table_info(api_admissions)")}
                for name, definition in (("reported_cached_input", "INTEGER"),
                                         ("reported_cache_write_input", "INTEGER"),
                                         ("legacy_estimate_micros", "INTEGER"),
                                         ("estimate_version", "INTEGER NOT NULL DEFAULT 0")):
                    if name not in columns:
                        connection.execute(f"ALTER TABLE api_admissions ADD COLUMN {name} {definition}")
                connection.execute("CREATE INDEX IF NOT EXISTS admission_day ON api_admissions(utc_day)")
                connection.execute("""CREATE TABLE IF NOT EXISTS budget_resets (
                    id INTEGER PRIMARY KEY, utc_day TEXT NOT NULL,
                    created REAL NOT NULL, admission_cutoff INTEGER NOT NULL)""")
                connection.execute("CREATE INDEX IF NOT EXISTS reset_day ON budget_resets(utc_day)")
                old_reports = connection.execute("""SELECT id,reported_input,reported_output,
                    reported_cached_input,reported_cache_write_input,reported_estimate_micros FROM api_admissions
                    WHERE reported_estimate_micros IS NOT NULL AND estimate_version<?""",
                    (USAGE_ESTIMATE_VERSION,)).fetchall()
                for row in old_reports:
                    incoming, outgoing = row["reported_input"], row["reported_output"]
                    if type(incoming) is not int or incoming < 0 or type(outgoing) is not int or outgoing < 0:
                        continue
                    cached = row["reported_cached_input"]
                    estimate = _usage_estimate_micros(incoming, outgoing, cached, row["reported_cache_write_input"])
                    connection.execute("""UPDATE api_admissions SET
                        legacy_estimate_micros=COALESCE(legacy_estimate_micros,reported_estimate_micros),
                        reported_estimate_micros=?,estimate_version=? WHERE id=?""",
                        (estimate, USAGE_ESTIMATE_VERSION, row["id"]))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        if self.path.exists():
            self.path.chmod(0o600)

    def _connect(self):
        connection = sqlite3.connect(self.path, timeout=15, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _day(self):
        return datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")

    @staticmethod
    def _current_reset(connection, day):
        row = connection.execute("""SELECT created,admission_cutoff FROM budget_resets
            WHERE utc_day=? ORDER BY id DESC LIMIT 1""", (day,)).fetchone()
        return (row["created"], row["admission_cutoff"]) if row else (None, 0)

    def reset_daily_allowance(self):
        """Explicit operator reset; retain every admission and reset audit row.

        Admission rows must never be deleted or their rowids reassigned. Results
        arriving after this reset still update their original excluded audit row.
        """
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                reset_at = self.clock()
                day = datetime.fromtimestamp(reset_at, timezone.utc).strftime("%Y-%m-%d")
                cutoff = connection.execute("""SELECT COALESCE(MAX(rowid),0)
                    FROM api_admissions WHERE utc_day=?""", (day,)).fetchone()[0]
                connection.execute("INSERT INTO budget_resets (utc_day,created,admission_cutoff) VALUES (?,?,?)",
                                   (day, reset_at, cutoff))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise
        return {"utc_day": day, "reset_at": reset_at, "admission_cutoff": cutoff}

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
                _, cutoff = self._current_reset(connection, day)
                spent = connection.execute("""SELECT
                    COALESCE(SUM(COALESCE(reported_estimate_micros,reserved_micros)),0)
                    FROM api_admissions WHERE utc_day=? AND rowid>?""", (day, cutoff)).fetchone()[0]
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
        if status != "completed" or type(incoming) is not int or incoming < 0 or type(outgoing) is not int or outgoing < 0:
            incoming = outgoing = cached = written = estimate = None
        else:
            details = usage.get("input_tokens_details")
            cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
            invalid_cached = type(cached) is not int or not 0 <= cached <= incoming
            if invalid_cached:
                cached = 0
            written = details.get("cache_write_tokens") if isinstance(details, dict) else None
            if invalid_cached or type(written) is not int or not 0 <= written <= incoming-cached:
                written = None
            estimate = _usage_estimate_micros(incoming, outgoing, cached, written)
        with closing(self._connect()) as connection:
            # The first complete usage report settles the admission permanently.
            # A duplicate completion or later transport failure cannot erase it;
            # a previously unknown outcome can still receive complete usage.
            connection.execute("""UPDATE api_admissions SET status=?,reported_input=?,
                reported_output=?,reported_cached_input=?,reported_cache_write_input=?,
                reported_estimate_micros=?,estimate_version=?
                WHERE id=? AND reported_estimate_micros IS NULL""",
                (status, incoming, outgoing, cached, written, estimate, USAGE_ESTIMATE_VERSION, admission_id))

    def snapshot(self):
        day = self._day()
        with closing(self._connect()) as connection:
            # Keep the cutoff, timestamp and totals in one consistent snapshot.
            connection.execute("BEGIN")
            reset_at, cutoff = self._current_reset(connection, day)
            row = connection.execute("""SELECT COUNT(*) AS requests,
                COALESCE(SUM(reserved_micros),0) AS admitted,
                COALESCE(SUM(COALESCE(reported_estimate_micros,reserved_micros)),0) AS charged,
                COALESCE(SUM(CASE WHEN reported_estimate_micros IS NULL
                    THEN reserved_micros ELSE 0 END),0) AS unsettled,
                COALESCE(SUM(reported_estimate_micros),0) AS reported,
                SUM(CASE WHEN reported_estimate_micros IS NULL THEN 1 ELSE 0 END) AS unknown
                FROM api_admissions WHERE utc_day=? AND rowid>?""", (day, cutoff)).fetchone()
            connection.execute("COMMIT")
        return {"utc_day": day, "daily_limit_usd": self.daily_limit_micros / 1_000_000,
                "reset_at": reset_at,
                "admitted_usd": row["admitted"] / 1_000_000,
                "charged_usd": row["charged"] / 1_000_000,
                "unsettled_reserved_usd": row["unsettled"] / 1_000_000,
                "remaining_usd": max(0, self.daily_limit_micros-row["charged"]) / 1_000_000,
                "reported_estimate_usd": row["reported"] / 1_000_000,
                "requests": row["requests"], "unknown_usage_requests": row["unknown"] or 0}
