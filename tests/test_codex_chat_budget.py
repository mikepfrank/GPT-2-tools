from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from gpt2_local.codex_chat_budget import DailyBudgetLedger, DailyBudgetExceeded, MAX_OUTPUT_TOKENS


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"budget.sqlite3"

    def test_restart_keeps_crashed_and_unknown_failure_reservations(self):
        ledger = DailyBudgetLedger(self.path)
        crashed = ledger.reserve("crashed", 100)
        failed = ledger.reserve("failed", 500)
        ledger.record_result(failed["id"], status="network_error")
        restarted = DailyBudgetLedger(self.path)
        state = restarted.snapshot()
        held = (crashed["reserved_micros"] + failed["reserved_micros"]) / 1_000_000
        self.assertEqual(state["admitted_usd"], held)
        self.assertEqual(state["charged_usd"], held)
        self.assertEqual(state["unsettled_reserved_usd"], held)
        self.assertEqual(state["reported_estimate_usd"], 0)
        self.assertEqual(state["remaining_usd"], 10-held)
        self.assertEqual(state["unknown_usage_requests"], 2)

    def test_concurrent_admission_is_atomic(self):
        ledger = DailyBudgetLedger(self.path, daily_limit_usd=1)
        capacity = 1_000_000 // ledger.estimate_micros(100)
        def admit(index):
            try:
                ledger.reserve(str(index), 100)
                return True
            except DailyBudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            admitted = sum(pool.map(admit, range(30)))
        self.assertEqual(admitted, capacity)
        self.assertLessEqual(ledger.snapshot()["charged_usd"], 1)

    def test_concurrent_admission_uses_settled_usage_without_losing_audit_history(self):
        ledger = DailyBudgetLedger(self.path, daily_limit_usd=1)
        capacity = 1_000_000 // ledger.estimate_micros(100)
        completed = [ledger.reserve(str(index), 100) for index in range(capacity)]
        with self.assertRaises(DailyBudgetExceeded):
            ledger.reserve("blocked", 100)
        for reservation in completed:
            ledger.record_result(reservation["id"], status="completed",
                                 usage={"input_tokens": 100, "output_tokens": 20})
        def admit(index):
            try:
                ledger.reserve(str(index), 100)
                return True
            except DailyBudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            admitted = sum(pool.map(admit, range(30)))
        self.assertEqual(admitted, capacity)
        state = ledger.snapshot()
        self.assertEqual(state["requests"], capacity * 2)
        self.assertEqual(state["unknown_usage_requests"], capacity)
        self.assertGreater(state["admitted_usd"], 1, "historical reservations must remain in the audit")
        self.assertLessEqual(state["charged_usd"], 1)
        self.assertAlmostEqual(state["charged_usd"], state["reported_estimate_usd"] + state["unsettled_reserved_usd"])
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute("SELECT id,status,reserved_micros,reported_estimate_micros FROM api_admissions").fetchall()
        self.assertEqual(len(rows), capacity * 2)
        by_id = {row[0]: row[1:] for row in rows}
        for reservation in completed:
            self.assertEqual(by_id[reservation["id"]], ("completed", reservation["reserved_micros"], 450))

    def test_day_boundary_is_utc(self):
        stamp = [datetime(2026,10,5,23,59,59,tzinfo=timezone.utc).timestamp()]
        ledger = DailyBudgetLedger(self.path, clock=lambda: stamp[0])
        ledger.reserve("chat", 100)
        self.assertEqual(ledger.snapshot()["utc_day"], "2026-10-05")
        stamp[0] += 2
        self.assertEqual(ledger.snapshot()["utc_day"], "2026-10-06")
        self.assertEqual(ledger.snapshot()["admitted_usd"], 0)
        self.assertEqual(ledger.snapshot()["charged_usd"], 0)
        self.assertEqual(ledger.snapshot()["unsettled_reserved_usd"], 0)

    def test_complete_usage_releases_allowance_and_admits_next_request(self):
        ledger = DailyBudgetLedger(self.path, daily_limit_usd=.4)
        reservation = ledger.reserve("chat", 500)
        with self.assertRaises(DailyBudgetExceeded):
            ledger.reserve("next", 500)
        ledger.record_result(reservation["id"], status="completed", usage={"input_tokens": 100, "output_tokens": 20})
        state = ledger.snapshot()
        self.assertEqual(state["reported_estimate_usd"], .00045)
        self.assertGreater(state["admitted_usd"], state["reported_estimate_usd"])
        self.assertEqual(state["charged_usd"], .00045)
        self.assertEqual(state["remaining_usd"], .39955)
        self.assertEqual(state["unsettled_reserved_usd"], 0)
        self.assertEqual(state["unknown_usage_requests"], 0)
        ledger.reserve("next", 500)
        self.assertEqual(ledger.snapshot()["requests"], 2)

    def test_known_usage_survives_duplicate_completion_failure_and_restart(self):
        ledger = DailyBudgetLedger(self.path)
        reservation = ledger.reserve("chat", 100)
        ledger.record_result(reservation["id"], status="completed", usage={"input_tokens": 100, "output_tokens": 20})
        original = ledger.snapshot()
        for status, usage in (("completed", {"input_tokens": 999, "output_tokens": 999}),
                              ("network_error", None), ("cancelled", {})):
            ledger.record_result(reservation["id"], status=status, usage=usage)
            self.assertEqual(ledger.snapshot(), original)
        self.assertEqual(DailyBudgetLedger(self.path).snapshot(), original)
        with closing(sqlite3.connect(self.path)) as connection:
            row = connection.execute("SELECT status,reported_input,reported_output FROM api_admissions WHERE id=?",
                                     (reservation["id"],)).fetchone()
        self.assertEqual(row, ("completed", 100, 20))

    def test_unknown_outcome_can_later_settle_with_zero_usage(self):
        ledger = DailyBudgetLedger(self.path)
        reservation = ledger.reserve("chat", 100)
        ledger.record_result(reservation["id"], status="network_error")
        self.assertGreater(ledger.snapshot()["charged_usd"], 0)
        ledger.record_result(reservation["id"], status="completed", usage={"input_tokens": 0, "output_tokens": 0})
        ledger.record_result(reservation["id"], status="cancelled")
        state = ledger.snapshot()
        self.assertEqual(state["charged_usd"], 0)
        self.assertEqual(state["remaining_usd"], 10)
        self.assertEqual(state["unknown_usage_requests"], 0)
        self.assertEqual(state["requests"], 1)

    def test_incomplete_or_invalid_usage_keeps_full_reservation(self):
        ledger = DailyBudgetLedger(self.path)
        for usage in ({"input_tokens": 100}, {"input_tokens": 100, "output_tokens": None},
                      {"input_tokens": True, "output_tokens": 20},
                      {"input_tokens": -1, "output_tokens": 20},
                      {"input_tokens": 100, "output_tokens": 2.5}):
            reservation = ledger.reserve("chat", 100)
            ledger.record_result(reservation["id"], status="completed", usage=usage)
        state = ledger.snapshot()
        self.assertEqual(state["charged_usd"], state["admitted_usd"])
        self.assertEqual(state["unsettled_reserved_usd"], state["charged_usd"])
        self.assertEqual(state["unknown_usage_requests"], 5)

    def test_legacy_schema_reprices_once_preserving_original_estimates_and_all_rows(self):
        stamp = datetime(2026, 10, 5, 12, tzinfo=timezone.utc).timestamp()
        # The fixture represents an old admission made under the 4096 limit.
        reserved = DailyBudgetLedger.estimate_micros(600_000, 4096)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("""CREATE TABLE api_admissions (
                id TEXT PRIMARY KEY, utc_day TEXT NOT NULL, session_id TEXT NOT NULL,
                created REAL NOT NULL,reserved_micros INTEGER NOT NULL,
                request_bytes INTEGER NOT NULL,max_output_tokens INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'reserved',reported_input INTEGER,
                reported_output INTEGER,reported_estimate_micros INTEGER)""")
            for index in range(3):
                known = index < 2
                connection.execute("INSERT INTO api_admissions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (str(index), "2026-10-05", "chat", stamp, reserved, 600_000, 4096,
                     "completed" if known else "reserved", 100 if known else None,
                     20 if known else None, 800 if known else None))
            connection.commit()
        with ThreadPoolExecutor(max_workers=4) as pool:
            ledgers = list(pool.map(lambda _: DailyBudgetLedger(self.path, clock=lambda: stamp), range(4)))
        restarted = ledgers[0]
        state = restarted.snapshot()
        self.assertGreater(state["admitted_usd"], 9)
        self.assertLess(state["charged_usd"], 4)
        restarted.reserve("new", 500_000)
        self.assertEqual(restarted.snapshot()["requests"], 4)
        self.assertGreater(restarted.snapshot()["admitted_usd"], 10)
        self.assertLess(restarted.snapshot()["charged_usd"], 10)
        DailyBudgetLedger(self.path, clock=lambda: stamp)
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute("""SELECT id,reported_estimate_micros,legacy_estimate_micros
                FROM api_admissions WHERE id IN ('0','1') ORDER BY id""").fetchall()
        self.assertEqual(rows, [("0", 450, 800), ("1", 450, 800)])

    def test_context_tier_boundary_uses_full_input_including_cache_reads(self):
        ledger = DailyBudgetLedger(self.path)
        for incoming, cached, expected_micros in ((272_000, 0, 681_000),
                                                  (272_001, 0, 1_361_505),
                                                  (272_001, 272_001, 55_901)):
            with self.subTest(incoming=incoming, cached=cached):
                before = ledger.snapshot()["reported_estimate_usd"]
                admission = ledger.reserve("chat", 100)
                ledger.record_result(admission["id"], status="completed", usage={
                    "input_tokens": incoming, "output_tokens": 100,
                    "input_tokens_details": {"cached_tokens": cached}})
                self.assertAlmostEqual(ledger.snapshot()["reported_estimate_usd"]-before,
                                       expected_micros/1_000_000)

    def test_known_cache_reads_and_writes_price_separate_buckets(self):
        ledger = DailyBudgetLedger(self.path)
        for details, expected_micros in (({"cached_tokens": 40, "cache_write_tokens": 30}, 159),
                                         ({"cached_tokens": 40, "cache_write_tokens": 0}, 144),
                                         ({"cached_tokens": 40}, 174)):
            admission = ledger.reserve("chat", 100)
            ledger.record_result(admission["id"], status="completed", usage={
                "input_tokens": 100, "output_tokens": 2, "input_tokens_details": details})
            with closing(sqlite3.connect(self.path)) as connection:
                row = connection.execute("""SELECT reported_estimate_micros,reported_cached_input,
                    reported_cache_write_input FROM api_admissions WHERE id=?""", (admission["id"],)).fetchone()
            self.assertEqual(row, (expected_micros, 40, details.get("cache_write_tokens")))

    def test_invalid_cache_metadata_keeps_conservative_noncache_write_estimate(self):
        ledger = DailyBudgetLedger(self.path)
        for details in ({"cached_tokens": True}, {"cached_tokens": -1}, {"cached_tokens": 101},
                        {"cached_tokens": "40"}, {"cached_tokens": 0, "cache_write_tokens": 101},
                        {"cached_tokens": 0, "cache_write_tokens": -1},
                        {"cached_tokens": 0, "cache_write_tokens": True},
                        {"cached_tokens": True, "cache_write_tokens": 0},
                        {"cached_tokens": 101, "cache_write_tokens": 0}):
            admission = ledger.reserve("chat", 100)
            ledger.record_result(admission["id"], status="completed", usage={
                "input_tokens": 100, "output_tokens": 2, "input_tokens_details": details})
            with closing(sqlite3.connect(self.path)) as connection:
                price = connection.execute("SELECT reported_estimate_micros FROM api_admissions WHERE id=?",
                                           (admission["id"],)).fetchone()[0]
            self.assertEqual(price, 270)

    def test_failed_or_cancelled_interim_usage_does_not_release_reservation(self):
        ledger = DailyBudgetLedger(self.path)
        for status in ("network_error", "cancelled", "upstream_error"):
            admission = ledger.reserve("chat", 100)
            ledger.record_result(admission["id"], status=status,
                                 usage={"input_tokens": 0, "output_tokens": 0})
        state = ledger.snapshot()
        self.assertEqual(state["unknown_usage_requests"], 3)
        self.assertEqual(state["charged_usd"], state["admitted_usd"])
        self.assertEqual(state["reported_estimate_usd"], 0)

    def test_fractional_microdollars_round_up_once_and_short_context_rates_apply(self):
        ledger = DailyBudgetLedger(self.path)
        for incoming, outgoing, details, expected in (
                (3, 0, {"cached_tokens": 1}, 6),
                (34_059, 420, {}, 89_348),
                (34_059, 420, {"cache_write_tokens": 0}, 72_318)):
            admission = ledger.reserve("chat", 100)
            ledger.record_result(admission["id"], status="completed", usage={
                "input_tokens": incoming, "output_tokens": outgoing, "input_tokens_details": details})
            with closing(sqlite3.connect(self.path)) as connection:
                actual = connection.execute("SELECT reported_estimate_micros FROM api_admissions WHERE id=?",
                                            (admission["id"],)).fetchone()[0]
            self.assertEqual(actual, expected)

    def test_explicit_reset_zeroes_counter_but_preserves_audit_and_late_completion(self):
        stamp = [datetime(2026, 10, 5, 12, tzinfo=timezone.utc).timestamp()]
        ledger = DailyBudgetLedger(self.path, clock=lambda: stamp[0])
        old = ledger.reserve("old-inflight", 100)
        failed = ledger.reserve("old-failed", 500)
        ledger.record_result(failed["id"], status="network_error")
        stamp[0] += 1
        reset = ledger.reset_daily_allowance()
        state = ledger.snapshot()
        self.assertEqual(state["charged_usd"], 0)
        self.assertEqual(state["admitted_usd"], 0)
        self.assertEqual(state["remaining_usd"], 10)
        self.assertEqual(state["reset_at"], reset["reset_at"])
        self.assertEqual(state["requests"], 0)
        ledger.record_result(old["id"], status="completed", usage={"input_tokens": 100, "output_tokens": 20})
        self.assertEqual(ledger.snapshot(), state, "late results remain in the excluded audit period")
        new = ledger.reserve("new", 100)
        restarted = DailyBudgetLedger(self.path, clock=lambda: stamp[0])
        self.assertEqual(restarted.snapshot()["charged_usd"], new["reserved_micros"]/1_000_000)
        self.assertEqual(restarted.snapshot()["requests"], 1)
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute("SELECT COUNT(*) FROM api_admissions").fetchone()[0]
            old_price = connection.execute("SELECT reported_estimate_micros FROM api_admissions WHERE id=?",
                                          (old["id"],)).fetchone()[0]
        self.assertEqual(rows, 3)
        self.assertEqual(old_price, 450)

    def test_resets_are_recorded_repeatedly_and_expire_at_utc_day_boundary(self):
        stamp = [datetime(2026, 10, 5, 23, 59, 55, tzinfo=timezone.utc).timestamp()]
        ledger = DailyBudgetLedger(self.path, clock=lambda: stamp[0])
        ledger.reset_daily_allowance()
        ledger.reserve("old", 100)
        stamp[0] += 1
        ledger.reset_daily_allowance()
        self.assertEqual(ledger.snapshot()["charged_usd"], 0)
        stamp[0] += 5
        state = ledger.snapshot()
        self.assertEqual(state["utc_day"], "2026-10-06")
        self.assertIsNone(state["reset_at"])
        new = ledger.reserve("today", 100)
        self.assertEqual(ledger.snapshot()["charged_usd"], new["reserved_micros"]/1_000_000)
        with closing(sqlite3.connect(self.path)) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM budget_resets").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM api_admissions").fetchone()[0], 2)

    def test_concurrent_reset_and_admission_have_a_single_auditable_cutoff(self):
        ledger = DailyBudgetLedger(self.path)
        ledger.reserve("prior", 100)
        def operation(index):
            return ledger.reset_daily_allowance() if index == 10 else ledger.reserve(str(index), 100)
        with ThreadPoolExecutor(max_workers=10) as pool:
            list(pool.map(operation, range(21)))
        with closing(sqlite3.connect(self.path)) as connection:
            cutoff = connection.execute("SELECT admission_cutoff FROM budget_resets ORDER BY id DESC LIMIT 1").fetchone()[0]
            total, included, held = connection.execute("""SELECT COUNT(*),
                SUM(CASE WHEN rowid>? THEN 1 ELSE 0 END),
                SUM(CASE WHEN rowid>? THEN reserved_micros ELSE 0 END) FROM api_admissions""",
                (cutoff, cutoff)).fetchone()
        self.assertEqual(total, 21)
        self.assertEqual(ledger.snapshot()["requests"], included)
        self.assertEqual(ledger.snapshot()["charged_usd"], held/1_000_000)

    def test_usage_above_reservation_is_fully_counted_and_blocks_new_admission(self):
        ledger = DailyBudgetLedger(self.path, daily_limit_usd=1)
        reservation = ledger.reserve("chat", 100)
        ledger.record_result(reservation["id"], status="completed",
                             usage={"input_tokens": 300_000, "output_tokens": 0})
        self.assertEqual(ledger.snapshot()["charged_usd"], 1.5)
        self.assertEqual(ledger.snapshot()["remaining_usd"], 0)
        with self.assertRaises(DailyBudgetExceeded):
            ledger.reserve("new", 100)

    def test_invalid_limits_and_requests_rejected(self):
        for limit in (0,-1,True,False,float("inf"),float("-inf"),float("nan"),"NaN","Infinity","invalid",None):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                DailyBudgetLedger(self.path, daily_limit_usd=limit)
        for size, output in ((0,MAX_OUTPUT_TOKENS),(750001,MAX_OUTPUT_TOKENS),
                             (100,MAX_OUTPUT_TOKENS+1),(100,True)):
            with self.assertRaises(ValueError):
                DailyBudgetLedger.estimate_micros(size,output)

    def test_larger_output_reservation_preserves_existing_admissions_and_daily_cap(self):
        self.assertEqual(MAX_OUTPUT_TOKENS, 16384)
        ledger = DailyBudgetLedger(self.path)
        original = ledger.reserve("original-output-limit", 100, 4096)
        expanded = ledger.reserve("expanded-output-limit", 100)
        self.assertEqual(expanded["reserved_micros"]-original["reserved_micros"], (16384-4096)*15)
        before = ledger.snapshot()
        self.assertEqual(before["daily_limit_usd"], 10)
        self.assertEqual(before["requests"], 2)
        self.assertEqual(before["charged_usd"],
                         (original["reserved_micros"]+expanded["reserved_micros"])/1_000_000)
        with self.assertRaises(ValueError):
            ledger.reserve("over-limit", 100, MAX_OUTPUT_TOKENS+1)
        self.assertEqual(ledger.snapshot(), before)
        self.assertEqual(DailyBudgetLedger(self.path).snapshot(), before)
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute("SELECT id,max_output_tokens,reserved_micros FROM api_admissions ORDER BY rowid").fetchall()
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM budget_resets").fetchone()[0], 0)
        self.assertEqual(rows, [(original["id"], 4096, original["reserved_micros"]),
                                (expanded["id"], MAX_OUTPUT_TOKENS, expanded["reserved_micros"])])

    def test_operator_may_configure_higher_allowance_without_resetting_charges(self):
        ledger=DailyBudgetLedger(self.path)
        reservation=ledger.reserve("chat",100)
        ledger.record_result(reservation["id"],status="network_error")
        reconfigured=DailyBudgetLedger(self.path,daily_limit_usd="20")
        state=reconfigured.snapshot()
        self.assertEqual(state["daily_limit_usd"],20)
        self.assertEqual(state["admitted_usd"],reservation["reserved_micros"]/1_000_000)
        self.assertEqual(state["charged_usd"], state["admitted_usd"])
        self.assertEqual(state["unknown_usage_requests"],1)
        self.assertEqual(state["requests"],1)


if __name__ == "__main__":
    unittest.main()
