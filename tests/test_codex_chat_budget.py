from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from gpt2_local.codex_chat_budget import DailyBudgetLedger, DailyBudgetExceeded


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"budget.sqlite3"

    def test_restart_and_failure_never_refund(self):
        ledger = DailyBudgetLedger(self.path)
        reservation = ledger.reserve("chat", 100)
        ledger.record_result(reservation["id"], status="network_error")
        restarted = DailyBudgetLedger(self.path)
        self.assertEqual(restarted.snapshot()["admitted_usd"], reservation["reserved_micros"]/1_000_000)
        self.assertEqual(restarted.snapshot()["unknown_usage_requests"], 1)

    def test_concurrent_admission_is_atomic(self):
        ledger = DailyBudgetLedger(self.path, daily_limit_usd=1)
        def admit(index):
            try:
                ledger.reserve(str(index), 100)
                return True
            except DailyBudgetExceeded:
                return False
        with ThreadPoolExecutor(max_workers=10) as pool:
            admitted = sum(pool.map(admit, range(30)))
        self.assertEqual(admitted, 6)
        self.assertLessEqual(ledger.snapshot()["admitted_usd"], 1)

    def test_day_boundary_is_utc(self):
        stamp = [datetime(2026,10,5,23,59,59,tzinfo=timezone.utc).timestamp()]
        ledger = DailyBudgetLedger(self.path, clock=lambda: stamp[0])
        ledger.reserve("chat", 100)
        self.assertEqual(ledger.snapshot()["utc_day"], "2026-10-05")
        stamp[0] += 2
        self.assertEqual(ledger.snapshot()["utc_day"], "2026-10-06")
        self.assertEqual(ledger.snapshot()["admitted_usd"], 0)

    def test_reported_usage_does_not_release_reservation(self):
        ledger = DailyBudgetLedger(self.path)
        reservation = ledger.reserve("chat", 500)
        ledger.record_result(reservation["id"], status="completed", usage={"input_tokens": 100, "output_tokens": 20})
        state = ledger.snapshot()
        self.assertEqual(state["reported_estimate_usd"], .0008)
        self.assertGreater(state["admitted_usd"], state["reported_estimate_usd"])
        self.assertEqual(state["unknown_usage_requests"], 0)

    def test_invalid_limits_and_requests_rejected(self):
        for limit in (0,-1,True,False,float("inf"),float("-inf"),float("nan"),"NaN","Infinity","invalid",None):
            with self.subTest(limit=limit),self.assertRaises(ValueError):
                DailyBudgetLedger(self.path, daily_limit_usd=limit)
        for size, output in ((0,4096),(750001,4096),(100,4097),(100,True)):
            with self.assertRaises(ValueError):
                DailyBudgetLedger.estimate_micros(size,output)

    def test_operator_may_configure_higher_allowance_without_resetting_charges(self):
        ledger=DailyBudgetLedger(self.path)
        reservation=ledger.reserve("chat",100)
        ledger.record_result(reservation["id"],status="network_error")
        reconfigured=DailyBudgetLedger(self.path,daily_limit_usd="20")
        state=reconfigured.snapshot()
        self.assertEqual(state["daily_limit_usd"],20)
        self.assertEqual(state["admitted_usd"],reservation["reserved_micros"]/1_000_000)
        self.assertEqual(state["unknown_usage_requests"],1)
        self.assertEqual(state["requests"],1)


if __name__ == "__main__":
    unittest.main()
