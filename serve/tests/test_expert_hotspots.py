"""Live route counters do not consume the generation channel or wait for the service FIFO."""
import io
import json
import queue
import unittest
from unittest import mock

from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine
from serve.tests import test_vram as vram_fixture


class Protocol(unittest.TestCase):
    def engine(self):
        e = ProcessEngine("missing-executable", [], lazy=True)
        e.proc = mock.Mock(pid=123)
        e.proc.poll.return_value = None
        e.proc.stdin = io.StringIO()
        e.ended, e.gen = False, 1
        e.info["expert_hotspots_protocol"] = 1
        e.lines, e.slot_q = queue.Queue(), [queue.Queue()]
        return e

    def test_isolated_sequence_and_no_generation_consumption(self):
        e = self.engine()
        e.lines.put("T 42\n")
        e.hotspot_q.put('HOTSPOTS 0 {"supported":true,"recording":false}\n')
        e.hotspot_q.put('HOTSPOTS 1 {"supported":true,"recording":true}\n')
        self.assertTrue(e.expert_hotspots("start")["recording"])
        self.assertEqual(e.proc.stdin.getvalue(), "HOTSPOTS 1 start\n")
        self.assertEqual(e.lines.get_nowait(), "T 42\n")

    def test_pump_separates_live_counters_and_does_not_log_them(self):
        e = self.engine()
        e.proc.stdout = io.StringIO('BT 0 42\nHOTSPOTS 1 {"supported":true}\nMEMORY {"host":{},"gpus":{}}\n')
        e.log = io.StringIO()
        e._pump()
        self.assertEqual(e.slot_q[0].get_nowait(), "BT 0 42\n")
        self.assertTrue(e.hotspot_q.get_nowait().startswith("HOTSPOTS 1 "))
        self.assertEqual(e.lines.get_nowait(), None)
        self.assertEqual(e.log.getvalue(), "")

    def test_old_build_is_never_sent_an_unknown_command(self):
        e = self.engine()
        e.info.clear()
        self.assertFalse(e.expert_hotspots()["supported"])
        self.assertEqual(e.proc.stdin.getvalue(), "")

    def test_timeout_exit_and_invalid_action(self):
        e = self.engine()
        with self.assertRaises(ValueError):
            e.expert_hotspots("nonsense")
        with self.assertRaisesRegex(ValueError, "did not answer"):
            e.expert_hotspots(timeout=.01)
        e.hotspot_q.put(None)
        with self.assertRaises(EngineDied):
            e.expert_hotspots()


class Route(unittest.TestCase):
    tearDown = vram_fixture.Route.tearDown
    request = vram_fixture.Route.request

    def setUp(self):
        vram_fixture.Route.setUp(self)
        self.engine.expert_hotspots = mock.Mock(return_value={"supported": True, "recording": True})

    def test_live_counter_controls(self):
        with self.svc.fifo:
            code, body = self.request("/api/expert-hotspots", {"action": "start"})
        self.assertEqual(code, 200, body)
        self.assertEqual(body["model"], self.svc.model)
        self.engine.expert_hotspots.assert_called_once_with("start")
        self.assertEqual(self.request("/api/expert-hotspots")[0], 200)
        self.assertIn("rejected", body["scope"])

    def test_control_validation(self):
        self.assertEqual(self.request("/api/expert-hotspots", {"action": "bad"})[0], 400)
        self.assertEqual(self.request("/api/expert-hotspots", {"action": "start", "model": "missing"})[0], 404)
        self.assertEqual(self.request("/api/expert-hotspots", {"action": "start"}, {"Origin": "https://foreign.example"})[0], 403)
        self.assertEqual(self.request("/api/expert-hotspots", {"action": "start"}, {"Content-Type": "text/plain"})[0], 415)
        self.engine.expert_hotspots.assert_not_called()


if __name__ == "__main__":
    unittest.main()
