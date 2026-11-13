import threading
import unittest

from serve.engine.mock import MockEngine
from serve.models import dialect_for
from serve.models.deepseek41_encoding import eos_token
from serve.service import Service, THINKING_LOOP_REMINDER
from serve.util.text import ByteTokenizer
from serve.util.thinking_loop import ThinkingLoop


class LoopTokenizer(ByteTokenizer):
    SPECIALS = [*ByteTokenizer.SPECIALS, eos_token]


class ThinkingLoopTest(unittest.TestCase):
    @staticmethod
    def detect(ids):
        guard = ThinkingLoop(256)
        for i, token in enumerate(ids, 1):
            if period := guard.push(token):
                return i, period
        return None

    def test_long_period_after_the_history_window_rolls(self):
        cycle = list(range(1409))
        before = list(range(2000, 20000))
        i, period = self.detect(before + cycle * 4)
        self.assertEqual(period, 1409)
        self.assertLess(i - len(before), len(cycle) * 3)

    def test_one_reread_and_separate_references_do_not_trigger(self):
        passage = list(range(1409))
        self.assertIsNone(self.detect(passage * 2 + [2000] * 4 + list(range(2100, 2500))))
        self.assertIsNone(self.detect(passage + [2000] + passage + [2001] + passage))

    def test_multi_token_loop_resumes_only_the_thinking_in_both_dialects(self):
        repeat = "Compute the same hex digit again. " * 30
        for name in ("qwen4exp", "deepseek41"):
            with self.subTest(name=name):
                dialect = dialect_for(name)
                tok = LoopTokenizer()
                next_thought = "The receipt is enough."
                answer = "Here is the result."
                end = tok.encode(dialect.stop_texts[0], parse_special=True)
                engine = MockEngine(tok, [repeat, next_thought + dialect.reasoning_close + answer])
                engine.scripts[-1] = tok.encode(next_thought + dialect.reasoning_close + answer) + end
                svc = Service(engine, tok, dialect.template(""), dialect=dialect)
                original = tok.encode("original prompt")
                events = list(svc.run(original, True, None, 1024, {}, threading.Event()))
                reasoning = "".join(v.text for k, v in events if k == "event" and v.kind == "reasoning")
                content = "".join(v.text for k, v in events if k == "event" and v.kind == "content")
                self.assertEqual(content, answer)
                self.assertEqual(reasoning.count(THINKING_LOOP_REMINDER), 1)
                first_thought, later = reasoning.split(THINKING_LOOP_REMINDER)
                self.assertTrue(repeat.startswith(first_thought))
                self.assertLess(len(first_thought), len(repeat))
                self.assertIn(next_thought, later)
                self.assertEqual(len(engine.requests), 2)
                self.assertEqual(engine.requests[1][0], original + tok.encode(first_thought + THINKING_LOOP_REMINDER))
                self.assertEqual(events[-1][1]["finish"], "stop")

    def test_second_loop_wraps_up_and_the_answer_is_not_checked(self):
        tok = ByteTokenizer()
        dialect = dialect_for("qwen4exp")
        repeat = "Compute the same hex digit again. " * 30
        answer = "Repeat as requested. " * 25
        engine = MockEngine(tok, [repeat, repeat, answer])
        svc = Service(engine, tok, dialect.template(""), dialect=dialect)
        events = list(svc.run(tok.encode("question"), True, None, 2000, {}, threading.Event()))
        reasoning = "".join(v.text for k, v in events if k == "event" and v.kind == "reasoning")
        content = "".join(v.text for k, v in events if k == "event" and v.kind == "content")
        self.assertEqual(reasoning.count(THINKING_LOOP_REMINDER), 1)
        self.assertEqual(content, answer)
        self.assertEqual(len(engine.requests), 3)
        self.assertTrue(tok.decode(engine.requests[-1][0]).endswith(dialect.reasoning_close))

    def test_reminder_uses_the_remaining_thinking_budget(self):
        tok = ByteTokenizer()
        dialect = dialect_for("qwen4exp")
        repeat = "Compute the same hex digit again. " * 30
        later = " ".join(f"Check input {i}." for i in range(80))
        engine = MockEngine(tok, [repeat, later, "Done."])
        svc = Service(engine, tok, dialect.template(""), dialect=dialect)
        budget = 600
        events = list(svc.run(tok.encode("question"), True, None, 1200,
                              {"reasoning_budget_tokens": budget}, threading.Event()))
        reasoning = "".join(v.text for k, v in events if k == "event" and v.kind == "reasoning")
        self.assertIn(THINKING_LOOP_REMINDER, reasoning)
        self.assertEqual(len(reasoning), budget + len(dialect.reasoning_close.split("</think>")[0]))
        self.assertEqual(events[-1][1]["finish"], "stop")

    def test_tight_budget_skips_the_reminder_but_disabled_guard_does_not_intervene(self):
        tok = ByteTokenizer()
        dialect = dialect_for("qwen4exp")
        repeat = "Compute the same hex digit again. " * 20
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                engine = MockEngine(tok, [repeat + dialect.reasoning_close + "Done.", "Done."])
                svc = Service(engine, tok, dialect.template(""), dialect=dialect)
                svc.answer_room = 0  # isolate the loop guard from the automatic answer budget
                svc.thinking_loop_tokens = 256 if enabled else 0
                sampling = {"reasoning_budget_tokens": 350} if enabled else {}
                events = list(svc.run(tok.encode("question"), True, None, 1000, sampling, threading.Event()))
                reasoning = "".join(v.text for k, v in events if k == "event" and v.kind == "reasoning")
                self.assertNotIn(THINKING_LOOP_REMINDER, reasoning)
                if enabled:
                    self.assertLess(len(reasoning), 350)
                    self.assertEqual(len(engine.requests), 2)
                else:
                    self.assertEqual(reasoning.rstrip(), repeat.rstrip())
                    self.assertEqual(len(engine.requests), 1)
                self.assertEqual(events[-1][1]["finish"], "stop")
