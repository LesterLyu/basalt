"""An orphaned call tail gets one model repair, never a text-to-command conversion."""
import http.client
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from serve.engine.mock import MockEngine
from serve.frontend import ChatTemplate
from serve.prompt import CTX_SLACK, PromptText
from serve.service import Service
from serve.tests.test_responses import ROOT, Server
from serve.tool_output import REPAIR_TURN, ToolOutputError, orphaned_tool_tail
from serve.util.text import ByteTokenizer


BAD = ('Command: `python3 -c "print(1)"\n</parameter>\n'
       '<parameter=yield_time_ms>\n15000\n</parameter>\n</function>\n</tool_call>')
CALL = ('<tool_call>\n<function=exec_command>\n<parameter=cmd>\n'
        'python3 -c "print(1)"\n</parameter>\n</function>\n</tool_call>')
TOOL = {"type": "function", "name": "exec_command", "parameters": {
    "type": "object", "properties": {"cmd": {"type": "string"}}}}


class Detection(unittest.TestCase):
    def test_orphaned_tail(self):
        self.assertTrue(orphaned_tool_tail(BAD))
        self.assertTrue(orphaned_tool_tail(BAD.replace('\n', '\r\n')))
        self.assertTrue(orphaned_tool_tail(BAD + '\n\n'))

    def test_examples_and_prose_remain_text(self):
        for text in ('All done.', 'The literal </parameter> ends a value.',
                     '```xml\n' + BAD + '\n```', '```xml\n' + BAD,
                     '~~~xml\n' + BAD, '````xml\n```\n' + BAD,
                     '`\n' + BAD + '\n`', '\n'.join('    ' + line for line in BAD.splitlines()),
                     '\n'.join('> ' + line for line in BAD.splitlines())):
            with self.subTest(text=text[:30]):
                self.assertFalse(orphaned_tool_tail(text))

    def test_closed_fence_does_not_hide_later_malformed_output(self):
        self.assertTrue(orphaned_tool_tail('```python\nprint(1)\n```\n' + BAD))


class RepairHttp(Server):
    script = ['</think>\n' + BAD, CALL]

    def ask(self, stream, **extra):
        return self.post({"model": "qwen3.8-flash-next", "input": "Read the local fixture.",
                          "tools": [TOOL], "max_output_tokens": 2048, "stream": stream, **extra})

    def test_repair_returns_a_real_call_in_both_modes(self):
        for stream in (False, True):
            self.engine.turns = 0
            self.engine.requests.clear()
            code, result = self.ask(stream)
            self.assertEqual(code, 200)
            if stream:
                self.assertEqual(result[-1]['type'], 'response.completed')
                result = result[-1]['response']
            calls = [item for item in result['output'] if item['type'] == 'function_call']
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['name'], 'exec_command')
            self.assertEqual(json.loads(calls[0]['arguments']), {'cmd': 'python3 -c "print(1)"'})
            self.assertEqual(len(self.engine.requests), 2)
            self.assertIn('no tool was invoked', self.tok.decode(self.engine.requests[-1][0]))

    def test_second_malformed_reply_fails_without_completed_event(self):
        self.engine.scripts = [self.tok.encode('</think>\n' + BAD) + list(self.svc.stop_ids)]
        self.engine.script = self.engine.scripts[0]
        code, result = self.ask(False)
        self.assertEqual(code, 502)
        self.assertEqual(result['error']['code'], 'tool_output_invalid')
        self.assertEqual(len(self.engine.requests), 2)
        self.engine.requests.clear()
        code, result = self.ask(True)
        self.assertEqual(code, 200)
        self.assertEqual(result[-1]['type'], 'response.failed')
        self.assertFalse(any(e['type'] == 'response.completed' for e in result))
        self.assertEqual(len(self.engine.requests), 2)

    def test_no_retry_when_tools_are_disabled(self):
        code, result = self.ask(False, tool_choice='none')
        self.assertEqual(code, 200)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.engine.requests), 1)

    def test_an_already_emitted_call_is_not_retried(self):
        self.engine.script = self.tok.encode('</think>\n' + CALL + '\n' + BAD) + list(self.svc.stop_ids)
        self.engine.scripts = [self.engine.script]
        code, result = self.ask(False)
        self.assertEqual(code, 200)
        self.assertEqual(len([item for item in result['output'] if item['type'] == 'function_call']), 1)
        self.assertEqual(len(self.engine.requests), 1)

    def test_fenced_example_does_not_retry(self):
        self.engine.script = self.tok.encode('</think>\n```xml\n' + BAD + '\n```') + list(self.svc.stop_ids)
        self.engine.scripts = [self.engine.script]
        code, result = self.ask(True)
        self.assertEqual(code, 200)
        self.assertEqual(result[-1]['type'], 'response.completed')
        self.assertEqual(len(self.engine.requests), 1)

    def test_empty_or_incomplete_repair_fails(self):
        for reply in ('', '<tool_call>\n<function=exec_command>\n<parameter=cmd>\nunclosed'):
            self.engine.scripts = [self.tok.encode('</think>\n' + BAD) + list(self.svc.stop_ids),
                                   self.tok.encode(reply) + list(self.svc.stop_ids)]
            self.engine.turns = 0
            code, result = self.ask(True)
            self.assertEqual(code, 200)
            self.assertEqual(result[-1]['type'], 'response.failed')
            self.assertFalse(any(e['type'] == 'response.completed' for e in result))

    def test_explicit_stop_does_not_retry(self):
        code, result = self.ask(False, stop='</tool_call>')
        self.assertEqual(code, 200)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.engine.requests), 1)

    def test_stop_completed_by_the_final_newline_does_not_retry(self):
        self.engine.script = self.tok.encode('</think>\n' + BAD + '\n') + list(self.svc.stop_ids)
        self.engine.scripts = [self.engine.script]
        code, result = self.ask(False, stop='</tool_call>\n')
        self.assertEqual(code, 200)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(self.engine.requests), 1)

    def test_other_apis_repair_and_report_failure(self):
        bodies = {
            '/v1/chat/completions': {'messages': [{'role': 'user', 'content': 'Read a fixture.'}],
                                     'tools': [{'type': 'function', 'function': TOOL}]},
            '/v1/messages': {'messages': [{'role': 'user', 'content': 'Read a fixture.'}],
                             'tools': [{'name': TOOL['name'], 'input_schema': TOOL['parameters']}]},
        }
        for path, fields in bodies.items():
            for failure in (False, True):
                for stream in (False, True):
                    self.engine.turns = 0
                    self.engine.scripts = [self.tok.encode('</think>\n' + BAD) + list(self.svc.stop_ids),
                                           self.tok.encode(BAD if failure else CALL) + list(self.svc.stop_ids)]
                    body = dict(fields, model='qwen3.8-flash-next', max_tokens=2048, stream=stream)
                    conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
                    try:
                        conn.request('POST', path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
                        response = conn.getresponse()
                        raw = response.read().decode()
                        self.assertEqual(response.status, 502 if failure and not stream else 200)
                        if stream:
                            if failure:
                                self.assertIn('"error"', raw)
                                self.assertIn('orphaned tool-call tags', raw)
                            else:
                                self.assertIn('exec_command', raw)
                        elif failure:
                            result = json.loads(raw)
                            self.assertEqual(result['error']['type'],
                                             'api_error' if path.endswith('messages') else 'server_error')
                        else:
                            result = json.loads(raw)
                            if path.endswith('completions'):
                                self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
                            else:
                                self.assertEqual(result['stop_reason'], 'tool_use')
                    finally:
                        conn.close()


class Budget(unittest.TestCase):
    def service(self, context):
        tok = ByteTokenizer()
        engine = MockEngine(tok, BAD, max_context=context)
        svc = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        return svc, tok, engine

    def test_repair_does_not_overrun_context_or_output_budget(self):
        tok = ByteTokenizer()
        ids = tok.encode('prompt')
        bad = tok.encode(BAD)
        overhead = tok.encode(REPAIR_TURN, parse_special=True)
        context = len(ids) + len(bad) + len(overhead) + CTX_SLACK + 12
        svc, tok, engine = self.service(context)
        events = list(svc.run(ids, False, [TOOL], 1500, {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(len(engine.requests), 2)
        self.assertLessEqual(len(engine.requests[1][0]) + 12 + CTX_SLACK, context)
        self.assertLessEqual(svc.history[-1]['output_tokens'], 1500)
        self.assertFalse(svc.status['busy'])

    def test_no_context_room_fails_without_retry(self):
        svc, tok, engine = self.service(len(BAD.encode()) + 40)
        with self.assertRaises(ToolOutputError):
            list(svc.run(tok.encode('prompt'), False, [TOOL], 1000, {}, threading.Event()))
        self.assertEqual(len(engine.requests), 1)

    def test_counted_text_prompt_keeps_the_smaller_repair_cap(self):
        class TextEngine(MockEngine):
            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                if isinstance(ids, PromptText):
                    ids, max_new = ids.resolve(max_new, self.max_context)
                self.limits.append((len(ids), max_new))
                yield from super().generate(ids, max_new, sampling, cancel, embeddings)

        tok = ByteTokenizer()
        overhead = tok.encode(REPAIR_TURN, parse_special=True)
        context = len(tok.encode('prompt' + BAD)) + len(overhead) + CTX_SLACK + 12
        engine = TextEngine(tok, BAD, max_context=context)
        engine.limits = []
        svc = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))
        events = list(svc.run(PromptText(tok, 'prompt', clip=True), False, [TOOL], 2000, {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(len(engine.limits), 2)
        for prompt_n, cap in engine.limits:
            self.assertLessEqual(prompt_n + cap + CTX_SLACK, context)

    def test_repair_shares_the_requested_output_cap(self):
        svc, tok, engine = self.service(4096)
        cap = len(tok.encode(BAD)) + 8
        events = list(svc.run(tok.encode('prompt'), False, [TOOL], cap, {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLessEqual(events[-1][1]['completion_tokens'], cap)

    def test_client_cancel_does_not_retry(self):
        svc, tok, engine = self.service(4096)
        cancel = threading.Event()
        for kind, value in svc.run(tok.encode('prompt'), False, [TOOL], 1000, {}, cancel):
            if kind == 'event' and value.kind == 'content':
                cancel.set()
        self.assertEqual(len(engine.requests), 1)

    def test_parallel_repair_does_not_change_the_other_request(self):
        tok = ByteTokenizer()

        def route(ids):
            text = tok.decode(ids)
            return 0 if 'repair_me' in text and 'no tool was invoked' not in text else 1

        engine = MockEngine(tok, [BAD, CALL], max_context=4096, batch=2, delay_s=0.0001, route=route)
        svc = Service(engine, tok, ChatTemplate(ROOT / 'serve/chat_template.jinja'))

        def run(text):
            return list(svc.run(tok.encode(text), False, [TOOL], 2000, {}, threading.Event()))

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, ['repair_me', 'normal_me']))
        for events in results:
            self.assertEqual(len([v for k, v in events if k == 'event' and v.kind == 'tool_call']), 1)
            self.assertEqual(events[-1][1]['finish'], 'stop')
        self.assertEqual(len(engine.requests), 3)
        self.assertFalse(svc.live_reqs)


if __name__ == '__main__':
    unittest.main()
