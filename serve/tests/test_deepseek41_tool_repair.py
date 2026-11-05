"""Malformed DSML gets one bounded repair, without replaying an announced call."""
import json
import threading
import unittest

from serve.engine.mock import MockEngine
from serve.handler import serve
from serve.models.deepseek41 import CALLS_START, CALLS_END, EOS, DeepSeek41, DeepseekParser, DeepseekTemplate
from serve.models.deepseek41_tools import REPAIR_TURN
from serve.prompt import CTX_SLACK
from serve.service import Service
from serve.tests.test_deepseek41_dialect import invoke
from serve.tests.test_responses import Server
from serve.tests.test_tool_output import TOOL
from serve.tool_output import ToolOutputError
from serve.util.text import ByteTokenizer

BAD = 'Command: `echo ok`\n</｜DSML｜ parameter>\n<｜DSML｜ parameter name="workdir" string="true">/workspace'
CALL = CALLS_START + '\n' + invoke('exec_command', cmd='echo ok') + CALLS_END


class Tokenizer(ByteTokenizer):
    SPECIALS = ByteTokenizer.SPECIALS + [EOS]
    ALWAYS = (EOS,)


def service(scripts, context=16384):
    tok = Tokenizer()
    engine = MockEngine(tok, [s + EOS for s in scripts], max_context=context)
    return Service(engine, tok, DeepseekTemplate(), dialect=DeepSeek41()), tok, engine


class Detection(unittest.TestCase):
    def test_orphan_tags_are_held_at_every_split(self):
        for bad in (BAD, '<｜DSML｜ parameter name="x" string="true">1',
                    '</｜DSML｜ invoke>', CALLS_START + '\n<｜DSML｜ parameter name="x"',
                    CALLS_START + '\n<｜DSML｜ invoke name="x"', '<｜DSML｜ parameter',
                    '<｜DSML｜ invoke>', '<｜DSML｜ parameter>', '<｜DSML｜ calls'):
            for cut in range(len(bad) + 1):
                with self.subTest(bad=bad, cut=cut):
                    p = DeepseekParser(False, [TOOL])
                    events = p.feed(bad[:cut]) + p.feed(bad[cut:]) + p.finish()
                    self.assertTrue(DeepSeek41().tool_repair_needed(p, ''))
                    self.assertFalse(any(e.kind.startswith('tool') for e in events))
                    self.assertNotIn('｜DSML｜', ''.join(e.text for e in events))

    def test_examples_are_not_orphan_calls(self):
        for text in ('```xml\n' + BAD + '\n```', '~~~xml\n' + BAD + '\n~~~',
                     '````xml\n```\n' + BAD, 'The literal </｜DSML｜ parameter> ends a value.',
                     '    </｜DSML｜ parameter>', '> </｜DSML｜ parameter>',
                     '```xml\n' + CALLS_START + '\n' + BAD + '\n```',
                     '```xml\n' + CALL + '\n```'):
            for chunks in ([text], list(text)):
                p = DeepseekParser(False, [TOOL])
                events = []
                for chunk in chunks:
                    events += p.feed(chunk)
                events += p.finish()
                self.assertEqual(''.join(e.text for e in events), text)
                self.assertFalse(any(e.kind.startswith('tool') for e in events))
                self.assertFalse(DeepSeek41().tool_repair_needed(p, text))

    def test_a_closed_fence_does_not_hide_later_orphan_tags(self):
        p = DeepseekParser(False, [TOOL])
        p.feed('```xml\nexample\n```\n' + BAD)
        self.assertTrue(p.orphaned_tool)

    def test_valid_calls_and_parameter_text_stream_unchanged(self):
        argument = 'echo ok\n<｜DSML｜ parameter name="literal" string="true">value'
        text = CALLS_START + '\n' + invoke('exec_command', cmd=argument) + CALLS_END
        p = DeepseekParser(False, [TOOL])
        events = []
        for ch in text:
            events += p.feed(ch)
        events += p.finish()
        self.assertFalse(p.orphaned_tool)
        self.assertFalse(p.incomplete_tool_call)
        calls = [e.call for e in events if e.kind == 'tool_call']
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {'cmd': argument})
        self.assertEqual(json.loads(''.join(e.text for e in events if e.kind == 'tool_args')), {'cmd': argument})

    def test_tools_disabled_leaves_orphan_text_alone(self):
        p = DeepseekParser(False, [])
        events = p.feed(BAD) + p.finish()
        self.assertEqual(''.join(e.text for e in events), BAD)
        self.assertFalse(p.orphaned_tool)


class RepairHttp(Server):
    def setUp(self):
        self.svc, self.tok, self.engine = service(['</think>\n' + BAD, CALL])
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def scripts(self, *texts):
        self.engine.scripts = [self.tok.encode(t + EOS, parse_special=True) for t in texts]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0
        self.engine.requests.clear()

    def ask(self, stream=False, **extra):
        return self.post(dict(model='basalt', input='Read the fixture.', tools=[TOOL],
                              stream=stream, max_output_tokens=2048, **extra))

    def test_repair_returns_one_real_call_and_hides_bad_markup(self):
        for stream in (False, True):
            self.scripts('</think>\n' + BAD, CALL)
            code, result = self.ask(stream)
            self.assertEqual(code, 200)
            if stream:
                self.assertEqual(result[-1]['type'], 'response.completed')
                self.assertNotIn('｜DSML｜', ''.join(e.get('delta', '') for e in result
                                                  if e['type'] == 'response.output_text.delta'))
                result = result[-1]['response']
            calls = [o for o in result['output'] if o['type'] == 'function_call']
            self.assertEqual(len(calls), 1)
            self.assertEqual(json.loads(calls[0]['arguments']), {'cmd': 'echo ok'})
            self.assertEqual(len(self.engine.requests), 2)
            prompt = self.tok.decode(self.engine.requests[-1][0])
            self.assertIn(REPAIR_TURN, prompt)
            self.assertNotIn('<|im_start|>', prompt)
            self.assertNotIn('<tool_call>', prompt)

    def test_repeated_malformed_or_empty_repair_fails_once(self):
        for retry in (BAD, '', CALLS_START + '\n<｜DSML｜ parameter name="x"'):
            for stream in (False, True):
                self.scripts('</think>\n' + BAD, retry)
                code, result = self.ask(stream)
                self.assertEqual(code, 200 if stream else 502)
                if stream:
                    self.assertEqual(result[-1]['type'], 'response.failed')
                    self.assertFalse(any(e['type'] == 'response.completed' for e in result))
                else:
                    self.assertEqual(result['error']['code'], 'tool_output_invalid')
                self.assertEqual(len(self.engine.requests), 2)

    def test_announced_incomplete_call_fails_without_replay(self):
        for stream in (False, True):
            self.scripts('</think>' + CALLS_START + '\n<｜DSML｜ invoke name="exec_command">\n'
                         '<｜DSML｜ parameter name="cmd" string="true">echo')
            code, result = self.ask(stream)
            self.assertEqual(code, 200 if stream else 502)
            if stream:
                self.assertEqual(result[-1]['type'], 'response.failed')
            self.assertEqual(len(self.engine.requests), 1)

    def test_completed_call_is_never_replayed(self):
        self.scripts('</think>' + CALL + '\n' + BAD)
        code, result = self.ask()
        self.assertEqual(code, 200)
        self.assertEqual(len([o for o in result['output'] if o['type'] == 'function_call']), 1)
        self.assertEqual(len(self.engine.requests), 1)

    def test_disabled_tools_and_explicit_stop_do_not_retry(self):
        for extra in ({'tool_choice': 'none'}, {'stop': 'Command:'}):
            self.scripts('</think>\n' + BAD)
            code, result = self.ask(**extra)
            self.assertEqual(code, 200)
            self.assertEqual(len(self.engine.requests), 1)

    def test_other_apis_repair_in_streaming_and_nonstreaming_modes(self):
        for path, fields in (
                ('/v1/chat/completions', {'tools': [{'type': 'function', 'function': TOOL}]}),
                ('/v1/messages', {'tools': [{'name': TOOL['name'], 'input_schema': TOOL['parameters']}]})):
            for stream in (False, True):
                self.scripts('</think>\n' + BAD, CALL)
                if stream:
                    # These APIs use their own SSE framing, not the Responses fixture's parser.
                    import http.client
                    conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
                    try:
                        body = dict(model='basalt', messages=[{'role': 'user', 'content': 'Read it.'}],
                                    max_tokens=2048, stream=True, **fields)
                        conn.request('POST', path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
                        response = conn.getresponse()
                        raw = response.read().decode()
                        self.assertEqual(response.status, 200)
                        self.assertIn('exec_command', raw)
                        self.assertNotIn('tool_output_invalid', raw)
                    finally:
                        conn.close()
                else:
                    code, result = self.post(dict(model='basalt', messages=[{'role': 'user', 'content': 'Read it.'}],
                                                  max_tokens=2048, **fields), path=path)
                    self.assertEqual(code, 200)
                    if path.endswith('completions'):
                        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
                    else:
                        self.assertEqual(result['stop_reason'], 'tool_use')
                self.assertEqual(len(self.engine.requests), 2)


class Budget(unittest.TestCase):
    def test_cancel_never_retries(self):
        svc, tok, engine = service([BAD])
        cancel = threading.Event()
        for kind, ev in svc.run(tok.encode('prompt'), False, [TOOL], 2048, {}, cancel):
            if kind == 'event' and ev.kind == 'content':
                cancel.set()
        self.assertEqual(len(engine.requests), 1)

    def test_context_and_output_cap_are_shared(self):
        overhead = len(Tokenizer().encode(REPAIR_TURN, parse_special=True))
        svc, tok, engine = service([BAD, CALL], context=overhead + CTX_SLACK + 200)
        events = list(svc.run(tok.encode('prompt'), False, [TOOL], 1500, {}, threading.Event()))
        self.assertEqual(len(engine.requests), 2)
        self.assertLessEqual(events[-1][1]['completion_tokens'], 1500)
        self.assertLessEqual(len(engine.requests[-1][0]) + CTX_SLACK, engine.max_context)

    def test_no_context_room_fails_before_retry(self):
        svc, tok, engine = service([BAD], context=CTX_SLACK + 200)
        with self.assertRaises(ToolOutputError):
            list(svc.run(tok.encode('prompt'), False, [TOOL], 2048, {}, threading.Event()))
        self.assertEqual(len(engine.requests), 1)


if __name__ == '__main__':
    unittest.main()
