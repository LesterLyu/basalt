import json
import io
import unittest

from serve.tool_output import ToolOutputError

from serve.engine import modelfile  # exposes the packaged tokenizer on sys.path
from serve.api.responses import ResponsesError, template_kwargs
from serve.deepseek import DeepSeekEngine, DeepSeekParser, DeepSeekTemplate, DeepSeekVision, deepseek_image_size, forced_prefix
from serve.frontend import anthropic_to_messages, forced_call, openai_to_messages


class DeepSeekProtocol(unittest.TestCase):
    tool = {"name": "get_weather", "parameters": {"type": "object"}}

    def test_fragmented_tool_call(self):
        _, prefix = forced_call({"type": "function", "name": "get_weather"}, [self.tool])
        raw = (forced_prefix(prefix)
               + '<｜DSML｜ parameter name="city" string="true">Toronto</｜DSML｜ parameter>\n'
                 '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
                 '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        for size in (1, 3, 9, len(raw)):
            parser = DeepSeekParser(tools=[self.tool])
            events = [e for i in range(0, len(raw), size) for e in parser.feed(raw[i:i+size])]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].call.name, "get_weather")
            self.assertEqual(events[0].call.arguments, {"city": "Toronto", "days": 2})

    def test_orphan_parameter_fragments_never_become_answer_text(self):
        tail = ('</｜DSML｜ parameter>\n'
                '<｜DSML｜ parameter name="workdir" string="true">/workspace</｜DSML｜ parameter>')
        for prefix in ("", "<｜DSML｜ calls>\n"):
            with self.subTest(prefix=prefix):
                parser = DeepSeekParser(tools=[self.tool])
                self.assertEqual([e for char in prefix + tail for e in parser.feed(char)] + parser.finish(), [])
                self.assertTrue(parser.orphaned_tool)
        for sample, tools in ((f"```xml\n{tail}\n```", [self.tool]), (tail, [])):
            with self.subTest(sample=sample):
                parser = DeepSeekParser(tools=tools)
                events = [e for char in sample for e in parser.feed(char)] + parser.finish()
                self.assertEqual("".join(e.text for e in events), sample)
                self.assertFalse(parser.orphaned_tool)

    def test_reports_pending_tool_without_emitting_a_call(self):
        parser = DeepSeekParser(tools=[{"name": "functions.apply_patch"}])
        opening = '<｜DSML｜ calls>\n<｜DSML｜ invoke name="apply_patch">\n'
        for char in opening:
            self.assertEqual(parser.feed(char), [])
        self.assertEqual(parser.pending_tool_name(), "functions.apply_patch")
        events = parser.feed('</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        self.assertEqual([e.call.name for e in events], ["functions.apply_patch"])
        self.assertIsNone(parser.pending_tool_name())

    def test_json_parameter_can_contain_dsml_closing_tags(self):
        value = {"code": "</｜DSML｜ calls> </｜DSML｜ parameter>"}
        _, prefix = forced_call({"type": "function", "name": "get_weather"}, [self.tool])
        raw = (forced_prefix(prefix) + '<｜DSML｜ parameter name="payload" string="false">'
               + json.dumps(value, ensure_ascii=False)
               + '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        parser = DeepSeekParser(tools=[self.tool])
        events = [ev for char in raw for ev in parser.feed(char)]
        self.assertEqual([ev.call.arguments for ev in events], [{"payload": value}])

    def test_truncated_and_duplicate_nested_keys_do_not_create_tool_calls(self):
        _, prefix = forced_call({"type": "function", "name": "get_weather"}, [self.tool])
        parser = DeepSeekParser(tools=[self.tool])
        self.assertEqual(parser.feed(forced_prefix(prefix)), [])
        with self.assertRaises(ToolOutputError):
            parser.finish()
        parser = DeepSeekParser(tools=[self.tool])
        with self.assertRaises(ToolOutputError):
            parser.feed(forced_prefix(prefix) + '<｜DSML｜ parameter name="payload" string="false">'
                        '{"x":1,"x":2}</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')

    def test_structured_answer_does_not_parse_dsml_inside_json(self):
        parser = DeepSeekParser(thinking=True, tools=[self.tool])
        parser.structured = True
        answer = '{"example":"<｜DSML｜ calls>"}'
        events = parser.feed("brief</think>\n" + answer[:7]) + parser.feed(answer[7:]) + parser.finish()
        self.assertEqual("".join(ev.text for ev in events if ev.kind == "content"), "\n" + answer)
        self.assertFalse(any(ev.call for ev in events))

    def test_reasoning_boundary(self):
        parser = DeepSeekParser(thinking=True)
        events = parser.feed("abc</thi") + parser.feed("nk>result") + parser.finish()
        self.assertEqual([(e.kind, e.text) for e in events], [("reasoning", "abc"), ("content", "result")])

    def test_numeric_reasoning_renders_model_effort_and_keeps_zero(self):
        template = DeepSeekTemplate()
        msg = [{"role": "user", "content": "Weather?"}]
        for requested, want in ((None, 75), (1, 1), (37, 37), (100, 100),
                                ("low", 50), ("medium", 62), ("high", 75), ("max", 100)):
            kw = template_kwargs({"reasoning": {"effort": requested}}, numeric_effort=True)
            prompt = template.render(msg, **kw)
            self.assertIn(f"Reasoning Effort: {want} ", prompt)
            self.assertTrue(prompt.endswith("<think>"))
        zero = {"enable_thinking": False}
        self.assertEqual(template_kwargs({"reasoning": {"effort": 0}}, numeric_effort=True), zero)
        self.assertEqual(openai_to_messages({"messages": msg, "reasoning_effort": 0,
                                             "reasoning": {"effort": 100}}, numeric_effort=True)[2], zero)
        self.assertEqual(anthropic_to_messages({"messages": msg, "output_config": {"effort": 0}},
                                               numeric_effort=True)[2], zero)
        off = template.render(msg, **zero)
        self.assertNotIn("Reasoning Effort:", off)
        self.assertTrue(off.endswith("</think>"))
        for value in (-1, 101, 4.5, True):
            with self.assertRaises(ResponsesError):
                template_kwargs({"reasoning": {"effort": value}}, numeric_effort=True)
        with self.assertRaises(ResponsesError):
            template_kwargs({"reasoning": {"effort": 37}})  # Qwen still takes named levels.

    def test_native_penalty_wire_values(self):
        keys = DeepSeekEngine.sampling_keys({"repetition_penalty": 1.1, "frequency_penalty": .6,
                                             "presence_penalty": -.2, "seed": -1})
        self.assertIn("penalty_repeat=1.1", keys)
        self.assertIn("penalty_freq=0.6", keys)
        self.assertIn("penalty_present=-0.2", keys)
        self.assertIn("penalty_last_n=64", keys)
        self.assertIn("seed=18446744073709551615", keys)
        self.assertIn("penalty_last_n=0", DeepSeekEngine.sampling_keys(
            {"repetition_penalty": 1.1, "penalty_last_n": 0}))

    def test_prompt_and_image_placeholder(self):
        template = DeepSeekTemplate()
        prompt = template.render([{"role": "user", "content": "Weather?"}], [self.tool], enable_thinking=False)
        self.assertIn("<｜DSML｜ calls>", prompt)
        self.assertTrue(prompt.endswith("<｜Assistant｜></think>"))
        image_prompt = template.render([{"role": "user", "content": [{"type": "image", "source": "image"}]}])
        self.assertEqual(image_prompt.count(template.image_placeholder), 1)

    def test_vision_resize_is_patch_aligned_and_bounded(self):
        self.assertEqual(deepseek_image_size(546, 546), (546, 546))
        for w, h in ((320, 200), (2560, 1440), (10000, 1), (1, 10000), (1000, 7000)):
            w, h = deepseek_image_size(w, h)
            self.assertEqual((w % 14, h % 14), (0, 0))
            self.assertLessEqual(((h+41)//42)*(((w+41)//42)+1)+2, 1024)

    def test_large_image_uses_detail_tiles_within_8192_token_budget(self):
        from PIL import Image

        def image_bytes(w, h):
            raw = io.BytesIO()
            Image.new("RGB", (w, h), "white").save(raw, format="PNG")
            return raw.getvalue()

        self.assertEqual(len(list(DeepSeekVision.tiles(image_bytes(320, 200)))), 1)
        sizes = [Image.open(io.BytesIO(png)).size for png in DeepSeekVision.tiles(image_bytes(3840, 2160))]
        counts = [((h + 41) // 42) * ((w + 41) // 42 + 1) + 2 for w, h in sizes]
        self.assertGreater(sum(counts), 1024)
        self.assertLessEqual(sum(counts), 8192)
        self.assertLessEqual(max(counts), 1024)

        # A 90x90 aligner grid plus row and image delimiters uses exactly 8192 tokens.
        whole = list(DeepSeekVision.tiles(image_bytes(3780, 3780), 8192))
        self.assertEqual(len(whole), 1)
        with Image.open(io.BytesIO(whole[0])) as image:
            self.assertEqual(image.size, (3780, 3780))
