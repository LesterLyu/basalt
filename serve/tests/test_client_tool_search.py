"""Client-executed Responses tool discovery and replay."""
import json
import unittest
from pathlib import Path

from serve.api.responses import Assembler, ResponsesError, input_messages, request_tools, template_kwargs
from serve.frontend import ChatTemplate, Event, ToolCall
from serve.tests.test_responses import Server
from serve.engine.mock import MockEngine
from serve.models.deepseek41 import DeepSeek41, DeepseekTemplate
from serve.service import Service
from serve.util.text import ByteTokenizer

SEARCH = {"type": "tool_search", "execution": "client", "description": "Find tools.",
          "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                         "required": ["query"]}}
FOUND = {"type": "namespace", "name": "mcp__docs", "tools": [
    {"type": "function", "name": "search", "parameters": {
        "type": "object", "properties": {"query": {"type": "string"}}}}]}


class ClientSearch(unittest.TestCase):
    def test_leading_tool_metadata_preserves_developer_instruction_priority(self):
        for instructions in (None, "API rules"):
            with self.subTest(instructions=instructions):
                req = {"instructions": instructions, "input": [
                    {"type": "additional_tools", "role": "developer", "tools": [FOUND]},
                    {"role": "developer", "content": "Agent rules"},
                    {"role": "system", "content": "Policy"},
                    {"role": "user", "content": "Do the task"},
                    {"type": "additional_tools", "tools": [FOUND]},
                    {"role": "developer", "content": "Later notes"}]}
                messages = input_messages(req)
                expected = "Agent rules\n\nPolicy"
                if instructions:
                    expected = instructions + "\n\n" + expected
                self.assertEqual(messages[0], {"role": "system", "content": expected})
                self.assertEqual(messages[1], {"role": "user", "content": "Do the task"})
                self.assertIn("Additional callable tools", messages[2]["content"])
                self.assertEqual(messages[3], {"role": "user", "content": "Later notes"})
                self.assertIn("mcp__docs.search", [tool["name"] for tool in request_tools(req)[0]])
                self.assertIn("mcp__docs.search", [tool["name"] for tool in template_kwargs(req)["_prompt_tools"]])

    def test_compacted_tools_keep_the_deepseek_native_call_protocol(self):
        req = {"input": [{"type": "additional_tools", "tools": [FOUND]},
                         {"role": "developer", "content": "Use tools to do the work."},
                         {"role": "user", "content": "Continue the task."}]}
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, DeepseekTemplate(), dialect=DeepSeek41())
        text, _ = svc.render_text(input_messages(req), request_tools(req)[0], template_kwargs(req))
        self.assertIn("### Available Tool Schemas", text)
        self.assertIn('<｜DSML｜ invoke name="$TOOL_NAME">', text)
        self.assertIn("mcp__docs.search", text)
        self.assertNotIn("Additional callable tools", text)

    def test_plugin_discovery_preserves_the_rendered_conversation_prefix(self):
        templates = [(ChatTemplate.from_file(Path(__file__).resolve().parents[1] / "chat_template.jinja"), None),
                     (DeepseekTemplate(), DeepSeek41())]
        for template, dialect in templates:
            with self.subTest(template=template.source[:35]):
                tok = ByteTokenizer()
                svc = Service(MockEngine(tok, "ok"), tok, template, **({"dialect": dialect} if dialect else {}))
                req = {"input": [
                    {"type": "additional_tools", "tools": [dict(SEARCH, description="Find <think>tools</think>.")]},
                    {"role": "user", "content": "Find the plugin I need."}]}
                def rendered():
                    return svc.render_text(input_messages(req), request_tools(req)[0], template_kwargs(req))[0]
                def encoded():
                    return svc.encode_prompt(input_messages(req), request_tools(req)[0], template_kwargs(req))
                before = rendered()
                before_ids = encoded()
                req["input"] += [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Find docs."}]},
                                 {"type": "tool_search_call", "execution": "client", "call_id": "s1",
                                  "arguments": '{"query":"docs"}'},
                                 {"type": "tool_search_output", "execution": "client", "call_id": "s1", "tools": [FOUND]}]
                after = rendered()
                self.assertTrue(after.startswith(before))
                self.assertEqual(encoded()[:len(before_ids)], before_ids)
                self.assertIn("mcp__docs", after[len(before):])
                self.assertIn("mcp__docs.search", [t["name"] for t in request_tools(req)[0]])
                self.assertIn("mcp__docs.search", request_tools(req)[1])
                another = dict(FOUND, name="mcp__chain")
                req["input"] += [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": "Find chain tools."}]},
                                 {"type": "tool_search_call", "execution": "client", "call_id": "s2", "arguments": '{}'},
                                 {"type": "tool_search_output", "execution": "client", "call_id": "s2", "tools": [another]}]
                self.assertTrue(rendered().startswith(after))

    def test_deferred_only_tools_keep_thinking_on_later_turns(self):
        req = {"input": [{"type": "additional_tools", "tools": [FOUND]},
                         {"role": "user", "content": "first question"},
                         {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "previous thought"}]},
                         {"role": "assistant", "content": "first answer"},
                         {"role": "user", "content": "next question"}]}
        tok = ByteTokenizer()
        svc = Service(MockEngine(tok, "ok"), tok, DeepseekTemplate(), dialect=DeepSeek41())
        text, _ = svc.render_text(input_messages(req), request_tools(req)[0], template_kwargs(req))
        self.assertIn("previous thought", text)
        self.assertIn("mcp__docs", text)
        self.assertIn("### Available Tool Schemas", text)

    def assembler(self):
        _, names, _ = request_tools({"tools": [SEARCH]})
        return Assembler({}, "model", 10, names, False)

    def test_client_search_is_declared_but_hosted_search_is_skipped(self):
        tools, names, skipped = request_tools({"tools": [SEARCH, dict(SEARCH, execution="server")]})
        self.assertEqual([tool["name"] for tool in tools], ["tool_search"])
        self.assertEqual(names["tool_search"], (None, "tool_search", "tool_search"))
        self.assertEqual(skipped, ["tool_search"])

    def test_replay_loads_the_discovered_namespace(self):
        req = {"tools": [SEARCH], "input": [
            {"role": "user", "content": "Find documentation."},
            {"type": "tool_search_call", "execution": "client", "call_id": "s1",
             "arguments": {"query": "docs"}},
            {"type": "tool_search_output", "execution": "client", "call_id": "s1", "tools": [FOUND]}]}
        msgs = input_messages(req)
        self.assertEqual(msgs[1]["tool_calls"][0]["function"],
                         {"name": "tool_search", "arguments": {"query": "docs"}})
        self.assertEqual(json.loads(msgs[2]["content"]), [FOUND])
        tools, names, _ = request_tools(req)
        self.assertIn("mcp__docs.search", [tool["name"] for tool in tools])
        self.assertEqual(names["mcp__docs__search"], ("mcp__docs", "search", "function"))

    def test_additional_tool_items_are_declarations(self):
        req = {"input": [{"role": "user", "content": "Hi"},
                         {"type": "additional_tools", "tools": [FOUND]}]}
        messages = input_messages(req)
        self.assertEqual(messages[0], {"role": "user", "content": "Hi"})
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("mcp__docs", messages[1]["content"])
        self.assertIn('"parameters"', messages[1]["content"])
        self.assertIn("mcp__docs.search", [tool["name"] for tool in request_tools(req)[0]])

    def test_hosted_search_replay_is_rejected(self):
        for kind in ("tool_search_call", "tool_search_output"):
            with self.subTest(kind=kind), self.assertRaises(ResponsesError):
                input_messages({"input": [{"type": kind, "execution": "server"}]})

    def test_streamed_search_has_object_arguments_and_no_function_deltas(self):
        a = self.assembler()
        call = ToolCall("tool_search", {"query": "docs"}, id="s1")
        events = a.start()
        events += a.feed(Event("tool_start", call=call))
        events += a.feed(Event("tool_args", text='{"query":', call=call))
        events += a.feed(Event("tool_args", text='"docs"}', call=call))
        events += a.feed(Event("tool_call", call=call))
        events += a.finish({"finish": "stop"})
        item = events[-1]["response"]["output"][0]
        self.assertEqual(item["type"], "tool_search_call")
        self.assertEqual(item["execution"], "client")
        self.assertEqual(item["arguments"], {"query": "docs"})
        self.assertEqual(item["call_id"], "s1")
        self.assertNotIn("_arguments_text", item)
        self.assertFalse(any("function_call_arguments" in event["type"] for event in events))

    def test_whole_and_incomplete_calls(self):
        a = self.assembler()
        a.feed(Event("tool_call", call=ToolCall("tool_search", {"query": "x"})))
        self.assertEqual(a.finish({})[-1]["response"]["output"][0]["arguments"], {"query": "x"})
        a = self.assembler()
        call = ToolCall("tool_search", {})
        a.feed(Event("tool_start", call=call))
        a.feed(Event("tool_args", text='{"query":', call=call))
        response = a.finish({"finish": "length"})[-1]["response"]
        self.assertEqual(response["status"], "incomplete")
        self.assertEqual(response["output"][0]["status"], "incomplete")


class ClientSearchHttp(Server):
    script = ('Search.</think>\n<tool_call>\n<function=tool_search>\n'
              '<parameter=query>\ndocs\n</parameter>\n</function>\n</tool_call>')

    def test_search_can_precede_a_structured_final_answer(self):
        code, events = self.post({"model": "basalt", "input": "Find tools", "tools": [SEARCH], "stream": True,
                                 "text": {"format": {"type": "json_object"}}})
        self.assertEqual(code, 200, events)
        self.assertEqual(events[-1]["type"], "response.completed")
        response = events[-1]["response"]
        self.assertEqual(response["output"][-1]["type"], "tool_search_call")
        self.assertEqual(response["output"][-1]["arguments"], {"query": "docs"})
        self.assertNotIn("response.output_text.delta", [event["type"] for event in events])

    def test_search_stream_and_replay_loaded_tool(self):
        code, events = self.post({"model": "basalt", "input": "Find tools", "tools": [SEARCH], "stream": True})
        self.assertEqual(code, 200)
        response = events[-1]["response"]
        call = next(item for item in response["output"] if item["type"] == "tool_search_call")
        self.assertEqual(call["arguments"], {"query": "docs"})
        self.engine.script = self.tok.encode(
            '</think>\n<tool_call>\n<function=mcp__docs__search>\n'
            '<parameter=query>\nprofiles\n</parameter>\n</function>\n</tool_call>')
        code, response = self.post({"model": "basalt", "tools": [SEARCH], "input": [
            {"role": "user", "content": "Find tools"}, call,
            {"type": "tool_search_output", "execution": "client", "call_id": call["call_id"], "tools": [FOUND]}]})
        self.assertEqual(code, 200)
        item = next(item for item in response["output"] if item["type"] == "function_call")
        self.assertEqual((item["namespace"], item["name"]), ("mcp__docs", "search"))
        self.assertEqual(json.loads(item["arguments"]), {"query": "profiles"})


if __name__ == "__main__":
    unittest.main()
