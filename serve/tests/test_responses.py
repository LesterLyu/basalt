"""The Responses API (POST /v1/responses) against the mock engine (no GPU, no pack)."""
from __future__ import annotations

import base64
import contextlib
import http.client
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.api.responses import (ENCRYPTED_PREFIX, ResponsesError, input_messages, request_tools,  # noqa: E402
                             template_kwargs, text_format)
from serve.api.structured import jsonschema_modules  # noqa: E402
from serve.engine import modelfile  # noqa: E402, F401 (adds the packaged tokenizer to sys.path)
from serve.engine.mock import MockEngine  # noqa: E402
from serve.handler import serve  # noqa: E402
from serve.service import Service  # noqa: E402
from serve.util.text import ByteTokenizer  # noqa: E402

HAVE_JSONSCHEMA = jsonschema_modules() is not None

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")

CALL = ("Let me look.\n</think>\n\nChecking.\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\ncat a.txt\n"
        "</parameter>\n</function>\n</tool_call>")
ANSWER = "Read it.\n</think>\n\nThe file says before."
TOOLS = [{"type": "function", "name": "exec_command", "description": "Runs a command.", "strict": False,
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}]


# ------------------------------------------------------------------------------------------------ parsing
class Parsing(unittest.TestCase):
    def test_a_string_input_is_one_user_message(self):
        self.assertEqual(input_messages({"model": "qwen3.8-flash-next", "input": "hi", "instructions": "Be brief."}),
                         [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}])

    def test_leading_developer_messages_join_the_system_message_later_ones_become_user(self):
        msgs = input_messages({"instructions": "I", "input": [
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "D1"},
                                                                 {"type": "input_text", "text": "D2"}]},
            {"role": "user", "content": "u"},
            {"type": "message", "role": "developer", "content": "late"}]})
        self.assertEqual(msgs, [{"role": "system", "content": "I\n\nD1D2"}, {"role": "user", "content": "u"},
                                {"role": "user", "content": "late"}])

    def test_codex_history_becomes_one_assistant_turn_per_answer(self):
        items = [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "edit it"}]},
                 {"type": "reasoning", "id": "rs_1", "summary": [],
                  "content": [{"type": "reasoning_text", "text": "look first"}]},
                 {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Looking."}]},
                 {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "exec_command",
                  "arguments": "{\"cmd\":\"cat a\"}"},
                 {"type": "function_call", "id": "fc_2", "call_id": "c2", "name": "spawn_agent",
                  "namespace": "multi_agent_v1", "arguments": "{}"},
                 # results in the order they finished, not the order of the calls
                 {"type": "function_call_output", "call_id": "c2", "output": "agent"},
                 {"type": "function_call_output", "call_id": "c1",
                  "output": [{"type": "input_text", "text": "before"}]},
                 {"type": "reasoning", "id": "rs_2", "summary": [{"type": "summary_text", "text": "s"}],
                  "content": [{"type": "reasoning_text", "text": "now write"}]},
                 {"type": "function_call", "call_id": "c3", "name": "exec_command", "arguments": "{\"cmd\":\"w\"}"},
                 {"type": "function_call_output", "call_id": "c3", "output": "ok"}]
        self.assertEqual(input_messages({"input": items}), [
            {"role": "user", "content": "edit it"},
            {"role": "assistant", "content": "Looking.", "reasoning_content": "look first", "tool_calls": [
                {"function": {"name": "exec_command", "arguments": {"cmd": "cat a"}}},
                {"function": {"name": "multi_agent_v1.spawn_agent", "arguments": {}}}]},
            {"role": "tool", "content": "before"}, {"role": "tool", "content": "agent"},
            {"role": "assistant", "content": "", "reasoning_content": "now write", "tool_calls": [
                {"function": {"name": "exec_command", "arguments": {"cmd": "w"}}}]},
            {"role": "tool", "content": "ok"}])

    def test_reasoning_comes_back_from_our_encrypted_content_and_foreign_tokens_are_ignored(self):
        enc = ENCRYPTED_PREFIX + base64.b64encode(json.dumps({"text": "hidden"}).encode()).decode()
        for item, want in (({"encrypted_content": enc, "summary": []}, "hidden"),
                           ({"encrypted_content": "gAAAAAB-someone-elses", "summary": []}, None)):
            msgs = input_messages({"input": [{"role": "user", "content": "q"}, {"type": "reasoning", **item},
                                             {"role": "assistant", "content": "a"}]})
            self.assertEqual(msgs[1].get("reasoning_content"), want)
        with self.assertRaises(ResponsesError) as e:
            input_messages({"input": [{"type": "reasoning", "encrypted_content": ENCRYPTED_PREFIX + "!!"}]})
        self.assertEqual(e.exception.param, "input[0].encrypted_content")

    def test_bad_arguments_from_a_cut_off_call_do_not_refuse_the_conversation(self):
        msgs = input_messages({"input": [{"role": "user", "content": "q"},
                                         {"type": "function_call", "call_id": "c", "name": "f", "arguments": "{\"a"},
                                         {"type": "function_call_output", "call_id": "c", "output": "x"}]})
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["arguments"], {"arguments": "{\"a"})

    def test_unsupported_items_are_named(self):
        for items, param in (([{"type": "item_reference", "id": "msg_1"}], "input[0]"),
                             ([{"type": "web_search_call"}], "input[0].type"),
                             ([{"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}],
                              "input[0].content[0].type"),
                             ([{"role": "robot", "content": "x"}], "input[0].role")):
            with self.assertRaises(ResponsesError) as e:
                input_messages({"input": items})
            self.assertEqual(e.exception.param, param)
        with self.assertRaises(ResponsesError):
            input_messages({})

    def test_tools_namespaces_custom_and_hosted(self):
        tools, names, skipped = request_tools({"tools": [
            *TOOLS, {"type": "namespace", "name": "mcp_fs", "description": "Files.",
                     "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}]},
            {"type": "custom", "name": "apply_patch", "description": "Patch.",
             "format": {"type": "grammar", "syntax": "lark", "definition": "start: \"x\""}},
            {"type": "web_search"}]})
        self.assertEqual([t["name"] for t in tools], ["exec_command", "mcp_fs.read", "apply_patch"])
        self.assertEqual(tools[1]["description"], "Files.")
        self.assertIn("start: \"x\"", tools[2]["description"])
        self.assertEqual(names["mcp_fs.read"], ("mcp_fs", "read", "function"))
        self.assertEqual(names["apply_patch"], (None, "apply_patch", "custom"))
        self.assertEqual(skipped, ["web_search"])
        self.assertIsNone(request_tools({"tools": TOOLS, "tool_choice": "none"})[0])

    def test_unqualified_names_resolve_only_unambiguous_namespace_tools(self):
        def namespace(name, kind="custom"):
            return {"type": "namespace", "name": name, "tools": [{"type": kind, "name": "apply_patch"}]}

        _, names, _ = request_tools({"tools": [namespace("functions")]})
        self.assertEqual(names["apply_patch"], ("functions", "apply_patch", "custom"))
        _, names, _ = request_tools({"tools": [namespace("first"), namespace("second")]})
        self.assertNotIn("apply_patch", names)
        for tools in ([namespace("functions"), {"type": "function", "name": "apply_patch"}],
                      [{"type": "function", "name": "apply_patch"}, namespace("functions")]):
            self.assertEqual(request_tools({"tools": tools})[1]["apply_patch"], (None, "apply_patch", "function"))

    def test_a_function_schema_that_is_not_an_object_is_refused(self):
        for bad in ("x", ["a"], 5, True):
            with self.assertRaises(ResponsesError) as e:
                request_tools({"tools": [{"type": "function", "name": "f", "parameters": bad}]})
            self.assertEqual(e.exception.param, "tools[0].parameters")
        tools = request_tools({"tools": [{"type": "function", "name": "f"},
                                         {"type": "function", "name": "g", "parameters": None}]})[0]
        self.assertEqual([t["parameters"]["type"] for t in tools], ["object", "object"])

    def test_effort_and_text_format(self):
        self.assertEqual(template_kwargs({"reasoning": {"effort": "minimal"}}), {"enable_thinking": False})
        self.assertEqual(template_kwargs({"reasoning": {"effort": "high"}}), {"reasoning_effort": "xhigh"})
        self.assertEqual(template_kwargs({"reasoning": {"summary": "auto"}}), {})      # the default
        with self.assertRaises(ResponsesError):
            template_kwargs({"reasoning": {"effort": "huge"}})
        self.assertEqual(text_format({"text": {"format": {"type": "json_schema", "name": "n", "schema": {}}}}),
                         {"type": "json_schema", "json_schema": {"name": "n", "schema": {}}})
        self.assertIsNone(text_format({"text": {"format": {"type": "text"}}}))


# ------------------------------------------------------------------------------------------------ over HTTP
class Server(unittest.TestCase):
    script = ANSWER

    def setUp(self):
        self.tok = ByteTokenizer()
        self.engine = MockEngine(self.tok, self.script, max_context=16384)
        self.svc = Service(self.engine, self.tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.port = self.httpd.server_address[1]

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def post(self, body, path="/v1/responses", headers=None):
        """(status, parsed body: a dict, or the list of SSE events of a stream)."""
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                c.request("POST", path, body=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", **(headers or {})})
                r = c.getresponse()
                raw = r.read().decode()
            if r.getheader("Content-Type") == "text/event-stream":
                events = []
                for block in raw.split("\n\n"):
                    lines = block.strip().splitlines()
                    if not lines or lines[0].startswith(":"):
                        continue
                    name = lines[0][len("event: "):]
                    data = json.loads(lines[1][len("data: "):])
                    self.assertEqual(name, data["type"])
                    events.append(data)
                return r.status, events
            return r.status, json.loads(raw)
        finally:
            c.close()



class OverHttp(Server):
    def test_deepseek_image_history_is_not_capped_at_eight(self):
        from serve.deepseek import DeepSeekTemplate

        self.svc.template = DeepSeekTemplate()

        class ImageTokenizer(ByteTokenizer):
            SPECIALS = ByteTokenizer.SPECIALS + [DeepSeekTemplate.image_placeholder]

        self.svc.tok = self.tok = ImageTokenizer()
        sources = ["data:image/png;base64," + base64.b64encode(str(i).encode()).decode() for i in range(10)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "request.sve"
            seen = []

            class Vision:
                def encode(self, images, *, token_budget):
                    seen.append(images)
                    if len(images) * 1024 > token_budget:
                        raise ValueError("image tokens exceed the remaining request context")
                    path.write_bytes(b"fixture")
                    return path, [1024] * len(images)

            self.svc.vision = Vision()
            history = [{"role": "user", "content": [
                {"type": "input_text", "text": f"Screenshot note {i}."},
                {"type": "input_image", "image_url": url},
            ]} for i, url in enumerate(sources)]
            # Codex can replay screenshots as tool results as well as user attachments.
            history[0:1] = [
                {"type": "function_call", "call_id": "snapshot", "name": "screenshot", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "snapshot", "output": history[0]["content"]},
            ]
            for count in (8, 10):
                with self.subTest(count=count):
                    code, response = self.post({"model": self.svc.model, "input": history[:count + 1], "stream": True})
                    self.assertEqual(code, 200, response)
                    self.assertEqual(response[-1]["type"], "response.completed")
                    self.assertEqual(seen[-1], sources[:count])
                    prompt = self.tok.decode(self.engine.last_prompt)
                    self.assertNotIn("Basalt omitted this image", prompt)
                    self.assertEqual(prompt.count(DeepSeekTemplate.image_placeholder), 1024 * count)
                    self.assertIn("Screenshot note 0.", prompt)
                    self.assertIn(f"Screenshot note {count - 1}.", prompt)
                    self.assertFalse(path.exists())

    def test_non_streaming_text_with_reasoning(self):
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "hi", "store": False})
        self.assertEqual(code, 200)
        self.assertEqual((r["object"], r["status"], r["error"]), ("response", "completed", None))
        rs, msg = r["output"]
        self.assertEqual(rs["type"], "reasoning")
        self.assertEqual(rs["content"], [{"type": "reasoning_text", "text": "Read it.\n"}])
        self.assertEqual(rs["summary"], [])
        self.assertNotIn("encrypted_content", rs)
        self.assertEqual(msg["type"], "message")
        self.assertEqual(msg["content"][0]["text"], "The file says before.")
        u = r["usage"]
        self.assertEqual(u["output_tokens"], len(self.tok.encode(ANSWER)) + 1)        # + the end-of-turn token
        self.assertEqual(u["output_tokens_details"]["reasoning_tokens"], len("Read it.\n</think>"))
        self.assertEqual(u["total_tokens"], u["input_tokens"] + u["output_tokens"])
        self.assertEqual(r["store"], False)

    def test_stream_event_order(self):
        code, events = self.post({"model": "qwen3.8-flash-next", "input": "hi", "stream": True,
                                  "include": ["reasoning.encrypted_content"]})
        self.assertEqual(code, 200)
        types = [e["type"] for e in events]
        deltas_r = types.count("response.reasoning_text.delta")
        deltas_t = types.count("response.output_text.delta")
        self.assertGreater(deltas_r, 0)
        self.assertGreater(deltas_t, 0)
        self.assertEqual(types, ["response.created", "response.in_progress",
                                 "response.output_item.added", *["response.reasoning_text.delta"] * deltas_r,
                                 "response.reasoning_text.done", "response.output_item.done",
                                 "response.output_item.added", "response.content_part.added",
                                 *["response.output_text.delta"] * deltas_t,
                                 "response.output_text.done", "response.content_part.done",
                                 "response.output_item.done", "response.completed"])
        self.assertEqual([e["sequence_number"] for e in events], list(range(len(events))))
        final = events[-1]["response"]
        self.assertEqual(final["status"], "completed")
        rs, msg = final["output"]
        self.assertEqual([e["item"]["id"] for e in events if e["type"] == "response.output_item.done"],
                         [rs["id"], msg["id"]])
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"),
                         msg["content"][0]["text"])
        self.assertTrue(rs["encrypted_content"].startswith(ENCRYPTED_PREFIX))
        # the encrypted_content alone brings the thinking back into the prompt
        replay = input_messages({"input": [{"role": "user", "content": "hi"},
                                           {"type": "reasoning", "summary": [], "encrypted_content":
                                            rs["encrypted_content"]}, msg]})
        self.assertEqual(replay[1]["reasoning_content"], "Read it.\n")

    def test_max_output_tokens_ends_incomplete(self):
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "hi", "max_output_tokens": 5})
        self.assertEqual(code, 200)
        self.assertEqual(r["status"], "incomplete")
        self.assertEqual(r["incomplete_details"], {"reason": "max_output_tokens"})
        self.assertEqual(r["output"][0]["status"], "incomplete")
        code, events = self.post({"model": "qwen3.8-flash-next", "input": "hi", "max_output_tokens": 5, "stream": True})
        self.assertEqual(events[-1]["type"], "response.incomplete")

    def test_effort_none_does_not_think(self):
        self.engine.scripts = [self.tok.encode("Hello.<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "hi", "reasoning": {"effort": "none"}})
        self.assertEqual([o["type"] for o in r["output"]], ["message"])
        self.assertIn("<think>\n\n</think>", self.tok.decode(self.engine.last_prompt))

    def test_errors_use_the_responses_format(self):
        for body, param in (({"model": "qwen3.8-flash-next"}, "input"), ({"model": "qwen3.8-flash-next", "input": "x", "previous_response_id": "r"},
                                                         "previous_response_id"),
                            ({"model": "qwen3.8-flash-next", "input": "x", "reasoning": {"effort": "huge"}}, "reasoning.effort")):
            code, r = self.post(body)
            self.assertEqual(code, 400)
            self.assertEqual(set(r["error"]), {"message", "type", "param", "code"})
            self.assertEqual(r["error"]["param"], param)
        code, r = self.post({}, path="/v1/responses/resp_123/cancel")
        self.assertEqual(code, 404)
        self.assertEqual(r["error"]["code"], "not_found")
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("POST", "/v1/responses", body=b"{not json", headers={"Content-Type": "application/json"})
        r = c.getresponse()
        err = json.loads(r.read())["error"]
        c.close()
        self.assertEqual((r.status, err["type"], err["param"]), (400, "invalid_request_error", None))

    def test_too_long_for_the_context(self):
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "x" * 20000})
        self.assertEqual(code, 400)
        self.assertEqual(r["error"]["code"], "context_length_exceeded")

    def test_api_key_and_foreign_pages(self):
        self.svc.api_key = "s3cret"
        self.assertEqual(self.post({"model": "qwen3.8-flash-next", "input": "hi"})[0], 401)
        self.assertEqual(self.post({"model": "qwen3.8-flash-next", "input": "hi"}, headers={"Authorization": "Bearer s3cret"})[0], 200)
        self.svc.api_key = None
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "hi"}, headers={"Origin": "https://evil.example.com"})
        self.assertEqual(code, 403)

    def test_json_schema_text_format(self):
        self.engine.scripts = [self.tok.encode("</think>\n\n{\"n\": 3}<|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        fmt = {"type": "json_schema", "name": "num", "strict": True,
               "schema": {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}}
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "a number", "text": {"format": fmt}})
        self.assertEqual(code, 200, r)
        self.assertEqual(r["output"][-1]["content"][0]["text"], "{\"n\": 3}")
        code, events = self.post({"model": "qwen3.8-flash-next", "input": "a number", "text": {"format": fmt}, "stream": True})
        self.assertEqual("".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"), "{\"n\": 3}")
        self.assertEqual(events[-1]["type"], "response.completed")

    def test_status_lists_the_dialect(self):
        self.post({"model": "qwen3.8-flash-next", "input": "hi", "stream": True})
        self.assertIn("/v1/responses", self.svc.v1_status()["dialects"])
        self.assertEqual(self.svc.history[-1]["path"], "/v1/responses")


class ToolsWithStructuredOutput(Server):
    fmt = {"type": "json_schema", "name": "num", "strict": True,
           "schema": {"type": "object", "properties": {"n": {"type": "integer"}},
                      "required": ["n"], "additionalProperties": False}}

    def scripts(self, *answers):
        self.engine.scripts = [self.tok.encode(answer + "<|im_end|>", parse_special=True)
                               for answer in answers]
        self.engine.script, self.engine.turns = self.engine.scripts[0], 0

    def test_final_json_with_available_tools(self):
        for fmt in (self.fmt, {"type": "json_object"}):
            for stream in (False, True):
                with self.subTest(format=fmt["type"], stream=stream):
                    self.scripts('</think>\n\n{"n": 3}')
                    code, result = self.post({"model": "qwen3.8-flash-next", "input": "a number", "tools": TOOLS,
                                              "text": {"format": fmt}, "stream": stream})
                    self.assertEqual(code, 200, result)
                    response = result[-1]["response"] if stream else result
                    self.assertEqual(response["status"], "completed")
                    self.assertEqual(response["output"][-1]["content"][0]["text"], '{"n":3}')
                    prompt = self.tok.decode(self.engine.last_prompt)
                    self.assertIn("exec_command", prompt)
                    self.assertIn("final answer", prompt)
                    grammar = self.engine.requests[-1][1]["_grammar"]
                    self.assertEqual(grammar["kind"], fmt["type"])
                    self.assertEqual(grammar["tools"][0]["function"]["name"], "exec_command")

    def test_tool_round_trip_validates_only_the_final_answer(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.scripts(CALL, '</think>\n\n{"n": 3}')
                user = {"role": "user", "content": "Read a.txt and return its number."}
                request = {"tools": TOOLS, "text": {"format": self.fmt}, "stream": stream}
                code, result = self.post({"model": "qwen3.8-flash-next", **request, "input": [user]})
                self.assertEqual(code, 200, result)
                response = result[-1]["response"] if stream else result
                self.assertEqual(response["status"], "completed")
                self.assertNotIn("message", [item["type"] for item in response["output"]])
                call = response["output"][-1]
                self.assertEqual((call["type"], call["status"]), ("function_call", "completed"))
                if stream:
                    self.assertNotIn("response.output_text.delta", [e["type"] for e in result])
                code, result = self.post({"model": "qwen3.8-flash-next", **request, "input": [user, *response["output"],
                    {"type": "function_call_output", "call_id": call["call_id"], "output": "3"}]})
                self.assertEqual(code, 200, result)
                response = result[-1]["response"] if stream else result
                self.assertEqual(response["output"][-1]["content"][0]["text"], '{"n":3}')

    def test_custom_tool_turn_does_not_need_a_final_json_answer(self):
        self.scripts('</think>\n\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n'
                     '*** Begin Patch\n*** Add File: test.txt\n+hello\n*** End Patch\n'
                     '</parameter>\n</function>\n</tool_call>')
        code, result = self.post({"model": "qwen3.8-flash-next", "input": "use the tool", "text": {"format": self.fmt},
                                 "tools": [{"type": "custom", "name": "apply_patch"}], "stream": True})
        self.assertEqual(code, 200, result)
        self.assertEqual(result[-1]["type"], "response.completed")
        self.assertEqual(result[-1]["response"]["output"][-1]["type"], "custom_tool_call")
        self.assertNotIn("response.output_text.delta", [e["type"] for e in result])

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is required for schema validation")
    def test_invalid_final_answer_after_a_tool_result_still_fails(self):
        self.scripts(CALL, '</think>\n\n{"n":"wrong"}')
        request = {"tools": TOOLS, "text": {"format": self.fmt}}
        user = {"role": "user", "content": "Read a.txt and return its number."}
        code, response = self.post({"model": "qwen3.8-flash-next", **request, "input": [user]})
        self.assertEqual(code, 200, response)
        call = response["output"][-1]
        code, result = self.post({"model": "qwen3.8-flash-next", **request, "input": [user, *response["output"],
            {"type": "function_call_output", "call_id": call["call_id"], "output": "3"}]})
        self.assertEqual((code, result["error"]["code"]), (502, "structured_output_failed"))

    @unittest.skipUnless(HAVE_JSONSCHEMA, "jsonschema is required for schema validation")
    def test_invalid_final_answers_with_tools_fail_without_streaming_text(self):
        for answer in ('not JSON', '{"n":"wrong"}', '{"n":3,"extra":true}', '{"n":3,"n":4}'):
            for stream in (False, True):
                with self.subTest(answer=answer, stream=stream):
                    self.scripts('</think>\n\n' + answer)
                    code, result = self.post({"model": "qwen3.8-flash-next", "input": "a number", "tools": TOOLS,
                                              "text": {"format": self.fmt}, "stream": stream})
                    if stream:
                        self.assertEqual(code, 200)
                        self.assertEqual(result[-1]["type"], "response.failed")
                        self.assertEqual(result[-1]["response"]["error"]["code"], "structured_output_failed")
                        self.assertNotIn("response.output_text.delta", [e["type"] for e in result])
                    else:
                        self.assertEqual((code, result["error"]["code"]), (502, "structured_output_failed"))

    def test_cut_off_tool_call_stays_incomplete(self):
        self.scripts(CALL)
        prefix = CALL[:CALL.index("cat a.txt") + 3]
        code, result = self.post({"model": "qwen3.8-flash-next", "input": "read the file", "tools": TOOLS,
                                 "text": {"format": self.fmt}, "stream": True,
                                 "max_output_tokens": len(self.tok.encode(prefix, parse_special=True))})
        self.assertEqual(code, 200, result)
        self.assertEqual(result[-1]["type"], "response.incomplete")
        self.assertEqual(result[-1]["response"]["output"][-1]["status"], "incomplete")


class ToolRoundTrip(Server):
    def test_patch_repair_replaces_one_line_before_streaming_to_codex(self):
        from serve.deepseek import DeepSeekTemplate

        fixed = ("*** Begin Patch\n*** Add File: first.txt\n" +
                 "".join(f"+original {i}\n" for i in range(253)) +
                 "*** Update File: second.txt\n@@\n unchanged\n-old\n+new\n*** End Patch")
        bad = fixed.replace("*** Update File:", "\n*** Update File:")
        tool = {"type": "namespace", "name": "functions", "tools": [{"type": "custom", "name": "apply_patch"}]}
        for native in (False, True):
            for repair_reply in ('{"lines":[]}', '{"lines":["*** Delete File: unrelated.txt"]}'):
                good = repair_reply == '{"lines":[]}'
                with self.subTest(native=native, valid_repair=good):
                    if native:
                        self.svc.template = DeepSeekTemplate()
                        script = ('</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="apply_patch">\n'
                                  '<｜DSML｜ parameter name="input" string="true">' + bad +
                                  '</｜DSML｜ parameter>\n</｜DSML｜ invoke>\n</｜DSML｜ calls>')
                    else:
                        self.svc.template = TEMPLATE
                        script = ("</think>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n" +
                                  bad + "\n</parameter>\n</function>\n</tool_call>")
                    self.engine.turns = 0
                    self.engine.requests.clear()
                    self.engine.scripts = [self.tok.encode(s + "<|im_end|>", parse_special=True)
                                           for s in (script, repair_reply)]
                    code, events = self.post({"model": self.svc.model, "input": "Apply the edit.",
                                              "tools": [tool], "stream": True, "max_output_tokens": 8192})
                    self.assertEqual(code, 200, events)
                    self.assertEqual(len(self.engine.requests), 2)
                    self.assertIn("apply_patch line 256:", self.tok.decode(self.engine.requests[1][0]))
                    if native:
                        self.assertEqual(self.engine.requests[1][1]["_grammar"]["kind"], "json_schema")
                    self.assertEqual(events[-1]["type"], "response.completed" if good else "response.failed")
                    sent = [e["delta"] for e in events if e["type"] == "response.custom_tool_call_input.delta"]
                    self.assertEqual(sent, [fixed] if good else [])
                    if not good:
                        self.assertFalse(any(e["type"] == "response.output_item.added" and
                                             e["item"]["type"] == "custom_tool_call" for e in events))
                        self.assertEqual(events[-1]["response"]["error"]["code"], "tool_output_invalid")

    def test_invalid_patch_with_no_repair_budget_fails_nonstreaming(self):
        script = ("</think>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n"
                  "*** Begin Patch\n*** Delete File: a.txt\n\n*** End Patch\n"
                  "</parameter>\n</function>\n</tool_call>")
        tokens = self.tok.encode(script, parse_special=True)
        self.engine.script = tokens
        self.engine.scripts = [tokens]
        code, result = self.post({"model": self.svc.model, "input": "Delete a.txt.",
                                  "tools": [{"type": "custom", "name": "apply_patch"}],
                                  "max_output_tokens": len(tokens)})
        self.assertEqual(code, 502, result)
        self.assertEqual(result["error"]["code"], "tool_output_invalid")
        self.assertIn("apply_patch line 3", result["error"]["message"])
        self.assertEqual(len(self.engine.requests), 1)

    def test_deepseek_retries_orphan_parameters_once(self):
        from serve.deepseek import DeepSeekTemplate

        self.svc.template = DeepSeekTemplate()
        bad = ('</think>\n</｜DSML｜ parameter>\n'
               '<｜DSML｜ parameter name="workdir" string="true">/workspace</｜DSML｜ parameter>')
        fixed = ('<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec_command">\n'
                 '<｜DSML｜ parameter name="cmd" string="true">pwd</｜DSML｜ parameter>\n'
                 '<｜DSML｜ parameter name="workdir" string="true">/workspace</｜DSML｜ parameter>\n'
                 '</｜DSML｜ invoke>\n</｜DSML｜ calls>')
        tool = {"type": "function", "name": "exec_command", "parameters": {"type": "object"}}
        for second, terminal in ((fixed, "response.completed"), (bad.removeprefix("</think>"), "response.failed")):
            with self.subTest(terminal=terminal):
                self.engine.turns = 0
                self.engine.requests.clear()
                self.engine.scripts = [self.tok.encode(reply + "<|im_end|>", parse_special=True)
                                       for reply in (bad, second)]
                code, events = self.post({"model": self.svc.model, "input": "Print the working directory.",
                                          "tools": [tool], "stream": True, "max_output_tokens": 1024})
                self.assertEqual(code, 200, events)
                self.assertEqual(events[-1]["type"], terminal)
                self.assertEqual(len(self.engine.requests), 2)
                self.assertIn("no tool ran", self.tok.decode(self.engine.requests[1][0]))
                self.assertNotIn("DSML", "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta"))
                result = events[-1]["response"]
                if terminal == "response.completed":
                    calls = [(item["name"], json.loads(item["arguments"])) for item in result["output"]
                             if item["type"] == "function_call"]
                    self.assertEqual(calls, [("exec_command", {"cmd": "pwd", "workdir": "/workspace"})])
                else:
                    self.assertEqual(result["error"]["code"], "tool_output_invalid")

    def test_deepseek_code_mode_accepts_only_an_unambiguous_short_name(self):
        from serve.deepseek import DeepSeekTemplate

        self.svc.template = DeepSeekTemplate()
        script = ('</think>\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="exec">\n'
                  '<｜DSML｜ parameter name="input" string="true">text(2 + 2)</｜DSML｜ parameter>\n'
                  '</｜DSML｜ invoke>\n</｜DSML｜ calls><|im_end|>')
        tokens = self.tok.encode(script, parse_special=True)
        def namespace(name):
            return {"type": "namespace", "name": name, "tools": [{"type": "custom", "name": "exec"}]}

        for namespaces, terminal in ((["functions"], "response.completed"),
                                     (["functions", "other"], "response.failed"),
                                     ([], "response.failed")):
            with self.subTest(namespaces=namespaces):
                self.engine.scripts = [tokens]
                self.engine.script = tokens
                code, events = self.post({"model": self.svc.model, "input": "calculate", "stream": True,
                                          "tools": [namespace(name) for name in namespaces]})
                self.assertEqual(code, 200, events)
                self.assertEqual(events[-1]["type"], terminal)
                result = events[-1]["response"]
                calls = [item for item in result["output"] if item["type"] == "custom_tool_call"]
                if len(namespaces) == 1:
                    self.assertEqual([(c["namespace"], c["name"], c["input"]) for c in calls],
                                     [("functions", "exec", "text(2 + 2)")])
                else:
                    self.assertFalse(calls)
                    self.assertEqual(result["error"]["code"], "tool_output_invalid")

    script = [CALL, ANSWER]

    def test_function_call_round_trip(self):
        user = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "what is in a.txt?"}]}
        code, events = self.post({"model": "qwen3.8-flash-next", "instructions": "You are a coding agent.", "input": [user],
                                  "tools": TOOLS, "store": False, "stream": True,
                                  "include": ["reasoning.encrypted_content"], "reasoning": {"summary": "auto"},
                                  "parallel_tool_calls": True, "tool_choice": "auto"})
        self.assertEqual(code, 200)
        first_prompt = self.tok.decode(self.engine.last_prompt)
        types = [e["type"] for e in events]
        n_args = types.count("response.function_call_arguments.delta")
        self.assertGreater(n_args, 0)
        call_part = types[types.index("response.output_text.done") + 2:]
        self.assertEqual(call_part, ["response.output_item.done", "response.output_item.added",
                                     *["response.function_call_arguments.delta"] * n_args,
                                     "response.function_call_arguments.done", "response.output_item.done",
                                     "response.completed"])
        output = events[-1]["response"]["output"]
        self.assertEqual([o["type"] for o in output], ["reasoning", "message", "function_call"])
        fc = output[2]
        self.assertEqual((fc["name"], json.loads(fc["arguments"]), fc["status"]),
                         ("exec_command", {"cmd": "cat a.txt"}, "completed"))
        self.assertTrue(fc["call_id"])
        added = next(e for e in events if e["type"] == "response.output_item.added"
                     and e["item"]["type"] == "function_call")
        self.assertEqual((added["item"]["call_id"], added["item"]["arguments"]), (fc["call_id"], ""))
        # the client sends everything back plus the result: the prompt continues where the model stopped
        code, r = self.post({"model": "qwen3.8-flash-next", "instructions": "You are a coding agent.", "tools": TOOLS, "store": False,
                             "input": [user, *output, {"type": "function_call_output", "call_id": fc["call_id"],
                                                       "output": "before"}]})
        self.assertEqual(code, 200, r)
        second_prompt = self.tok.decode(self.engine.last_prompt)
        self.assertTrue(second_prompt.startswith(first_prompt + CALL + "<|im_end|>"), second_prompt[-600:])
        self.assertIn("<tool_response>\nbefore\n</tool_response>", second_prompt)
        self.assertEqual(r["output"][-1]["content"][0]["text"], "The file says before.")

    def test_namespace_and_custom_calls_come_back_under_their_own_names(self):
        patch = "*** Begin Patch\n*** Add File: test.txt\n+hello\n*** End Patch"
        self.engine.scripts = [self.tok.encode(
            "</think>\n\n<tool_call>\n<function=multi_agent_v1.spawn_agent>\n<parameter=message>\nhi\n</parameter>\n"
            "</function>\n</tool_call>\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n" + patch +
            "\n</parameter>\n</function>\n</tool_call><|im_end|>", parse_special=True)]
        self.engine.script = self.engine.scripts[0]
        tools = [{"type": "namespace", "name": "multi_agent_v1", "description": "Agents.", "tools": [
            {"type": "function", "name": "spawn_agent", "strict": False,
             "parameters": {"type": "object", "properties": {"message": {"type": "string"}}}}]},
            {"type": "custom", "name": "apply_patch", "description": "Edit files."}]
        code, r = self.post({"model": "qwen3.8-flash-next", "input": "go", "tools": tools})
        self.assertEqual(code, 200, r)
        ns, custom = r["output"]
        self.assertEqual((ns["type"], ns["namespace"], ns["name"], json.loads(ns["arguments"])),
                         ("function_call", "multi_agent_v1", "spawn_agent", {"message": "hi"}))
        self.assertEqual((custom["type"], custom["name"], custom["input"]),
                         ("custom_tool_call", "apply_patch", patch))
        msgs = input_messages({"input": [{"role": "user", "content": "go"}, ns, custom,
                                         {"type": "function_call_output", "call_id": ns["call_id"], "output": "1"},
                                         {"type": "custom_tool_call_output", "call_id": custom["call_id"],
                                          "output": "2"}]})
        self.assertEqual([c["function"]["name"] for c in msgs[1]["tool_calls"]],
                         ["multi_agent_v1.spawn_agent", "apply_patch"])

    def test_unqualified_namespaced_patch_preserves_custom_payload_and_replay(self):
        patch = "*** Begin Patch\n*** Add File: test.txt\n+hello\n*** End Patch"
        script = ("</think>\n\n<tool_call>\n<function=apply_patch>\n<parameter=input>\n" + patch +
                  "\n</parameter>\n</function>\n</tool_call><|im_end|>")
        tool = {"type": "namespace", "name": "functions", "tools": [
            {"type": "custom", "name": "apply_patch", "description": "Edit files."}]}
        for lite, stream in ((False, False), (False, True), (True, False), (True, True)):
            with self.subTest(lite=lite, stream=stream):
                self.engine.scripts = [self.tok.encode(script, parse_special=True)]
                self.engine.script = self.engine.scripts[0]
                request = {"model": "qwen3.8-flash-next", "input": [{"role": "user", "content": "edit"}], "stream": stream}
                if lite:
                    request["input"].append({"type": "additional_tools", "tools": [tool]})
                else:
                    request["tools"] = [tool]
                code, result = self.post(request)
                self.assertEqual(code, 200, result)
                response = result[-1]["response"] if stream else result
                custom = response["output"][-1]
                self.assertEqual((custom["type"], custom["namespace"], custom["name"], custom["input"]),
                                 ("custom_tool_call", "functions", "apply_patch", patch))
                if stream:
                    deltas = [e["delta"] for e in result if e["type"] == "response.custom_tool_call_input.delta"]
                    self.assertEqual("".join(deltas), patch)
                    self.assertFalse(any("function_call_arguments" in e["type"] for e in result))
                messages = input_messages({"input": [{"role": "user", "content": "edit"}, custom,
                    {"type": "custom_tool_call_output", "call_id": custom["call_id"], "output": "ok"}]})
                self.assertEqual(messages[1]["tool_calls"][0]["function"],
                                 {"name": "functions.apply_patch", "arguments": {"input": patch}})


if __name__ == "__main__":
    unittest.main()
