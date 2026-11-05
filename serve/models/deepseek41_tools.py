"""Hold orphaned DSML control tags before they become streamed answer text."""
import re
from copy import copy

from serve.models import deepseek41_encoding as enc

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADS = tuple(f"<{slash}{enc.dsml_token}{tag}{end}" for slash, tag, end in (
    ("", enc.tool_parameter_tag_name, " "), ("", enc.tool_parameter_tag_name, ">"),
    ("/", enc.tool_parameter_tag_name, ">"),
    ("", enc.tool_call_tag_name, " "), ("", enc.tool_call_tag_name, ">"),
    ("/", enc.tool_call_tag_name, ">"), ("", enc.tool_calls_block_name, ">"),
    ("/", enc.tool_calls_block_name, ">")))

REPAIR_TURN = (
    enc.eos_token + enc.USER_SP_TOKEN +
    "Your previous tool call was incomplete, so no tool was invoked. "
    "Reissue the intended call using a listed tool and its parameter schema. "
    'Start with <｜DSML｜ calls> and <｜DSML｜ invoke name="TOOL"> before any '
    'parameters, then close every parameter, invoke and calls tag. '
    "Keep the call outside Markdown and prose. Do not claim it ran. "
    "If you intended to quote an example, use a fenced code block instead." +
    enc.ASSISTANT_SP_TOKEN + enc.thinking_end_token + "\n\n"
)


class OrphanFilter:
    """Recognize control tags at a line's start, outside Markdown fences, across deltas."""

    def __init__(self):
        self.found = False
        self.fence = self.head = self.pending = ""
        self.checking = True

    def feed(self, text):
        out = []
        for ch in text:
            if self.found:
                break
            if ch == "\n":
                m = _FENCE.match(self.head.rstrip("\r"))
                if m:
                    ticks, rest = m.groups()
                    if self.fence:
                        if ticks[0] == self.fence[0] and len(ticks) >= len(self.fence) and not rest.strip():
                            self.fence = ""
                    elif ticks[0] != "`" or "`" not in rest:
                        self.fence = ticks
                out.extend((self.pending, ch))
                self.head = self.pending = ""
                self.checking = not self.fence
                continue
            if len(self.head) < 256:
                self.head += ch
            if not self.checking:
                out.append(ch)
                continue
            self.pending += ch
            marker = self.pending.lstrip(" ")
            if len(self.pending) - len(marker) <= 3:
                if marker in _HEADS:
                    self.found, self.pending = True, ""
                    break
                if any(tag.startswith(marker) for tag in _HEADS):
                    continue
            out.append(self.pending)
            self.pending, self.checking = "", False
        return "".join(out)

    def fenced_after(self, text):
        probe = copy(self)
        probe.feed(text)
        return bool(probe.fence)

    def finish(self):
        text, self.pending = self.pending, ""
        if enc.dsml_token in text and any(tag.startswith(text.lstrip(" ")) for tag in _HEADS):
            self.found = True
        return "" if self.found else text
