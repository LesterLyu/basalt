"""Recognize a tool-call tail accidentally emitted as an ordinary answer."""
import re


class ToolOutputError(RuntimeError):
    pass


_TAIL = re.compile(r"(?m)^ {0,3}</parameter>[ \t]*\r?\n(?:[ \t]*\r?\n)*"
                   r" {0,3}</function>[ \t]*\r?\n(?:[ \t]*\r?\n)*"
                   r" {0,3}</tool_call>\s*\Z")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def orphaned_tool_tail(text: str) -> bool:
    match = _TAIL.search(text)
    if match is None:
        return False
    fence = ""
    for line in text[:match.start()].splitlines():
        found = _FENCE.match(line)
        if found is None:
            continue
        ticks, rest = found.groups()
        if fence:
            if ticks[0] == fence[0] and len(ticks) >= len(fence) and not rest.strip():
                fence = ""
        elif ticks[0] != "`" or "`" not in rest:
            fence = ticks
    return not fence


REPAIR_TURN = (
    "<|im_end|>\n<|im_start|>user\n"
    "Your previous reply ended with incomplete tool-call markup, so no tool was invoked. "
    "Reissue the intended call using a listed tool and its parameter schema, with the complete "
    "<tool_call><function=NAME><parameter=NAME>value</parameter></function></tool_call> structure. "
    "Keep the actual call outside Markdown and prose. Do not claim it ran. "
    "If you only intended to quote an example, put it in a fenced code block instead."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
