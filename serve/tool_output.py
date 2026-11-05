"""Recognize a tool-call tail accidentally emitted as an ordinary answer."""
import re


class ToolOutputError(RuntimeError):
    pass


_TAIL = re.compile(r"(?m)^ {0,3}</parameter>[ \t]*\r?\n(?:[ \t]*\r?\n)*"
                   r" {0,3}</function>[ \t]*\r?\n(?:[ \t]*\r?\n)*"
                   r" {0,3}</tool_call>\s*\Z")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")


def _fence_after(line: str, fence: str) -> str:
    if found := _FENCE.match(line):
        ticks, rest = found.groups()
        if fence:
            if ticks[0] == fence[0] and len(ticks) >= len(fence) and not rest.strip():
                return ""
        elif ticks[0] != "`" or "`" not in rest:
            return ticks
    return fence


def orphaned_tool_tail(text: str) -> bool:
    match = _TAIL.search(text)
    if match is None:
        return False
    fence = ""
    for line in text[:match.start()].splitlines():
        fence = _fence_after(line, fence)
    return not fence


DSML_PARAMETER_TAILS = ("<｜DSML｜ parameter ", "</｜DSML｜ parameter>")


class DsmlOrphanFilter:
    """Hold possible orphan DSML prefixes at line starts until they are identified."""

    def __init__(self):
        self.found = False
        self.fence = ""
        self.head = ""
        self.pending = ""
        self.checking = True

    def feed(self, delta: str) -> str:
        out = []
        for ch in delta:
            if self.found:
                break
            if ch == "\n":
                self.fence = _fence_after(self.head.rstrip("\r"), self.fence)
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
                if marker in DSML_PARAMETER_TAILS:
                    self.found = True
                    self.pending = ""
                    break
                if any(tag.startswith(marker) for tag in DSML_PARAMETER_TAILS):
                    continue
            out.append(self.pending)
            self.pending = ""
            self.checking = False
        return "".join(out)

    def finish(self) -> str:
        text, self.pending = self.pending, ""
        return text


REPAIR_TURN = (
    "<|im_end|>\n<|im_start|>user\n"
    "Your previous reply ended with incomplete tool-call markup, so no tool was invoked. "
    "Reissue the intended call using a listed tool and its parameter schema, with the complete "
    "<tool_call><function=NAME><parameter=NAME>value</parameter></function></tool_call> structure. "
    "Keep the actual call outside Markdown and prose. Do not claim it ran. "
    "If you only intended to quote an example, put it in a fenced code block instead."
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
