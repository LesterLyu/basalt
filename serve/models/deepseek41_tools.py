"""Hold orphaned DSML control tags before they become streamed answer text."""
import re
from copy import copy

from serve.models import deepseek41_encoding as enc

_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_HEADS = (f"<{enc.dsml_token}", f"</{enc.dsml_token}", enc.dsml_token)

REPAIR_TURN = (
    enc.eos_token + enc.USER_SP_TOKEN +
    "Your previous tool call was incomplete, so no tool was invoked. "
    "Reissue the intended call using a listed tool and its parameter schema. "
    'Start with <｜DSML｜ calls> and <｜DSML｜ invoke name="TOOL"> before any '
    'parameters, then close every parameter, invoke and calls tag. '
    "Keep the call outside Markdown and prose. Do not claim it ran. "
    "If you intended to quote an example, describe it without literal DSML markers." +
    enc.ASSISTANT_SP_TOKEN + enc.thinking_end_token + "\n\n"
)


class OrphanFilter:
    """Hold unconsumed DSML markers anywhere in answer text, across deltas."""

    def __init__(self):
        self.found = False
        self.fence = self.head = self.pending = ""

    def _track_fences(self, text):
        for ch in text:
            if ch == "\n":
                m = _FENCE.match(self.head.rstrip("\r"))
                if m:
                    ticks, rest = m.groups()
                    if self.fence:
                        if ticks[0] == self.fence[0] and len(ticks) >= len(self.fence) and not rest.strip():
                            self.fence = ""
                    elif ticks[0] != "`" or "`" not in rest:
                        self.fence = ticks
                self.head = ""
            elif len(self.head) < 256:
                self.head += ch

    def feed(self, text):
        if self.found:
            return ""
        text = self.pending + text
        at = text.find(enc.dsml_token)
        if at >= 0:
            # Tag names and whitespace may be malformed; the marker alone is enough.
            text = re.sub(r"</?\s*$", "", text[:at])
            self.found, self.pending = True, ""
        else:
            keep = max((n for head in _HEADS for n in range(1, len(head))
                        if text.endswith(head[:n])), default=0)
            text, self.pending = (text[:-keep], text[-keep:]) if keep else (text, "")
        self._track_fences(text)
        return text

    def fenced_after(self, text):
        probe = copy(self)
        probe.feed(text)
        return bool(probe.fence)

    def finish(self):
        text, self.pending = self.pending, ""
        if "DSML" in text and any(tag.startswith(text) for tag in _HEADS):
            self.found = True
        return "" if self.found else text
