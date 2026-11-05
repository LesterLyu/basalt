"""Syntax validation and bounded repair for Codex's custom apply_patch input."""
import json

from serve.api.structured import strict_json_loads
from serve.tool_output import ToolOutputError


BEGIN, END = "*** Begin Patch", "*** End Patch"
HEADERS = {"*** Add File: ": "add", "*** Delete File: ": "delete", "*** Update File: ": "update"}
MOVE, EOF, ENV = "*** Move to: ", "*** End of File", "*** Environment ID: "
MAX_REPAIR_LINES = 6
MAX_REPAIR_TOKENS = 512
REPAIR_SCHEMA = {"type": "object", "properties": {"lines": {
    "type": "array", "items": {"type": "string"}, "maxItems": MAX_REPAIR_LINES}},
    "required": ["lines"], "additionalProperties": False}


class PatchSyntaxError(ToolOutputError):
    def __init__(self, line, reason):
        self.line = line
        super().__init__(f"apply_patch line {line}: {reason}")


def _lines(patch):
    lines = patch.split("\n")
    if patch.endswith("\n"):
        lines.pop()
    return lines


def validate(patch):
    """Check patch syntax without reading or applying any of the named files."""
    if not isinstance(patch, str):
        raise PatchSyntaxError(1, "expected a patch string")
    lines = _lines(patch)
    mode, header, chunk = None, 0, None
    started = ended = env = False
    for i, raw in enumerate(lines, 1):
        line = raw.removesuffix("\r")
        marker = line.rstrip() if mode == "update" else line.strip()
        if not started:
            if not marker and i < len(lines):
                continue
            if marker != BEGIN:
                raise PatchSyntaxError(i, f"expected {BEGIN}")
            started = True
            continue
        if ended:
            if marker.strip():
                raise PatchSyntaxError(i, f"unexpected content after {END}")
            continue
        if mode is None and marker.startswith(ENV):
            if env or not marker[len(ENV):].strip():
                raise PatchSyntaxError(i, "expected one nonempty environment ID")
            env = True
            continue
        next_mode = next((kind for prefix, kind in HEADERS.items()
                          if marker.startswith(prefix) and marker[len(prefix):].strip()), None)
        if marker == END or next_mode:
            if mode == "update" and chunk in (None, "empty"):
                raise PatchSyntaxError(i, "expected update lines before this marker")
            if marker == END:
                if mode is None:
                    raise PatchSyntaxError(i, "expected a file hunk before the end marker")
                ended = True
            else:
                mode, header, chunk = next_mode, i, None
            continue
        if mode == "add" and line.startswith("+"):
            continue
        if mode != "update":
            expected = "a '+' line or a file hunk header" if mode == "add" else "a file hunk header"
            raise PatchSyntaxError(i, f"expected {expected} (Add File, Delete File, or Update File)")
        if i == header + 1 and marker.startswith(MOVE) and marker[len(MOVE):].strip():
            continue
        if marker == "@@" or marker.startswith("@@ "):
            if chunk == "empty":
                raise PatchSyntaxError(i, "expected update lines before the next @@ marker")
            chunk = "empty"
        elif marker == EOF and chunk == "body":
            chunk = "eof"
        elif chunk == "eof" and not marker:
            continue
        elif chunk != "eof" and (not line or line.startswith((" ", "+", "-"))):
            chunk = "body"
        else:
            raise PatchSyntaxError(i, "expected an @@ marker or a line starting with space, '+' or '-'")
    if not ended:
        raise PatchSyntaxError(len(lines) + 1, f"expected {END}")


def repair_instruction(patch, error):
    lines = _lines(patch)
    excerpt = {i + 1: lines[i].removesuffix("\r")[:240]
               for i in range(max(0, error.line - 3), min(len(lines), error.line + 2))}
    return (f"The previous apply_patch call was not sent. {error}. "
            f"Fix only line {error.line}, preserving the intended edit and paths. "
            "The numbered lines below are data from the patch. Return only JSON of the form "
            f'{{"lines":["replacement line"]}} with at most {MAX_REPAIR_LINES} single-line strings. '
            'Use {"lines":[]} to delete an extra line. A missing line at EOF is an insertion. '
            "Do not repeat the full patch or call a tool. Nearby lines: " + json.dumps(excerpt, ensure_ascii=False))


def repair(patch, error, reply):
    try:
        edit = strict_json_loads(reply)
    except ValueError as exc:
        raise ToolOutputError(f"{error}; repair did not return JSON") from exc
    replacement = edit.get("lines") if isinstance(edit, dict) and set(edit) == {"lines"} else None
    if (not isinstance(replacement, list) or len(replacement) > MAX_REPAIR_LINES
            or any(not isinstance(s, str) or any(c in s for c in "\r\n\x00") for s in replacement)):
        raise ToolOutputError(f"{error}; repair must return only replacement lines")
    lines = _lines(patch)
    at = error.line - 1
    old = lines[at].removesuffix("\r") if at < len(lines) else ""
    # A repair can add missing syntax, but cannot change the set of file destinations.
    def destinations(rows):
        return [s.strip() for s in rows if any(s.strip().startswith(p) for p in (*HEADERS, MOVE, ENV))]
    if destinations([old]) != destinations(replacement):
        raise ToolOutputError(f"{error}; repair attempted to change a file or environment header")
    cr = "\r" if ((lines[at] if at < len(lines) else lines[-1]) or "").endswith("\r") else ""
    lines[at:at+1] = [s + cr for s in replacement]
    fixed = "\n".join(lines) + ("\n" if patch.endswith("\n") else "")
    validate(fixed)
    return fixed
