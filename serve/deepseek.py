"""Basalt HTTP adapters for the DeepSeek V4.1 text protocol."""
from __future__ import annotations

import base64
import binascii
import io
import itertools
import math
import json
import queue
import re
import shutil
import time
import threading
import uuid

from serve import deepseek_encoding as codec
from serve.api.grammar import REQUEST_KEY as GRAMMAR_KEY, engine_key
from serve.api.structured import strict_json_loads
from serve.tool_output import DSML_PARAMETER_TAILS, DsmlOrphanFilter, ToolOutputError
from serve.engine.errors import EngineDied
from serve.engine.process import ProcessEngine, request_err
from serve.engine.vision import IMAGE_MAX_PIXELS, Vision, vision_card
from serve.frontend import Event, ToolCall
from serve.prompt import PromptText
from tokenizer import Tokenizer

CALL = "<｜DSML｜ calls>"
END = "</｜DSML｜ calls>"
INVOKE_HEADER = re.compile(r'\n<｜DSML｜ invoke name="([^"<>\r\n]+)">\n')
ORPHAN_DSML_HEAD = re.compile(r"\s*(?:" + "|".join(map(re.escape, DSML_PARAMETER_TAILS)) + ")")
VISION_TILE_TOKENS = 1024
VISION_IMAGE_TOKENS = 8192


def forced_prefix(text):
    if not text:
        return text
    name = re.search(r"<function=([^>]+)>", text)
    return CALL + ('\n<｜DSML｜ invoke name="' + name[1] + '">\n' if name else "\n<｜DSML｜ invoke")


class DeepSeekTokenizer(Tokenizer):
    def token_bytes(self, index):
        if 0 <= index < len(self.tokens) and self.token_types[index] in (3, 4):
            return self.tokens[index].encode("utf-8")
        return super().token_bytes(index)


def deepseek_image_size(width, height, budget=VISION_TILE_TOKENS):
    """Patch-aligned resize under the image token budget."""
    patch, cell = 14, 42
    area = width * height
    if not 0 < area <= IMAGE_MAX_PIXELS:
        raise ValueError("image dimensions exceed the decode limit")
    if area < 295936:
        scale = math.sqrt(295936 / area)
        width, height = max(1, int(width*scale)), max(1, int(height*scale))
    w, h = math.ceil(width/patch)*patch, math.ceil(height/patch)*patch
    if math.ceil(h/cell)*(math.ceil(w/cell)+1)+2 <= budget:
        return w, h
    aspect = height/width
    max_w = math.sqrt((budget-2)/aspect + .25) - .5
    max_h = max_w*aspect
    if max_w < 1:
        return cell, (budget-2)//2*cell
    if max_h < 1:
        return (budget-3)*cell, cell
    scale = min(math.floor(max_w)*cell/width, math.floor(max_h)*cell/height)
    return max(patch, math.floor(width*scale/patch)*patch), max(patch, math.floor(height*scale/patch)*patch)


class DeepSeekVision(Vision):
    def __init__(self, cfg, **kwargs):
        self.vram_gpu = vision_card(cfg.get("mode"))
        self.whole_image_tokens = cfg.get("whole_image_tokens")
        self._vision_memory = (0, time.monotonic())
        super().__init__(cfg, **kwargs)

    def _command(self, cfg):
        command = [cfg["vision_bin"], cfg["mmproj"]]
        if self.whole_image_tokens is not None:
            command += ["--max-tokens", str(self.whole_image_tokens)]
        return command

    def _encode_command(self, img, out):
        command = "ENCSTAT" if "mem-events" in self.ready_fields else "ENC"
        return f"{command} {img}\t{out}\n"

    def _readline(self, timeout, what):
        line = super()._readline(timeout, what)
        if line.startswith("MEM "):
            self._vision_memory = (int(line.split()[1]), time.monotonic())
            line = super()._readline(timeout, what)
        return line

    def _enc(self, img, out):
        try:
            return super()._enc(img, out)
        finally:
            if self._vision_memory[0]:
                self._vision_memory = (0, time.monotonic())

    def vram_usage(self):
        if self.vram_gpu is None or not self.alive() or "mem-events" not in self.ready_fields:
            return None
        return self.vram_gpu, *self._vision_memory

    @staticmethod
    def load(source):
        # Codex and the browser upload inline images; do not let HTTP clients read local paths.
        header, sep, payload = source.partition(",")
        if not sep or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("DeepSeek images must be uploaded as base64 data URLs")
        if len(payload) > 32 * 1024 * 1024:
            raise ValueError("image upload exceeds 24 MiB")
        try:
            return base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ValueError("invalid base64 image") from exc

    @staticmethod
    def tiles(data, whole_image_tokens=None):
        from PIL import Image, ImageOps
        try:
            with Image.open(io.BytesIO(data)) as source:
                if source.width * source.height > IMAGE_MAX_PIXELS:
                    raise ValueError("image dimensions exceed the decode limit")
                image = ImageOps.exif_transpose(source).convert("RGB")
                w, h = image.size
                ow, oh = deepseek_image_size(w, h, whole_image_tokens or VISION_TILE_TOKENS)

                def tile(part, size):
                    resized = ImageOps.pad(part, size, color=(127, 127, 127))
                    out = io.BytesIO()
                    resized.save(out, format="PNG")
                    return out.getvalue()

                yield tile(image, (ow, oh))
                if whole_image_tokens is not None or (w <= ow and h <= oh):
                    return
                max_tiles = VISION_IMAGE_TOKENS // VISION_TILE_TOKENS - 1
                detail = min(max_tiles, max(2, math.ceil(w * h / (ow * oh))))
                cols, rows = min(((x, y) for x in range(1, min(w, max_tiles) + 1)
                                  for y in range(1, min(h, max_tiles) + 1)
                                  if 2 <= x * y <= max_tiles),
                                 key=lambda g: abs(math.log(w * g[1] / (h * g[0])))
                                               + abs(math.log(g[0] * g[1] / detail)))
                for y in range(rows):
                    for x in range(cols):
                        part = image.crop((x * w // cols, y * h // rows, (x + 1) * w // cols, (y + 1) * h // rows))
                        yield tile(part, deepseek_image_size(*part.size))
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("invalid image") from exc

    def encode(self, images: list[bytes], *, token_budget: int):
        combined = self.dir / f"req-{uuid.uuid4().hex[:12]}.sve"
        counts = []
        try:
            with combined.open("wb") as out:
                for image in images:
                    count = 0
                    for tile in self.tiles(image, self.whole_image_tokens):
                        pins = []
                        try:
                            path, rows, _ = self._encode_one(tile, pins)
                            part_limit = self.whole_image_tokens or VISION_TILE_TOKENS
                            if not 0 < rows <= part_limit or count + rows > VISION_IMAGE_TOKENS:
                                raise ValueError("invalid DeepSeek vision tile size")
                            token_budget -= rows
                            if token_budget < 0:
                                raise ValueError("image tokens exceed the remaining request context")
                            with path.open("rb") as inp:
                                shutil.copyfileobj(inp, out, 1 << 20)
                            count += rows
                        finally:
                            with self.lock:
                                for key in pins:
                                    self.pinned[key] -= 1
                                    if not self.pinned[key]:
                                        del self.pinned[key]
                                self._evict()
                    counts.append(count)
        except BaseException:
            combined.unlink(missing_ok=True)
            raise
        return combined, counts


class DeepSeekTemplate:
    patch_repair_turn = (codec.eos_token + codec.USER_SP_TOKEN,
                         codec.ASSISTANT_SP_TOKEN + codec.thinking_end_token + "\n\n")
    native_grammar = True
    numeric_effort = True
    forced_tail_separate = True
    image_placeholder = codec.IMAGE_PLACEHOLDER

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs):
        items = []
        for msg in messages:
            item = dict(msg)
            item["content"] = item.get("content") or ""
            if item.get("role") == "tool":
                item["tool_call_id"] = item.pop("_call_id", None)
            if item.get("tool_calls"):
                item["tool_calls"] = [
                    {"id": call.get("_call_id"), "type": "function",
                     "function": {**call["function"],
                                  "arguments": json.dumps(call["function"]["arguments"], ensure_ascii=False)}}
                    for call in item["tool_calls"]]
            items.append(item)
        if tools:
            if not items or items[0]["role"] != "system":
                items.insert(0, {"role": "system", "content": ""})
            items[0]["tools"] = [{"type": "function", "function": t} for t in tools]
        effort = kwargs.get("reasoning_effort")
        mode = "chat" if kwargs.get("enable_thinking") is False or effort == 0 else "thinking"
        if mode == "chat":
            effort = None
        return codec.encode_messages(items, thinking_mode=mode, reasoning_effort=effort)


class DeepSeekParser:
    repair_turn = (
        codec.eos_token + codec.USER_SP_TOKEN +
        "Your tool call was incomplete and no tool ran. Reissue only the complete tool call with a listed tool. "
        "Start with <｜DSML｜ calls> and <｜DSML｜ invoke name=\"TOOL\"> before any parameters, "
        "then close the invoke and calls tags. Do not claim the tool ran." +
        codec.ASSISTANT_SP_TOKEN + codec.thinking_end_token + "\n\n"
    )

    def __init__(self, thinking=False, tools=None, stream_tools=True):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.dsml_scan = 0
        self.structured = False
        self.json_answer = None
        self.tools = {t["name"] for t in tools or []}
        self.orphan_filter = DsmlOrphanFilter()
        self.orphaned_tool = False
        by_member = {}
        for name in self.tools:
            _, separator, member = name.rpartition(".")
            if separator:
                by_member.setdefault(member, []).append(name)
        self.tool_aliases = {member: names[0] for member, names in by_member.items()
                             if len(names) == 1 and member not in self.tools}

    def pending_tool_name(self):
        if self.state != "dsml" or not (header := INVOKE_HEADER.match(self.buf)):
            return None
        name = self._tool_name(header[1])
        return name if name in self.tools else None

    def _tool_name(self, name):
        if "::" in name:
            return name.replace("::", ".", 1)
        return self.tool_aliases.get(name, name)

    def feed(self, delta):
        if self.orphaned_tool:
            return []
        events = []
        for ev in self._feed(delta):
            if self.tools and ev.kind == "content" and not self.json_answer:
                visible = self.orphan_filter.feed(ev.text)
                if visible:
                    events.append(Event("content", visible))
                if self.orphan_filter.found:
                    self.orphaned_tool = True
                    break
            else:
                if ev.kind == "tool_call" and (pending := self.orphan_filter.finish()):
                    events.append(Event("content", pending))
                events.append(ev)
        return events

    def _feed(self, delta):
        self.buf += delta
        out = []
        while self.buf:
            if self.state == "dsml":
                if self.tools and ORPHAN_DSML_HEAD.match(self.buf):
                    self.orphaned_tool = True
                    self.buf, self.state = "", "content"
                    break
                at = self.buf.find(END, self.dsml_scan)
                if at < 0:
                    self.dsml_scan = max(self.dsml_scan, len(self.buf)-len(END)+1)
                    break
                end = at + len(END)
                try:
                    _, stop, calls = codec.parse_tool_calls(len(CALL)-1, CALL+self.buf[:end])
                except ValueError:
                    # A JSON parameter may contain a quoted closing marker.
                    self.dsml_scan = at + 1
                    continue
                if not stop or not calls:
                    raise ToolOutputError("incomplete DeepSeek tool call")
                self.buf = self.buf[end:]
                self.state, self.dsml_scan = "content", 0
                for call in calls:
                    # Codex namespaces can be omitted by the model; only resolve a unique registered member.
                    name = self._tool_name("::".join(filter(None, (call.get("namespace"), call["name"]))))
                    if name not in self.tools:
                        raise ToolOutputError(f"unknown tool: {name}")
                    try:
                        args = strict_json_loads(call["arguments"])
                    except ValueError as exc:
                        raise ToolOutputError(f"invalid arguments for {name}") from exc
                    out.append(Event("tool_call", call=ToolCall(name, args)))
                continue
            if self.state == "content" and self.structured:
                if self.json_answer is None and self.buf.lstrip():
                    self.json_answer = self.buf.lstrip().startswith("{")
                if self.json_answer:
                    out.append(Event("content", self.buf))
                    self.buf = ""
                    break
            marker = "</think>" if self.state == "reasoning" else CALL
            at = self.buf.find(marker)
            if at >= 0:
                if at:
                    out.append(Event(self.state, self.buf[:at]))
                self.buf = self.buf[at+len(marker):]
                self.state = "content" if marker == "</think>" else "dsml"
                continue
            n = next((i for i in range(min(len(marker)-1, len(self.buf)), 0, -1)
                      if self.buf.endswith(marker[:i])), 0)
            end = len(self.buf)-n
            if end:
                out.append(Event(self.state, self.buf[:end]))
                self.buf = self.buf[end:]
            break
        return out

    def finish(self):
        buf, self.buf = self.buf, ""
        if self.orphaned_tool:
            return []
        if self.state == "dsml":
            raise ToolOutputError("DeepSeek stopped with an incomplete tool call")
        if self.state == "content" and self.tools and not self.json_answer:
            visible = self.orphan_filter.feed(buf)
            if self.orphan_filter.found:
                self.orphaned_tool = True
            buf = visible + self.orphan_filter.finish()
        return [Event(self.state, buf)] if buf else []


class DeepSeekEngine(ProcessEngine):
    """Route GEN replies by request ID; Basalt owns the physical slot assignment."""
    def _pump(self):
        proc, lines = self.proc, self.lines
        for raw in proc.stdout:
            line = raw.strip()
            kind, _, tail = line.partition(" ")
            if kind == "RAM":
                try:
                    parts = dict(entry.split(":", 1) for entry in tail.split())
                    usage = {key: int(value) for key, value in parts.items()}
                    if any(value < 0 for value in usage.values()):
                        raise ValueError("invalid host allocation")
                    with self.request_lock:
                        if self.proc is proc:
                            self._host_memory = usage
                except ValueError:
                    pass
                continue
            if kind == "MEM":
                try:
                    snapshot = {}
                    for entry in tail.split():
                        card, component, count = entry.split(":")
                        card, count = int(card), int(count)
                        if card < 0 or count < 0:
                            raise ValueError("invalid allocation")
                        parts = snapshot.setdefault(card, {})
                        parts[component] = parts.get(component, 0) + count
                    with self.request_lock:
                        if self.proc is proc:
                            self._memory = snapshot
                except ValueError:
                    pass
                continue
            rid, _, body = tail.partition(" ")
            if kind in ("PT", "PP", "T", "STAT", "DONE", "ERR") and rid.isdecimal():
                with self.request_lock:
                    target = self.requests.get(int(rid))
                    if target is not None:
                        target.put((kind, body))
        with self.request_lock:
            if self.proc is proc:
                self.ended = True
                self._memory = {}
                self._host_memory = {}
                for target in self.requests.values():
                    target.put(None)
        lines.put(None)

    def __init__(self, *args, **kwargs):
        self._memory = {}
        self._host_memory = {}
        if not hasattr(self, "request_lock"):
            self.request_lock = threading.Lock()
            self.requests = {}
            self.request_ids = itertools.count(1)
            self._decode_metrics = threading.local()
        super().__init__(*args, **kwargs)
        self.request_slots = threading.BoundedSemaphore(max(1, self.batch))
        self.info["logit_bias"] = 0
        if self.info.get("memory") == 1 and self.alive():
            self._send("MEMORY")

    def vram_usage(self):
        return self._memory

    def ram_usage(self):
        return self._host_memory

    def batch_slots(self):
        return int(self.info.get("slots") or 1)

    def decode_progress(self):
        return getattr(self._decode_metrics, "progress", None)

    @staticmethod
    def _routes(values):
        return dict(zip(("primary", "helper", "ram"), map(int, values), strict=True))

    def _parse_done(self, line):
        base, _, routes = line.partition(" experts=")
        super()._parse_done(base)
        if routes:
            self.last["expert_routes"] = self._routes(routes.split(","))

    @property
    def takes_text(self):
        return bool(self.info.get("text_prompt"))

    text_batch = True

    @staticmethod
    def sampling_keys(sampling):
        keys = ("temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                "frequency_penalty", "presence_penalty", "penalty_last_n")
        selected = {key: sampling[key] for key in keys if sampling.get(key) is not None}
        if selected.get("seed", 0) < 0:
            selected["seed"] = int(selected["seed"]) % (1 << 64)
        return ProcessEngine.sampling_keys(selected)

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self._decode_metrics.progress = None
        image_key = " image_file="+str(embeddings).encode("utf-8").hex() if embeddings else ""
        if not self.alive():
            raise EngineDied("DeepSeek engine is unavailable")
        sampling = sampling or {}
        if not self.info.get("deepseek_penalties") and any(
                sampling.get(k) not in (None, neutral) for k, neutral in
                (("repetition_penalty", 1), ("frequency_penalty", 0), ("presence_penalty", 0))):
            raise ValueError("this DeepSeek engine does not support sampling penalties")
        spec = sampling.get(GRAMMAR_KEY)
        if spec and self.info.get("deepseek_grammar") != 1:
            raise ValueError("grammar: this DeepSeek engine does not support constrained decoding")
        keys = self.sampling_keys(sampling) + engine_key(spec)
        text = ids if isinstance(ids, PromptText) else None
        if text is not None and (embeddings or not self.takes_text):
            ids, max_new = text.resolve(max_new, self.max_context)
            text = None
        slots = self.request_slots
        while not slots.acquire(timeout=.25):
            if cancel.is_set():
                return
        with self.request_lock:
            rid = next(self.request_ids)
            proc = self.proc
            target = self.requests[rid] = queue.Queue()
        done = False
        try:
            self.read_ms = 0.0
            self._reset_windows()
            if text is None:
                self._send(f"GEN {int(max_new)} rid={rid}{keys}{image_key} " + ",".join(map(str, ids)))
            else:
                size = len(text.text.encode("utf-8"))
                if size > self.max_context * 32 + (1 << 20):
                    done = True
                    raise ValueError("request exceeds the context window: the text prompt is too large")
                spans = text.byte_spans()
                if spans:
                    keys += " plain=" + ",".join(f"{a}:{b}" for a, b in spans)
                if text.clip:
                    keys += " clip=1"
                if text.tail:
                    keys += f" tail={len(text.tail.encode('utf-8'))}"
                try:
                    with self.wlock:
                        proc.stdin.write(f"GENT {size} {int(max_new)} rid={rid}{keys}\n{text.text}")
                        proc.stdin.flush()
                except (OSError, AttributeError):
                    raise EngineDied("DeepSeek engine closed its input") from None
            heard = beat = time.monotonic()
            while not cancel.is_set():
                try:
                    reply = target.get(timeout=.25)
                except queue.Empty:
                    now = time.monotonic()
                    if self.silence_s and now-heard > self.silence_s:
                        raise self._silent("DeepSeek engine stopped reporting progress")
                    if now-beat >= 10:
                        beat = now
                        yield None
                    continue
                if reply is None:
                    done = True
                    raise EngineDied("DeepSeek engine closed its output")
                kind, body = reply
                heard = beat = time.monotonic()
                if kind == "ERR":
                    done = True
                    if text is not None:
                        body = text.explain(body)
                        if body.startswith("request exceeds the context window"):
                            raise ValueError(body)
                    raise request_err(body)
                if kind == "PT" and text is not None:
                    n, cap = map(int, body.split())
                    text.set_counted(n, cap)
                    yield None
                if kind == "DONE":
                    done = True
                    self._parse_done("DONE " + body)
                    return
                if kind == "STAT":
                    accepted, offered, *routes = body.split()
                    self._decode_metrics.progress = {"drafts_accepted": int(accepted), "drafts_offered": int(offered),
                                                     "expert_routes": self._routes(routes)}
                    continue
                if kind == "T":
                    yield int(body)
                elif kind == "PP":
                    yield None
        finally:
            try:
                if not done and self.proc is proc and self.alive():
                    self._send(f"STOP {rid}")
                    deadline = time.monotonic()+5
                    while time.monotonic()<deadline:
                        try:
                            reply = target.get(timeout=.25)
                        except queue.Empty:
                            continue
                        if reply is None or reply[0] in ("DONE", "ERR"):
                            if reply is not None and reply[0] == "DONE":
                                self._parse_done("DONE " + reply[1])
                            break
            finally:
                with self.request_lock:
                    self.requests.pop(rid, None)
                slots.release()

DeepSeekTemplate.parser = DeepSeekParser
DeepSeekTemplate.forced_prefix = staticmethod(forced_prefix)
