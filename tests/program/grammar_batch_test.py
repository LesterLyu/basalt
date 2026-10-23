#!/usr/bin/env python3
"""Real-engine grammar isolation, slot reuse and solo/batch handovers.

    grammar_batch_test.py ENGINE MODEL.basalt [ENGINE ARGS...]

Needs a GPU and model pack. Runs two incompatible schemas concurrently, then moves one reply from solo to batch
and back to solo. No tools are executed. The engine log must show masked resampling in the batch windows.
"""
import json
import pathlib
import queue
import re
import subprocess
import sys
import tempfile
import threading

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from grammar_engine_test import PROMPT, load_tokenizer, text_of  # noqa: E402


class Protocol:
    def __init__(self, proc):
        self.proc = proc
        self.lines = queue.Queue()
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)

    def send(self, *lines):
        self.proc.stdin.write("\n".join(lines) + "\n")
        self.proc.stdin.flush()

    def line(self, timeout=120):
        line = self.lines.get(timeout=timeout)
        if line is None:
            raise RuntimeError("the engine ended")
        if line.startswith("ERR"):
            raise RuntimeError(line)
        return line

    def solo(self, command):
        self.send(command)
        tokens = []
        while True:
            line = self.line()
            if line.startswith("T "):
                tokens.append(int(line.split()[1]))
            elif line.startswith("DONE "):
                return tokens

    def pair(self, commands, cancel_first=False):
        self.send(*commands)
        tokens = [[], []]
        admitted, done, finish = set(), set(), {}
        control = 0
        cancelled = overlap = False
        widths = []
        while len(done) < 2:
            fields = self.line().split()
            if not fields:
                continue
            if fields[0] == "T":
                tokens[control].append(int(fields[1]))
            elif fields[0] == "BADM":
                slot = int(fields[1])
                if int(fields[2]) == 0:
                    done.add(slot)
                admitted.add(slot)
                control = 1
            elif fields[0] == "BT":
                slot = int(fields[1])
                tokens[slot].extend(map(int, fields[2:]))
                widths.append(len(fields) - 2)
                overlap |= len(admitted) == 2 and not done
                if cancel_first and slot == 0 and not cancelled:
                    self.send("BSTOP 0")
                    cancelled = True
            elif fields[0] == "BDONE":
                slot = int(fields[1])
                done.add(slot)
                finish[slot] = fields[3]
        return tokens, finish, overlap, max(widths, default=0)


def grammar(expected):
    schema = {"type": "object", "properties": {k: {"const": v} for k, v in expected.items()},
              "required": list(expected), "additionalProperties": False}
    return "grammar=" + json.dumps({"kind": "json_schema", "schema": schema}, separators=(",", ":")).replace(" ", "\\u0020")


def main():
    engine, model, *extra = sys.argv[1:]
    tok, vocab = load_tokenizer(model)
    prompt = tok.encode(PROMPT, parse_special=True)
    expected = [{"slot": "left", "payload": "a" * 2048}, {"slot": "right", "payload": "b" * 2048}]

    def command(kind, slot, doc, prefix=(), count=1024):
        ids = prompt + list(prefix)
        return (f"{kind} {slot} {count} temperature=0.7 top_p=0.95 top_k=20 seed={101 + slot} "
                f"{grammar(doc)} grammar_start={len(prompt)} " + ",".join(map(str, ids)))

    def check(ids, doc):
        text = text_of(tok, vocab, ids)
        assert json.loads(text) == doc, text

    with tempfile.TemporaryFile("w+") as log:
        proc = subprocess.Popen([engine, "--serve", "--model", model, *extra, "--batch", "2", "--grammar", "on"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        protocol = Protocol(proc)
        try:
            info = ""
            while True:
                line = protocol.line(timeout=300)
                if line.startswith("INFO "):
                    info += " " + line
                if line.startswith("READY "):
                    break
            assert "batch_slots=2" in info and "solo_any_slot=1" in info, info
            pair, finishes, overlap, width = protocol.pair([command("BGEN", i, doc) for i, doc in enumerate(expected)])
            assert overlap, "the requests never occupied both slots together"
            assert all(finishes.get(i) == "stop" for i in range(2)), finishes
            for i in range(2):
                check(pair[i], expected[i])
            print(f"independent schemas: ok; largest emitted segment {width} tokens")

            # Swap the schemas on reused slots, carrying an incomplete reply across both transitions.
            docs = expected[::-1]
            prefix = protocol.solo(command("SGEN", 0, docs[0], count=4))
            pair, finishes, overlap, _ = protocol.pair(
                [command("BGEN", 0, docs[0], prefix), command("BGEN", 1, docs[1])], cancel_first=True)
            assert finishes.get(0) == "cancel", finishes
            prefix += pair[0]
            tail = protocol.solo(command("SGEN", 0, docs[0], prefix))
            check(prefix + tail, docs[0])
            check(pair[1], docs[1])
            print("slot reuse and solo/batch/solo handover: ok")
            protocol.send("QUIT")
            proc.wait(timeout=120)
            assert proc.returncode == 0, proc.returncode
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-12000:], file=sys.stderr)
            raise
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=30)
        log.seek(0)
        content = log.read()
    resamples = [int(n) for n in re.findall(r"slot \d+: \d+ windows, (\d+) masked resamples", content)]
    assert len(resamples) >= 4 and sum(resamples) > 0, resamples
    print(f"batch masked resamples: {resamples}\ngrammar_batch_test: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
