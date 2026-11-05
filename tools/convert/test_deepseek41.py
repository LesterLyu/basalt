"""Byte-preservation, layout and failed-conversion ownership checks on small GGUFs."""
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import deepseek41 as D
import gguf_writer as W
import model_sections as S
import tokenizer as T
from model_file import ModelFile


class DeepSeekPackTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original = {}
        self.target = self.model("target", 2, ("IQ1_S", "IQ2_XXS"))
        self.draft = self.model("draft", 1, ("MXFP4",))
        self.out = self.root / "model.basalt"
        self.tok = self.root / "tokenizer.json"
        self.tok.write_bytes(b'{"pre":"joyai-llm"}')

    def tearDown(self):
        self.target.close()
        self.draft.close()
        self.temp.cleanup()

    def model(self, label, layers, formats):
        g = W.GGUFWriter()
        g.add("general.architecture", "deepseek41" if label == "target" else "dflash")
        g.add("deepseek41.engram.token_map", list(range(100)))
        for l in range(layers):
            g.add_raw(f"blk.{l}.ffn_gate_inp.weight", [256, 3], "F32", bytes(256 * 3 * 4))
            for r, role in enumerate(D.ROLES):
                form = formats[(l + (role == "down")) % len(formats)]
                block_bytes = {"IQ1_S": 50, "IQ2_XXS": 66, "MXFP4": 136}[form]
                size = 256 * 256 // 256 * block_bytes
                raw = b"".join(bytes([(l * 31 + r * 7 + e) % 256]) * size for e in range(3))
                name = f"blk.{l}.ffn_{role}_exps.weight"
                g.add_raw(name, [256, 256, 3], form, raw)
                self.original[label, name] = raw
        raw = b"\x17\x80" * 512
        g.add_raw("per_layer_token_embd.weight", [256, 1], "F32", raw)
        self.original[label, "per_layer_token_embd.weight"] = raw
        with patch.dict(W.TYPE_IDS, {"IQ1_S": 19, "IQ2_XXS": 16, "MXFP4": 39}):
            path = g.write(self.root / (label + ".gguf"))
        return D.sources([path])

    def write(self):
        return D.write_pack(self.target, self.draft, self.out, [self.tok], list(range(100)), "MIT fixture", "fixture")

    def test_lossless_sections(self):
        self.write()
        m = ModelFile(str(self.out))
        meta = json.loads(m.read("meta"))
        self.assertNotIn("deepseek41.engram.token_map", meta["gguf_kv"])
        self.assertEqual(struct.unpack("<100I", m.read("engram.map")), tuple(range(100)))
        for section in m.names():
            self.assertTrue(m.verify(section), section)
        for label, model, prefix, view in (("target", self.target, "", meta), ("draft", self.draft, "dspark/", meta["dspark"])):
            data = m.read(prefix + "experts")
            for l, _, _, offset, blob, names in S.Experts(model).layout:
                for expert in range(3):
                    want = b"".join(self.original[label, n][expert * (len(self.original[label, n]) // 3):
                                                          (expert + 1) * (len(self.original[label, n]) // 3)] for n in names)
                    self.assertEqual(data[offset + expert * blob:offset + (expert + 1) * blob], want)
            dense = m.read(prefix + "gguf")
            for name, info in view["tensors"].items():
                if (label, name) in self.original:
                    self.assertEqual(dense[info["offset"]:info["offset"] + info["bytes"]], self.original[label, name])
        self.assertFalse(Path(str(self.out) + ".partial").exists())
        self.assertEqual(meta["conversions"], [])

    def test_existing_partial_not_touched(self):
        partial = Path(str(self.out) + ".partial")
        partial.write_bytes(b"another process")
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(partial.read_bytes(), b"another process")

    def test_source_output_refused(self):
        self.out = self.target.paths[0]
        original = self.out.read_bytes()
        with self.assertRaisesRegex(S.ConvertError, "source GGUF"):
            self.write()
        self.assertEqual(original, self.out.read_bytes())

    def test_bad_checksum_keeps_previous_output(self):
        self.out.write_bytes(b"previous output")
        with patch.object(ModelFile, "verify", return_value=False):
            with self.assertRaisesRegex(IOError, "checksum"):
                self.write()
        self.assertEqual(self.out.read_bytes(), b"previous output")
        self.assertFalse(Path(str(self.out) + ".partial").exists())

    def test_source_change_refused(self):
        chunks = D.expert_chunks
        def change(plan):
            yield from chunks(plan)
            if plan.model is self.target:
                with self.target.paths[0].open("ab") as f:
                    f.write(b"changed")
        with patch.object(D, "expert_chunks", change):
            with self.assertRaisesRegex(IOError, "source changed"):
                self.write()
        self.assertFalse(self.out.exists())

    def test_joyai_splits_and_roundtrip(self):
        tk = T.Tokenizer(list(T.BYTE_TO_UNICODE.values()), [], pre="joyai-llm")
        self.assertEqual(tk.pieces("hi123456中文+test\n"), ["hi", "123", "456", "中文", "+test", "\n"])
        for text in ("Hello world 1234567", "中文かなカナ abc", " e\u0301 😀\r\n", "\x00\t<>!!!"):
            self.assertEqual(tk.decode(tk.encode(text)), text)


if __name__ == "__main__":
    unittest.main()
