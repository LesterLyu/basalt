"""Pack the DeepSeek V4.1 Flash 1017 Maya XS2 drafter as a Basalt companion.

The reference pack provides DFlash metadata and the destination tensor schema.
Only the draft's weights are written; the target pack is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_file import ModelFile, ModelWriter


N_LAYERS, N_EXPERTS, HIDDEN, FF = 3, 128, 5120, 2304
IQ2_XXS = 16
ENCODINGS = {"F32": (0, 1, 4), "BF16": (30, 1, 2), "Q8_0": (8, 32, 34), "IQ2_XXS": (IQ2_XXS, 256, 66)}
DENSE_NAMES = {
    "attn.attn_sink": "attn_sinks.weight", "attn.kv_norm.weight": "attn_kv_a_norm.weight",
    "attn.q_norm.weight": "attn_q_a_norm.weight", "attn.wkv.weight": "attn_kv.weight",
    "attn.wo_a.weight": "attn_output_a.weight", "attn.wo_b.weight": "attn_output_b.weight",
    "attn.wq_a.weight": "attn_q_a.weight", "attn.wq_b.weight": "attn_q_b.weight",
    "attn_norm.weight": "attn_norm.weight", "ffn.gate.bias": "exp_probs_b.bias",
    "ffn.gate.weight": "ffn_gate_inp.weight", "ffn_norm.weight": "ffn_norm.weight",
    "ffn.shared_experts.w1.weight": "ffn_gate_shexp.weight",
    "ffn.shared_experts.w3.weight": "ffn_up_shexp.weight",
    "ffn.shared_experts.w2.weight": "ffn_down_shexp.weight",
    **{f"hc_{group}_{part}": f"hc_{group}_{part}.weight"
       for group in ("attn", "ffn") for part in ("base", "fn", "scale")},
}
GLOBAL_NAMES = {
    "mtp.0.main_norm.weight": "enc.output_norm.weight", "mtp.0.main_proj.weight": "fc.weight",
    "mtp.2.confidence_head.proj.weight": "conf_proj.weight", "mtp.2.norm.weight": "output_norm.weight",
    "mtp.2.markov_head.embed.weight": "markov_w1.weight", "mtp.2.markov_head.head.weight": "markov_w2.weight",
}


def get_entries(manifest: Path, size: int):
    desc = json.loads(manifest.read_bytes())
    if (desc.get("experts"), desc.get("dense")) != ("IQ2_XXS", "Q8_0"):
        raise ValueError("expected the Maya XS2 IQ2_XXS/Q8_0 manifest")
    entries = {}
    end = 0
    for name, encoding, offset, n, shape in sorted(desc["tensors"], key=lambda t: t[2]):
        if (encoding not in ENCODINGS or not isinstance(name, str) or not shape
                or any(type(dim) is not int or dim <= 0 for dim in shape)
                or type(offset) is not int or type(n) is not int or offset < end or n <= 0 or n > size - offset
                or name in entries):
            raise ValueError(f"invalid or overlapping XS2 tensor: {name}")
        tid, block, block_bytes = ENCODINGS[encoding]
        if shape[-1] % block or math.prod(shape) // block * block_bytes != n:
            raise ValueError(f"invalid XS2 tensor size: {name}")
        entries[name] = (tid, offset, n, shape)
        end = offset + n
    return entries


def bf16_to_f32(raw):
    """Widen BF16 to the F32 storage expected by Basalt's norms and router."""
    data = bytearray(len(raw) * 2)
    data[2::4], data[3::4] = raw[0::2], raw[1::2]
    return bytes(data)


def read_at(handle, offset, n):
    handle.seek(offset)
    data = handle.read(n)
    if len(data) != n:
        raise IOError("short read from XS2 weights")
    return data


def dense_parts(entries, schema, binary):
    names = dict(GLOBAL_NAMES)
    names.update({f"mtp.{layer}.{src}": f"blk.{layer}.{dst}"
                  for layer in range(N_LAYERS) for src, dst in DENSE_NAMES.items()})
    if set(names.values()) != set(schema):
        raise ValueError("the reference pack does not have the expected 1017 DFlash tensor schema")
    metas, parts = {}, []
    with binary.open("rb") as handle:
        for src, dst in names.items():
            tid, offset, n, shape = entries[src]
            target = schema[dst]
            if shape[::-1] != target["shape"]:
                raise ValueError(f"XS2 tensor geometry differs: {src}")
            if tid == 30 and target["type_id"] == 0:
                parts.append(bf16_to_f32(read_at(handle, offset, n)))
                tid, n = 0, 2 * n
            else:
                parts.append((binary, offset, n))
            if tid != target["type_id"] or n != target["bytes"]:
                raise ValueError(f"XS2 tensor encoding differs: {src}")
            metas[dst] = {"type_id": tid, "shape": target["shape"], "bytes": n}
    return metas, parts


def expert_ranges(entries):
    for layer in range(N_LAYERS):
        for expert in range(N_EXPERTS):
            for role, shape in (("w1", [FF, HIDDEN]), ("w3", [FF, HIDDEN]), ("w2", [HIDDEN, FF])):
                name = f"mtp.{layer}.ffn.experts.{expert}.{role}.weight"
                tid, offset, n, got = entries[name]
                if tid != IQ2_XXS or got != shape:
                    raise ValueError(f"invalid routed XS2 tensor: {name}")
                yield offset, n


def pack(reference, manifest, binary, output):
    if output.resolve() in {p.resolve() for p in (reference, manifest, binary)}:
        raise ValueError("output overlaps an input")
    if output.exists():
        raise FileExistsError(output)
    meta = json.loads(ModelFile(str(reference)).read("meta"))
    draft = meta["dspark"]
    kv = draft["gguf_kv"]
    if (meta.get("architecture"), kv.get("general.architecture"), kv.get("dflash.block_count"),
            kv.get("dflash.expert_count"), kv.get("dflash.target_layers")) != (
            "deepseek41", "dflash", N_LAYERS, N_EXPERTS, [37, 38, 39]):
        raise ValueError("expected a DeepSeek V4.1 Flash 1017 target pack")
    snapshot = (binary.stat().st_size, binary.stat().st_mtime_ns)
    entries = get_entries(manifest, snapshot[0])
    tensors, parts = dense_parts(entries, draft["tensors"], binary)
    ranges = list(expert_ranges(entries))
    blob = 3 * HIDDEN * FF // 256 * 66
    stride = N_EXPERTS * blob
    total = N_LAYERS * stride
    layout = f"# experts v5: layer gu_type d_type offset blob_bytes (n_expert {N_EXPERTS}, total {total})\n"
    layout += "".join(f"{i} {IQ2_XXS} {IQ2_XXS} {i*stride} {blob}\n" for i in range(N_LAYERS))

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with partial.open("xb"):
        pass
    writer = None
    try:
        writer = ModelWriter(str(partial))
        for entry, offset in zip(tensors.values(), writer.add_parts("dspark/gguf", parts), strict=True):
            entry["offset"] = offset
        with binary.open("rb") as handle:
            if writer.add_chunks("dspark/experts", (read_at(handle, off, n) for off, n in ranges)) != total:
                raise IOError("XS2 expert section size mismatch")
        writer.add_bytes("dspark/experts.layout", layout.encode())
        writer.add_bytes("meta", json.dumps({"format": "basalt", "version": 2, "architecture": "deepseek41-dspark",
            "name": "DeepSeek V4.1 Flash 1017 DSpark XS2", "dspark": {"gguf_kv": kv, "tensors": tensors},
            "source": "peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF", "license": meta.get("license")},
            separators=(",", ":")).encode())
        writer.close()
        if (binary.stat().st_size, binary.stat().st_mtime_ns) != snapshot:
            raise IOError("XS2 source changed during packing")
        os.replace(partial, output)
    finally:
        if writer is not None and not writer.f.closed:
            writer.f.close()
        partial.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("reference", "manifest", "weights", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    pack(args.reference, args.manifest, args.weights, args.out)


if __name__ == "__main__":
    main()
