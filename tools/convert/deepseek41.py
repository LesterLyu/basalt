"""Lossless DeepSeek/DSpark GGUF packaging into the .basalt v2 container."""
from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import model_sections as S
from model_file import HEADER_BYTES, SECTION_ALIGN, TENSOR_ALIGN, ModelFile, ModelWriter

CHUNK = 4 << 20
ROLES = ("gate", "up", "down")


def sources(paths):
    expanded = S.expand_shards(Path(paths[0])) if len(paths) == 1 else list(map(Path, paths))
    return S.SourceModel(expanded)


def metadata(model):
    kv = {}
    for g in model.files:
        for key, value in g.metadata.items():
            kv.setdefault(key, value)
    return kv


def check_contract(target, draft):
    for model, arch, layers, experts, top_k in ((target, "deepseek41", 40, 384, 6), (draft, "dflash", 3, 128, 3)):
        m = metadata(model)
        wanted = {"general.architecture": arch, f"{arch}.block_count": layers,
                  f"{arch}.embedding_length": 5120, f"{arch}.expert_feed_forward_length": 2304,
                  f"{arch}.expert_count": experts, f"{arch}.expert_used_count": top_k,
                  f"{arch}.expert_gating_func": 4, f"{arch}.expert_weights_scale": 1.5}
        for key, value in wanted.items():
            if m.get(key) != value:
                raise S.ConvertError(f"unsupported {key}: {m.get(key)!r}, expected {value!r}")
        for layer in range(layers):
            types = []
            for role in ROLES:
                name = f"blk.{layer}.ffn_{role}_exps.weight"
                entry = model.where.get(name)
                shape = [2304, 5120, experts] if role == "down" else [5120, 2304, experts]
                if entry is None or entry[1].shape != shape or entry[1].type_id not in ((39,) if arch == "dflash" else (19, 16)):
                    raise S.ConvertError(f"unsupported expert tensor: {name}")
                types.append(entry[1].type_id)
            if types[0] != types[1]:
                raise S.ConvertError(f"layer {layer}: gate/up types differ")
    tm, dm = metadata(target), metadata(draft)
    if tm.get("deepseek41.dspark.target_layers") != [37, 38, 39] or dm.get("dflash.target_layers") != [37, 38, 39]:
        raise S.ConvertError("target/DSpark anchor layers differ")
    if tm.get("deepseek41.dspark.block_size") != 5 or dm.get("dflash.block_size") != 5:
        raise S.ConvertError("unsupported DSpark block size")
    if "per_layer_token_embd.weight" not in target.where:
        raise S.ConvertError("missing Engram table")


def ranges(model, names):
    for name in names:
        g, t, path = model.where[name]
        yield path, g.data_start + t.offset, t.expected_bytes()


def expert_chunks(plan):
    """Interleave each expert's roles without mapping the input or staging whole layers."""
    with ExitStack() as stack:
        handles = {p: stack.enter_context(p.open("rb")) for p in plan.model.paths}
        for layer, _, _, _, blob, names in plan.layout:
            entries = list(ranges(plan.model, names))
            for expert in range(plan.n_expert):
                for path, offset, size in entries:
                    per = size // plan.n_expert
                    handle = handles[path]
                    handle.seek(offset + per * expert)
                    left = per
                    while left:
                        chunk = handle.read(min(CHUNK, left))
                        if not chunk:
                            raise IOError(f"{path.name}: short expert read")
                        yield chunk
                        left -= len(chunk)
            print(f"  expert layer {layer + 1}/{plan.n_layers}: {blob * plan.n_expert / (1 << 30):.3f} GiB", flush=True)


def dense_plan(model):
    order = {p: i for i, p in enumerate(model.paths)}
    names = [n for n in model.where if not S.is_expert(n)]
    names.sort(key=lambda n: (order[model.where[n][2]], model.where[n][0].data_start + model.where[n][1].offset))
    return names


def dense_size(model, names):
    pos = 0
    for _, _, size in ranges(model, names):
        pos += (-pos) % TENSOR_ALIGN + size
    return pos


def write_dense(writer, model, prefix):
    names = dense_plan(model)
    offsets = writer.add_parts(prefix + "gguf", ranges(model, names))
    return {name: {"type_id": model.where[name][1].type_id, "shape": model.where[name][1].shape,
                   "bytes": model.where[name][1].expected_bytes(), "offset": offset}
            for name, offset in zip(names, offsets)}


def write_pack(target, draft, output, tokenizer_files, token_map, license_text, name):
    """Write to an exclusively owned partial file; publish only after section checksums pass."""
    plans = [S.Experts(target), S.Experts(draft)]
    expected = sum(p.total + dense_size(p.model, dense_plan(p.model)) for p in plans)
    expected += sum(p.stat().st_size for p in tokenizer_files) + len(token_map) * 4 + HEADER_BYTES + 64 * SECTION_ALIGN
    output = Path(output)
    all_inputs = target.paths + draft.paths
    if output.resolve() in {p.resolve() for p in all_inputs}:
        raise S.ConvertError("output is a source GGUF")
    snapshots = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in all_inputs}
    if shutil.disk_usage(output.parent).free < expected + (2 << 30):
        raise S.ConvertError(f"conversion needs about {expected / (1 << 30):.2f} GiB plus 2 GiB free space")
    partial = output.with_name(output.name + ".partial")
    # Ownership is acquired before ModelWriter reopens it; never touch another conversion's partial.
    with partial.open("xb"):
        pass
    writer = None
    try:
        writer = ModelWriter(str(partial))
        for p in tokenizer_files:
            writer.add_file("tokenizer/" + p.name, str(p))
        writer.add_bytes("engram.map", struct.pack(f"<{len(token_map)}I", *token_map))
        views = []
        for plan, prefix in zip(plans, ("", "dspark/")):
            written = writer.add_chunks(prefix + "experts", expert_chunks(plan))
            if written != plan.total:
                raise IOError("expert section size mismatch")
            writer.add_bytes(prefix + "experts.layout", plan.layout_text())
            print(f"  {prefix}gguf: copying non-expert weights", flush=True)
            views.append({"gguf_kv": metadata(plan.model), "tensors": write_dense(writer, plan.model, prefix)})
        # The complete map is binary so native metadata readers can retain bounded JSON samples.
        views[0]["gguf_kv"].pop("deepseek41.engram.token_map", None)
        meta = {"format": "basalt", "version": 2, "name": name, "architecture": "deepseek41",
                **views[0], "dspark": views[1], "engram_map": {"section": "engram.map", "type": "U32", "count": len(token_map)},
                "sources": [{"file": p.name, "bytes": snapshots[p][0]} for p in all_inputs],
                "license": {"name": "MIT", "text": license_text}, "conversions": []}
        writer.add_bytes("meta", json.dumps(meta, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode())
        writer.close()
        for p, snap in snapshots.items():
            if (p.stat().st_size, p.stat().st_mtime_ns) != snap:
                raise IOError(f"source changed during conversion: {p.name}")
        packed = ModelFile(str(partial))
        for section in packed.names():
            print(f"  verify {section}", flush=True)
            if not packed.verify(section):
                raise IOError(f"checksum mismatch: {section}")
        os.replace(partial, output)
        return output.stat().st_size
    except BaseException:
        if writer is not None and not writer.f.closed:
            writer.f.close()
        partial.unlink(missing_ok=True)
        raise


def convert(args):
    if not args.dspark:
        raise S.ConvertError("deepseek41 needs --dspark DSpark.gguf")
    target, draft = sources(args.gguf), sources([args.dspark])
    check_contract(target, draft)
    output = Path(args.out)
    if output.exists() and not args.force:
        raise S.ConvertError(f"output exists: {output}; use --force to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    kv = metadata(target)
    token_map = kv.get("deepseek41.engram.token_map")
    compressed = kv.get("deepseek41.engram.compressed_vocab_size", 0)
    if not isinstance(token_map, list) or len(token_map) != 129280 or any(type(x) is not int or x < 0 or x >= compressed for x in token_map):
        raise S.ConvertError("invalid Engram token map")
    import tokenizer
    license_text = (Path(__file__).parent / "deepseek-mit-license.txt").read_text(encoding="utf-8")
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory() as td:
        tokenizer.extract_metadata(kv, td)
        files = sorted(p for p in (Path(td) / "tokenizer").iterdir() if p.is_file())
        size = write_pack(target, draft, output, files, token_map, license_text,
                          args.name or "DeepSeek-V4.1-Flash-1017 UD-IQ1_S")
    print(f"done: {size / (1 << 30):.3f} GiB, {time.monotonic() - t0:.1f} s -> {output}", flush=True)
