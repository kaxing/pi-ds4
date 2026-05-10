#!/usr/bin/env python3
"""Convert cyberneurova GGUF tensors that ds4 expects as raw F16 (currently Q8_0 or F32)
into F16, keeping all other tensors byte-identical. Routed experts (Q2_K) are passed
through; the engine-side dispatch handles them."""

import os, sys, re, time
import numpy as np
from gguf import GGUFReader, GGUFWriter
from gguf.constants import GGMLQuantizationType, GGUFValueType
from gguf.quants import dequantize

F16_PATTERNS = [re.compile(p) for p in [
    r'^token_embd\.weight$',
    r'^output_hc_fn\.weight$',
    r'^blk\.\d+\.hc_attn_fn\.weight$',
    r'^blk\.\d+\.hc_ffn_fn\.weight$',
    r'^blk\.\d+\.attn_compressor_(ape|gate|kv)\.weight$',
    r'^blk\.\d+\.indexer\.(attn_q_b|proj)\.weight$',
    r'^blk\.\d+\.indexer_compressor_(ape|gate|kv)\.weight$',
    r'^blk\.\d+\.ffn_gate_inp\.weight$',
]]

def needs_f16(name): return any(p.match(name) for p in F16_PATTERNS)

def field_value(field):
    """Return (vtype, value, sub_type) for a GGUFReader field, suitable for writer."""
    vtype = field.types[0]
    if vtype == GGUFValueType.STRING:
        return vtype, bytes(field.parts[field.data[0]]).decode('utf-8', errors='replace'), None
    if vtype == GGUFValueType.ARRAY:
        sub_type = field.types[1]
        if sub_type == GGUFValueType.STRING:
            vals = [bytes(field.parts[i]).decode('utf-8', errors='replace') for i in field.data]
        else:
            vals = []
            for i in field.data:
                p = field.parts[i]
                if hasattr(p, 'item'): vals.append(p.item())
                elif hasattr(p, 'tolist'):
                    t = p.tolist()
                    vals.append(t[0] if isinstance(t, list) and len(t) == 1 else t)
                else: vals.append(p)
        return vtype, vals, sub_type
    # Primitive scalar
    p = field.parts[field.data[0]]
    if hasattr(p, 'item'): return vtype, p.item(), None
    return vtype, p, None

SKIP_KEYS = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count', 'general.architecture'}

def main():
    src_path, dst_path = sys.argv[1], sys.argv[2]
    print(f"reading: {src_path}", flush=True)
    r = GGUFReader(src_path)

    arch_field = r.fields['general.architecture']
    arch = bytes(arch_field.parts[arch_field.data[0]]).decode()
    print(f"arch: {arch}, fields: {len(r.fields)}, tensors: {len(r.tensors)}", flush=True)

    w = GGUFWriter(dst_path, arch=arch, endianess=r.endianess)

    # Copy metadata
    for key in r.fields:
        if key in SKIP_KEYS: continue
        vtype, val, sub_type = field_value(r.fields[key])
        if vtype == GGUFValueType.STRING:
            w.add_string(key, val)
        elif vtype == GGUFValueType.ARRAY:
            w.add_array(key, val)
        else:
            w.add_key_value(key, val, vtype)
    print(f"copied {len(r.fields) - sum(1 for k in r.fields if k in SKIP_KEYS)} metadata fields", flush=True)

    # Tensor pass: convert or passthrough
    converted, copied = 0, 0
    t_start = time.time()
    for ti, t in enumerate(r.tensors):
        name = t.name
        src_type = GGMLQuantizationType(int(t.tensor_type))
        # Element shape: gguf stores dims little-endian (innermost first); numpy wants outermost first
        elem_shape = tuple(reversed(t.shape.tolist()))

        if needs_f16(name) and src_type != GGMLQuantizationType.F16:
            if src_type == GGMLQuantizationType.F32:
                f32 = np.array(t.data, dtype=np.float32, copy=False).reshape(elem_shape)
            else:
                f32 = dequantize(t.data, src_type)
                if f32.shape != elem_shape:
                    f32 = f32.reshape(elem_shape)
            f16 = f32.astype(np.float16)
            w.add_tensor(name, f16, raw_dtype=GGMLQuantizationType.F16)
            converted += 1
            if converted <= 5 or converted % 100 == 0:
                print(f"  [{ti+1}/{len(r.tensors)}] convert {name}: {src_type.name} -> F16 ({f16.nbytes/1e6:.1f} MB)", flush=True)
        else:
            # Pass through as raw bytes; writer infers element shape from byte shape + raw_dtype.
            raw = np.asarray(t.data)
            w.add_tensor(name, raw, raw_dtype=src_type)
            copied += 1
            if copied % 200 == 0 or ti == len(r.tensors) - 1:
                print(f"  [{ti+1}/{len(r.tensors)}] passthrough — {time.time()-t_start:.0f}s", flush=True)

    print(f"\nconverted={converted}, copied={copied}", flush=True)
    print("writing header...", flush=True); w.write_header_to_file()
    print("writing kv...", flush=True); w.write_kv_data_to_file()
    print("writing tensors...", flush=True); w.write_tensors_to_file(progress=True)
    w.close()
    sz = os.path.getsize(dst_path)
    print(f"done. {dst_path} ({sz/1e9:.1f} GB)", flush=True)

if __name__ == "__main__":
    main()
