#!/usr/bin/env python3
"""Write the control vector fixtures used by test-cvec-model.

Usage: gen-cvec-fixtures.py <out-dir> <n_embd> <n_layer>

Directions are unit-norm because projective mode needs them to be: the removal is
alpha*(h.d)d, which is quadratic in ||d||, so a non-unit direction silently over-
or under-ablates.
"""
import sys
import numpy as np
import gguf

out_dir, n_embd, n_layer = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])


def write(name, mode=None, hook="residual_stream_post_layer", alpha=4.0, seed=7):
    r = np.random.default_rng(seed)
    w = gguf.GGUFWriter(f"{out_dir}/{name}", "controlvector")
    w.add_string("controlvector.model_hint", "llama")
    w.add_uint32("controlvector.layer_count", n_layer)
    if mode is not None:
        w.add_string("dspark.mode", mode)
        w.add_uint32("dspark.spec_version", 1)
        w.add_float32("dspark.alpha_default", alpha)
        if hook is not None:
            w.add_string("dspark.hook_point", hook)
    for il in range(1, n_layer + 1):          # direction.N is 1-based
        v = r.standard_normal(n_embd).astype(np.float32)
        v /= np.linalg.norm(v)
        w.add_tensor(f"direction.{il}", v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  {name}")


write("cv-project.gguf", mode="project")
write("cv-project2.gguf", mode="project", seed=99)
write("cv-badmode.gguf", mode="sculpt")                    # must be refused
write("cv-badhook.gguf", mode="project", hook="attn_wo_output")  # must be refused
write("cv-add.gguf", mode=None)   # legacy additive: no dspark.* keys at all


def write_dup_layer(name):
    """Two directions parsing to the same layer -- must be refused in project mode.

    gguf tensor names are unique, so the collision is expressed as "direction.2"
    and "direction.02": distinct names, same parsed layer index.
    """
    r = np.random.default_rng(11)
    w = gguf.GGUFWriter(f"{out_dir}/{name}", "controlvector")
    w.add_string("controlvector.model_hint", "llama")
    w.add_uint32("controlvector.layer_count", 2)
    w.add_string("dspark.mode", "project")
    w.add_uint32("dspark.spec_version", 1)
    w.add_float32("dspark.alpha_default", 1.0)
    w.add_string("dspark.hook_point", "residual_stream_post_layer")
    for nm in ("direction.2", "direction.02"):
        v = r.standard_normal(n_embd).astype(np.float32)
        v /= np.linalg.norm(v)
        w.add_tensor(nm, v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  {name}")


write_dup_layer("cv-duplayer.gguf")


def write_single(name, slot):
    """One direction in one slot, for test-cvec-layer-map."""
    r = np.random.default_rng(slot)
    w = gguf.GGUFWriter(f"{out_dir}/{name}", "controlvector")
    w.add_string("controlvector.model_hint", "llama")
    w.add_uint32("controlvector.layer_count", 1)
    w.add_string("dspark.mode", "project")
    w.add_uint32("dspark.spec_version", 1)
    w.add_float32("dspark.alpha_default", 1.0)
    w.add_string("dspark.hook_point", "residual_stream_post_layer")
    v = r.standard_normal(n_embd).astype(np.float32)
    v /= np.linalg.norm(v)
    w.add_tensor(f"direction.{slot}", v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  {name}")


write_single("cv-single3.gguf", 3)
