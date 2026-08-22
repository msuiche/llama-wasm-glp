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
        w.add_string("glp.mode", mode)
        w.add_uint32("glp.spec_version", 1)
        w.add_float32("glp.alpha_default", alpha)
        if hook is not None:
            w.add_string("glp.hook_point", hook)
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
write("cv-add.gguf", mode=None)   # legacy additive: no glp.* keys at all


def write_dup_layer(name):
    """Two directions parsing to the same layer -- must be refused in project mode.

    gguf tensor names are unique, so the collision is expressed as "direction.2"
    and "direction.02": distinct names, same parsed layer index.
    """
    r = np.random.default_rng(11)
    w = gguf.GGUFWriter(f"{out_dir}/{name}", "controlvector")
    w.add_string("controlvector.model_hint", "llama")
    w.add_uint32("controlvector.layer_count", 2)
    w.add_string("glp.mode", "project")
    w.add_uint32("glp.spec_version", 1)
    w.add_float32("glp.alpha_default", 1.0)
    w.add_string("glp.hook_point", "residual_stream_post_layer")
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
    w.add_string("glp.mode", "project")
    w.add_uint32("glp.spec_version", 1)
    w.add_float32("glp.alpha_default", 1.0)
    w.add_string("glp.hook_point", "residual_stream_post_layer")
    v = r.standard_normal(n_embd).astype(np.float32)
    v /= np.linalg.norm(v)
    w.add_tensor(f"direction.{slot}", v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  {name}")


write_single("cv-single3.gguf", 3)


def write_shifted(name, offset=1):
    """Tensor names disagreeing with glp.layer_ids_zero_based.

    Reproduces a real exporter bug: directions derived at layers 1..3 written as
    direction.2..4 while the metadata still declared 1,2,3. The names are what get
    applied, so the whole stack lands one layer late -- which does not fail, it
    degrades, because adjacent layers' directions are highly correlated. Must be
    refused.
    """
    r = np.random.default_rng(5)
    layers = [1, 2, 3]
    w = gguf.GGUFWriter(f"{out_dir}/{name}", "controlvector")
    w.add_string("controlvector.model_hint", "llama")
    w.add_uint32("controlvector.layer_count", len(layers))
    w.add_string("glp.mode", "project")
    w.add_uint32("glp.spec_version", 1)
    w.add_float32("glp.alpha_default", 1.0)
    w.add_string("glp.hook_point", "residual_stream_post_layer")
    w.add_string("glp.layer_ids_zero_based", ",".join(str(l) for l in layers))
    for l in layers:
        v = r.standard_normal(n_embd).astype(np.float32)
        v /= np.linalg.norm(v)
        w.add_tensor(f"direction.{l + offset}", v)   # the bug
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  {name}")


write_shifted("cv-shifted.gguf")


def write_consistent(name):
    """Same as cv-shifted but correct, so the check is shown not to false-positive."""
    r = np.random.default_rng(5)
    layers = [1, 2, 3]
    w = gguf.GGUFWriter(f"{out_dir}/{name}", "controlvector")
    w.add_string("controlvector.model_hint", "llama")
    w.add_uint32("controlvector.layer_count", len(layers))
    w.add_string("glp.mode", "project")
    w.add_uint32("glp.spec_version", 1)
    w.add_float32("glp.alpha_default", 1.0)
    w.add_string("glp.hook_point", "residual_stream_post_layer")
    w.add_string("glp.layer_ids_zero_based", ",".join(str(l) for l in layers))
    for l in layers:
        v = r.standard_normal(n_embd).astype(np.float32)
        v /= np.linalg.norm(v)
        w.add_tensor(f"direction.{l}", v)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"  {name}")


write_consistent("cv-consistent.gguf")
