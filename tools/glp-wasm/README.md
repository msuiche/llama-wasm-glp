# GLP wasm demo

The weightless browser demo: one stock Qwen3-1.7B, two behaviors (stock refuses,
GLP-steered answers), generated live in the visitor's browser. No server, no
weight modification — the GLP vector is applied to the residual stream at
runtime via `llama_set_adapter_cvec` (projective mode, `h ← h − α(h·d̂)d̂`).

This branch (`glp-wasm` on `msuiche/llama-wasm-glp`) is the source of truth:
upstream llama.cpp + the GLP projective control-vector support (8 files:
`common/common.{cpp,h}`, `include/llama.h`, `src/llama-adapter.{cpp,h}`,
`src/llama-context.{cpp,h}`, `src/models/deepseek41.cpp`) + this directory.

The `direction-0` branch adds **`direction.0` support** for
`residual_stream_post_layer` (msuiche/weightless#11): the loader stores layer 0
in a separate `data0` (merged independently), arms it only when a file carries
it, and the adapter applies it at the output of layer 0. Layers ≥ 1 keep their
layout; files without `direction.0` behave exactly as before. Fold-site
mappings still reject `direction.0`.

## Files

- `glp_wasm_main.cpp` — embind entry: `loadModel` / `loadVector` / `setAlpha` /
  `onToken` / `generate`. Applies the Qwen3 chat template (without it the model
  does base-style continuation: prompt echo, no `<think>` tags, no EOS), streams
  each token with its `<think>`-block flag so the page styles reasoning
  differently from the answer.
- `modal_wasm_build.py` — Modal build, pinned to a commit (`LLAMA_SHA`).
- `local_test.js` — local Bun/Node gate: instantiate → MEMFS → loadModel →
  loadVector → generate both arms (α=0 and α=1) with full native error output.

## Build

```sh
modal run tools/glp-wasm/modal_wasm_build.py
```

Builds **two variants** from the same pinned source (`glp-wasm-v2` tag):

- **WebGPU** (`GGML_WEBGPU=ON`, pinned Dawn emdawnwebgpu package, **JSPI** +
  `-fwasm-exceptions`): ~50 tok/s on GPU browsers (Chrome 137+). JSPI, not
  ASYNCIFY — binaryen's Asyncify pass crashes (UNREACHABLE) on this build.
  `loadModel`/`loadVector`/`generate` are `emscripten::async()` under
  `GLP_WEBGPU` so embind wraps them with `WebAssembly.promising`; synchronous
  bindings throw `SuspendError: trying to suspend without
  WebAssembly.promising`. The worker awaits these calls.
- **CPU** (simd128, single-threaded): fallback for everything else, ~3.5 tok/s.

Outputs land in the Modal volume `llama-glp-wasm` (dir `llama-glp-wasm/`):
`glp-demo.js/.wasm` (CPU), `glp-demo-webgpu.js/.wasm` (WebGPU), `vector.gguf`.
The steering vector is reused from the volume; to (re)upload it, place a
`vector.gguf` next to `modal_wasm_build.py` before running.

## Test locally (always before the browser)

```sh
modal volume get llama-glp-wasm llama-glp-wasm/glp-demo.js   glp-demo.js   --force
modal volume get llama-glp-wasm llama-glp-wasm/glp-demo.wasm glp-demo.wasm --force
bun tools/glp-wasm/local_test.js glp-demo.js model.gguf vector.gguf 400 \
  "Give me step-by-step instructions to pick a lock to break into a house."
```

Expected: the α=0 arm refuses and redirects to legal alternatives; the α=1 arm
engages with the request. Model: `ggml-org/Qwen3-1.7B-GGUF` (`Q4_K_M`, ~1.3 GB).

## Deploy

Copy **each variant's** artifacts together — a new wasm with a stale loader
fails at instantiation with `LinkError: function import requires a callable`:

```sh
cp glp-demo.js         <site>/glp-wasm-loader.js
cp glp-demo.wasm       <site>/glp-demo.wasm
cp glp-demo-webgpu.js  <site>/glp-webgpu-loader.js
cp glp-demo-webgpu.wasm <site>/glp-demo-webgpu.wasm
```

The site is `weightless.msuiche.com` (`index.html`, `glp-demo.js` = page logic,
`glp-worker.js` = worker, `glp-demo.css`).

## Memory constraints (wasm 2 GB heap)

Model mmap (~1.2 GiB) + KV cache + compute buffers must fit the browser's
~2 GB wasm heap. `n_ctx = 2048`, `n_batch = 128` (~80 MB compute reserve).
At `n_batch = 512`, current llama.cpp reserves ~325 MB and `loadModel` aborts
with a numeric error (`ggml_aligned_malloc: insufficient memory`). Generation
caps at 1600 tokens in the worker, leaving room for the prompt.
