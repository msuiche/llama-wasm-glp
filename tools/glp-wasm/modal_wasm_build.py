"""Modal job: compile this GLP-patched llama.cpp to wasm for the browser demo.

The GLP projective apply is built into this branch natively (stock refuses,
the vector steers). This compiles it to wasm via emscripten, so the demo runs
in the browser with a runtime GLP alpha toggle (0 = stock, >0 = steered).

The source is pinned by LLAMA_SHA — upstream drift broke a previous build
(a new llama.cpp graph reserve overflowed the 2 GB wasm heap), so never build
from a moving HEAD.

Usage: modal run modal_wasm_build.py

Outputs (Modal volume "llama-glp-wasm", dir llama-glp-wasm/):
  glp-demo.js    emscripten loader (deploy as glp-wasm-loader.js)
  glp-demo.wasm  the module       (deploy as glp-demo.wasm)
  vector.gguf    the GLP steering vector (reused from the volume if present;
                 otherwise uploaded from a local vector.gguf next to this file)
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).resolve().parent

LLAMA_REPO = "https://github.com/msuiche/llama.cpp"
LLAMA_SHA = "PIN_ME"  # commit of the glp-wasm branch to build

EMSDK_VER = "4.0.10"

app = modal.App("llama-glp-wasm")
vol = modal.Volume.from_name("llama-glp-wasm", create_if_missing=True)

img = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "cmake", "build-essential", "curl", "git-lfs",
                 "nodejs", "npm", "python3-dev", "ninja-build", "unzip", "xz-utils")
    .pip_install("huggingface_hub", "hf_transfer")
)
if (HERE / "vector.gguf").exists():
    img = img.add_local_file(HERE / "vector.gguf", "/work/vector.gguf", copy=True)

ENV = {"HF_HOME": "/data/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"}


@app.function(image=img, volumes={"/data": vol}, timeout=6 * 3600, env=ENV, cpu=16, memory=65536)
def build():
    import glob
    import os
    import shutil
    import subprocess

    assert LLAMA_SHA != "PIN_ME", "set LLAMA_SHA to the glp-wasm branch commit"

    os.makedirs("/work", exist_ok=True)
    os.chdir("/work")

    # 1. emscripten (pinned version)
    if not os.path.isdir("/work/emsdk"):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/emscripten-core/emsdk.git"], check=True, cwd="/work")
    subprocess.run(["./emsdk", "install", EMSDK_VER], check=True, cwd="/work/emsdk")
    subprocess.run(["./emsdk", "activate", EMSDK_VER], check=True, cwd="/work/emsdk")
    env = dict(os.environ)
    env["PATH"] = f"/work/emsdk:/work/emsdk/upstream/emscripten:{env['PATH']}"
    env["EMSDK"] = "/work/emsdk"
    print("emscripten ready", flush=True)

    # 2. clone the GLP fork at the pinned commit (fetch by SHA: no moving HEAD)
    os.makedirs("/work/llama.cpp", exist_ok=True)
    subprocess.run(["git", "init"], check=True, cwd="/work/llama.cpp")
    subprocess.run(["git", "remote", "add", "origin", LLAMA_REPO], check=True, cwd="/work/llama.cpp")
    subprocess.run(["git", "fetch", "--depth", "1", "origin", LLAMA_SHA], check=True, cwd="/work/llama.cpp")
    subprocess.run(["git", "checkout", "FETCH_HEAD"], check=True, cwd="/work/llama.cpp")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd="/work/llama.cpp",
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == LLAMA_SHA, f"expected {LLAMA_SHA}, got {head}"
    content = open("/work/llama.cpp/src/models/deepseek41.cpp").read()
    assert "chunk_a = (n_tok + 63)/64" in content, "deepseek41 max fix missing"
    print(f"llama.cpp {LLAMA_REPO} @ {LLAMA_SHA[:12]} (GLP verified)", flush=True)

    # 3. configure: CPU-only wasm (WebGPU is a follow-up)
    cmake = [
        "emcmake", "cmake", "-B", "build-wasm",
        "-DGGML_WEBGPU=OFF",
        "-DGGML_NATIVE=OFF",
        "-DLLAMA_BUILD_COMMON=ON",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_UI_GZIP=OFF",
        "-DLLAMA_WASM_MEM64=OFF",
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    r = subprocess.run(cmake, cwd="/work/llama.cpp", env=env, capture_output=True, text=True)
    print("cmake stdout:", r.stdout[-2000:], flush=True)
    print("cmake stderr:", r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, cmake)

    # 4. build the llama.cpp library (not a cli target — we link a custom main)
    r = subprocess.run(["cmake", "--build", "build-wasm", "--config", "Release", "-j", "16"],
                       cwd="/work/llama.cpp", env=env, capture_output=True, text=True)
    print("build stdout:", r.stdout[-2000:], flush=True)
    print("build stderr:", r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, ["cmake", "--build"])

    # 5. compile the GLP demo main (from the repo) against the built libraries
    libs = []
    for pat in ["build-wasm/src/libllama.a", "build-wasm/common/libcommon.a",
                "build-wasm/common/libllama-common.a",
                "build-wasm/ggml/src/libggml*.a"]:
        libs.extend(sorted(glob.glob(f"/work/llama.cpp/{pat}")))
    emcc = ["/work/emsdk/upstream/emscripten/em++",
            "-O3", "-std=c++17", "-msimd128",
            "-I", "include", "-I", "ggml/include", "-I", "common",
            "/work/llama.cpp/tools/glp-wasm/glp_wasm_main.cpp",
            *libs,
            "--bind",
            "-s", "WASM_BIGINT=1",
            "-s", "ALLOW_MEMORY_GROWTH=1",
            "-s", "EXPORTED_RUNTIME_METHODS=['FS','callMain','ccall','cwrap']",
            "-s", "FORCE_FILESYSTEM=1",
            "-s", "MODULARIZE=1",
            "-s", "EXPORT_NAME='createGlpDemo'",
            "-o", "/work/glp-demo.js",
           ]
    print("+", " ".join(emcc[:20]), "...", flush=True)
    r = subprocess.run(emcc, cwd="/work/llama.cpp", env=env, capture_output=True, text=True)
    print("emcc stdout:", r.stdout[-2000:], flush=True)
    print("emcc stderr:", r.stderr[-2000:], flush=True)
    if r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, emcc[:2])

    # 6. outputs to the volume (always deploy the .js and .wasm together)
    out_dir = "/data/llama-glp-wasm"
    os.makedirs(out_dir, exist_ok=True)
    for f in ["/work/glp-demo.js", "/work/glp-demo.wasm"]:
        shutil.copy(f, out_dir + "/")
    if os.path.exists("/work/vector.gguf"):
        shutil.copy("/work/vector.gguf", out_dir + "/")
    elif not os.path.exists(out_dir + "/vector.gguf"):
        raise RuntimeError("no vector.gguf: place one next to this script, or pre-upload it to the volume")
    vol.commit()
    print("done; wasm outputs committed to volume", flush=True)
    print("outputs:", os.listdir(out_dir), flush=True)


@app.local_entrypoint()
def main():
    build.remote()
