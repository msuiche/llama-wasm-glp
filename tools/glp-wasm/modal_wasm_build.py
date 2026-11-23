"""Modal job: compile this GLP-patched llama.cpp to wasm for the browser demo.

Builds TWO variants from the same pinned source:
  - WebGPU (GGML_WEBGPU=ON, emdawnwebgpu port, ASYNCIFY): fast on GPU browsers
  - CPU (simd128, single-threaded): fallback for everything else

Outputs (Modal volume "llama-glp-wasm", dir llama-glp-wasm/):
  glp-demo.js / glp-demo.wasm              CPU       -> site: glp-wasm-loader.js / glp-demo.wasm
  glp-demo-webgpu.js / glp-demo-webgpu.wasm WebGPU   -> site: glp-webgpu-loader.js / glp-demo-webgpu.wasm
  vector.gguf    the GLP steering vector (reused from the volume if present;
                 otherwise uploaded from a local vector.gguf next to this file)

The source is pinned by LLAMA_REF (an immutable tag) — upstream drift broke a
previous build (a new llama.cpp graph reserve overflowed the 2 GB wasm heap),
so never build from a moving HEAD.

Usage: modal run modal_wasm_build.py
"""
import pathlib
import modal

HERE = pathlib.Path(__file__).resolve().parent

LLAMA_REPO = "https://github.com/msuiche/llama-wasm-glp"
# immutable tag on the glp-wasm branch; cut a new tag (glp-wasm-v2, ...) to
# update the build deliberately. Never build from a moving branch tip.
LLAMA_REF = "glp-wasm-v2"

EMSDK_VER = "4.0.10"
DAWN_VER = "v20261119.211954"
DAWN_PKG = f"https://github.com/google/dawn/releases/download/{DAWN_VER}/emdawnwebgpu_pkg-{DAWN_VER}.zip"

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

    assert LLAMA_REF.startswith("glp-wasm-v"), "set LLAMA_REF to an immutable release tag"

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

    # 2. emdawnwebgpu package (pinned Dawn release) for the WebGPU variant
    subprocess.run(["curl", "-sL", "-o", "dawn.zip", DAWN_PKG], check=True, cwd="/work")
    subprocess.run(["unzip", "-q", "-o", "dawn.zip", "-d", "."], check=True, cwd="/work")
    port = "/work/emdawnwebgpu_pkg/emdawnwebgpu.port.py"
    assert os.path.exists(port), "emdawnwebgpu.port.py missing from the Dawn package"
    print("emdawnwebgpu ready", flush=True)

    # 3. clone the GLP fork at the immutable tag (no moving HEAD)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", LLAMA_REF, LLAMA_REPO, "llama.cpp"],
                   check=True, cwd="/work")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd="/work/llama.cpp",
                          capture_output=True, text=True, check=True).stdout.strip()
    content = open("/work/llama.cpp/src/models/deepseek41.cpp").read()
    assert "chunk_a = (n_tok + 63)/64" in content, "deepseek41 max fix missing"
    print(f"llama.cpp {LLAMA_REPO} @ {LLAMA_REF} ({head[:12]}, GLP verified)", flush=True)

    # 4. build both variants
    for variant, webgpu in [("cpu", False), ("webgpu", True)]:
        suffix = "-webgpu" if webgpu else ""
        bdir = f"build-wasm{suffix}"
        cmake = [
            "emcmake", "cmake", "-B", bdir,
            f"-DGGML_WEBGPU={'ON' if webgpu else 'OFF'}",
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
        if webgpu:
            cmake.append("-DEMDAWNWEBGPU_DIR=/work/emdawnwebgpu_pkg")
            # JSPI instead of ASYNCIFY: binaryen's Asyncify pass crashes
            # (UNREACHABLE at Asyncify.cpp:1146) on this build. JSPI needs
            # Chrome 137+; other browsers get the CPU fallback.
            cmake.append("-DGGML_WEBGPU_JSPI=ON")
        r = subprocess.run(cmake, cwd="/work/llama.cpp", env=env, capture_output=True, text=True)
        print(f"[{variant}] cmake stdout:", r.stdout[-1500:], flush=True)
        print(f"[{variant}] cmake stderr:", r.stderr[-4000:], flush=True)
        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, cmake)

        r = subprocess.run(["cmake", "--build", bdir, "--config", "Release", "-j", "16"],
                           cwd="/work/llama.cpp", env=env, capture_output=True, text=True)
        print(f"[{variant}] build stdout:", r.stdout[-1500:], flush=True)
        print(f"[{variant}] build stderr:", r.stderr[-4000:], flush=True)
        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, ["cmake", "--build"])

        libs = []
        for pat in [f"{bdir}/src/libllama.a", f"{bdir}/common/libcommon.a",
                    f"{bdir}/common/libllama-common.a",
                    f"{bdir}/ggml/src/libggml*.a",
                    f"{bdir}/ggml/src/ggml-webgpu/libggml*.a"]:
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
                "-o", f"/work/glp-demo{suffix}.js",
               ]
        if webgpu:
            # ggml-webgpu's INTERFACE link options don't propagate to a manual
            # em++ link, so pass the port + JSPI explicitly. JSPI_EXPORTS lists
            # every export that may suspend (emwgpuWaitAny during ctors / model
            # load / decode), otherwise V8 throws SuspendError.
            emcc += [f"--use-port={port}", "-s", "JSPI", "-fwasm-exceptions",
                     "-DGLP_WEBGPU",
                     "-s", "JSPI_EXPORTS=__wasm_call_ctors,_glp_load_model,_glp_load_vector,_glp_generate"]
        print(f"[{variant}] +", " ".join(emcc[:20]), "...", flush=True)
        r = subprocess.run(emcc, cwd="/work/llama.cpp", env=env, capture_output=True, text=True)
        print(f"[{variant}] emcc stdout:", r.stdout[-1500:], flush=True)
        print(f"[{variant}] emcc stderr:", r.stderr[-4000:], flush=True)
        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, emcc[:2])

    # 5. outputs to the volume (always deploy each .js with its own .wasm)
    out_dir = "/data/llama-glp-wasm"
    os.makedirs(out_dir, exist_ok=True)
    for f in ["/work/glp-demo.js", "/work/glp-demo.wasm",
              "/work/glp-demo-webgpu.js", "/work/glp-demo-webgpu.wasm"]:
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
