// GLP browser demo entry: load a model + GLP vector, generate with a runtime
// alpha toggle (0 = stock, 1 = steered), exposed to JS via embind.
//
// Compiled to wasm with emscripten + emdawnwebgpu (WebGPU backend).
#include "llama.h"
#include "common.h"

#include <emscripten/bind.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static llama_model * g_model = nullptr;
static llama_context * g_ctx = nullptr;
static const llama_vocab * g_vocab = nullptr;
static int g_n_embd = 0;
static std::vector<float> g_glp_data; // GLP vector data (loaded once)

// load the model from a path (the file is fetched by JS and written to the FS)
bool glp_load_model(const std::string & model_path) {
    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    g_model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!g_model) return false;
    g_vocab = llama_model_get_vocab(g_model);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 2048; // KV cache must fit the wasm 2 GB heap cap alongside the model
    // small prefill batch: the graph reserve scales with n_batch (~325 MB at
    // 512 on current llama.cpp), and model mmap + KV already use most of the
    // 2 GB wasm heap. 128 keeps the compute buffer ~80 MB.
    cparams.n_batch = 128;
    g_ctx = llama_init_from_model(g_model, cparams);
    return g_ctx != nullptr;
}

// load the GLP vector data (directions per layer, F32) into memory
bool glp_load_vector(const std::string & vec_path) {
    // use the common control-vector loader so we get glp.* metadata handling
    std::vector<common_control_vector_load_info> infos = { {1.0f, vec_path} };
    auto cvec = common_control_vector_load(infos, llama_model_n_layer(g_model));
    if (cvec.n_embd == -1) return false;
    g_glp_data = std::move(cvec.data);
    g_n_embd = cvec.n_embd;
    return true;
}

// set the GLP steering strength (projective). alpha = 0 -> stock, >0 -> steered.
bool glp_set_alpha(float alpha) {
    if (!g_ctx || g_glp_data.empty()) return false;
    int err = llama_set_adapter_cvec(
        g_ctx,
        g_glp_data.data(),
        g_glp_data.size(),
        g_n_embd,
        alpha <= 0.0f ? LLAMA_CVEC_APPLY_ADD : LLAMA_CVEC_APPLY_PROJECT,
        alpha,
        1,
        llama_model_n_layer(g_model));
    return err == 0;
}

// optional per-token streaming callback: set by JS to receive each generated
// token as it's produced (so the page can render text as it generates).
static emscripten::val g_on_token = emscripten::val::undefined();
void glp_on_token(emscripten::val cb) {
    g_on_token = cb;
}

// generate text from a prompt, greedy, up to max_new tokens (stops at EOS).
// reasoning is kept in the output: the token callback tags each token with its
// <think>-block state and the page styles reasoning differently from the answer.
std::string glp_generate(const std::string & prompt, int max_new) {
    if (!g_ctx || !g_vocab) return "";

    // Qwen3 chat template: without it the model does base-style continuation
    // (echoes the prompt, never emits <think> tags or the <|im_end|> EOS).
    const std::string chat = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";

    const bool add_bos = llama_vocab_get_add_bos(g_vocab);
    std::vector<llama_token> tokens(chat.size() + 2);
    int n_tokens = llama_tokenize(g_vocab, chat.c_str(), chat.size(), tokens.data(), tokens.size(), add_bos, true);
    if (n_tokens < 0) return "";
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(g_ctx), true);

    std::string out;
    int n_cur = 0;
    llama_batch_ext * batch = llama_batch_ext_init(g_ctx);
    if (!batch) return "";

    // prefill
    for (int i = 0; i < n_tokens; i++) {
        const int idx = llama_batch_ext_add_token(batch, 0, tokens[i]);
        const llama_pos pos = i;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, n_tokens - 1, true);
    if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) {
        llama_batch_ext_free(batch);
        return "";
    }
    n_cur = n_tokens;

    // decode: use a proper sampler chain with a repetition penalty (greedy alone
    // loops badly on a small model). Greedy + repeat-penalty keeps output
    // deterministic but breaks the "same sentence repeated N times" degeneracy.
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
        llama_vocab_n_tokens(g_vocab), 64, 1.3f, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // track whether we're inside a <think> block so the page can style
    // reasoning tokens differently from answer tokens (visual comparison).
    bool in_think = false;
    for (int i = 0; i < max_new; i++) {
        const llama_token next = llama_sampler_sample(smpl, g_ctx, -1);
        if (next == llama_vocab_eos(g_vocab)) break;
        char buf[128];
        int len = llama_token_to_piece(g_vocab, next, buf, sizeof(buf), 0, true);
        if (len > 0) {
            out.append(buf, len);
            // toggle thinking state on <think> / </think> markers
            std::string tok(buf, len);
            if (tok.find("<think>") != std::string::npos) in_think = true;
            if (tok.find("</think>") != std::string::npos) in_think = false;
            // stream each token + its thinking flag to JS (live feedback)
            if (!g_on_token.isUndefined() && !g_on_token.isNull()) {
                g_on_token(tok, in_think);
            }
        }

        llama_batch_ext_clear(batch);
        const int idx = llama_batch_ext_add_token(batch, 0, next);
        const llama_pos pos = n_cur;
        llama_batch_ext_set_pos(batch, idx, &pos);
        llama_batch_ext_set_output_logits(batch, 0, true);
        if (llama_process(g_ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0) break;
        n_cur += 1;
    }
    llama_sampler_free(smpl);
    llama_batch_ext_free(batch);
    return out;
}

// WebGPU/JSPI builds: these entry points can suspend (emwgpuWaitAny during
// model load and decode), so embind must wrap them with WebAssembly.promising
// via emscripten::async() — synchronous wrappers throw SuspendError. The CPU
// build stays synchronous (also keeps Bun's JSCore, which lacks
// WebAssembly.promising, able to run the local test).
EMSCRIPTEN_BINDINGS(glp_demo) {
#ifdef GLP_WEBGPU
    emscripten::function("loadModel", &glp_load_model, emscripten::async());
    emscripten::function("loadVector", &glp_load_vector, emscripten::async());
    emscripten::function("generate", &glp_generate, emscripten::async());
#else
    emscripten::function("loadModel", &glp_load_model);
    emscripten::function("loadVector", &glp_load_vector);
    emscripten::function("generate", &glp_generate);
#endif
    emscripten::function("setAlpha", &glp_set_alpha);
    emscripten::function("onToken", &glp_on_token);
}

int main() { return 0; }
