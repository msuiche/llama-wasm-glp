// Control vector apply, checked against a real model's logits.
//
// Text-level comparison is too weak here: a broken apply can still emit fluent
// output, and greedy decoding hides small perturbations. So decode a fixed token
// sequence and compare logits directly. The important assertion is that alpha=0
// is BIT-IDENTICAL to no control vector at all -- the vLLM port of this shipped a
// bug where the alpha=0 control silently ran a different graph, which made the
// whole ablation look like it did nothing.
//
// Also exercises the loader's refusal paths, which must be hard failures: a
// projective direction applied additively is not an error at runtime, just wrong
// output, so there is nothing safe to fall back to.

#include "llama.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;
static void ok(const char * s)            { printf("  PASS  %s\n", s); }
static void no(const char * s)            { printf("  FAIL  %s\n", s); n_fail++; }
static void expect(bool c, const char * s) { c ? ok(s) : no(s); }

static std::vector<float> logits_for(
        llama_model * model, llama_context * ctx,
        const std::vector<llama_token> & toks) {
    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = i + 1 == toks.size();
    }
    batch.n_tokens = (int32_t) toks.size();

    std::vector<float> out;
    if (llama_decode(ctx, batch) == 0) {
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * l = llama_get_logits_ith(ctx, (int32_t) toks.size() - 1);
        if (l) { out.assign(l, l + n_vocab); }
    }
    llama_batch_free(batch);
    return out;
}

static bool identical(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return false;
    return memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

static double maxdiff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return -1.0;
    double m = 0.0;
    for (size_t i = 0; i < a.size(); i++) { m = std::max(m, (double) std::fabs(a[i]-b[i])); }
    return m;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        printf("usage: %s <model.gguf> <cvec-dir>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string dir        = argv[2];

    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr); // quiet the loader

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { printf("  FAIL  could not load %s\n", model_path.c_str()); return 1; }

    auto cparams = llama_context_default_params();
    cparams.n_ctx = 64;
    cparams.n_batch = 64;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { printf("  FAIL  could not create context\n"); return 1; }

    const int n_embd  = llama_model_n_embd(model);
    const int n_layer = llama_model_n_layer(model);
    printf("test-cvec-model: n_embd=%d n_layer=%d\n", n_embd, n_layer);

    const std::vector<llama_token> toks = {1, 4, 9, 16, 25};

    auto load = [&](const char * name) {   // as --control-vector: strength unset
        std::vector<common_control_vector_load_info> in;
        in.push_back({1.0f, dir + "/" + name, false});
        return common_control_vector_load(in);
    };
    auto load_scaled = [&](const char * name, float s) {  // as --control-vector-scaled
        std::vector<common_control_vector_load_info> in;
        in.push_back({s, dir + "/" + name, true});
        return common_control_vector_load(in);
    };
    auto set = [&](const common_control_vector_data & cv, float alpha) {
        return llama_set_adapter_cvec_ex(ctx, cv.data.data(), cv.data.size(),
                cv.n_embd, cv.mode, alpha, 1, n_layer) == 0;
    };
    auto clear = [&]() {
        llama_set_adapter_cvec_ex(ctx, nullptr, 0, n_embd, LLAMA_CVEC_APPLY_ADD, 1.0f, 1, n_layer);
    };

    const auto base = logits_for(model, ctx, toks);
    expect(!base.empty(), "baseline decode produced logits");

    // ---- mode is read from the file, not guessed ---------------------------
    const auto proj = load("cv-project.gguf");
    expect(proj.n_embd == n_embd && proj.mode == LLAMA_CVEC_APPLY_PROJECT,
           "project file loads with mode=project");
    expect(std::fabs(proj.alpha - 4.0f) < 1e-6,
           "alpha taken from dspark.alpha_default");

    const auto add = load("cv-add.gguf");
    expect(add.n_embd == n_embd && add.mode == LLAMA_CVEC_APPLY_ADD,
           "file without dspark.mode defaults to add");

    // ---- alpha = 0 must be bit-identical to no vector at all --------------
    expect(set(proj, 0.0f), "project apply accepted at alpha=0");
    const auto a0 = logits_for(model, ctx, toks);
    expect(identical(a0, base), "project alpha=0 is bit-identical to baseline");

    // ---- and the op is actually in the graph ------------------------------
    expect(set(proj, 1.0f), "project apply accepted at alpha=1");
    const auto a1 = logits_for(model, ctx, toks);
    expect(!identical(a1, base), "project alpha=1 changes logits");
    printf("        max|logit delta| at alpha=1: %.6f\n", maxdiff(a1, base));

    expect(set(proj, 4.0f), "project apply accepted at alpha=4");
    const auto a4 = logits_for(model, ctx, toks);
    expect(!identical(a4, a1), "alpha=4 differs from alpha=1");
    expect(maxdiff(a4, base) > maxdiff(a1, base),
           "larger alpha moves logits further from baseline");
    printf("        max|logit delta| at alpha=4: %.6f\n", maxdiff(a4, base));

    // ---- repeatable --------------------------------------------------------
    expect(set(proj, 1.0f), "re-apply at alpha=1");
    expect(identical(logits_for(model, ctx, toks), a1), "project apply is deterministic");

    // ---- clearing restores the baseline exactly ----------------------------
    clear();
    expect(identical(logits_for(model, ctx, toks), base), "clearing the vector restores baseline");

    // ---- the additive path is untouched ------------------------------------
    expect(set(add, 1.0f), "additive apply accepted");
    const auto ad = logits_for(model, ctx, toks);
    expect(!identical(ad, base), "additive vector changes logits");
    expect(!identical(ad, a1), "additive and projective differ on the same directions");
    clear();

    // ---- project and add are genuinely different operations ---------------
    // Same direction data, both modes, alpha/strength 1. If these matched, the
    // whole reason for the mode key would be gone.
    {
        auto as_add = proj;
        as_add.mode = LLAMA_CVEC_APPLY_ADD;
        set(as_add, 1.0f);
        const auto forced_add = logits_for(model, ctx, toks);
        expect(!identical(forced_add, a1),
               "same data added vs projected gives different logits");
        printf("        max|add - project| on identical data: %.6f\n", maxdiff(forced_add, a1));
        clear();
    }

    // ---- loader refusals ---------------------------------------------------
    expect(load("cv-badmode.gguf").n_embd == -1, "unknown dspark.mode is refused");
    expect(load("cv-badhook.gguf").n_embd == -1, "mismatched dspark.hook_point is refused");
    expect(load("cv-duplayer.gguf").n_embd == -1, "two directions for one layer are refused");

    // A file whose tensor names disagree with dspark.layer_ids_zero_based was
    // produced by a broken exporter. The names are what get applied, so it would
    // steer the wrong layers -- silently, since a one-layer shift degrades rather
    // than fails. Refuse it, and do not false-positive on the correct file.
    expect(load("cv-shifted.gguf").n_embd == -1,
           "tensor names disagreeing with declared layer ids are refused");
    expect(load("cv-consistent.gguf").n_embd == n_embd,
           "agreeing layer ids load fine");

    {   // two projective vectors do not sum into a projection axis
        std::vector<common_control_vector_load_info> in;
        in.push_back({1.0f, dir + "/cv-project.gguf"});
        in.push_back({1.0f, dir + "/cv-project2.gguf"});
        expect(common_control_vector_load(in).n_embd == -1, "two project vectors are refused");
    }
    {   // nor with an additive one
        std::vector<common_control_vector_load_info> in;
        in.push_back({1.0f, dir + "/cv-project.gguf"});
        in.push_back({1.0f, dir + "/cv-add.gguf"});
        expect(common_control_vector_load(in).n_embd == -1, "project + additive is refused");
    }
    {   // but additive vectors must still combine -- the check must not be broad
        std::vector<common_control_vector_load_info> in;
        in.push_back({1.0f, dir + "/cv-add.gguf"});
        in.push_back({1.0f, dir + "/cv-add.gguf"});
        expect(common_control_vector_load(in).n_embd == n_embd, "additive vectors still combine");
    }
    {   // strength must not be folded into projective data: alpha carries it
        const auto cv = load_scaled("cv-project.gguf", 2.5f);
        expect(cv.n_embd == n_embd && std::fabs(cv.alpha - 2.5f) < 1e-6,
               "strength becomes alpha in project mode");
        double sq = 0.0;
        for (int j = 0; j < n_embd; j++) { sq += (double) cv.data[j]*cv.data[j]; }
        expect(std::fabs(sq - 1.0) < 1e-3, "project data stays unit-norm despite strength");
    }

    // An explicitly requested alpha=1.0 must not be replaced by the file's
    // dspark.alpha_default of 4.0. Keying on "strength != 1.0" would lose this.
    expect(std::fabs(load_scaled("cv-project.gguf", 1.0f).alpha - 1.0f) < 1e-6,
           "explicit alpha=1.0 beats dspark.alpha_default");
    expect(std::fabs(load("cv-project.gguf").alpha - 4.0f) < 1e-6,
           "unset strength falls through to dspark.alpha_default");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf(n_fail == 0 ? "test-cvec-model: OK\n" : "test-cvec-model: %d FAILURES\n", n_fail);
    return n_fail == 0 ? 0 : 1;
}
