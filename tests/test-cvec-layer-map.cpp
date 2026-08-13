// Which graph layer does direction.N actually apply to?
//
// The 1-based naming invites two readings. common_control_vector_load_one()
// stores direction.N at data offset (N-1)*n_embd, which reads like "N-1 is the
// layer". But llama_adapter_cvec::apply() then fills tensors[il] from offset
// (il-1)*n_embd, so the two -1s cancel and tensors[il] holds direction.il --
// applied at graph layer il. Reading it the first way shifts an entire vector by
// one layer, which degrades quietly instead of failing: adjacent-layer refusal
// directions have cosine similarity 0.83-0.91, so the output stays plausible.
//
// So pin it by measurement rather than by reading. Put a direction in exactly one
// slot, then use the il_start/il_end range to isolate single layers: the only
// range that changes logits identifies the layer the direction landed on.

#include "llama.h"
#include "common.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int n_fail = 0;

static std::vector<float> logits_for(llama_model * model, llama_context * ctx,
        const std::vector<llama_token> & toks) {
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch b = llama_batch_init((int32_t) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        b.token[i] = toks[i]; b.pos[i] = (llama_pos) i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = i + 1 == toks.size();
    }
    b.n_tokens = (int32_t) toks.size();
    std::vector<float> out;
    if (llama_decode(ctx, b) == 0) {
        const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
        const float * l = llama_get_logits_ith(ctx, (int32_t) toks.size() - 1);
        if (l) out.assign(l, l + nv);
    }
    llama_batch_free(b);
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 3) { printf("usage: %s <model.gguf> <cvec-single.gguf>\n", argv[0]); return 1; }

    llama_backend_init();
    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);

    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { printf("  FAIL load model\n"); return 1; }

    auto cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 64;
    llama_context * ctx = llama_init_from_model(model, cp);

    const int n_layer = llama_model_n_layer(model);
    const int n_embd  = llama_model_n_embd(model);

    // the fixture carries a direction in exactly one slot; find which
    std::vector<common_control_vector_load_info> in;
    in.push_back({1.0f, argv[2]});
    const auto cv = common_control_vector_load(in);
    if (cv.n_embd != n_embd) { printf("  FAIL cvec n_embd %d != %d\n", cv.n_embd, n_embd); return 1; }

    int slot = -1;
    for (size_t il = 0; il*cv.n_embd < cv.data.size(); il++) {
        double sq = 0.0;
        for (int j = 0; j < cv.n_embd; j++) {
            const float v = cv.data[il*cv.n_embd + j]; sq += (double) v*v;
        }
        if (sq > 1e-12) {
            if (slot >= 0) { printf("  FAIL fixture has more than one populated slot\n"); return 1; }
            slot = (int) il;
        }
    }
    // data index `slot` is where direction.(slot+1) was stored
    const int tensor_n = slot + 1;
    printf("test-cvec-layer-map: n_layer=%d, fixture carries direction.%d\n", n_layer, tensor_n);

    const std::vector<llama_token> toks = {1, 4, 9, 16, 25};
    const auto base = logits_for(model, ctx, toks);

    // sweep single-layer ranges and record which one moves the logits
    int hit = -1, n_hits = 0;
    for (int il = 0; il < n_layer; il++) {
        llama_set_adapter_cvec_ex(ctx, cv.data.data(), cv.data.size(), cv.n_embd,
                LLAMA_CVEC_APPLY_PROJECT, 1.0f, il, il);
        const auto out = logits_for(model, ctx, toks);
        const bool moved = !(out.size() == base.size() &&
                memcmp(out.data(), base.data(), base.size()*sizeof(float)) == 0);
        if (moved) { hit = il; n_hits++; printf("  layer range [%d,%d] moves logits\n", il, il); }
    }

    if (n_hits != 1) {
        printf("  FAIL expected exactly one layer to respond, got %d\n", n_hits);
        n_fail++;
    } else {
        printf("  direction.%d applies at graph layer %d\n", tensor_n, hit);
        if (hit == tensor_n) {
            printf("  PASS  mapping is direction.N -> layer N\n");
            printf("        an exporter with 0-based layer ids must write direction.<id>,\n");
            printf("        NOT direction.<id+1>\n");
        } else if (hit == tensor_n - 1) {
            printf("  PASS  mapping is direction.N -> layer N-1\n");
            printf("        an exporter with 0-based layer ids must write direction.<id+1>\n");
        } else {
            printf("  FAIL  direction.%d landed on layer %d, neither N nor N-1\n", tensor_n, hit);
            n_fail++;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf(n_fail == 0 ? "test-cvec-layer-map: OK\n" : "test-cvec-layer-map: FAILURES\n");
    return n_fail == 0 ? 0 : 1;
}
